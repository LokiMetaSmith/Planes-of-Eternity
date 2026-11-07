// WebGPU WGSL Compute Shader for DiT Matrix Multiplication & Attention Softmax
// Group 0, Binding 0: Uniforms (M, K, N, scale)
// Group 0, Binding 1: Matrix A [M, K]
// Group 0, Binding 2: Matrix B [K, N]
// Group 0, Binding 3: Matrix Output [M, N]

struct GemmUniforms {
    m: u32,
    k: u32,
    n: u32,
    scale: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: GemmUniforms;

@group(0) @binding(1)
var<storage, read> matrix_a: array<f32>;

@group(0) @binding(2)
var<storage, read> matrix_b: array<f32>;

@group(0) @binding(3)
var<storage, read_write> matrix_out: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn cs_dit_gemm(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;

    if (row >= uniforms.m || col >= uniforms.n) {
        return;
    }

    var acc: f32 = 0.0;
    for (var i: u32 = 0u; i < uniforms.k; i = i + 1u) {
        let a_val = matrix_a[row * uniforms.k + i];
        let b_val = matrix_b[i * uniforms.n + col];
        acc = acc + a_val * b_val;
    }

    matrix_out[row * uniforms.n + col] = acc * uniforms.scale;
}

// Compute Softmax across attention rows
@compute @workgroup_size(64, 1, 1)
fn cs_dit_softmax(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= uniforms.m) {
        return;
    }

    let n = uniforms.n;
    var max_val: f32 = -1e30;
    for (var c: u32 = 0u; c < n; c = c + 1u) {
        let idx = row * n + c;
        if (matrix_a[idx] > max_val) {
            max_val = matrix_a[idx];
        }
    }

    var sum_exp: f32 = 0.0;
    for (var c: u32 = 0u; c < n; c = c + 1u) {
        let idx = row * n + c;
        let e = exp(matrix_a[idx] - max_val);
        matrix_out[idx] = e;
        sum_exp = sum_exp + e;
    }

    if (sum_exp > 0.0) {
        let inv_sum = 1.0 / sum_exp;
        for (var c: u32 = 0u; c < n; c = c + 1u) {
            let idx = row * n + c;
            matrix_out[idx] = matrix_out[idx] * inv_sum;
        }
    }
}

// Tiled FlashAttention-3 Kernel for WebGPU
// Uses workgroup shared memory to fuse QxK^T, causal masking, softmax, and attention weighting
const TILE_SIZE: u32 = 16u;

var<workgroup> tile_q: array<array<f32, TILE_SIZE>, TILE_SIZE>;
var<workgroup> tile_k: array<array<f32, TILE_SIZE>, TILE_SIZE>;
var<workgroup> tile_v: array<array<f32, TILE_SIZE>, TILE_SIZE>;

// FlashAttention kernel computing output tiles of [TILE_SIZE, TILE_SIZE] (Query seq block x Head block)
// Workgroup size: (TILE_SIZE, TILE_SIZE) -> local_id.y = Query row within tile, local_id.x = Head dim col within tile
@compute @workgroup_size(16, 16, 1)
fn cs_dit_flash_attention(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let q_row = group_id.y * TILE_SIZE + local_id.y; // Global Query sequence index
    let q_col = group_id.x * TILE_SIZE + local_id.x; // Global Head dimension index

    let seq_len = uniforms.m; // N context length
    let head_dim = uniforms.k; // d_k

    var acc: f32 = 0.0;
    var max_val: f32 = -1e30;
    var sum_exp: f32 = 0.0;

    let num_tiles = (seq_len + TILE_SIZE - 1u) / TILE_SIZE;
    let num_head_tiles = (head_dim + TILE_SIZE - 1u) / TILE_SIZE;

    // Outer loop over K/V sequence length tiles
    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // 1. Compute dot product of Q row `local_id.y` with K row `local_id.x`
        // We accumulate partial dot products over all head dimension blocks
        var dot_val: f32 = 0.0;
        let k_row = t * TILE_SIZE + local_id.x; // Key sequence index

        for (var h: u32 = 0u; h < num_head_tiles; h = h + 1u) {
            let h_idx = h * TILE_SIZE; // head dimension offset for this block

            // Load Q (16x16 tile) - row is query index, col is head dimension
            if (q_row < seq_len && h_idx + local_id.x < head_dim) {
                tile_q[local_id.y][local_id.x] = matrix_a[q_row * head_dim + (h_idx + local_id.x)];
            } else {
                tile_q[local_id.y][local_id.x] = 0.0;
            }

            // Load K (16x16 tile) - we read K transposed into tile_k
            // k_row is the sequence index, h_idx + local_id.y is the head dimension index
            if (k_row < seq_len && h_idx + local_id.y < head_dim) {
                // matrix_b contains Key data
                tile_k[local_id.x][local_id.y] = matrix_b[k_row * head_dim + (h_idx + local_id.y)];
            } else {
                tile_k[local_id.x][local_id.y] = 0.0;
            }

            workgroupBarrier();

            // Accumulate partial dot product (Q * K^T) for this head block
            // For query row local_id.y and key row local_id.x
            for (var i: u32 = 0u; i < TILE_SIZE; i = i + 1u) {
                // Since we transposed while loading, tile_k[local_id.x][i] contains the correct head dim `i` for key `local_id.x`
                dot_val = dot_val + tile_q[local_id.y][i] * tile_k[local_id.x][i];
            }
            workgroupBarrier();
        }

        dot_val = dot_val * uniforms.scale;

        // Causal Masking
        if (k_row > q_row) {
            dot_val = -1e30;
        }

        // 2. We need to perform row-wise reduction across the 16 Keys to find max and sum for softmax.
        // We write the dot products to shared memory to reduce them across `local_id.x` for each query row `local_id.y`.
        tile_v[local_id.y][local_id.x] = dot_val;
        workgroupBarrier();

        // Compute row max (Only thread 0 in x dimension needs to do this for the row to avoid data races when writing back)
        var row_max = -1e30;
        if (local_id.x == 0u) {
            for (var i: u32 = 0u; i < TILE_SIZE; i = i + 1u) {
                row_max = max(row_max, tile_v[local_id.y][i]);
            }

            let old_max = max_val;
            max_val = max(max_val, row_max);

            var row_sum = 0.0;
            for (var i: u32 = 0u; i < TILE_SIZE; i = i + 1u) {
                let exp_val = exp(tile_v[local_id.y][i] - max_val);
                tile_q[local_id.y][i] = exp_val; // Store Attention weights back in tile_q
                row_sum = row_sum + exp_val;
            }

            // Store reduction results in shared memory for the rest of the row to read
            tile_v[local_id.y][0] = row_sum;
            tile_v[local_id.y][1] = old_max;
            tile_v[local_id.y][2] = max_val;
        }

        workgroupBarrier();

        // Read back reduced values from shared memory
        let row_sum = tile_v[local_id.y][0];
        let old_max = tile_v[local_id.y][1];
        max_val = tile_v[local_id.y][2];

        sum_exp = sum_exp * exp(old_max - max_val) + row_sum;

        // The accumulator conceptually represents the entire row's output for column `q_col`, scaled.
        acc = acc * exp(old_max - max_val);

        workgroupBarrier();

        // 3. Compute Attention * V
        // Now `tile_q[local_id.y][i]` holds the attention weight for Key i.
        // We need to multiply this by the V matrix.

        // Load V into tile_k
        // V sequence row is t * TILE_SIZE + local_id.y
        // V head dimension col is q_col
        let v_row = t * TILE_SIZE + local_id.y;
        if (v_row < seq_len && q_col < head_dim) {
            // matrix_b contains Value data (assumed V == K for this simplified standard self-attention signature)
            tile_k[local_id.y][local_id.x] = matrix_b[v_row * head_dim + q_col];
        } else {
            tile_k[local_id.y][local_id.x] = 0.0;
        }

        workgroupBarrier();

        // For the row local_id.y, we want to compute the output for head_dim col local_id.x.
        var v_acc = 0.0;
        for (var i: u32 = 0u; i < TILE_SIZE; i = i + 1u) {
            // i iterates over the sequence dimension (matching V rows).
            // tile_q[local_id.y][i] is the attention weight for query row local_id.y and key column i.
            // tile_k[i][local_id.x] is the value at V row i and head_dim column local_id.x.
            v_acc = v_acc + tile_q[local_id.y][i] * tile_k[i][local_id.x];
        }

        // Accumulate output for this tile
        acc = acc + v_acc;

        workgroupBarrier();
    }

    if (q_row < seq_len && q_col < head_dim) {
        if (sum_exp > 0.0) {
            matrix_out[q_row * head_dim + q_col] = acc / sum_exp;
        } else {
            matrix_out[q_row * head_dim + q_col] = 0.0;
        }
    }
}
