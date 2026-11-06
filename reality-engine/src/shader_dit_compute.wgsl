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

@compute @workgroup_size(16, 16, 1)
fn cs_dit_flash_attention(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let row = global_id.x;
    let col = global_id.y;
    let seq_len = uniforms.m; // N context length
    let head_dim = uniforms.k; // d_k

    var acc: f32 = 0.0;
    var max_val: f32 = -1e30;
    var sum_exp: f32 = 0.0;

    let num_tiles = (seq_len + TILE_SIZE - 1u) / TILE_SIZE;

    let num_head_tiles = (head_dim + TILE_SIZE - 1u) / TILE_SIZE;

    // Outer loop over K/V sequence length tiles
    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        var dot_val: f32 = 0.0;
        let k_idx_base = t * TILE_SIZE + local_id.x; // Key sequence index

        // Compute Q * K^T block by iterating over head_dim tiles
        for (var h: u32 = 0u; h < num_head_tiles; h = h + 1u) {
            let h_idx = h * TILE_SIZE + local_id.y;

            // Load Q (size: TILE_SIZE x TILE_SIZE for this block)
            // local_id.x -> row within block, local_id.y -> col within block
            let q_row = group_id.x * TILE_SIZE + local_id.x;
            let q_col = h_idx;
            if (q_row < seq_len && q_col < head_dim) {
                tile_q[local_id.x][local_id.y] = matrix_a[q_row * head_dim + q_col];
            } else {
                tile_q[local_id.x][local_id.y] = 0.0;
            }

            // Load K^T (size: TILE_SIZE x TILE_SIZE for this block)
            // K is [seq_len, head_dim], so K^T is [head_dim, seq_len]
            // local_id.x -> row within K^T block (col in K), local_id.y -> col in K^T block (row in K)
            let k_row = t * TILE_SIZE + local_id.y;
            let k_col = h_idx;
            if (k_row < seq_len && k_col < head_dim) {
                // Notice we transpose while loading into tile_k
                tile_k[local_id.x][local_id.y] = matrix_b[k_row * head_dim + k_col];
            } else {
                tile_k[local_id.x][local_id.y] = 0.0;
            }

            workgroupBarrier();

            // Accumulate partial dot product for this head_dim tile
            for (var i: u32 = 0u; i < TILE_SIZE; i = i + 1u) {
                dot_val = dot_val + tile_q[local_id.x][i] * tile_k[i][local_id.x];
            }
            workgroupBarrier();
        }

        dot_val = dot_val * uniforms.scale;

        // Causal Masking (if key index > query index)
        let q_row_idx = group_id.x * TILE_SIZE + local_id.x;
        let current_k = k_idx_base;
        if (current_k > q_row_idx) {
            dot_val = -1e30;
        }

        // 2. Online Softmax update
        let old_max = max_val;
        max_val = max(max_val, dot_val);
        let exp_val = exp(dot_val - max_val);

        // Rescale previous sum and accumulate
        sum_exp = sum_exp * exp(old_max - max_val) + exp_val;

        // Rescale previous V accumulator
        acc = acc * exp(old_max - max_val);

        // 3. Compute Attention * V
        // For simplicity in this WebGPU port, we'll do a sequential V load here.
        // A fully optimized kernel would load V into shared memory via subgroup operations.
        // Here we read V directly from global memory to multiply with the scalar `exp_val`.
        if (current_k < seq_len && col < head_dim) {
            // Using matrix_a as V for this combined kernel structure
            // (Assuming Q, K, V are packed or this is a self-attention demo)
            let v_val = matrix_a[current_k * head_dim + col];
            acc = acc + exp_val * v_val;
        }
    }

    // Final normalization
    if (sum_exp > 0.0) {
        acc = acc / sum_exp;
    }

    if (row < seq_len && col < head_dim) {
        matrix_out[row * head_dim + col] = acc;
    }
}
