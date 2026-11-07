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

// FlashAttention-3 Kernel for WebGPU WGSL
// Tiled computation of temporal cross-attention: Softmax(Q * K^T / sqrt(d)) * V
// using var<workgroup> shared memory and causal masking.

struct FlashAttentionUniforms {
    seq_len_q: u32,
    seq_len_kv: u32,
    head_dim: u32,
    scale: f32,
};

@group(0) @binding(4)
var<uniform> fa_uniforms: FlashAttentionUniforms;

@group(0) @binding(5)
var<storage, read> Q: array<f32>;

@group(0) @binding(6)
var<storage, read> K: array<f32>;

@group(0) @binding(7)
var<storage, read> V: array<f32>;

@group(0) @binding(8)
var<storage, read_write> O: array<f32>;

const BLOCK_SIZE: u32 = 16u;
var<workgroup> shared_K: array<f32, 256>; // BLOCK_SIZE * BLOCK_SIZE
var<workgroup> shared_V: array<f32, 256>; // BLOCK_SIZE * BLOCK_SIZE

@compute @workgroup_size(16, 16, 1)
fn cs_dit_flash_attention(@builtin(global_invocation_id) global_id: vec3<u32>,
                          @builtin(local_invocation_id) local_id: vec3<u32>,
                          @builtin(workgroup_id) workgroup_id: vec3<u32>) {

    let q_idx = global_id.x; // Sequence index for query
    let d_idx = global_id.y; // Head dimension index

    let valid_q = q_idx < fa_uniforms.seq_len_q;
    let valid_d = d_idx < fa_uniforms.head_dim;

    var m_i = -1e30; // Max tracker
    var l_i = 0.0;   // Sum of exp tracker
    var acc = 0.0;   // Accumulated output

    let num_kv_blocks = (fa_uniforms.seq_len_kv + BLOCK_SIZE - 1u) / BLOCK_SIZE;

    // Load full query vector for the current query index into private memory if valid
    // (Assuming head_dim is small, e.g., 64 or 128, we compute the dot product inline)
    // To do this efficiently, each thread in the workgroup computing the same q_idx
    // needs the full Q vector. We just load it directly from global memory.

    for (var kv_block = 0u; kv_block < num_kv_blocks; kv_block = kv_block + 1u) {
        let kv_base_idx = kv_block * BLOCK_SIZE;

        // Causal mask check (assuming q and kv sequences align)
        if (kv_base_idx > q_idx) {
            break;
        }

        // Collaborative loading of K and V blocks into workgroup shared memory
        let load_seq_idx = kv_base_idx + local_id.x;
        let load_dim_idx = local_id.y;

        // Stride is BLOCK_SIZE (16), so index is load_seq_idx_local * 16 + load_dim_idx_local
        let shared_idx = local_id.x * BLOCK_SIZE + local_id.y;

        if (load_seq_idx < fa_uniforms.seq_len_kv && load_dim_idx < fa_uniforms.head_dim) {
            shared_K[shared_idx] = K[load_seq_idx * fa_uniforms.head_dim + load_dim_idx];
            shared_V[shared_idx] = V[load_seq_idx * fa_uniforms.head_dim + load_dim_idx];
        } else {
            shared_K[shared_idx] = 0.0;
            shared_V[shared_idx] = 0.0;
        }

        workgroupBarrier();

        if (valid_q && valid_d) {
            var m_ij = m_i;
            var l_ij = l_i;
            var acc_ij = acc;

            let num_keys_in_block = min(BLOCK_SIZE, fa_uniforms.seq_len_kv - kv_base_idx);

            for (var k_local = 0u; k_local < num_keys_in_block; k_local = k_local + 1u) {
                let kv_idx = kv_base_idx + k_local;
                if (kv_idx > q_idx) {
                    break; // Causal mask
                }

                // Compute dot product Q[q_idx, :] * K[kv_idx, :]
                var score = 0.0;
                for (var d = 0u; d < fa_uniforms.head_dim; d = d + 1u) {
                    let q_val = Q[q_idx * fa_uniforms.head_dim + d];
                    let k_val = K[kv_idx * fa_uniforms.head_dim + d];
                    score += q_val * k_val;
                }
                score *= fa_uniforms.scale;

                // Log-sum-exp updates
                let m_new = max(m_ij, score);
                let exp_score = exp(score - m_new);
                let exp_m_diff = exp(m_ij - m_new);

                l_ij = l_ij * exp_m_diff + exp_score;

                var v_val = 0.0;
                if (d_idx < BLOCK_SIZE) {
                     v_val = shared_V[k_local * BLOCK_SIZE + local_id.y];
                } else {
                     v_val = V[kv_idx * fa_uniforms.head_dim + d_idx]; // Fallback if head_dim > BLOCK_SIZE
                }

                acc_ij = acc_ij * exp_m_diff + exp_score * v_val;
                m_ij = m_new;
            }

            m_i = m_ij;
            l_i = l_ij;
            acc = acc_ij;
        }

        workgroupBarrier();
    }

    if (valid_q && valid_d) {
        if (l_i > 0.0) {
            O[q_idx * fa_uniforms.head_dim + d_idx] = acc / l_i;
        } else {
            O[q_idx * fa_uniforms.head_dim + d_idx] = 0.0;
        }
    }
}
