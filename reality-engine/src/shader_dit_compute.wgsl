enable subgroups;
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

// Temporal FlashAttention-3 Kernel for WebGPU WGSL
// Utilizes subgroup operations and workgroup shared memory
const TILE_SIZE = 16u;
const HEAD_DIM = 64u;

@group(0) @binding(4)
var<storage, read> q_matrix: array<f32>;

@group(0) @binding(5)
var<storage, read> k_matrix: array<f32>;

@group(0) @binding(6)
var<storage, read> v_matrix: array<f32>;

@group(0) @binding(7)
var<storage, read_write> out_matrix: array<f32>;

var<workgroup> k_shared: array<f32, 1024>; // TILE_SIZE * HEAD_DIM (16 * 64)
var<workgroup> v_shared: array<f32, 1024>; // TILE_SIZE * HEAD_DIM (16 * 64)

@compute @workgroup_size(16, 16, 1)
fn cs_dit_flash_attention(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = global_id.x;
    let local_row = local_id.x;
    let local_col = local_id.y;
    let linear_local_id = local_row * 16u + local_col;

    let seq_len = uniforms.m;
    let is_valid = row < seq_len;

    var m_i = -1e30;
    var l_i = 0.0;

    var out_acc: array<f32, HEAD_DIM>;
    for (var d = 0u; d < HEAD_DIM; d++) {
        out_acc[d] = 0.0;
    }

    var q_local: array<f32, HEAD_DIM>;
    if (is_valid) {
        for (var d = 0u; d < HEAD_DIM; d++) {
            q_local[d] = q_matrix[row * HEAD_DIM + d];
        }
    }

    for (var start_col = 0u; start_col < seq_len; start_col += TILE_SIZE) {

        for (var idx = linear_local_id; idx < 1024u; idx += 256u) {
            let k_row = idx / HEAD_DIM;
            let k_col = idx % HEAD_DIM;
            let global_k_row = start_col + k_row;

            if (global_k_row < seq_len) {
                k_shared[idx] = k_matrix[global_k_row * HEAD_DIM + k_col];
                v_shared[idx] = v_matrix[global_k_row * HEAD_DIM + k_col];
            } else {
                k_shared[idx] = 0.0;
                v_shared[idx] = 0.0;
            }
        }

        workgroupBarrier();

        if (local_col == 0u) {
            var m_ij = m_i;
            var s_local: array<f32, TILE_SIZE>;

            for (var i = 0u; i < TILE_SIZE; i++) {
                let key_col = start_col + i;
                var s_val = 0.0;

                if (is_valid && key_col < seq_len) {
                    for (var d = 0u; d < HEAD_DIM; d++) {
                        s_val += q_local[d] * k_shared[i * HEAD_DIM + d];
                    }

                    s_val *= uniforms.scale;

                    if (key_col > row) {
                        s_val = -1e30;
                    }
                } else {
                    s_val = -1e30;
                }

                s_local[i] = s_val;

                if (s_val > m_ij) {
                    m_ij = s_val;
                }
            }

            var l_ij = 0.0;
            for (var i = 0u; i < TILE_SIZE; i++) {
                s_local[i] = exp(s_local[i] - m_ij);
                l_ij += s_local[i];
            }

            let m_new = max(m_i, m_ij);
            let exp_diff = exp(m_i - m_new);
            let exp_ij = exp(m_ij - m_new);
            let l_new = l_i * exp_diff + l_ij * exp_ij;

            for (var d = 0u; d < HEAD_DIM; d++) {
                var v_val = 0.0;
                for (var i = 0u; i < TILE_SIZE; i++) {
                    if ((start_col + i) < seq_len) {
                        v_val += s_local[i] * v_shared[i * HEAD_DIM + d];
                    }
                }

                out_acc[d] = out_acc[d] * exp_diff + v_val * exp_ij;
            }

            m_i = m_new;
            l_i = l_new;
        }

        workgroupBarrier();
    }

    if (is_valid && local_col == 0u) {
        if (l_i > 0.0) {
            for (var d = 0u; d < HEAD_DIM; d++) {
                out_matrix[row * HEAD_DIM + d] = out_acc[d] / l_i;
            }
        }
    }
}
