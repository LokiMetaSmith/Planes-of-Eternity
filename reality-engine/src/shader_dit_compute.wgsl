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
