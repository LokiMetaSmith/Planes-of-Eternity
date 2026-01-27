struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(1) @binding(0)
var<uniform> camera: CameraUniform;

struct RealityUniform {
    blend_color: vec4<f32>,
    blend_params: vec4<f32>, // x = alpha, y = roughness, z = scale, w = distortion
};
@group(2) @binding(0)
var<uniform> reality: RealityUniform;

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
};

// Pseudo-random number generator
fn hash(p: vec2<f32>) -> f32 {
    var p2 = fract(p * vec2<f32>(123.34, 456.21));
    p2 = p2 + dot(p2, p2 + 45.32);
    return fract(p2.x * p2.y);
}

// 2D Noise
fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i + vec2<f32>(0.0,0.0)),
                   hash(i + vec2<f32>(1.0,0.0)), u.x),
               mix(hash(i + vec2<f32>(0.0,1.0)),
                   hash(i + vec2<f32>(1.0,1.0)), u.x), u.y);
}

// Fractional Brownian Motion
fn fbm(p: vec2<f32>, octaves: i32, roughness: f32) -> f32 {
    var v = 0.0;
    var a = 0.5;
    var shift = vec2<f32>(100.0);
    var p2 = p;
    // Manually unrolled loop for compatibility/performance or just fixed size
    // WGSL requires constant loops usually for unrolling, or dynamic loops are fine.
    // We will do a fixed loop for simplicity.
    for (var i = 0; i < 5; i = i + 1) {
        if (i >= octaves) { break; }
        v = v + a * noise(p2);
        p2 = p2 * 2.0 + shift;
        a = a * roughness;
    }
    return v;
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) height: f32, // Pass displaced height to fragment
    @location(2) world_pos: vec3<f32>,
};

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;

    let roughness = reality.blend_params.y;
    let scale = reality.blend_params.z;
    let distortion = reality.blend_params.w;

    var pos = model.position;

    // Generative Displacement
    // Use XZ plane for noise input
    let noise_val = fbm(pos.xz * scale, 4, roughness);

    // Apply displacement (Y-up)
    // Scale displacement by distortion and blend alpha (implied by params)
    pos.y = pos.y + noise_val * distortion;

    out.height = pos.y;
    out.world_pos = pos;
    out.clip_position = camera.view_proj * vec4<f32>(pos, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let base_texture = textureSample(t_diffuse, s_diffuse, in.tex_coords);
    let blend_color = reality.blend_color;
    let blend_alpha = reality.blend_params.x; // How much reality is "invading"

    // Generative Pattern
    let roughness = reality.blend_params.y;
    let scale = reality.blend_params.z;

    // Create a procedural pattern based on world position and height
    // "Stable Diffusion" style latent noise visualization
    let n = fbm(in.world_pos.xz * scale * 2.0, 3, roughness);

    // Create a color ramp based on height/noise
    let pattern_color = vec3<f32>(n, n * 0.8 + 0.2, n * 0.5 + 0.5); // Cyan-ish noise

    // Mix the procedural color with the "Archetype Color" (blend_color)
    let generative_look = mix(pattern_color, blend_color.rgb, 0.5);

    // Final blend: Base Texture vs Generative Reality
    // If blend_alpha is high, we see more of the generated reality

    // Add glowing edge effect based on height (Nanite wireframe-ish look?)
    let edge = step(0.9, fract(in.world_pos.x * 10.0)) + step(0.9, fract(in.world_pos.z * 10.0));
    let wireframe = clamp(edge, 0.0, 0.2);

    let final_gen_color = generative_look + vec3<f32>(wireframe);

    let result = mix(base_texture.rgb, final_gen_color, blend_alpha);

    return vec4<f32>(result, 1.0);
}
