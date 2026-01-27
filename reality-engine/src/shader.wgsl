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

struct InstanceInput {
    @location(2) model_matrix_0: vec4<f32>,
    @location(3) model_matrix_1: vec4<f32>,
    @location(4) model_matrix_2: vec4<f32>,
    @location(5) model_matrix_3: vec4<f32>,
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
    for (var i = 0; i < 5; i = i + 1) {
        if (i >= octaves) { break; }
        v = v + a * noise(p2);
        p2 = p2 * 2.0 + shift;
        a = a * roughness;
    }
    return v;
}

// Voronoi / Cellular Noise for SciFi/Tech look
fn voronoi(x: vec2<f32>) -> f32 {
    let p = floor(x);
    let f = fract(x);

    var min_dist = 1.0;

    for (var j = -1; j <= 1; j = j + 1) {
        for (var i = -1; i <= 1; i = i + 1) {
            let b = vec2<f32>(f32(i), f32(j));
            let r = b - f + hash(p + b); // hash returns 0..1, treat as random offset
            let d = dot(r, r);
            min_dist = min(min_dist, d);
        }
    }
    // Return inverted distance for "cells"
    return 1.0 - sqrt(min_dist);
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) height: f32, // Pass displaced height to fragment
    @location(2) world_pos: vec3<f32>,
};

@vertex
fn vs_main(model: VertexInput, instance: InstanceInput) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );

    var out: VertexOutput;
    out.tex_coords = model.tex_coords;

    let roughness = reality.blend_params.y;
    let scale = reality.blend_params.z;
    let distortion = reality.blend_params.w;

    // Transform vertex to world space using instance matrix
    var pos = (model_matrix * vec4<f32>(model.position, 1.0)).xyz;

    // Generative Displacement
    // Use XZ plane for noise input
    var noise_val = 0.0;

    // Hybrid blend: If roughness is high (> 0.6), start mixing in Voronoi (SciFi)
    // Otherwise stick to FBM (Organic)
    if (roughness > 0.6) {
        // SciFi / Tech
        // Snap coordinates for "digital" look
        let snap = 5.0;
        let p = floor(pos.xz * scale * snap) / snap;
        noise_val = voronoi(p);
        // Make it blocky
        noise_val = step(0.5, noise_val) * 0.5;
    } else {
        // Organic / Fantasy
        noise_val = fbm(pos.xz * scale, 4, roughness);
    }

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

    var pattern_color = vec3<f32>(0.0);

    if (roughness > 0.6) {
        // SciFi Look: Voronoi / Circuitry
        let v = voronoi(in.world_pos.xz * scale * 2.0);
        let circuit = step(0.95, v) + step(v, 0.05);
        pattern_color = mix(blend_color.rgb * 0.5, vec3<f32>(0.0, 1.0, 1.0), circuit);
    } else {
        // Fantasy Look: Organic FBM
        let n = fbm(in.world_pos.xz * scale * 2.0, 3, roughness);
        // Earthy tones + Magic
        pattern_color = mix(vec3<f32>(0.4, 0.3, 0.2), blend_color.rgb, n);
    }

    // Mix the procedural color with the "Archetype Color"
    let generative_look = mix(pattern_color, blend_color.rgb, 0.3);

    // Add glowing edge effect based on height (Nanite wireframe-ish look?)
    let edge = step(0.9, fract(in.world_pos.x * 5.0)) + step(0.9, fract(in.world_pos.z * 5.0));
    let wireframe = clamp(edge, 0.0, 0.2) * roughness; // Only wireframe if rough/techy

    let final_gen_color = generative_look + vec3<f32>(wireframe);

    let result = mix(base_texture.rgb, final_gen_color, blend_alpha);

    return vec4<f32>(result, 1.0);
}
