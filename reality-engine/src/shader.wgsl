struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(1) @binding(0)
var<uniform> camera: CameraUniform;

struct RealityUniform {
    proj1_pos_fid: vec4<f32>,
    proj1_params: vec4<f32>, // x=roughness, y=scale, z=distortion, w=archetype_id
    proj1_color: vec4<f32>,
    proj2_pos_fid: vec4<f32>,
    proj2_params: vec4<f32>,
    proj2_color: vec4<f32>,
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

struct BlendResult {
    weight1: f32,
    weight2: f32,
    total_strength: f32,
};

fn calculate_blend(pos: vec3<f32>) -> BlendResult {
    let dist1 = max(distance(pos, reality.proj1_pos_fid.xyz), 1.0);
    let dist2 = max(distance(pos, reality.proj2_pos_fid.xyz), 1.0);

    let strength1 = reality.proj1_pos_fid.w / dist1;
    let strength2 = reality.proj2_pos_fid.w / dist2;
    let total = strength1 + strength2;

    var w1 = 0.5;
    var w2 = 0.5;

    if (total > 0.0001) {
        w1 = strength1 / total;
        w2 = strength2 / total;
    } else {
        w1 = 0.0;
        w2 = 0.0;
    }

    return BlendResult(w1, w2, total);
}

fn get_displacement(xz: vec2<f32>, params: vec4<f32>) -> f32 {
    let roughness = params.x;
    let scale = params.y;
    // archetype_id is mixed. 0.0 = Fantasy, 1.0 = SciFi.
    let arch_mix = clamp(params.w, 0.0, 1.0);

    // Fantasy (FBM)
    var val_fantasy = fbm(xz * scale, 4, roughness);

    // SciFi (Voronoi)
    let snap = 5.0;
    let p = floor(xz * scale * snap) / snap;
    var val_scifi = voronoi(p);
    val_scifi = step(0.5, val_scifi) * 0.5;

    return mix(val_fantasy, val_scifi, arch_mix);
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) height: f32,
    @location(2) world_pos: vec3<f32>,
    @location(3) params: vec4<f32>,
    @location(4) color: vec4<f32>,
    @location(5) visibility: f32,
};

@vertex
fn vs_main(model: VertexInput, instance: InstanceInput) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );

    // Transform vertex to world space using instance matrix
    var pos = (model_matrix * vec4<f32>(model.position, 1.0)).xyz;

    // Calculate Blend
    let blend = calculate_blend(pos);

    // Mix Parameters
    let params = reality.proj1_params * blend.weight1 + reality.proj2_params * blend.weight2;
    let color = reality.proj1_color * blend.weight1 + reality.proj2_color * blend.weight2;
    let visibility = min(blend.total_strength, 1.0);

    let distortion = params.z;

    // Calculate Displacement
    let noise_val = get_displacement(pos.xz, params);

    // Apply displacement
    pos.y = pos.y + noise_val * distortion * visibility; // Fade displacement with visibility too? Or just keep it?
    // Logic: If visibility is 0 (far away), we should probably not distort to match base plane.
    // So multiplying by visibility is good.

    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    out.height = pos.y;
    out.world_pos = pos;
    out.clip_position = camera.view_proj * vec4<f32>(pos, 1.0);
    out.params = params;
    out.color = color;
    out.visibility = visibility;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let base_texture = textureSample(t_diffuse, s_diffuse, in.tex_coords);

    let roughness = in.params.x;
    let scale = in.params.y;
    let arch_mix = clamp(in.params.w, 0.0, 1.0);

    // Generate Patterns

    // Fantasy Pattern
    let n_fantasy = fbm(in.world_pos.xz * scale * 2.0, 3, roughness);
    let color_fantasy = mix(vec3<f32>(0.4, 0.3, 0.2), in.color.rgb, n_fantasy);

    // SciFi Pattern
    let v = voronoi(in.world_pos.xz * scale * 2.0);
    let circuit = step(0.95, v) + step(v, 0.05);
    let color_scifi = mix(in.color.rgb * 0.5, vec3<f32>(0.0, 1.0, 1.0), circuit);

    // Mix Patterns based on Archetype ID
    var pattern_color = mix(color_fantasy, color_scifi, arch_mix);

    // Mix with Archetype Color (General Tint)
    let generative_look = mix(pattern_color, in.color.rgb, 0.3);

    // Wireframe / Edge effect
    let edge = step(0.9, fract(in.world_pos.x * 5.0)) + step(0.9, fract(in.world_pos.z * 5.0));
    let wireframe = clamp(edge, 0.0, 0.2) * roughness;

    let final_gen_color = generative_look + vec3<f32>(wireframe);

    let result = mix(base_texture.rgb, final_gen_color, in.visibility);

    return vec4<f32>(result, 1.0);
}
