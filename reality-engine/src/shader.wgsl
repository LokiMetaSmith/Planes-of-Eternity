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
    global_offset: vec4<f32>,
};
@group(2) @binding(0)
var<uniform> reality: RealityUniform;

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;
@group(0) @binding(2)
var t_density: texture_3d<u32>;

fn ray_march_shadow(origin: vec3<f32>, direction: vec3<f32>) -> f32 {
    let max_dist = 60.0;
    let step_size = 0.5;
    var current_pos = origin + direction * 1.5; // Start bias
    var dist = 0.0;

    loop {
        if (dist > max_dist) { break; }

        // Map to Texture Space (World -64..64 -> 0..128)
        // Offset: X+64, Y+32, Z+64
        let tx = i32(floor(current_pos.x + 64.0));
        let ty = i32(floor(current_pos.y + 32.0));
        let tz = i32(floor(current_pos.z + 64.0));

        // Bounds Check
        if (tx >= 0 && tx < 128 && ty >= 0 && ty < 128 && tz >= 0 && tz < 128) {
            let val = textureLoad(t_density, vec3<i32>(tx, ty, tz), 0).r;
            if (val > 0u) {
                return 0.0; // Shadow
            }
        }

        current_pos += direction * step_size;
        dist += step_size;
    }
    return 1.0; // Lit
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
};

struct InstanceInput {
    @location(2) model_matrix_0: vec4<f32>,
    @location(3) model_matrix_1: vec4<f32>,
    @location(4) model_matrix_2: vec4<f32>,
    @location(5) model_matrix_3: vec4<f32>,
    @location(6) stability: f32,
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

// Domain Warping for organic, natural terrain
fn domain_warp(p: vec2<f32>) -> f32 {
    let q = vec2<f32>(
        fbm(p + vec2<f32>(0.0, 0.0), 3, 0.5),
        fbm(p + vec2<f32>(5.2, 1.3), 3, 0.5)
    );
    return fbm(p + 4.0 * q, 3, 0.5);
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
    let id = params.w;

    // Apply global offset (Geolocated map extraction)
    let pos = xz + reality.global_offset.xy;

    if (id < -0.5) {
        // Void (Flat)
        return 0.0;
    } else if (id < 0.5) {
        // Fantasy (FBM)
        return fbm(pos * scale, 4, roughness);
    } else if (id < 1.5) {
        // SciFi (Voronoi)
        let snap = 5.0;
        let p = floor(pos * scale * snap) / snap;
        var val = voronoi(p);
        return step(0.5, val) * 0.5;
    } else if (id < 2.5) {
        // Horror (Ridged / Jagged)
        // High frequency, spiky
        let n = fbm(pos * scale * 2.0, 5, roughness + 0.2);
        return (1.0 - abs(n * 2.0 - 1.0)) * 0.8;
    } else if (id < 3.5) {
        // Toon (Stepped)
        let n = fbm(pos * scale, 3, 0.5); // Smooth base
        // Quantize
        let steps = 4.0;
        return floor(n * steps) / steps;
    } else if (id < 4.5) {
        // HyperNature (Domain Warp + Erosion)
        let n = domain_warp(pos * scale * 0.5);
        // Create "canyons" or "rivers" by using inverted ridges on top
        let ridges = 1.0 - abs(fbm(pos * scale * 1.5, 4, roughness) * 2.0 - 1.0);
        return n * 0.7 + ridges * 0.3 * params.z; // Use distortion param for ridge intensity
    } else if (id < 5.5) {
        // Genie (Generative Dream)
        let time = reality.global_offset.z;
        let p = pos * scale * 0.5;
        // Animated domain warp
        let q = vec2<f32>(
            fbm(p + vec2<f32>(time * 0.1, time * 0.2), 3, roughness),
            fbm(p + vec2<f32>(5.2 - time * 0.1, 1.3 + time * 0.05), 3, roughness)
        );
        let n = fbm(p + 4.0 * q, 4, roughness);
        return n * params.z; // Use distortion to control height
    } else if (id < 6.5) {
        // Glitch (Digital Distortion)
        let time = reality.global_offset.z;

        // Quantized grid base
        let block_size = 2.0;
        let p = floor(pos * scale * block_size) / block_size;

        // Fast flashing noise
        let t_snap = floor(time * 20.0);
        let n1 = hash(p + vec2<f32>(t_snap, t_snap * 0.5));

        // Random "staircase" tearing
        let tear = step(0.8, hash(vec2<f32>(p.y, t_snap)));
        let offset_x = (hash(vec2<f32>(p.x, t_snap)) - 0.5) * 5.0 * tear;

        let final_p = pos + vec2<f32>(offset_x, 0.0);

        // Sharp spiky noise combined with blocky quantization
        let base_n = fbm(final_p * scale, 3, roughness);
        let blocky = floor(base_n * 5.0) / 5.0;

        // Use distortion to create extreme sharp height jumps
        let spike = step(0.9, hash(final_p + vec2<f32>(t_snap))) * 2.0;

        return (blocky + spike * tear) * params.z;
    } else if (id < 7.5) {
        // Steampunk (Brass and Steam - blocky/gears and steam vents)
        // We'll create a layered, terraced look for brass plates/gears
        let base_scale = scale * 0.5;
        let p = pos * base_scale;

        // Base structural noise (large plates)
        let n1 = fbm(p, 4, roughness);
        // Create sharp terraces (stepped)
        let terraces = floor(n1 * 6.0) / 6.0;

        // Add some "rivet" or "gear" circular patterns
        let grid_p = fract(pos * scale * 2.0) - vec2<f32>(0.5);
        let dist = length(grid_p);
        // Small raised bumps if distance is small
        var rivet = 0.0;
        if (dist < 0.2) {
            rivet = 0.1;
        }

        // Combine
        let distortion = params.z;
        let final_h = mix(n1, terraces, distortion) + rivet;
        return final_h * params.z;
    } else if (id < 8.5) {
        // Vaporwave (Synthwave / Outrun - Flat neon grid with digital sun/mountains)
        let time = reality.global_offset.z;
        let base_scale = scale * 0.5;
        let p = pos * base_scale;

        // Very flat ground near the viewer, mountains far away (use noise with distance bias)
        let dist = length(pos);
        // Only start generating mountains if we are far away
        let mountain_zone = smoothstep(10.0, 30.0, dist);

        // Procedural wireframe mountains
        let n = fbm(p * 0.5, 4, roughness);

        // Combine flatness with distant mountains
        return n * 5.0 * mountain_zone * params.z;
    } else if (id < 9.5) {
        // Noir (Monochrome, high contrast, wet streets)
        let block_scale = scale * 2.0;
        let p = pos * block_scale;

        // Blocky city streets / buildings
        let grid_x = step(0.1, fract(p.x));
        let grid_y = step(0.1, fract(p.y));
        let is_building = grid_x * grid_y;

        // Random building heights
        let h = hash(floor(p)) * 2.0;
        let base_height = h * is_building;

        return base_height * params.z;
    }

    return 0.0;
}

// Helper to calculate final height at a point, including blending and visibility
fn get_height_at(pos: vec3<f32>) -> f32 {
    let blend = calculate_blend(pos);

    let disp1 = get_displacement(pos.xz, reality.proj1_params);
    let disp2 = get_displacement(pos.xz, reality.proj2_params);

    // Apply distortion amount
    let val1 = disp1 * reality.proj1_params.z;
    let val2 = disp2 * reality.proj2_params.z;

    let total_displacement = val1 * blend.weight1 + val2 * blend.weight2;
    let visibility = min(blend.total_strength, 1.0);

    return total_displacement * visibility;
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) height: f32,
    @location(2) world_pos: vec3<f32>,
    @location(3) weight1: f32,
    @location(4) weight2: f32,
    @location(5) visibility: f32,
    @location(6) normal: vec3<f32>,
    @location(7) instability: f32,
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
    let flat_pos = pos; // Keep original position for calculating blending/noise

    // Calculate height
    let h = get_height_at(flat_pos);

    // Apply Instability (Glitch)
    let instability = 1.0 - instance.stability;
    var glitch_offset = 0.0;
    if (instability > 0.01) {
        // High frequency noise based on time
        let t = reality.global_offset.z * 10.0;
        let n = hash(flat_pos.xz + vec2<f32>(t));
        if (n > 0.95 - (instability * 0.5)) {
             glitch_offset = (n - 0.5) * instability * 5.0; // Spikes
        }
    }

    // Calculate Normal using Finite Difference
    let e = 0.05;
    let hx = get_height_at(flat_pos + vec3<f32>(e, 0.0, 0.0));
    let hz = get_height_at(flat_pos + vec3<f32>(0.0, 0.0, e));

    // N = normalize(vec3(h - hx, e, h - hz))
    let normal = normalize(vec3<f32>(h - hx, e, h - hz));

    // Apply displacement
    pos.y = pos.y + h + glitch_offset;

    // Recalculate blend for Fragment Shader usage
    let blend = calculate_blend(flat_pos);

    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    out.height = pos.y;
    out.world_pos = pos;
    out.clip_position = camera.view_proj * vec4<f32>(pos, 1.0);
    out.weight1 = blend.weight1;
    out.weight2 = blend.weight2;
    out.visibility = min(blend.total_strength, 1.0);
    out.normal = normal;
    out.instability = instability;

    return out;
}

fn get_pattern_color(pos_in: vec3<f32>, params: vec4<f32>, base_color: vec3<f32>) -> vec3<f32> {
    let roughness = params.x;
    let scale = params.y;
    let id = params.w;

    let pos = pos_in + vec3<f32>(reality.global_offset.x, 0.0, reality.global_offset.y);

    if (id < -0.5) {
        // Void - Dark grid
        let gridSize = 1.0;
        let lineThickness = 0.05;
        let g = step(1.0 - lineThickness, fract(pos.x * gridSize)) + step(1.0 - lineThickness, fract(pos.z * gridSize));
        return mix(vec3<f32>(0.01, 0.01, 0.01), vec3<f32>(0.2, 0.0, 0.2), clamp(g, 0.0, 1.0));
    } else if (id < 0.5) {
        // Fantasy
        let n = fbm(pos.xz * scale * 2.0, 3, roughness);
        return mix(vec3<f32>(0.4, 0.3, 0.2), base_color, n);
    } else if (id < 1.5) {
        // SciFi
        let v = voronoi(pos.xz * scale * 2.0);
        let circuit = step(0.95, v) + step(v, 0.05);
        return mix(base_color * 0.5, vec3<f32>(0.0, 1.0, 1.0), circuit);
    } else if (id < 2.5) {
        // Horror
        let n = fbm(pos.xz * scale * 3.0, 4, 0.8);
        // Blood red and darkness
        return mix(vec3<f32>(0.1, 0.0, 0.0), vec3<f32>(0.6, 0.0, 0.0), n);
    } else if (id < 3.5) {
        // Toon
        let n = fbm(pos.xz * scale, 3, 0.5);
        // Cell shading bands
        let band = floor(n * 3.0) / 3.0;
        return mix(base_color, vec3<f32>(1.0), band * 0.5);
    } else if (id < 4.5) {
        // HyperNature
        // Biomes based on height-like value
        let h = domain_warp(pos.xz * scale * 0.5);

        let water = vec3<f32>(0.0, 0.3, 0.8);
        let grass = vec3<f32>(0.1, 0.6, 0.1);
        let rock = vec3<f32>(0.5, 0.5, 0.5);
        let snow = vec3<f32>(0.9, 0.9, 0.9);

        // Smooth blending using mix
        let c1 = mix(water, grass, smoothstep(0.3, 0.35, h));
        let c2 = mix(c1, rock, smoothstep(0.6, 0.65, h));
        return mix(c2, snow, smoothstep(0.8, 0.85, h));
    } else if (id < 5.5) {
        // Genie (Generative Dream)
        let time = reality.global_offset.z;
        let n = fbm(pos.xz * scale + vec2<f32>(time * 0.1), 3, roughness);

        let c1 = vec3<f32>(1.0, 0.8, 0.2); // Gold
        let c2 = vec3<f32>(0.5, 0.0, 0.8); // Purple
        let c3 = vec3<f32>(0.0, 0.8, 1.0); // Cyan

        // Cyclic blending
        let phase = n + time * 0.2;
        let w1 = 0.5 + 0.5 * sin(phase * 6.28);
        let w2 = 0.5 + 0.5 * sin((phase + 0.33) * 6.28);
        let w3 = 0.5 + 0.5 * sin((phase + 0.66) * 6.28);

        return (c1 * w1 + c2 * w2 + c3 * w3) / (w1 + w2 + w3);
    } else if (id < 6.5) {
        // Glitch (Digital Distortion)
        let time = reality.global_offset.z;
        let t_snap = floor(time * 15.0);

        // Create scanning bands
        let scanline = step(0.9, fract(pos.z * scale + time * 5.0));

        // Create static blocks
        let block_p = floor(pos.xz * scale * 4.0) / 4.0;
        let static_noise = hash(block_p + vec2<f32>(t_snap));

        // Base dark color
        var col = vec3<f32>(0.05, 0.05, 0.05);

        // Add neon magenta and cyan based on noise and scanlines
        if (static_noise > 0.8) {
            col = vec3<f32>(1.0, 0.0, 1.0); // Magenta
        } else if (static_noise < 0.2) {
            col = vec3<f32>(0.0, 1.0, 1.0); // Cyan
        }

        // Intense scanlines
        col = mix(col, vec3<f32>(1.0, 1.0, 1.0), scanline * 0.5);

        return col;
    } else if (id < 7.5) {
        // Steampunk (Brass and Copper)
        let p = pos_in.xz * scale;
        // Use height-like value to color differently based on terraces
        let h_val = floor(fbm(p, 4, roughness) * 6.0) / 6.0;

        let bronze = vec3<f32>(0.8, 0.5, 0.2); // Bright bronze/brass
        let copper = vec3<f32>(0.7, 0.3, 0.1); // Reddish copper
        let dark_iron = vec3<f32>(0.2, 0.2, 0.2); // Iron framework

        // Add dirt/grease via high-frequency noise
        let dirt = fbm(p * 5.0, 2, 0.8) * 0.5;

        var col = bronze;
        if (h_val < 0.3) {
            col = dark_iron;
        } else if (h_val > 0.7) {
            col = copper;
        }

        // Apply grease/dirt
        return mix(col, vec3<f32>(0.05), dirt);
    } else if (id < 8.5) {
        // Vaporwave (Synthwave / Outrun - Flat neon grid)
        let time = reality.global_offset.z;

        let p = pos_in.xz * scale;

        // Base grid (neon cyan)
        let grid_thickness = 0.05;
        let p_moving = p + vec2<f32>(0.0, -time * 2.0); // Grid moving towards camera
        let g = step(1.0 - grid_thickness, fract(p_moving.x)) + step(1.0 - grid_thickness, fract(p_moving.y));

        // Base dark purple ground
        let ground_col = vec3<f32>(0.05, 0.0, 0.1); // Deep synthwave purple
        let grid_col = vec3<f32>(0.0, 1.0, 1.0); // Neon cyan grid

        return mix(ground_col, grid_col, clamp(g, 0.0, 1.0));
    } else if (id < 9.5) {
        // Noir
        // High contrast black/white/grey
        let p = pos_in.xz * scale;

        let block_scale = scale * 2.0;
        let bp = pos_in.xz * block_scale;
        let grid_x = step(0.1, fract(bp.x));
        let grid_y = step(0.1, fract(bp.y));
        let is_building = grid_x * grid_y;

        // Dark streets, gray buildings
        var col = vec3<f32>(0.05, 0.05, 0.05); // Asphalt

        if (is_building > 0.0) {
            // Concrete buildings
            let n = fbm(pos_in.xz * scale * 5.0, 3, roughness);
            col = mix(vec3<f32>(0.2, 0.2, 0.2), vec3<f32>(0.6, 0.6, 0.6), n);

            // Neon / Lit windows occasionally
            let window_grid = step(0.8, fract(pos_in.y * 10.0)) * step(0.8, fract(pos_in.x * 10.0 + pos_in.z * 10.0));
            let is_lit = step(0.9, hash(floor(bp) + floor(pos_in.y * 2.0)));
            if (window_grid * is_lit > 0.0) {
                 col = vec3<f32>(0.9, 0.9, 0.8); // Warm yellow/white light
            }
        } else {
            // Wet streets - add some bright reflections based on noise
            let wetness = fbm(p * 2.0, 2, roughness);
            let reflection = step(0.8, wetness);
            col = mix(col, vec3<f32>(0.8, 0.8, 0.9), reflection * 0.5);
        }

        // Overarching monochrome grading (desaturate base color just in case)
        let lum = dot(col, vec3<f32>(0.299, 0.587, 0.114));
        // Add a slight blueish/cyan tint for the cinematic Noir feel
        return mix(vec3<f32>(lum), vec3<f32>(lum * 0.9, lum * 0.95, lum * 1.0), 0.5);
    }

    return base_color;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let base_texture = textureSample(t_diffuse, s_diffuse, in.tex_coords);

    // Generate Patterns Independently
    let c1 = get_pattern_color(in.world_pos, reality.proj1_params, reality.proj1_color.rgb);
    let c2 = get_pattern_color(in.world_pos, reality.proj2_params, reality.proj2_color.rgb);

    // Blend Patterns
    let pattern_color = c1 * in.weight1 + c2 * in.weight2;

    // Wireframe effect
    let blended_roughness = reality.proj1_params.x * in.weight1 + reality.proj2_params.x * in.weight2;
    let edge = step(0.9, fract(in.world_pos.x * 5.0)) + step(0.9, fract(in.world_pos.z * 5.0));
    let wireframe = clamp(edge, 0.0, 0.2) * blended_roughness;

    let final_gen_color = pattern_color + vec3<f32>(wireframe);

    // Lighting
    let time = reality.global_offset.z;
    let cycle = time * 0.1;
    let light_x = sin(cycle);
    let light_y = cos(cycle);
    let light_dir = normalize(vec3<f32>(light_x, light_y, 0.5));

    // Shadow
    var shadow = 1.0;
    if (light_y > 0.0) {
        shadow = ray_march_shadow(in.world_pos, light_dir);
    } else {
        shadow = 0.0; // No direct sun at night
    }

    let diffuse = max(dot(in.normal, light_dir), 0.0) * shadow;

    // Ambient varies with Day/Night
    var ambient = 0.3;
    if (light_y < 0.0) { ambient = 0.05; }

    let lighting = diffuse + ambient;

    // Apply lighting to reality
    let lit_reality = final_gen_color * lighting;

    // Apply lighting to base texture
    let base_normal = vec3<f32>(0.0, 1.0, 0.0);
    let base_diffuse = max(dot(base_normal, light_dir), 0.0) * shadow;
    let lit_base = base_texture.rgb * (base_diffuse + ambient);

    let result = mix(lit_base, lit_reality, in.visibility);

    // Visual Glitch Overlay (Chromatic Aberration simulation via color shift)
    if (in.instability > 0.1) {
         let shift = in.instability * 0.1;
         return vec4<f32>(result.r + shift, result.g - shift, result.b, 1.0);
    }

    return vec4<f32>(result, 1.0);
}
