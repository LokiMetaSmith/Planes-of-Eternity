struct CameraUniform {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    time: vec4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var t_atlas: texture_2d<f32>;
@group(1) @binding(1)
var s_atlas: sampler;
@group(1) @binding(2)
var t_density: texture_3d<u32>;

struct RealityUniform {
    proj_pos_fid: array<vec4<f32>, 5>,
    proj_params: array<vec4<f32>, 5>,
    proj_color: array<vec4<f32>, 5>,
    global_offset: vec4<f32>,
    nodes_pos_fid: array<vec4<f32>, 15>,
    nodes_params: array<vec4<f32>, 15>,
    nodes_color: array<vec4<f32>, 15>,
    num_nodes: vec4<u32>,
};
@group(2) @binding(0) var<uniform> reality: RealityUniform;


fn get_lighting_info(id: f32) -> vec2<f32> {
    if (id < -0.5) { return vec2<f32>(0.0, 0.4); }
    else if (id < 0.5) { return vec2<f32>(1.0, 0.3); }
    else if (id < 1.5) { return vec2<f32>(0.0, 0.5); }
    else if (id < 2.5) { return vec2<f32>(0.0, 0.2); }
    else if (id < 3.5) { return vec2<f32>(1.0, 0.3); }
    else if (id < 4.5) { return vec2<f32>(1.0, 0.3); }
    else if (id < 5.5) { return vec2<f32>(1.0, 0.3); }
    else if (id < 6.5) { return vec2<f32>(0.0, 0.5); }
    else if (id < 7.5) { return vec2<f32>(1.0, 0.3); }
    else if (id < 8.5) { return vec2<f32>(0.0, 0.5); }
    else if (id < 9.5) { return vec2<f32>(0.0, 0.4); }
    else if (id < 10.5) { return vec2<f32>(0.0, 0.5); }
    else if (id < 11.5) { return vec2<f32>(0.0, 0.6); }
    else if (id < 12.5) { return vec2<f32>(1.0, 0.3); }
    else if (id < 13.5) { return vec2<f32>(1.0, 0.3); }
    else if (id < 14.5) { return vec2<f32>(1.0, 0.3); }
    else if (id < 15.5) { return vec2<f32>(0.0, 0.5); }
    else if (id < 16.5) { return vec2<f32>(0.0, 0.6); }
    else if (id < 17.5) { return vec2<f32>(0.0, 0.7); }
    else if (id < 18.5) { return vec2<f32>(1.0, 0.3); }
    else if (id < 19.5) { return vec2<f32>(1.0, 0.3); }
    return vec2<f32>(0.5, 0.3);
}

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
        // Optimization: Single bounds check utilizing unsigned cast. Negative values wrap to high unsigned values
        if (u32(tx) < 128u && u32(ty) < 128u && u32(tz) < 128u) {
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
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) ao: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) ao: f32,
};

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let time = reality.global_offset.z;
    var animated_pos = model.position;

        let r = model.color.r;
    let g = model.color.g;
    let b = model.color.b;

    // Liquid and Gas logic
    if (abs(r - 0.2) < 0.01 && abs(g - 1.0) < 0.01 && abs(b - 0.2) < 0.01) {
        // Acid
        animated_pos.y += sin(time * 3.0 + model.position.x * 2.0 + model.position.z * 2.0) * 0.1;
    } else if (abs(r - 0.0) < 0.01 && abs(g - 0.5) < 0.01 && abs(b - 1.0) < 0.01) {
        // Water
        animated_pos.y += sin(time * 1.5 + model.position.x + model.position.z) * 0.15;
    } else if (abs(r - 1.0) < 0.01 && abs(g - 0.3) < 0.01 && abs(b - 0.0) < 0.01) {
        // Lava
        animated_pos.y += sin(time * 0.5 + model.position.x * 0.5 + model.position.z * 0.5) * 0.05;
    } else if ((r > 0.7 && g > 0.7 && b > 0.7) || (b > 0.9 && r > 0.4 && g > 0.4 && r < 0.6)) {
        // Gasses/Weather (Fog, Cloud, Rain)
        animated_pos.x += sin(time * 0.5 + model.position.y) * 0.2;
        animated_pos.z += cos(time * 0.5 + model.position.y) * 0.2;
    } else if (abs(r - 0.2) < 0.01 && abs(g - 0.8) < 0.01 && abs(b - 0.2) < 0.01) || (abs(r - 0.2) < 0.01 && abs(g - 0.6) < 0.01 && abs(b - 0.2) < 0.01) {
        // Existing Landscape (Grass tops / leaves wobble slightly in wind)
        if (model.normal.y > 0.5) {
            let is_stormy = sin(time * 0.05) > 0.8;
            let wind_strength = select(0.05, 0.2, is_stormy);
            let wind_speed = select(2.0, 8.0, is_stormy);
            animated_pos.x += sin(time * wind_speed + model.position.y) * wind_strength;
            if (is_stormy) {
                animated_pos.z += cos(time * wind_speed + model.position.y) * wind_strength;
            }
        }
    }

    out.world_pos = animated_pos;
    out.color = model.color;
    out.normal = model.normal;
    out.ao = model.ao;
    out.clip_position = camera.view_proj * vec4<f32>(animated_pos, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
        // 1. Material Logic
    var offset = vec2<f32>(0.0, 0.0); // Stone
    var specular_strength = 0.0;
    var emissive_strength = 0.0;

    let r = in.color.r;
    let g = in.color.g;
    let b = in.color.b;

    // Check Material based on exact color mapping
    if (abs(r - 0.5) < 0.01 && abs(g - 0.5) < 0.01 && abs(b - 0.5) < 0.01) {
        // 1: Stone
        offset = vec2<f32>(0.0, 0.0);
    } else if (abs(r - 1.0) < 0.01 && abs(g - 0.3) < 0.01 && abs(b - 0.0) < 0.01) {
        // 2: Lava
        offset = vec2<f32>(0.25, 0.0);
        specular_strength = 0.5;
        emissive_strength = 0.8;
    } else if (abs(r - 1.0) < 0.01 && abs(g - 0.8) < 0.01 && abs(b - 0.0) < 0.01) {
        // 3: Fire
        offset = vec2<f32>(0.5, 0.0);
        emissive_strength = 1.0;
    } else if (abs(r - 0.0) < 0.01 && abs(g - 0.5) < 0.01 && abs(b - 1.0) < 0.01) {
        // 4: Water
        offset = vec2<f32>(0.0, 0.5);
        specular_strength = 1.0;
    } else if (abs(r - 0.2) < 0.01 && abs(g - 0.8) < 0.01 && abs(b - 0.2) < 0.01) {
        // 5: Grass
        offset = vec2<f32>(0.0, 0.25);
    } else if (abs(r - 0.4) < 0.01 && abs(g - 0.2) < 0.01 && abs(b - 0.0) < 0.01) {
        // 6: Wood
        offset = vec2<f32>(0.75, 0.0);
    } else if (abs(r - 0.2) < 0.01 && abs(g - 0.6) < 0.01 && abs(b - 0.2) < 0.01) {
        // 7: Leaves
        offset = vec2<f32>(0.25, 0.25);
    } else if (abs(r - 0.4) < 0.01 && abs(g - 0.3) < 0.01 && abs(b - 0.2) < 0.01) {
        // 8: Dirt
        offset = vec2<f32>(0.5, 0.25);
    } else if (abs(r - 0.8) < 0.01 && abs(g - 0.8) < 0.01 && abs(b - 0.6) < 0.01) {
        // 9: Sand
        offset = vec2<f32>(0.75, 0.25);
    } else if (abs(r - 0.2) < 0.01 && abs(g - 1.0) < 0.01 && abs(b - 0.2) < 0.01) {
        // 10: Acid
        offset = vec2<f32>(0.0, 0.0); // using stone as base, colored by albedo
        specular_strength = 0.8;
        emissive_strength = 0.5;
    } else if ((in.color.r > 0.7 && in.color.g > 0.7 && in.color.b > 0.7) || (in.color.b > 0.9 && in.color.r > 0.4 && in.color.g > 0.4 && in.color.r < 0.6)) {
        // 11/12/13: Fog/Cloud/Rain
        offset = vec2<f32>(0.0, 0.0);
        specular_strength = 0.1;
    }

    // 2. Triplanar UV
    let n = abs(in.normal);
    var uv = vec2<f32>(0.0, 0.0);
    if (n.y > 0.5) {
        uv = in.world_pos.xz;
    } else if (n.x > 0.5) {
        uv = in.world_pos.yz;
    } else {
        uv = in.world_pos.xy;
    }

    // Scale to 0.25 quadrant size (since it's a 4x4 grid now)
    // fract(uv) is 0..1 per block
    let uv_scaled = fract(uv) * 0.25;
    let final_uv = uv_scaled + offset;

    let tex_color = textureSample(t_atlas, s_atlas, final_uv);

    // 3. Lighting
    let time = reality.global_offset.z;
    let cycle = time * 0.1;
    let light_x = sin(cycle);
    let light_y = cos(cycle);
    let light_dir = normalize(vec3<f32>(light_x, light_y, 0.5));

    let view_dir = normalize(camera.camera_pos.xyz - in.world_pos);

    var total_str = 0.0;
    var strengths: array<f32, 5>;

    for (var i = 0u; i < 5u; i = i + 1u) {
        let dist = max(distance(in.world_pos, reality.proj_pos_fid[i].xyz), 1.0);
        let strength = reality.proj_pos_fid[i].w / dist;
        strengths[i] = strength;
        total_str = total_str + strength;
    }

    var directional_weight = 0.0;
    var ambient_strength = 0.0;

    if (total_str > 0.0001) {
        for (var i = 0u; i < 5u; i = i + 1u) {
            let weight = strengths[i] / total_str;
            if (weight > 0.001) {
                let l_info = get_lighting_info(reality.proj_params[i].w);
                directional_weight = directional_weight + l_info.x * weight;
                ambient_strength = ambient_strength + l_info.y * weight;
            }
        }
    } else {
        let l_info = get_lighting_info(-1.0);
        directional_weight = l_info.x;
        ambient_strength = l_info.y;
    }

    // Ambient varies with Day/Night ONLY if there is directional light.
    var ambient = ambient_strength;
    if (light_y < 0.0) {
        ambient = mix(ambient_strength, 0.05, directional_weight);
    }

    // Raytraced Shadow (only if sun is up)
    var shadow = 1.0;
    if (light_y > 0.0 && directional_weight > 0.001) {
        shadow = ray_march_shadow(in.world_pos, light_dir);
    } else {
        shadow = 0.0; // No direct sun at night or if ambient only
    }

    // Storm Lightning Flash
    let is_stormy = sin(time * 0.05) > 0.8;
    if (is_stormy && light_y < 0.2) {
        let lightning_noise = fract(sin(dot(vec2<f32>(time * 10.0, 0.0), vec2<f32>(12.9898, 78.233))) * 43758.5453);
        if (lightning_noise > 0.98) {
            ambient = ambient + 1.5;
        }
    }

    let diff = max(dot(in.normal, light_dir), 0.0) * shadow * directional_weight;

    // AO Factor
    let ao_factor = in.ao * 0.8 + 0.2;

    // Specular
    let half_dir = normalize(light_dir + view_dir);
    let spec_angle = max(dot(in.normal, half_dir), 0.0);
    let specular = pow(spec_angle, 32.0) * specular_strength * shadow * directional_weight;

    // Combine
    let albedo = tex_color.rgb * in.color;

    let lighting = (ambient + diff) * ao_factor;

    let emission = albedo * emissive_strength; // Glowing things glow at night too

    var final_rgb = albedo * lighting + vec3<f32>(specular) + emission;

    // Reflections (Procedural Sky)
    if (specular_strength > 0.0) {
        let r = reflect(-view_dir, in.normal);
        // Simple Sky Gradient based on Y
        let t = 0.5 * (r.y + 1.0);

        // Day: Blue Sky
        let day_top = vec3<f32>(0.2, 0.6, 1.0);
        let day_bot = vec3<f32>(0.7, 0.8, 1.0);

        // Sunset: Orange/Purple
        let set_top = vec3<f32>(0.8, 0.3, 0.1);
        let set_bot = vec3<f32>(0.9, 0.6, 0.3);

        // Night: Black/Stars
        let night_top = vec3<f32>(0.0, 0.0, 0.1);
        let night_bot = vec3<f32>(0.0, 0.0, 0.05);

        var sky_color = vec3<f32>(0.0);
        if (light_y > 0.2) {
            sky_color = mix(day_bot, day_top, t);
        } else if (light_y > -0.2) {
            sky_color = mix(set_bot, set_top, t);
        } else {
            sky_color = mix(night_bot, night_top, t);
        }

        // Fresnel Effect
        let f0 = 0.04;
        let fresnel = f0 + (1.0 - f0) * pow(1.0 - max(dot(view_dir, in.normal), 0.0), 5.0);

        final_rgb = mix(final_rgb, sky_color, fresnel * specular_strength * 0.8);
    }

    return vec4<f32>(final_rgb, 1.0);
}
