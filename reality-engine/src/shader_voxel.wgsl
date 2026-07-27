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
@group(1) @binding(3)
var t_normal: texture_2d<f32>;
@group(1) @binding(4)
var s_normal: sampler;
@group(1) @binding(5)
var t_pbr: texture_2d<f32>;
@group(1) @binding(6)
var s_pbr: sampler;

struct RealityUniform {
    proj_pos_fid: array<vec4<f32>, 5>,
    proj_params: array<vec4<f32>, 5>,
    proj_color: array<vec4<f32>, 5>,
    global_offset: vec4<f32>,
    nodes_pos_fid: array<vec4<f32>, 15>,
    nodes_params: array<vec4<f32>, 15>,
    nodes_color: array<vec4<f32>, 15>,
    num_nodes: vec4<u32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    ambient_color: vec4<f32>,
};
@group(2) @binding(0) var<uniform> reality: RealityUniform;



// PBR Math Helpers
const PI = 3.14159265359;

fn get_roughness_metallic(id: f32) -> vec2<f32> {
    if (id < -0.5) { return vec2<f32>(0.6, 0.0); }
    else if (id < 0.5) { return vec2<f32>(0.9, 0.0); }
    else if (id < 1.5) { return vec2<f32>(0.8, 0.0); }
    else if (id < 2.5) { return vec2<f32>(0.5, 0.1); }
    else if (id < 3.5) { return vec2<f32>(0.7, 0.0); }
    else if (id < 4.5) { return vec2<f32>(0.2, 0.0); }
    else if (id < 5.5) { return vec2<f32>(0.8, 0.0); }
    else if (id < 6.5) { return vec2<f32>(0.5, 0.0); }
    else if (id < 7.5) { return vec2<f32>(0.4, 0.0); }
    else if (id < 8.5) { return vec2<f32>(0.2, 0.8); }
    else if (id < 9.5) { return vec2<f32>(0.9, 0.0); }
    else { return vec2<f32>(0.5, 0.0); }
}

fn DistributionGGX(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness*roughness;
    let a2 = a*a;
    let NdotH = max(dot(N, H), 0.0);
    let NdotH2 = NdotH*NdotH;
    let num = a2;
    var denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    return num / denom;
}

fn GeometrySchlickGGX(NdotV: f32, roughness: f32) -> f32 {
    let r = (roughness + 1.0);
    let k = (r*r) / 8.0;
    let num = NdotV;
    let denom = NdotV * (1.0 - k) + k;
    return num / denom;
}

fn GeometrySmith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    let ggx2 = GeometrySchlickGGX(NdotV, roughness);
    let ggx1 = GeometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

fn fresnelSchlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

fn calculate_pbr(
    N: vec3<f32>, V: vec3<f32>, L: vec3<f32>,
    albedo: vec3<f32>, roughness: f32, metallic: f32,
    light_color: vec3<f32>, attenuation: f32
) -> vec3<f32> {
    let H = normalize(V + L);
    let radiance = light_color * attenuation;

    var F0 = vec3<f32>(0.04);
    F0 = mix(F0, albedo, metallic);

    let NDF = DistributionGGX(N, H, roughness);
    let G   = GeometrySmith(N, V, L, roughness);
    let F   = fresnelSchlick(max(dot(H, V), 0.0), F0);

    let numerator    = NDF * G * F;
    let denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    let specular = numerator / denominator;

    let kS = F;
    var kD = vec3<f32>(1.0) - kS;
    kD *= 1.0 - metallic;

    let NdotL = max(dot(N, L), 0.0);
    return (kD * albedo / PI + specular) * radiance * NdotL;
}

// ACES Tonemapping
fn ACESFilm(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), vec3<f32>(0.0), vec3<f32>(1.0));
}

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
    @location(4) tangent: vec3<f32>,
    @location(5) bitangent: vec3<f32>,
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

    // Compute basic tangent space aligned with triplanar mapping
    let n = abs(model.normal);
    var t = vec3<f32>(1.0, 0.0, 0.0);
    if (n.y > 0.5) {
        t = vec3<f32>(1.0, 0.0, 0.0);
    } else if (n.x > 0.5) {
        t = vec3<f32>(0.0, 0.0, 1.0);
    }
    t = normalize(t - dot(t, model.normal) * model.normal);
    let bitan = cross(model.normal, t);

    out.tangent = t;
    out.bitangent = bitan;
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

    // 3. Lighting (PBR & Point Lights)
    let time = reality.global_offset.z;
    let cycle = time * 0.1;
    let light_x = sin(cycle);
    let light_y = cos(cycle);

    // Sample Normal & PBR maps
    let normal_map = textureSample(t_normal, s_normal, final_uv).rgb * 2.0 - 1.0;
    let pbr_map = textureSample(t_pbr, s_pbr, final_uv).rgb;

    // TBN Matrix
    let T = normalize(in.tangent);
    let B = normalize(in.bitangent);
    let N_geom = normalize(in.normal);
    let TBN = mat3x3<f32>(T, B, N_geom);
    let N = normalize(TBN * normal_map); // Perturbed normal

    // Procedural Screen-Space SSAO using derivatives
    let dx = dpdx(in.world_pos);
    let dy = dpdy(in.world_pos);
    let cross_der = normalize(cross(dx, dy));
    let normal_diff = max(0.0, 1.0 - dot(N_geom, cross_der));
    let ssao = max(0.0, 1.0 - normal_diff * 5.0); // Cavity approximation

    // Mix PBR properties
    // texture: R = roughness, G = metallic, B = AO
    let roughness = mix(1.0 - specular_strength, pbr_map.r, 0.8);
    let metallic = mix(0.0, pbr_map.g, 0.8);
    let map_ao = pbr_map.b;
    let ao_factor = in.ao * ssao * map_ao;

    var sun_direction = normalize(reality.sun_dir.xyz);
    if (length(reality.sun_dir.xyz) < 0.1) {
        sun_direction = normalize(vec3<f32>(light_x, max(light_y, 0.0), 0.5));
    }
    var sun_col = reality.sun_color.rgb;
    if (length(sun_col) < 0.1) {
        if (light_y > 0.0) { sun_col = vec3<f32>(1.0, 0.95, 0.9); }
        else { sun_col = vec3<f32>(0.1, 0.1, 0.3); }
    }

    var ambient_color = reality.ambient_color.rgb;
    if (length(ambient_color) < 0.1) {
        ambient_color = vec3<f32>(0.1, 0.15, 0.2);
    }

    let view_dir = normalize(camera.camera_pos.xyz - in.world_pos);
    let V = view_dir;

    let albedo = tex_color.rgb * in.color;
    var Lo = vec3<f32>(0.0);

    var shadow = 1.0;
    if (sun_direction.y > 0.0) {
        shadow = ray_march_shadow(in.world_pos, sun_direction);
    } else {
        shadow = 0.0;
    }

    let sun_irradiance = sun_col * 2.0;
    Lo = Lo + calculate_pbr(N, V, sun_direction, albedo, roughness, metallic, sun_irradiance, shadow);

    for (var i = 0u; i < 5u; i = i + 1u) {
        let p_pos = reality.proj_pos_fid[i].xyz;
        if (p_pos.y > -999.0 && reality.proj_pos_fid[i].w > 0.0) {
            let dist = distance(in.world_pos, p_pos);
            let L = normalize(p_pos - in.world_pos);
            let atten = 1.0 / (dist * dist + 1.0);
            Lo = Lo + calculate_pbr(N, V, L, albedo, roughness, metallic, reality.proj_color[i].rgb, atten * 10.0);
        }
    }

    var ambient = ambient_color * albedo * ao_factor * (1.0 - metallic);

    let is_stormy = sin(time * 0.05) > 0.8;
    if (is_stormy && sun_direction.y < 0.2) {
        let lightning_noise = fract(sin(dot(vec2<f32>(time * 10.0, 0.0), vec2<f32>(12.9898, 78.233))) * 43758.5453);
        if (lightning_noise > 0.98) {
            ambient = ambient + vec3<f32>(1.5);
        }
    }

    let emission = albedo * emissive_strength;
    var color = ambient + Lo + emission;

    // --- Peek Effect Cutout & Rim ---
    var peek_rim_color = vec3<f32>(0.0);

    // Check against permanent anomaly nodes
    let num_nodes = reality.num_nodes.x;
    for (var i = 0u; i < num_nodes; i++) {
        let n_pos = reality.nodes_pos_fid[i].xyz;
        let n_params = reality.nodes_params[i];
        let n_archetype = u32(n_params.w);

        let d = distance(in.world_pos, n_pos);
        // radius = scale (y component of params)
        let radius = n_params.y;

        // Discard geometry inside the peek sphere
        if (d < radius) {
            discard;
        }

        // Add a glowing rim just outside the threshold
        let rim_width = 0.5;
        if (d >= radius && d < radius + rim_width) {
            // HDR glow color (pinkish purple from engine)
            let rim = (1.0 - (d - radius) / rim_width);
            peek_rim_color = vec3<f32>(1.0, 0.2, 0.8) * rim * 2.0;
        }
    }
    
    // Apply the rim glow to our final color
    color = color + peek_rim_color;
    // ---------------------------------

    // Reflections (Procedural Sky)
    if (roughness < 0.3 || specular_strength > 0.0) {
        var r: vec3<f32>; // Declared properly so it survives the if/else
        
        if (roughness < 0.3) {
            r = reflect(-V, N);
        } else {
            r = reflect(-view_dir, in.normal);
        }
        
        // Simple Sky Gradient based on Y
        let t = 0.5 * (r.y + 1.0);
        let sky_color = mix(vec3<f32>(0.2, 0.6, 1.0), vec3<f32>(0.7, 0.8, 1.0), t);
        let F = fresnelSchlick(max(dot(N, V), 0.0), mix(vec3<f32>(0.04), albedo, metallic));
        color = color + sky_color * F * (1.0 - roughness);
    }

    // Tonemapping & Gamma Correction
    color = ACESFilm(color);
    color = pow(color, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(color, 1.0);
}
