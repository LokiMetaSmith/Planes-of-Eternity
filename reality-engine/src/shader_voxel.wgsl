struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var t_atlas: texture_2d<f32>;
@group(1) @binding(1)
var s_atlas: sampler;
@group(1) @binding(2)
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
    out.world_pos = model.position;
    out.color = model.color;
    out.normal = model.normal;
    out.ao = model.ao;
    out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // 1. Material Logic
    var offset = vec2<f32>(0.0, 0.0); // Stone
    var specular_strength = 0.0;
    var emissive_strength = 0.0;

    // Check Material based on color
    if (in.color.r > 0.8) {
        if (in.color.g < 0.6) {
            // Lava (TR)
            offset = vec2<f32>(0.5, 0.0);
            specular_strength = 0.5;
            emissive_strength = 0.8;
        } else {
            // Fire (BL)
            offset = vec2<f32>(0.0, 0.5);
            emissive_strength = 1.0;
        }
    } else if (in.color.b > 0.8) {
        // Water (Stone texture + Blue tint + Shiny)
        offset = vec2<f32>(0.0, 0.0);
        specular_strength = 1.0;
    } else if (in.color.g > 0.6 || (in.color.r > 0.4 && in.color.b < 0.4)) {
        // Wood/Grass (BR)
        offset = vec2<f32>(0.5, 0.5);
    } else {
        // Stone (TL)
        offset = vec2<f32>(0.0, 0.0);
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

    // Scale to 0.5 quadrant size
    // fract(uv) is 0..1 per block
    let uv_scaled = fract(uv) * 0.5;
    let final_uv = uv_scaled + offset;

    let tex_color = textureSample(t_atlas, s_atlas, final_uv);

    // 3. Lighting
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.5));
    let view_dir = normalize(camera.camera_pos.xyz - in.world_pos);

    let ambient = 0.3;

    // Raytraced Shadow
    let shadow = ray_march_shadow(in.world_pos, light_dir);

    let diff = max(dot(in.normal, light_dir), 0.0) * shadow;

    // AO Factor
    let ao_factor = in.ao * 0.8 + 0.2;

    // Specular
    let half_dir = normalize(light_dir + view_dir);
    let spec_angle = max(dot(in.normal, half_dir), 0.0);
    let specular = pow(spec_angle, 32.0) * specular_strength * shadow;

    // Combine
    let albedo = tex_color.rgb * in.color;

    let lighting = (ambient + diff) * ao_factor;

    let emission = albedo * emissive_strength;

    var final_rgb = albedo * lighting + vec3<f32>(specular) + emission;

    // Reflections (Procedural Sky)
    if (specular_strength > 0.0) {
        let r = reflect(-view_dir, in.normal);
        // Simple Sky Gradient based on Y
        let t = 0.5 * (r.y + 1.0);
        let sky_color = mix(vec3<f32>(0.1, 0.1, 0.2), vec3<f32>(0.5, 0.7, 1.0), t);

        // Fresnel Effect
        let f0 = 0.04;
        let fresnel = f0 + (1.0 - f0) * pow(1.0 - max(dot(view_dir, in.normal), 0.0), 5.0);

        final_rgb = mix(final_rgb, sky_color, fresnel * specular_strength * 0.8);
    }

    return vec4<f32>(final_rgb, 1.0);
}
