struct CameraUniform {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    time: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    // Generate a fullscreen quad
    let uv = vec2<f32>(vec2<u32>(
        (in_vertex_index << 1u) & 2u,
        in_vertex_index & 2u
    ));

    // Convert UV to clip space (-1 to 1)
    let clip_pos = vec4<f32>(uv * 2.0 - 1.0, 1.0, 1.0); // Z = 1.0 for far plane

    out.clip_position = vec4<f32>(clip_pos.x, -clip_pos.y, clip_pos.z, clip_pos.w);
    out.uv = uv;

    return out;
}

// Basic noise functions
fn hash(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453123);
}

fn hash2(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453123);
}

fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash2(i + vec2<f32>(0.0,0.0)),
                   hash2(i + vec2<f32>(1.0,0.0)), u.x),
               mix(hash2(i + vec2<f32>(0.0,1.0)),
                   hash2(i + vec2<f32>(1.0,1.0)), u.x), u.y);
}

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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Reconstruct view ray from clip space
    let clip_space_pos = vec4<f32>(in.uv * 2.0 - 1.0, 1.0, 1.0);
    var world_space_pos = camera.inv_view_proj * vec4<f32>(clip_space_pos.x, -clip_space_pos.y, 1.0, 1.0);
    world_space_pos = world_space_pos / world_space_pos.w;

    let view_dir = normalize(world_space_pos.xyz - camera.camera_pos.xyz);
    let time = camera.time.x;

    // Day/Night cycle
    let cycle = time * 0.1;
    let light_x = sin(cycle);
    let light_y = cos(cycle);
    let sun_dir = normalize(vec3<f32>(light_x, light_y, 0.5));
    let moon_dir = normalize(vec3<f32>(-light_x, -light_y, -0.5));

    // Simple Sky Gradient based on Y
    let t = 0.5 * (view_dir.y + 1.0);

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
        // Sunset transition
        let st = (light_y + 0.2) / 0.4;
        let day_col = mix(day_bot, day_top, t);
        let set_col = mix(set_bot, set_top, t);
        let night_col = mix(night_bot, night_top, t);

        if (st > 0.5) {
            let t_val = (st - 0.5) * 2.0;
            sky_color = mix(set_col, day_col, t_val);
        } else {
            let t_val = st * 2.0;
            sky_color = mix(night_col, set_col, t_val);
        }
    } else {
        sky_color = mix(night_bot, night_top, t);
    }

    // Sun
    let sun_dist = distance(view_dir, sun_dir);
    if (sun_dist < 0.05) {
        let sun_glow = smoothstep(0.05, 0.0, sun_dist);
        sky_color += vec3<f32>(1.0, 0.9, 0.5) * sun_glow * max(light_y + 0.2, 0.0);
    }

    // Moon
    let moon_dist = distance(view_dir, moon_dir);
    if (moon_dist < 0.05) {
        let moon_glow = smoothstep(0.05, 0.0, moon_dist);
        // Moon surface craters
        let moon_surface = fbm(view_dir.xy * 20.0, 3, 0.5);
        let moon_color = mix(vec3<f32>(0.7, 0.7, 0.8), vec3<f32>(1.0, 1.0, 1.0), moon_surface);
        sky_color += moon_color * moon_glow * max(-light_y + 0.2, 0.0);
    }

    // Stars
    if (light_y < 0.1) {
        let star_noise = hash2(view_dir.xy * 200.0);
        let star_thresh = 0.99;
        if (star_noise > star_thresh) {
            let star_glow = (star_noise - star_thresh) / (1.0 - star_thresh);
            // Twinkle
            let twinkle = sin(time * 5.0 + view_dir.x * 100.0) * 0.5 + 0.5;
            let star_alpha = star_glow * twinkle * smoothstep(0.1, -0.1, light_y);
            sky_color += vec3<f32>(1.0, 1.0, 1.0) * star_alpha;
        }
    }

    // Clouds
    if (view_dir.y > 0.0) {
        // Project onto sky plane
        let cloud_uv = view_dir.xz / view_dir.y;
        let cloud_time = time * 0.05;
        let cloud_noise = fbm(cloud_uv * 0.5 + vec2<f32>(cloud_time, cloud_time * 0.5), 4, 0.5);

        // Cloud shape
        let cloud_cover = smoothstep(0.4, 0.8, cloud_noise);

        // Cloud color depends on time of day
        var cloud_color = vec3<f32>(1.0, 1.0, 1.0);
        if (light_y > 0.2) {
            cloud_color = vec3<f32>(1.0, 1.0, 1.0);
        } else if (light_y > -0.2) {
            let st = (light_y + 0.2) / 0.4;
            let set_c = vec3<f32>(1.0, 0.6, 0.3);
            let night_c = vec3<f32>(0.2, 0.2, 0.3);
            if (st > 0.5) {
                cloud_color = mix(set_c, vec3<f32>(1.0), (st - 0.5) * 2.0);
            } else {
                cloud_color = mix(night_c, set_c, st * 2.0);
            }
        } else {
            cloud_color = vec3<f32>(0.2, 0.2, 0.3);
        }

        // Storm Lightning (Occurs randomly)
        // High frequency time noise
        let is_stormy = sin(time * 0.05) > 0.8;
        if (is_stormy && light_y < 0.2) {
            cloud_color *= 0.5; // Darker clouds
            let lightning_noise = hash2(vec2<f32>(time * 10.0, 0.0));
            if (lightning_noise > 0.98) {
                // Flash
                cloud_color += vec3<f32>(1.0, 1.0, 1.5) * 2.0;
                sky_color += vec3<f32>(0.5, 0.5, 0.8);
            }
        }

        sky_color = mix(sky_color, cloud_color, cloud_cover * 0.8);
    }

    // Add lightning to the whole sky if stormy
    let is_stormy = sin(time * 0.05) > 0.8;
    if (is_stormy && light_y < 0.2) {
        let lightning_noise = hash2(vec2<f32>(time * 10.0, 0.0));
        if (lightning_noise > 0.98) {
            sky_color += vec3<f32>(0.2, 0.2, 0.4);
        }
    }

    return vec4<f32>(sky_color, 1.0);
}
