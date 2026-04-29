struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct SplatVertexInput {
    @location(0) position: vec3<f32>, // Not used, quad generated internally
};

struct SplatInstanceInput {
    @location(2) position: vec3<f32>,
    @location(3) rotation: vec4<f32>, // quaternion
    @location(4) scale: vec3<f32>,
    @location(5) color: vec4<f32>,
};

struct SplatVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) conic: vec3<f32>, // 2D covariance conic parameters
    @location(2) center_ndc: vec2<f32>, // Splat center in NDC
    @location(3) ray_dir: vec2<f32>, // Ray direction from center
    @location(4) world_pos: vec3<f32>,
};

// --- Shared Shadow Logic ---
@group(1) @binding(0)
var t_density: texture_3d<u32>;

struct RealityUniform {
    proj1_pos_fid: vec4<f32>,
    proj1_params: vec4<f32>,
    proj1_color: vec4<f32>,
    proj2_pos_fid: vec4<f32>,
    proj2_params: vec4<f32>,
    proj2_color: vec4<f32>,
    global_offset: vec4<f32>, // z is time
};
@group(2) @binding(0) var<uniform> reality: RealityUniform;

fn ray_march_shadow(origin: vec3<f32>, direction: vec3<f32>) -> f32 {
    let max_dist = 60.0;
    let step_size = 0.5;
    var current_pos = origin + direction * 1.5; // Start bias
    var dist = 0.0;

    loop {
        if (dist > max_dist) { break; }

        let tx = i32(floor(current_pos.x + 64.0));
        let ty = i32(floor(current_pos.y + 32.0));
        let tz = i32(floor(current_pos.z + 64.0));

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
// ---------------------------


@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    instance: SplatInstanceInput
) -> SplatVertexOutput {
    var out: SplatVertexOutput;

    // Generate quad vertices
    var offsets = array<vec2<f32>, 6>(
        vec2<f32>(-2.0, -2.0),
        vec2<f32>( 2.0, -2.0),
        vec2<f32>(-2.0,  2.0),
        vec2<f32>( 2.0, -2.0),
        vec2<f32>( 2.0,  2.0),
        vec2<f32>(-2.0,  2.0)
    );
    let offset = offsets[in_vertex_index % 6u];

    let pos = vec4<f32>(instance.position, 1.0);
    let view_pos = camera.view_proj * pos;

    let depth = view_pos.w;

    // Projective Transformation (Simplified)
    // Map 3D scale to 2D screen space based on depth
    let screen_scale = instance.scale.x / depth * 50.0;

    out.clip_position = vec4<f32>(
        view_pos.x / depth + offset.x * screen_scale / 800.0,
        view_pos.y / depth + offset.y * screen_scale / 600.0,
        view_pos.z / depth,
        1.0
    );

    out.color = instance.color;
    out.center_ndc = vec2<f32>(view_pos.x / depth, view_pos.y / depth);
    out.ray_dir = offset;
    out.world_pos = instance.position;

    out.conic = vec3<f32>(1.0, 0.0, 1.0); // Simplified circle conic

    return out;
}

@fragment
fn fs_main(in: SplatVertexOutput) -> @location(0) vec4<f32> {
    // EWA Gaussian Evaluation
    let x = in.ray_dir.x;
    let y = in.ray_dir.y;
    let power = -0.5 * (in.conic.x * x * x + in.conic.z * y * y + 2.0 * in.conic.y * x * y);

    // Fragment Discarding (> 3 sigma equivalent roughly)
    if (power > 0.0 || power < -9.0) {
        discard;
    }

    var alpha = exp(power) * in.color.a;

    // Lighting Integration
    let time = reality.global_offset.z;
    let cycle = time * 0.1;
    let light_x = sin(cycle);
    let light_y = cos(cycle);
    let light_dir = normalize(vec3<f32>(light_x, light_y, 0.5));

    var shadow = 1.0;
    if (light_y > 0.0) {
        shadow = ray_march_shadow(in.world_pos, light_dir);
    } else {
        shadow = 0.0;
    }

    // Simple shading based on normal facing light + ambient
    let ambient = 0.4;
    let lighting = ambient + (0.6 * shadow);

    return vec4<f32>(in.color.rgb * lighting, alpha);
}
