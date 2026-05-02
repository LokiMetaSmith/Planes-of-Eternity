struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct RealityUniform {
    proj1_pos_fid: vec4<f32>,
    proj1_params: vec4<f32>,
    proj1_color: vec4<f32>,
    proj2_pos_fid: vec4<f32>,
    proj2_params: vec4<f32>,
    proj2_color: vec4<f32>,
    global_offset: vec4<f32>, // z is time
};
@group(1) @binding(0) var<uniform> reality: RealityUniform;

@group(1) @binding(1)
var t_density: texture_3d<u32>;

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

struct SplatVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) rotation: vec4<f32>,
    @location(2) scale: vec3<f32>,
    @location(3) color: vec4<f32>,
};

struct SplatOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) world_pos: vec3<f32>,
};

// 6 vertices per splat quad
var<private> QUAD_UVS: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-2.0, -2.0),
    vec2<f32>( 2.0, -2.0),
    vec2<f32>(-2.0,  2.0),
    vec2<f32>( 2.0, -2.0),
    vec2<f32>( 2.0,  2.0),
    vec2<f32>(-2.0,  2.0)
);

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    instance: SplatVertexInput
) -> SplatOutput {
    var out: SplatOutput;

    let uv = QUAD_UVS[vertex_index];
    out.uv = uv;

    // Calculate rotation matrix from quaternion
    let q = instance.rotation;
    let R = mat3x3<f32>(
        1.0 - 2.0*q.y*q.y - 2.0*q.z*q.z, 2.0*q.x*q.y - 2.0*q.z*q.w, 2.0*q.x*q.z + 2.0*q.y*q.w,
        2.0*q.x*q.y + 2.0*q.z*q.w, 1.0 - 2.0*q.x*q.x - 2.0*q.z*q.z, 2.0*q.y*q.z - 2.0*q.x*q.w,
        2.0*q.x*q.z - 2.0*q.y*q.w, 2.0*q.y*q.z + 2.0*q.x*q.w, 1.0 - 2.0*q.x*q.x - 2.0*q.y*q.y
    );

    let S = mat3x3<f32>(
        instance.scale.x, 0.0, 0.0,
        0.0, instance.scale.y, 0.0,
        0.0, 0.0, instance.scale.z
    );

    let M = R * S;

    // For a 2D quad billboard facing the camera:
    // Extract camera right and up vectors from view_proj
    // (A more accurate implementation would compute 2D covariance projection, but
    //  a simple billboarded quad scaled by projected covariance is sufficient for prototype)

    let camera_dir = normalize(camera.camera_pos.xyz - instance.position);
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(dot(camera_dir, up)) > 0.99) {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let right = normalize(cross(up, camera_dir));
    let final_up = normalize(cross(camera_dir, right));

    // Project uv onto billboard plane
    // apply scaling based on rotation/scale (approximate)
    let local_pos = (right * uv.x + final_up * uv.y) * max(max(instance.scale.x, instance.scale.y), instance.scale.z);

    let world_pos = instance.position + local_pos;

    out.world_pos = world_pos;
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = instance.color;

    return out;
}

@fragment
fn fs_main(in: SplatOutput) -> @location(0) vec4<f32> {
    // Distance from center
    let dist2 = dot(in.uv, in.uv);

    // Discard fragments exceeding 3 sigma distance to maintain performance
    if (dist2 > 9.0) {
        discard;
    }

    // Gaussian density
    let alpha = exp(-0.5 * dist2) * in.color.a;

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
        shadow = 0.0;
    }

    let ambient = 0.4;
    let lighting = ambient + (0.6 * shadow);

    let final_color = in.color.rgb * lighting;

    return vec4<f32>(final_color, alpha);
}
