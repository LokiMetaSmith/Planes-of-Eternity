struct CameraUniform {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    time: vec4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

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

@group(1) @binding(2)
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
    @location(4) previous_position: vec3<f32>,
    @location(5) archetype_id: u32,
    @location(6) target_archetype_id: u32,
    @location(7) morph_weight: f32,
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

    // Neural Morphing & Filtering weights
    var f_motion = 1.0;
    var f_scale = 1.0;
    var f_color = 1.0;

    // Apply Neural Filtering based on both current and target archetypes (Morphed)
    for (var i = 0u; i < 5u; i++) {
        let p_pos = reality.proj_pos_fid[i].xyz;
        let p_fid = reality.proj_pos_fid[i].w;
        let p_params = reality.proj_params[i];
        let p_archetype = u32(p_params.w);

        let dist = distance(instance.position, p_pos);
        if (dist < p_fid) {
            let influence = 1.0 - (dist / p_fid);

            // Weight filtering influence by morph progress
            if (instance.archetype_id == p_archetype) {
                let weight = (1.0 - instance.morph_weight) * influence;
                f_motion += p_params.z * weight;
                f_scale += p_params.y * weight;
                f_color += p_params.x * weight;
            }
            if (instance.target_archetype_id == p_archetype) {
                let weight = instance.morph_weight * influence;
                f_motion += p_params.z * weight;
                f_scale += p_params.y * weight;
                f_color += p_params.x * weight;
            }
        }
    }

    // Calculate rotation matrix from quaternion
    let q = instance.rotation;
    let R = mat3x3<f32>(
        1.0 - 2.0*q.y*q.y - 2.0*q.z*q.z, 2.0*q.x*q.y - 2.0*q.z*q.w, 2.0*q.x*q.z + 2.0*q.y*q.w,
        2.0*q.x*q.y + 2.0*q.z*q.w, 1.0 - 2.0*q.x*q.x - 2.0*q.z*q.z, 2.0*q.y*q.z - 2.0*q.x*q.w,
        2.0*q.x*q.z - 2.0*q.y*q.w, 2.0*q.y*q.z + 2.0*q.x*q.w, 1.0 - 2.0*q.x*q.x - 2.0*q.y*q.y
    );

    let S = mat3x3<f32>(
        instance.scale.x * f_scale, 0.0, 0.0,
        0.0, instance.scale.y * f_scale, 0.0,
        0.0, 0.0, instance.scale.z * f_scale
    );

    let M = R * S;

    let tick_alpha = clamp(reality.global_offset.w, 0.0, 1.0);
    // Apply Neural Motion Filtering (F_motion affects the delta between frames)
    let delta = instance.position - instance.previous_position;
    let interpolated_pos = instance.previous_position + (delta * tick_alpha * f_motion);

    let camera_dir = normalize(camera.camera_pos.xyz - interpolated_pos);
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(dot(camera_dir, up)) > 0.99) {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let right = normalize(cross(up, camera_dir));
    let final_up = normalize(cross(camera_dir, right));

    // Project uv onto billboard plane
    // apply scaling based on rotation/scale (approximate)
    let local_pos = (right * uv.x + final_up * uv.y) * max(max(instance.scale.x, instance.scale.y), instance.scale.z);

    let world_pos = interpolated_pos + local_pos;

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

    // Apply Neural Color Filtering
    // (In a full implementation, we'd have decoupled tracks, for now we modulate the base color)
    let final_color = in.color.rgb * lighting;

    return vec4<f32>(final_color, alpha);
}
