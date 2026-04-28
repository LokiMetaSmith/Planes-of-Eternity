// shader_lambda.wgsl

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    time: vec4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct InstanceInput {
    @location(2) instance_pos: vec3<f32>,
    @location(3) instance_color: vec4<f32>,
    @location(4) instance_scale: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    // Procedural Wobble/Squash Animation
    let t = camera.time.x;

    // Scale animation intensity based on scale (bubbles vs. small objects)
    let is_animated = step(0.1, instance.instance_scale);
    let is_tron_bit = step(instance.instance_scale, -0.5); // If scale is negative, it's the "Bit"
    let real_scale = abs(instance.instance_scale);

    let wobble = sin(t * 3.0 + instance.instance_pos.x * 0.5 + instance.instance_pos.z * 0.5) * 0.1 * is_animated * (1.0 - is_tron_bit);
    let squash = cos(t * 5.0 + instance.instance_pos.y * 0.5) * 0.05 * is_animated * (1.0 - is_tron_bit);

    var animated_pos = model.position;
    animated_pos.x *= 1.0 + squash;
    animated_pos.z *= 1.0 + squash;
    animated_pos.y *= 1.0 - squash * 2.0; // Preserve volume roughly
    animated_pos += model.normal * wobble;

    // Tron Bit geometric animation: shape morphs blocky based on time
    if (is_tron_bit > 0.5) {
        let bit_state = step(0.0, sin(t * 2.0 + instance.instance_pos.x));
        // State 0: Diamond shape
        let diamond_pos = sign(animated_pos) * min(abs(animated_pos.x) + abs(animated_pos.y) + abs(animated_pos.z), 1.0);
        // State 1: Blocky cube
        let cube_pos = sign(animated_pos);
        animated_pos = mix(diamond_pos, cube_pos, bit_state) * 0.5;
        // Float animation
        animated_pos.y += sin(t * 4.0) * 0.2;
    }

    let world_pos = (animated_pos * real_scale) + instance.instance_pos;
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = instance.instance_color;
    out.normal = normalize(model.normal);
    out.world_pos = world_pos;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let view_dir = normalize(camera.camera_pos.xyz - in.world_pos);
    let N = normalize(in.normal);

    // Fresnel / Rim Light
    let NdotV = max(dot(N, view_dir), 0.0);
    let fresnel = pow(1.0 - NdotV, 3.0);

    let base_alpha = in.color.a;

    var final_color = in.color.rgb;
    var final_alpha = base_alpha;

    if (base_alpha < 0.95) {
        // Translucent Bubble
        // Rim is more opaque
        final_alpha = max(base_alpha, fresnel * 0.9);
        // Rim is whiteish
        final_color = mix(in.color.rgb, vec3<f32>(0.8, 0.9, 1.0), fresnel);
    } else {
        // Opaque Object
        let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.5));
        let diffuse = max(dot(N, light_dir), 0.0);
        let ambient = 0.3;
        final_color = in.color.rgb * (ambient + diffuse);
    }

    return vec4<f32>(final_color, final_alpha);
}

// Line Pipeline

struct LineVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_line(
    model: LineVertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);
    out.color = model.color;
    out.normal = vec3<f32>(0.0, 1.0, 0.0);
    out.world_pos = model.position;
    return out;
}

@fragment
fn fs_line(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
