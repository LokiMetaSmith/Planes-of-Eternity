// shader_lambda.wgsl

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
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
    let world_pos = (model.position * instance.instance_scale) + instance.instance_pos;
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
