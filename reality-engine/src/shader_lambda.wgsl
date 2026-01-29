// shader_lambda.wgsl

struct CameraUniform {
    view_proj: mat4x4<f32>,
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
    out.normal = model.normal; // Assume scale is uniform, so normal is valid
    out.world_pos = world_pos;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Simple Lighting
    let light_dir = normalize(vec3<f32>(1.0, 2.0, 3.0));
    let diffuse = max(dot(in.normal, light_dir), 0.0);
    let ambient = 0.3;

    // Add "glow" (rim lighting)
    let view_dir = normalize(vec3<f32>(0.0, 0.0, 1.0)); // Rough approximation or pass camera pos
    // Actually, simple emissive color is enough for "glowing".
    // Let's make the edges glow more.

    let lighting = ambient + diffuse;
    let final_color = in.color.rgb * lighting;

    return vec4<f32>(final_color, in.color.a);
}
