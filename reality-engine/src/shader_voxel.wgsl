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
    out.color = model.color; // Contains Material ID or Color tint?
    // In current impl, 'color' is rgb. We need material ID.
    // Actually, color passed is RGB. But we can deduce material from RGB since we hardcoded it in voxel.rs.
    // Or we can just use the tint for now.
    // To support textures properly, we should pass Material ID as an attribute.
    // But since I can't change vertex struct in this step without re-doing previous work,
    // I will use World Pos to generate UVs and map "regions" based on simple logic or just use one texture for now + color tint.

    // Better: Assume the atlas quadrants are mixed by color.
    // Stone is grey, Lava is orange.

    out.normal = model.normal;
    out.ao = model.ao;
    out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Lighting
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.5));
    let diffuse = max(dot(in.normal, light_dir), 0.2);

    // AO
    let ao_factor = in.ao * 0.8 + 0.2;

    // Texture Mapping (Triplanar-ish logic)
    // Map world pos to UVs.
    // If Normal is Y, use XZ. If X, use YZ. If Z, use XY.
    var uv = vec2<f32>(0.0, 0.0);
    let n = abs(in.normal);
    if (n.y > 0.5) {
        uv = in.world_pos.xz;
    } else if (n.x > 0.5) {
        uv = in.world_pos.yz;
    } else {
        uv = in.world_pos.xy;
    }

    // Determine Material Quadrant based on "Color" passed from vertex
    // Color was: Stone(0.5,0.5,0.5), Lava(1.0,0.3,0.0), Fire(1.0,0.8,0.0), Water(0,0,1), Grass(0,0.8,0), Wood(0.6,0.4,0.2)

    var offset = vec2<f32>(0.0, 0.0);

    if (in.color.r > 0.9 && in.color.g < 0.5) { // Lava (Reddish)
        offset = vec2<f32>(0.5, 0.0);
    } else if (in.color.r > 0.9 && in.color.g > 0.7) { // Fire (Yellow)
        offset = vec2<f32>(0.0, 0.5);
    } else if (in.color.g > 0.7) { // Grass (Green) -> Wood slot for now
        offset = vec2<f32>(0.5, 0.5);
    } else if (in.color.b > 0.8) { // Water (Blue) -> Stone slot (tinted blue)
        offset = vec2<f32>(0.0, 0.0);
    } else { // Stone/Wood/Other
        if (in.color.r > 0.5 && in.color.b < 0.3) { // Wood
             offset = vec2<f32>(0.5, 0.5);
        } else { // Stone
             offset = vec2<f32>(0.0, 0.0);
        }
    }

    // Scale UV to 0.5 range (quadrant size)
    let uv_scaled = fract(uv) * 0.5;
    let final_uv = uv_scaled + offset;

    let tex_color = textureSample(t_atlas, s_atlas, final_uv);

    // Mix texture with vertex color tint
    let final_color = tex_color.rgb * in.color * diffuse * ao_factor;

    return vec4<f32>(final_color, 1.0);
}
