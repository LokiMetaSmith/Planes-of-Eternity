import re

with open("reality-engine/src/shader_voxel.wgsl", "r") as f:
    content = f.read()

new_logic = """    // 1. Material Logic
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
    let final_uv = uv_scaled + offset;"""

# Need to replace from "// 1. Material Logic" down to "let final_uv = uv_scaled + offset;"
content = re.sub(r'// 1\. Material Logic.*?let final_uv = uv_scaled \+ offset;', new_logic, content, flags=re.DOTALL)

with open("reality-engine/src/shader_voxel.wgsl", "w") as f:
    f.write(content)
