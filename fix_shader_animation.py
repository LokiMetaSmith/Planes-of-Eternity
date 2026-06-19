import re

with open("reality-engine/src/shader_voxel.wgsl", "r") as f:
    content = f.read()

new_anim = """    let r = model.color.r;
    let g = model.color.g;
    let b = model.color.b;

    // Liquid and Gas logic
    if (abs(r - 0.2) < 0.01 && abs(g - 1.0) < 0.01 && abs(b - 0.2) < 0.01) {
        // Acid
        animated_pos.y += sin(time * 3.0 + model.position.x * 2.0 + model.position.z * 2.0) * 0.1;
    } else if (abs(r - 0.0) < 0.01 && abs(g - 0.5) < 0.01 && abs(b - 1.0) < 0.01) {
        // Water
        animated_pos.y += sin(time * 1.5 + model.position.x + model.position.z) * 0.15;
    } else if (abs(r - 1.0) < 0.01 && abs(g - 0.3) < 0.01 && abs(b - 0.0) < 0.01) {
        // Lava
        animated_pos.y += sin(time * 0.5 + model.position.x * 0.5 + model.position.z * 0.5) * 0.05;
    } else if ((r > 0.7 && g > 0.7 && b > 0.7) || (b > 0.9 && r > 0.4 && g > 0.4 && r < 0.6)) {
        // Gasses/Weather (Fog, Cloud, Rain)
        animated_pos.x += sin(time * 0.5 + model.position.y) * 0.2;
        animated_pos.z += cos(time * 0.5 + model.position.y) * 0.2;
    } else if (abs(r - 0.2) < 0.01 && abs(g - 0.8) < 0.01 && abs(b - 0.2) < 0.01) || (abs(r - 0.2) < 0.01 && abs(g - 0.6) < 0.01 && abs(b - 0.2) < 0.01) {
        // Existing Landscape (Grass tops / leaves wobble slightly in wind)
        if (model.normal.y > 0.5) {
            animated_pos.x += sin(time * 2.0 + model.position.y) * 0.05;
        }
    }"""

content = re.sub(r'// Liquid and Gas logic.*?if \(model\.normal\.y > 0\.5\) \{\n            animated_pos\.x \+= sin\(time \* 2\.0 \+ model\.position\.y\) \* 0\.05;\n        \}\n    \}', new_anim, content, flags=re.DOTALL)

with open("reality-engine/src/shader_voxel.wgsl", "w") as f:
    f.write(content)
