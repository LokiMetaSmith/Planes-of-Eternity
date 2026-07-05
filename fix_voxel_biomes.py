import sys

with open('reality-engine/src/voxel.rs', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'Some(crate::reality_types::RealityArchetype::SciFi) => {' in line:
        new_lines.append(line)
        new_lines.append('                            // Metal/Stone Ground\n')
        new_lines.append('                            if wy == -1 {\n')
        new_lines.append('                                voxel.id = 1;\n')
        new_lines.append('                            }\n')
        new_lines.append('\n')
        new_lines.append('                            // Procedural Pillars\n')
        new_lines.append('                            if wx % 10 == 0 && wz % 10 == 0 && (0..=10).contains(&wy) {\n')
        new_lines.append('                                if wy == 10 {\n')
        new_lines.append('                                    voxel.id = 4; // Glowing energy top\n')
        new_lines.append('                                } else {\n')
        new_lines.append('                                    voxel.id = 1; // Metal pillar\n')
        new_lines.append('                                }\n')
        new_lines.append('                            }\n')
        new_lines.append('                        }\n')
        new_lines.append('                        Some(crate::reality_types::RealityArchetype::Fractal) => {\n')
        new_lines.append('                            // Geometric Floating Islands\n')
        new_lines.append('                            let fx = wx as f32 * 0.1;\n')
        new_lines.append('                            let fy = wy as f32 * 0.1;\n')
        new_lines.append('                            let fz = wz as f32 * 0.1;\n')
        new_lines.append('                            let val = (fx.sin() * fy.cos() + fy.sin() * fz.cos() + fz.sin() * fx.cos()).abs();\n')
        new_lines.append('                            if val > 1.2 && wy > 5 {\n')
        new_lines.append('                                voxel.id = 1; // Stone islands\n')
        new_lines.append('                            }\n')
        # Skip the original SciFi block to avoid duplicates
        skip = True
        continue
    if 'Some(crate::reality_types::RealityArchetype::Horror) => {' in line:
        skip = False

    if not globals().get('skip', False):
        new_lines.append(line)

with open('reality-engine/src/voxel.rs', 'w') as f:
    f.writelines(new_lines)
