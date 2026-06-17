import sys
import re

filepath = "reality-engine/src/shader_voxel.wgsl"
with open(filepath, 'r') as f:
    content = f.read()

fs_replacement = """
    // 3. Lighting
    let time = reality.global_offset.z;
    let cycle = time * 0.1;
    let light_x = sin(cycle);
    let light_y = cos(cycle);
    let light_dir = normalize(vec3<f32>(light_x, light_y, 0.5));

    let view_dir = normalize(camera.camera_pos.xyz - in.world_pos);

    let dist1 = max(distance(in.world_pos, reality.proj1_pos_fid.xyz), 1.0);
    let str1 = reality.proj1_pos_fid.w / dist1;
    let dist2 = max(distance(in.world_pos, reality.proj2_pos_fid.xyz), 1.0);
    let str2 = reality.proj2_pos_fid.w / dist2;

    let total_str = str1 + str2;
    var w1 = 0.0;
    var w2 = 0.0;
    if (total_str > 0.0001) {
        w1 = str1 / total_str;
        w2 = str2 / total_str;
    }

    let l1 = get_lighting_info(reality.proj1_params.w);
    let l2 = get_lighting_info(reality.proj2_params.w);

    let directional_weight = l1.x * w1 + l2.x * w2;
    let ambient_strength = l1.y * w1 + l2.y * w2;

    // Ambient varies with Day/Night ONLY if there is directional light.
    var ambient = ambient_strength;
    if (light_y < 0.0) {
        ambient = mix(ambient_strength, 0.05, directional_weight);
    }

    // Raytraced Shadow (only if sun is up)
    var shadow = 1.0;
    if (light_y > 0.0 && directional_weight > 0.001) {
        shadow = ray_march_shadow(in.world_pos, light_dir);
    } else {
        shadow = 0.0; // No direct sun at night or if ambient only
    }

    let diff = max(dot(in.normal, light_dir), 0.0) * shadow * directional_weight;

    // AO Factor
    let ao_factor = in.ao * 0.8 + 0.2;

    // Specular
    let half_dir = normalize(light_dir + view_dir);
    let spec_angle = max(dot(in.normal, half_dir), 0.0);
    let specular = pow(spec_angle, 32.0) * specular_strength * shadow * directional_weight;
"""

fs_search = """
    // 3. Lighting
    let time = reality.global_offset.z;
    let cycle = time * 0.1;
    let light_x = sin(cycle);
    let light_y = cos(cycle);
    let light_dir = normalize(vec3<f32>(light_x, light_y, 0.5));

    let view_dir = normalize(camera.camera_pos.xyz - in.world_pos);

    // Ambient varies with Day/Night
    var ambient = 0.3;
    if (light_y < 0.0) { ambient = 0.05; } // Darker at night

    // Raytraced Shadow (only if sun is up)
    var shadow = 1.0;
    if (light_y > 0.0) {
        shadow = ray_march_shadow(in.world_pos, light_dir);
    } else {
        shadow = 0.0; // No direct sun at night
    }

    let diff = max(dot(in.normal, light_dir), 0.0) * shadow;

    // AO Factor
    let ao_factor = in.ao * 0.8 + 0.2;

    // Specular
    let half_dir = normalize(light_dir + view_dir);
    let spec_angle = max(dot(in.normal, half_dir), 0.0);
    let specular = pow(spec_angle, 32.0) * specular_strength * shadow;
"""

if fs_search.strip() not in content:
    print("Could not find voxel fs_search block")
else:
    print("Found voxel fs_search block!")

content = content.replace(fs_search.strip(), fs_replacement.strip())

with open(filepath, 'w') as f:
    f.write(content)
