import sys
import re

filepath = "reality-engine/src/shader.wgsl"
with open(filepath, 'r') as f:
    content = f.read()

# Add get_lighting_info
# Ambient: 0, 1, 2, 6, 8, 9, 10, 11, 15, 16, 17
# Directional: 3, 4, 5, 7, 12, 13, 14, 18, 19
# We can do this based on the ID

lighting_func = """
fn get_lighting_info(id: f32) -> vec2<f32> {
    // x = directional weight, y = ambient strength
    if (id < -0.5) {
        return vec2<f32>(0.0, 0.4); // Void (ambient only)
    } else if (id < 0.5) {
        return vec2<f32>(1.0, 0.3); // Fantasy
    } else if (id < 1.5) {
        return vec2<f32>(0.0, 0.5); // SciFi (ambient)
    } else if (id < 2.5) {
        return vec2<f32>(0.0, 0.2); // Horror (ambient)
    } else if (id < 3.5) {
        return vec2<f32>(1.0, 0.3); // Toon
    } else if (id < 4.5) {
        return vec2<f32>(1.0, 0.3); // HyperNature
    } else if (id < 5.5) {
        return vec2<f32>(1.0, 0.3); // Genie
    } else if (id < 6.5) {
        return vec2<f32>(0.0, 0.5); // Glitch (ambient)
    } else if (id < 7.5) {
        return vec2<f32>(1.0, 0.3); // Steampunk
    } else if (id < 8.5) {
        return vec2<f32>(0.0, 0.5); // Vaporwave (ambient)
    } else if (id < 9.5) {
        return vec2<f32>(0.0, 0.4); // Noir (ambient)
    } else if (id < 10.5) {
        return vec2<f32>(0.0, 0.5); // CyberSpace (ambient)
    } else if (id < 11.5) {
        return vec2<f32>(0.0, 0.6); // Dream (ambient)
    } else if (id < 12.5) {
        return vec2<f32>(1.0, 0.3); // ObraDinn
    } else if (id < 13.5) {
        return vec2<f32>(1.0, 0.3); // SolarPunk
    } else if (id < 14.5) {
        return vec2<f32>(1.0, 0.3); // Biopunk
    } else if (id < 15.5) {
        return vec2<f32>(0.0, 0.5); // Tron (ambient)
    } else if (id < 16.5) {
        return vec2<f32>(0.0, 0.6); // ColdStorage (ambient)
    } else if (id < 17.5) {
        return vec2<f32>(0.0, 0.7); // LiminalSpace (ambient)
    } else if (id < 18.5) {
        return vec2<f32>(1.0, 0.3); // Clockwork
    } else if (id < 19.5) {
        return vec2<f32>(1.0, 0.3); // Cottagecore
    }

    return vec2<f32>(0.5, 0.3); // fallback
}
"""

content = content.replace("fn get_displacement", lighting_func + "\nfn get_displacement")

fs_replacement = """
    // Lighting
    let time = reality.global_offset.z;
    let cycle = time * 0.1;
    let light_x = sin(cycle);
    let light_y = cos(cycle);
    let light_dir = normalize(vec3<f32>(light_x, light_y, 0.5));

    var directional_weight = 0.0;
    var ambient_strength = 0.0;

    for (var i = 0u; i < 5u; i = i + 1u) {
        if (weights[i] > 0.001) {
            let l_info = get_lighting_info(reality.proj_params[i].w);
            directional_weight = directional_weight + l_info.x * weights[i];
            ambient_strength = ambient_strength + l_info.y * weights[i];
        }
    }

    for (var i = 0u; i < 4u; i = i + 1u) {
        if (node_weights[i] > 0.001) {
            let l_info = get_lighting_info(reality.nodes_params[i].w);
            directional_weight = directional_weight + l_info.x * node_weights[i];
            ambient_strength = ambient_strength + l_info.y * node_weights[i];
        }
    }

    // Shadow
    var shadow = 1.0;
    if (light_y > 0.0 && directional_weight > 0.001) {
        shadow = ray_march_shadow(in.world_pos, light_dir);
    } else {
        shadow = 0.0; // No direct sun at night or if no directional light
    }

    let diffuse = max(dot(in.normal, light_dir), 0.0) * shadow * directional_weight;

    // Ambient varies with Day/Night ONLY if there is directional light.
    // If it's pure ambient, keep it constant (e.g. indoor/cyberpunk doesn't get dark at "night")
    var ambient = ambient_strength;
    if (light_y < 0.0) {
        // Fade out ambient slightly at night based on how directional the area is
        ambient = mix(ambient_strength, 0.05, directional_weight);
    }

    let lighting = diffuse + ambient;
"""

fs_search = """
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
        shadow = 0.0; // No direct sun at night
    }

    let diffuse = max(dot(in.normal, light_dir), 0.0) * shadow;

    // Ambient varies with Day/Night
    var ambient = 0.3;
    if (light_y < 0.0) { ambient = 0.05; }

    let lighting = diffuse + ambient;
"""

if fs_search.strip() not in content:
    print("Could not find fs_search block")

content = content.replace(fs_search.strip(), fs_replacement.strip())

with open(filepath, 'w') as f:
    f.write(content)
