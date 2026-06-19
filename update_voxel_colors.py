import re

with open("reality-engine/src/voxel.rs", "r") as f:
    content = f.read()

colors = """        match id {
            1 => [0.5, 0.5, 0.5],   // 1: Stone
            2 => [1.0, 0.3, 0.0],   // 2: Lava
            3 => [1.0, 0.8, 0.0],   // 3: Fire
            4 => [0.0, 0.5, 1.0],   // 4: Water
            5 => [0.2, 0.8, 0.2],   // 5: Grass
            6 => [0.4, 0.2, 0.0],   // 6: Wood
            7 => [0.2, 0.6, 0.2],   // 7: Leaves
            8 => [0.4, 0.3, 0.2],   // 8: Dirt
            9 => [0.8, 0.8, 0.6],   // 9: Sand
            10 => [0.2, 1.0, 0.2],  // 10: Acid
            11 => [0.7, 0.7, 0.8],  // 11: Fog
            12 => [0.9, 0.9, 0.95], // 12: Cloud
            13 => [0.4, 0.5, 0.8],  // 13: Rain
            _ => [1.0, 0.0, 1.0],   // Magenta error
        }"""

# Replace in `get_color_for_id`
content = re.sub(r'match id \{\n(?:.|\n)*?\}', colors, content, count=1)

# The second `get_color` function has a slightly different fallback, so let's just replace both.
content = re.sub(r'match id \{\n(?:.|\n)*?\}', colors, content)

with open("reality-engine/src/voxel.rs", "w") as f:
    f.write(content)
