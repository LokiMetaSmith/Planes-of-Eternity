import re

with open("reality-engine/src/voxel.rs", "r") as f:
    content = f.read()

# Add 2D noise helper
helper = """fn noise2d(x: f32, z: f32) -> f32 {
    let xi = x.floor() as i32;
    let zi = z.floor() as i32;
    let xf = x - x.floor();
    let zf = z - z.floor();

    let bl = hash(xi, 0, zi).abs();
    let br = hash(xi + 1, 0, zi).abs();
    let tl = hash(xi, 0, zi + 1).abs();
    let tr = hash(xi + 1, 0, zi + 1).abs();

    let u = xf * xf * (3.0 - 2.0 * xf);
    let v = zf * zf * (3.0 - 2.0 * zf);

    let b = bl + u * (br - bl);
    let t = tl + u * (tr - tl);
    b + v * (t - b)
}

fn hash("""

content = content.replace("fn hash(", helper)

# Update Fantasy Archetype to use procedural terrain
new_fantasy = """                        Some(crate::reality_types::RealityArchetype::Fantasy) | None => {
                            // Procedural Rolling Hills Terrain
                            let nx = wx as f32 * 0.05;
                            let nz = wz as f32 * 0.05;
                            // Add octaves for more organic hills
                            let height_noise = (noise2d(nx, nz) * 0.5 + noise2d(nx * 2.0, nz * 2.0) * 0.25) * 20.0 - 5.0;
                            let terrain_height = height_noise as i32;

                            if wy <= terrain_height {
                                if wy == terrain_height {
                                    voxel.id = 5; // Grass
                                } else if wy > terrain_height - 3 {
                                    voxel.id = 8; // Dirt
                                } else {
                                    voxel.id = 1; // Stone
                                }
                            }

                            // Water Level (Ocean/Lakes)
                            if wy <= -2 && voxel.id == 0 {
                                voxel.id = 4; // Water
                            }

                            // Sand on shores
                            if wy == terrain_height && wy >= -2 && wy <= 0 && voxel.id == 5 {
                                voxel.id = 9; // Sand
                            }

                            // Trees
                            // We check tree placement deterministically on a local grid (e.g. 10x10)
                            if wy > terrain_height && terrain_height > 0 && voxel.id == 0 {
                                let cell_x = wx / 8;
                                let cell_z = wz / 8;
                                // Tree anchor point in this cell
                                let tx = cell_x * 8 + (hash(cell_x, 0, cell_z).abs() * 8.0) as i32;
                                let tz = cell_z * 8 + (hash(cell_z, 0, cell_x).abs() * 8.0) as i32;

                                if wx == tx && wz == tz {
                                    // Tree trunk
                                    if wy <= terrain_height + 4 {
                                        voxel.id = 6; // Wood
                                    }
                                }

                                // Tree canopy (Leaves)
                                let dx = (wx - tx).abs();
                                let dy = (wy - (terrain_height + 4)).abs();
                                let dz = (wz - tz).abs();

                                if dy <= 2 && dx * dx + dy * dy + dz * dz <= 5 && voxel.id == 0 {
                                    if hash(wx, wy, wz).abs() > 0.1 {
                                        voxel.id = 7; // Leaves
                                    }
                                }
                            }

                            // Castle (Procedural) - Keeping for backwards compatibility
                            if (-9..=9).contains(&wx)
                                && (-9..=9).contains(&wz)
                                && (0..10).contains(&wy)
                                && (wx.abs() > 8 || wz.abs() > 8 || wy == 0)
                            {
                                voxel.id = 1; // Stone
                            }

                            // Fire Noise (Fireflies or embers)
                            if voxel.id == 0 && wy > terrain_height + 5 && wy < terrain_height + 10 {
                                let n = hash(wx, wy, wz);
                                if n > 0.999 {
                                    voxel.id = 3; // Fire
                                }
                            }
                        }"""

content = re.sub(r'Some\(crate::reality_types::RealityArchetype::Fantasy\) \| None => \{.*?\}(?=\s*Some\(crate::reality_types::RealityArchetype::SciFi\))', new_fantasy, content, flags=re.DOTALL)

with open("reality-engine/src/voxel.rs", "w") as f:
    f.write(content)
