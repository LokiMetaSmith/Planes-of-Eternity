import re

with open("reality-engine/src/texture.rs", "r") as f:
    content = f.read()

new_func = """    pub fn create_procedural_atlas(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let size = 512;
        let mut pixels = vec![0u8; size * size * 4];

        for y in 0..size {
            for x in 0..size {
                let idx = (x + y * size) * 4;
                let u = x as f32 / size as f32;
                let v = y as f32 / size as f32;

                // 4x4 Grid (16 Tiles, each 128x128)
                let col = (u * 4.0).floor() as usize;
                let row = (v * 4.0).floor() as usize;
                let tile_idx = row * 4 + col;

                let tile_x = x % 128;
                let tile_y = y % 128;
                let lx = tile_x as f32 * 0.1;
                let ly = tile_y as f32 * 0.1;

                let mut r = 255u8;
                let mut g = 0u8;
                let mut b = 255u8;

                match tile_idx {
                    0 => {
                        // 0: Stone
                        let n = fbm(lx, ly, 3);
                        let val = (100.0 + n * 50.0) as u8;
                        let gx = (lx * 2.0).fract();
                        let gy = (ly * 2.0).fract();
                        let edge = if gx < 0.1 || gy < 0.1 { 0.5 } else { 1.0 };
                        let f = (val as f32 * edge) as u8;
                        r = f; g = f; b = f;
                    }
                    1 => {
                        // 1: Lava
                        let lx2 = lx * 0.5;
                        let ly2 = ly * 0.5;
                        let n = fbm(lx2, ly2, 4);
                        let heat = (n * 2.0).clamp(0.0, 1.0);
                        r = lerp(50.0, 255.0, heat) as u8;
                        g = lerp(0.0, 200.0, heat * heat) as u8;
                        b = 0;
                    }
                    2 => {
                        // 2: Fire
                        let lx2 = lx * 0.5;
                        let ly2 = ly;
                        let n = fbm(lx2, ly2 - (lx2 * 0.5), 3);
                        r = (200.0 + 55.0 * n) as u8;
                        g = (100.0 + 100.0 * n) as u8;
                        b = 0;
                    }
                    3 => {
                        // 3: Wood
                        let lx2 = lx;
                        let ly2 = ly * 0.2;
                        let grain = noise_2d(lx2 * 2.0, ly2 * 0.5);
                        let wave = ((lx2 * 2.0 + grain).sin() + 1.0) * 0.5;
                        let base = 80.0;
                        let var = 40.0 * wave;
                        r = (base + var) as u8;
                        g = ((base + var) * 0.6) as u8;
                        b = ((base + var) * 0.3) as u8;
                    }
                    4 => {
                        // 4: Grass
                        let n = fbm(lx * 2.0, ly * 2.0, 4);
                        r = (30.0 + n * 20.0) as u8;
                        g = (120.0 + n * 60.0) as u8;
                        b = (30.0 + n * 20.0) as u8;
                    }
                    5 => {
                        // 5: Leaves
                        let lx2 = lx * 1.5;
                        let ly2 = ly * 1.5;
                        let n = noise_2d(lx2, ly2);
                        // Make it look a bit cellular or spotty
                        let spots = (n * 10.0).sin() * 0.5 + 0.5;
                        r = (20.0 + spots * 30.0) as u8;
                        g = (80.0 + spots * 80.0) as u8;
                        b = (20.0 + spots * 30.0) as u8;
                    }
                    6 => {
                        // 6: Dirt
                        let n = fbm(lx * 1.5, ly * 1.5, 3);
                        r = (80.0 + n * 30.0) as u8;
                        g = (50.0 + n * 20.0) as u8;
                        b = (30.0 + n * 10.0) as u8;
                    }
                    7 => {
                        // 7: Sand
                        let n = fbm(lx * 3.0, ly * 3.0, 2);
                        r = (210.0 + n * 30.0) as u8;
                        g = (190.0 + n * 30.0) as u8;
                        b = (140.0 + n * 20.0) as u8;
                    }
                    8 => {
                        // 8: Water
                        let n = fbm(lx * 1.2, ly * 1.2, 3);
                        r = (20.0 + n * 20.0) as u8;
                        g = (60.0 + n * 40.0) as u8;
                        b = (180.0 + n * 50.0) as u8;
                    }
                    _ => {
                        // Fallback
                        let n = fbm(lx, ly, 2);
                        let f = (n * 255.0) as u8;
                        r = f; g = 0; b = f;
                    }
                }

                pixels[idx] = r;
                pixels[idx + 1] = g;
                pixels[idx + 2] = b;
                pixels[idx + 3] = 255;
            }
        }

        let texture_size = wgpu::Extent3d {"""

content = re.sub(r'pub fn create_procedural_atlas.*?(?=width: size as u32,)', new_func, content, flags=re.DOTALL)

with open("reality-engine/src/texture.rs", "w") as f:
    f.write(content)
