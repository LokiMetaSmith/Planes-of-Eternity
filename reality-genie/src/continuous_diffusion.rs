fn hash(x: i32, y: i32, z: i32) -> f32 {
    let mut n = x.wrapping_mul(374761393) ^ y.wrapping_mul(668265263) ^ z.wrapping_mul(393555907);
    n = (n ^ (n >> 13)).wrapping_mul(1274126177);
    (n as f32) / (i32::MAX as f32)
}

fn noise2d(x: f32, z: f32) -> f32 {
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

/// A scaffold for a continuous diffusion pipeline capable of generating
/// float-based terrain heightmaps.
pub struct ContinuousDiffusionGenerator;

impl Default for ContinuousDiffusionGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl ContinuousDiffusionGenerator {
    pub fn new() -> Self {
        Self
    }

    /// Simulates a non-blocking forward pass of continuous terrain diffusion.
    /// In a fully trained ML scenario, this would evaluate a diffusion model
    /// using SafeTensors weights. Here, we use procedural noise mapping.
    pub fn generate_heightmap(&self, chunk_x: i32, chunk_z: i32, size: usize) -> Vec<f32> {
        let mut heightmap = vec![0.0; size * size];

        let wx_base = chunk_x * size as i32;
        let wz_base = chunk_z * size as i32;

        for z in 0..size {
            for x in 0..size {
                let wx = wx_base + x as i32;
                let wz = wz_base + z as i32;

                let nx = wx as f32;
                let nz = wz as f32;

                let height_noise =
                    noise2d(nx * 0.02, nz * 0.02) * 16.0 +
                    noise2d(nx * 0.04, nz * 0.04) * 8.0 +
                    noise2d(nx * 0.08, nz * 0.08) * 4.0 +
                    noise2d(nx * 0.16, nz * 0.16) * 2.0;

                let terrain_height = height_noise - 5.0;

                heightmap[x + z * size] = terrain_height;
            }
        }

        heightmap
    }
}
