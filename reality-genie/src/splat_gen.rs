pub trait SplatGenerator {
    fn generate_splats_from_prompt(&self, prompt: &str) -> Vec<[f32; 14]>;
}

pub struct DummySplatGenerator;

impl DummySplatGenerator {
    pub fn new() -> Self {
        Self
    }
}

impl SplatGenerator for DummySplatGenerator {
    fn generate_splats_from_prompt(&self, prompt: &str) -> Vec<[f32; 14]> {
        let mut splats = Vec::new();
        let color = if prompt.to_lowercase().contains("fire") {
            [1.0, 0.2, 0.0, 0.8]
        } else if prompt.to_lowercase().contains("water") {
            [0.1, 0.3, 1.0, 0.6]
        } else {
            [0.8, 0.8, 0.8, 0.9]
        };

        for i in 0..10 {
            splats.push([
                (i as f32) * 0.1, 0.5, (i as f32) * 0.1,
                0.0, 0.0, 0.0, 1.0,
                0.1, 0.1, 0.1,
                color[0], color[1], color[2], color[3]
            ]);
        }
        splats
    }
}


use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use crate::vqvae::{VqVae, VqVaeConfig};

pub struct GenieSplatGenerator {
    pub vqvae: VqVae,
}

impl GenieSplatGenerator {
    pub fn new() -> Self {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(candle_core::DType::F32, &device);

        let mut cfg = VqVaeConfig::default();
        cfg.in_channels = 14;

        let vqvae = VqVae::new(&cfg, vb).unwrap();

        Self { vqvae }
    }
}

impl SplatGenerator for GenieSplatGenerator {
    fn generate_splats_from_prompt(&self, _prompt: &str) -> Vec<[f32; 14]> {
        let device = Device::Cpu;
        let b = 1;
        let h = 16;
        let w = 16;
        let max_vocab = 512;

        let mut rng_data = Vec::with_capacity(b * h * w);
        for i in 0..(b * h * w) {
            rng_data.push((i % max_vocab) as u32);
        }
        let indices = Tensor::from_vec(rng_data, (b, h, w), &device).unwrap();

        let decoded = self.vqvae.decode_from_indices(&indices).unwrap();
        let permuted = decoded.permute((0, 2, 3, 1)).unwrap();
        let flat = permuted.flatten_all().unwrap();

        let flat_data = flat.to_vec1::<f32>().unwrap();

        let mut splats = Vec::new();
        for chunk in flat_data.chunks_exact(14) {
            let mut s = [0.0; 14];
            s.copy_from_slice(chunk);
            splats.push(s);
        }

        splats
    }
}
