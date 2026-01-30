use candle_core::{Result, Tensor, Module};
use candle_nn::{Linear, VarBuilder};

#[derive(Debug, Clone)]
pub struct LamConfig {
    pub input_dim: usize, // e.g. embedding_dim * H * W (flattened) or compressed
    pub hidden_dim: usize,
    pub num_actions: usize,
    pub num_layers: usize,
}

impl Default for LamConfig {
    fn default() -> Self {
        Self {
            input_dim: 64 * 8 * 8, // Assuming 128x128 image downsampled by 4 layers (1/16) -> 8x8? No, 2 layers (1/4) -> 32x32.
            // Let's assume the VQ-VAE bottleneck is roughly H/4 x W/4.
            // If we flatten the whole latent, it's huge.
            // A real LAM usually uses a visual encoder (like a ResNet or Transformer) to process the tokens.
            // For this simpler implementation, we'll assume we take the raw latents, flatten them, and pass through MLP.
            hidden_dim: 256,
            num_actions: 8,
            num_layers: 2,
        }
    }
}

#[derive(Debug)]
pub struct LatentActionModel {
    layers: Vec<Linear>,
    head: Linear,
}

impl LatentActionModel {
    pub fn new(cfg: &LamConfig, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        let mut curr_dim = cfg.input_dim * 2; // Concatenate 2 frames

        for i in 0..cfg.num_layers {
            let layer = candle_nn::linear(curr_dim, cfg.hidden_dim, vb.pp(format!("fc_{}", i)))?;
            layers.push(layer);
            curr_dim = cfg.hidden_dim;
        }

        let head = candle_nn::linear(curr_dim, cfg.num_actions, vb.pp("head"))?;

        Ok(Self { layers, head })
    }

    pub fn forward(&self, frame1_latents: &Tensor, frame2_latents: &Tensor) -> Result<Tensor> {
        // frames: [B, C, H, W] or similar. We need to flatten.
        let (_b, _, _, _) = frame1_latents.dims4()?;

        let f1_flat = frame1_latents.flatten_from(1)?;
        let f2_flat = frame2_latents.flatten_from(1)?;

        let mut xs = Tensor::cat(&[&f1_flat, &f2_flat], 1)?;

        for layer in &self.layers {
            xs = layer.forward(&xs)?;
            xs = xs.relu()?;
        }

        self.head.forward(&xs)
        // Returns logits [B, num_actions]
    }

    pub fn predict_action(&self, frame1: &Tensor, frame2: &Tensor) -> Result<Tensor> {
        let logits = self.forward(frame1, frame2)?;
        logits.argmax(1)
    }
}
