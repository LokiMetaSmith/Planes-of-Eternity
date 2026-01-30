use candle_core::{Result, Tensor, Module};
use candle_nn::{Conv2d, ConvTranspose2d, Conv2dConfig, ConvTranspose2dConfig, VarBuilder};

#[derive(Debug, Clone)]
pub struct VqVaeConfig {
    pub in_channels: usize,
    pub hidden_channels: usize,
    pub num_embeddings: usize,
    pub embedding_dim: usize,
    pub num_residual_layers: usize,
    pub num_downsample_layers: usize,
}

impl Default for VqVaeConfig {
    fn default() -> Self {
        Self {
            in_channels: 3,
            hidden_channels: 128,
            num_embeddings: 512,
            embedding_dim: 64,
            num_residual_layers: 2,
            num_downsample_layers: 2,
        }
    }
}

// Helper for residual block
#[derive(Debug)]
struct ResidualBlock {
    conv1: Conv2d,
    conv2: Conv2d,
}

impl ResidualBlock {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig { padding: 1, ..Default::default() };
        let conv1 = candle_nn::conv2d(channels, channels, 3, cfg, vb.pp("conv1"))?;
        let conv2 = candle_nn::conv2d(channels, channels, 3, cfg, vb.pp("conv2"))?;
        Ok(Self { conv1, conv2 })
    }
}

impl Module for ResidualBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let ys = self.conv1.forward(xs)?;
        let ys = ys.relu()?;
        let ys = self.conv2.forward(&ys)?;
        let ys = ys.relu()?;
        ys + residual
    }
}

#[derive(Debug)]
pub struct Encoder {
    conv_in: Conv2d,
    downsamples: Vec<Conv2d>,
    residuals: Vec<ResidualBlock>,
    conv_out: Conv2d,
}

impl Encoder {
    pub fn new(cfg: &VqVaeConfig, vb: VarBuilder) -> Result<Self> {
        let conv_in = candle_nn::conv2d(cfg.in_channels, cfg.hidden_channels, 3, Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("conv_in"))?;

        let mut downsamples = Vec::new();
        let curr_channels = cfg.hidden_channels;
        for i in 0..cfg.num_downsample_layers {
            // Strided conv for downsampling
            let conv = candle_nn::conv2d(
                curr_channels,
                curr_channels, // keeping channels same for simplicity in this demo, or usually double
                4,
                Conv2dConfig { stride: 2, padding: 1, ..Default::default() },
                vb.pp(format!("down_{}", i))
            )?;
            downsamples.push(conv);
        }

        let mut residuals = Vec::new();
        for i in 0..cfg.num_residual_layers {
            residuals.push(ResidualBlock::new(curr_channels, vb.pp(format!("res_{}", i)))?);
        }

        let conv_out = candle_nn::conv2d(curr_channels, cfg.embedding_dim, 1, Default::default(), vb.pp("conv_out"))?;

        Ok(Self { conv_in, downsamples, residuals, conv_out })
    }
}

impl Module for Encoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut ys = self.conv_in.forward(xs)?;
        ys = ys.relu()?;

        for down in &self.downsamples {
            ys = down.forward(&ys)?;
            ys = ys.relu()?;
        }

        for res in &self.residuals {
            ys = res.forward(&ys)?;
        }

        self.conv_out.forward(&ys)
    }
}

#[derive(Debug)]
pub struct VectorQuantizer {
    embedding: Tensor, // [num_embeddings, embedding_dim]
    num_embeddings: usize,
    embedding_dim: usize,
}

impl VectorQuantizer {
    pub fn new(cfg: &VqVaeConfig, vb: VarBuilder) -> Result<Self> {
        let embedding = vb.get((cfg.num_embeddings, cfg.embedding_dim), "embedding")?;
        Ok(Self {
            embedding,
            num_embeddings: cfg.num_embeddings,
            embedding_dim: cfg.embedding_dim,
        })
    }

    pub fn forward(&self, z: &Tensor) -> Result<(Tensor, Tensor)> {
        // z: [B, C, H, W] -> permute to [B, H, W, C]
        let (b, c, h, w) = z.dims4()?;
        let z_permuted = z.permute((0, 2, 3, 1))?;
        let z_flat = z_permuted.reshape((b * h * w, c))?;

        // Distances: (z - e)^2 = z^2 + e^2 - 2ze
        // z_flat: [N, C]
        // embedding: [K, C]

        let z2 = z_flat.sqr()?.sum_keepdim(1)?; // [N, 1]
        let e2 = self.embedding.sqr()?.sum_keepdim(1)?; // [K, 1]
        let e2 = e2.reshape((1, self.num_embeddings))?; // [1, K]

        let ze = z_flat.matmul(&self.embedding.t()?)?; // [N, K]

        // z2 [N, 1] broadcasts to [N, K]
        // e2 [1, K] broadcasts to [N, K]

        let e2_broadcast = e2.broadcast_as((b*h*w, self.num_embeddings))?;
        let z2_broadcast = z2.broadcast_as((b*h*w, self.num_embeddings))?;

        let dist = ((z2_broadcast + e2_broadcast)? - (ze * 2.0)?)?;

        let encoding_indices = dist.argmin(1)?; // [N]

        // Quantize
        // embedding: [K, C]
        // indices: [N]
        // We want to gather embeddings.
        let quantized_flat = self.embedding.index_select(&encoding_indices, 0)?; // [N, C]

        let quantized = quantized_flat.reshape((b, h, w, c))?.permute((0, 3, 1, 2))?; // [B, C, H, W]

        // Straight-through estimator:
        // We want gradients to flow from Decoder to Encoder, bypassing the quantization step mathematically during backprop.
        // In PyTorch: z_q = z + (z_q - z).detach()
        // In Candle, we can use `detach()`

        let quantized_final = ((z - z.detach())? + quantized.detach())?;

        Ok((quantized_final, encoding_indices.reshape((b, h, w))?))
    }

    // Helper to decode indices directly
    pub fn decode_indices(&self, indices: &Tensor) -> Result<Tensor> {
        // indices: [B, H, W] or [N]
        // Flatten
        let _shape = indices.dims();
        let indices_flat = indices.flatten_all()?;
        let quantized_flat = self.embedding.index_select(&indices_flat, 0)?;

        // If input was [B, H, W], output should be [B, C, H, W] (needs permutation)
        // Here we just return flat or guess based on original shape?
        // Better to assume caller handles reshape or we pass dims.
        // For simplicity, assuming [B, H, W] input -> return [B, C, H, W] logic needs C.

        // Actually this function is usually used by decoder which takes quantized tensor.
        // But the Transformer predicts indices. So we need this.

        Ok(quantized_flat)
    }
}

#[derive(Debug)]
pub struct Decoder {
    conv_in: Conv2d,
    upsamples: Vec<ConvTranspose2d>,
    residuals: Vec<ResidualBlock>,
    conv_out: Conv2d,
}

impl Decoder {
    pub fn new(cfg: &VqVaeConfig, vb: VarBuilder) -> Result<Self> {
        let conv_in = candle_nn::conv2d(cfg.embedding_dim, cfg.hidden_channels, 3, Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("conv_in"))?;

        let mut residuals = Vec::new();
        let curr_channels = cfg.hidden_channels;

        for i in 0..cfg.num_residual_layers {
            residuals.push(ResidualBlock::new(curr_channels, vb.pp(format!("res_{}", i)))?);
        }

        let mut upsamples = Vec::new();
        for i in 0..cfg.num_downsample_layers {
            let conv = candle_nn::conv_transpose2d(
                curr_channels,
                curr_channels,
                4,
                ConvTranspose2dConfig { stride: 2, padding: 1, ..Default::default() },
                vb.pp(format!("up_{}", i))
            )?;
            upsamples.push(conv);
        }

        let conv_out = candle_nn::conv2d(curr_channels, cfg.in_channels, 3, Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("conv_out"))?;

        Ok(Self { conv_in, residuals, upsamples, conv_out })
    }
}

impl Module for Decoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut ys = self.conv_in.forward(xs)?;
        ys = ys.relu()?;

        for res in &self.residuals {
            ys = res.forward(&ys)?;
        }

        for up in &self.upsamples {
            ys = up.forward(&ys)?;
            ys = ys.relu()?;
        }

        self.conv_out.forward(&ys)
    }
}

#[derive(Debug)]
pub struct VqVae {
    pub encoder: Encoder,
    pub quantizer: VectorQuantizer,
    pub decoder: Decoder,
}

impl VqVae {
    pub fn new(cfg: &VqVaeConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = Encoder::new(cfg, vb.pp("encoder"))?;
        let quantizer = VectorQuantizer::new(cfg, vb.pp("quantizer"))?;
        let decoder = Decoder::new(cfg, vb.pp("decoder"))?;
        Ok(Self { encoder, quantizer, decoder })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        // Returns (reconstruction, quantization_loss_info_ignored_here, indices)
        let z = self.encoder.forward(xs)?;
        let (quantized, indices) = self.quantizer.forward(&z)?;
        let reconstruction = self.decoder.forward(&quantized)?;
        Ok((reconstruction, quantized, indices))
    }

    // For inference from tokens
    pub fn decode_from_indices(&self, indices: &Tensor) -> Result<Tensor> {
         // indices: [B, H, W]
         let (b, h, w) = indices.dims3()?;
         let indices_flat = indices.flatten_all()?;
         let quantized_flat = self.quantizer.embedding.index_select(&indices_flat, 0)?; // [B*H*W, C]
         let c = self.quantizer.embedding_dim;

         // Reshape back to [B, H, W, C] then permute to [B, C, H, W]
         let quantized = quantized_flat.reshape((b, h, w, c))?.permute((0, 3, 1, 2))?;

         self.decoder.forward(&quantized)
    }

    // For encoding to tokens
    pub fn encode_to_indices(&self, xs: &Tensor) -> Result<Tensor> {
        let z = self.encoder.forward(xs)?;
        let (_, indices) = self.quantizer.forward(&z)?;
        Ok(indices)
    }
}
