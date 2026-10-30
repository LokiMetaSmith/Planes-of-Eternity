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
            let conv = candle_nn::conv2d(
                curr_channels,
                curr_channels,
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
    embedding: Tensor,
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
        let (b, c, h, w) = z.dims4()?;
        let z_permuted = z.permute((0, 2, 3, 1))?;
        let z_flat = z_permuted.reshape((b * h * w, c))?;

        let z2 = z_flat.sqr()?.sum_keepdim(1)?;
        let e2 = self.embedding.sqr()?.sum_keepdim(1)?;
        let e2 = e2.reshape((1, self.num_embeddings))?;

        let ze = z_flat.matmul(&self.embedding.t()?)?;

        let e2_broadcast = e2.broadcast_as((b*h*w, self.num_embeddings))?;
        let z2_broadcast = z2.broadcast_as((b*h*w, self.num_embeddings))?;

        let dist = ((z2_broadcast + e2_broadcast)? - (ze * 2.0)?)?;

        let encoding_indices = dist.argmin(1)?;

        let quantized_flat = self.embedding.index_select(&encoding_indices, 0)?;
        let quantized = quantized_flat.reshape((b, h, w, c))?.permute((0, 3, 1, 2))?;
        let quantized_final = ((z - z.detach())? + quantized.detach())?;

        Ok((quantized_final, encoding_indices.reshape((b, h, w))?))
    }

    pub fn decode_indices(&self, indices: &Tensor) -> Result<Tensor> {
        let indices_flat = indices.flatten_all()?;
        let quantized_flat = self.embedding.index_select(&indices_flat, 0)?;
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
        let z = self.encoder.forward(xs)?;
        let (quantized, indices) = self.quantizer.forward(&z)?;
        let reconstruction = self.decoder.forward(&quantized)?;
        Ok((reconstruction, quantized, indices))
    }

    pub fn decode_from_indices(&self, indices: &Tensor) -> Result<Tensor> {
         let (b, h, w) = indices.dims3()?;
         let indices_flat = indices.flatten_all()?;
         let quantized_flat = self.quantizer.embedding.index_select(&indices_flat, 0)?;
         let c = self.quantizer.embedding_dim;

         let quantized = quantized_flat.reshape((b, h, w, c))?.permute((0, 3, 1, 2))?;
         self.decoder.forward(&quantized)
    }

    pub fn encode_to_indices(&self, xs: &Tensor) -> Result<Tensor> {
        let z = self.encoder.forward(xs)?;
        let (_, indices) = self.quantizer.forward(&z)?;
        Ok(indices)
    }
}

/// Latent-to-Voxel & Splat Visual Decoder
#[derive(Debug)]
pub struct LatentVoxelDecoder {
    linear_out: candle_nn::Linear,
    pub vocab_size: usize,
}

impl LatentVoxelDecoder {
    pub fn new(d_model: usize, vocab_size: usize, vb: VarBuilder) -> Result<Self> {
        let linear_out = candle_nn::linear(d_model, vocab_size, vb.pp("linear_out"))?;
        Ok(Self {
            linear_out,
            vocab_size,
        })
    }

    /// Decode continuous latents [B, T, S, C] or [B, S, C] into discrete Voxel IDs
    pub fn decode_voxels(&self, latents: &Tensor) -> Result<Tensor> {
        let logits = self.linear_out.forward(latents)?;
        let last_dim = logits.rank() - 1;
        logits.argmax(last_dim)
    }

    /// Decode continuous latents [B, T, S, C] into 3D Splat vertices float attributes
    pub fn decode_splats(&self, latents: &Tensor) -> Result<Vec<Vec<f32>>> {
        let flat_vec = latents.flatten_all()?.to_vec1::<f32>()?;
        let chunk_len = 14; // SplatVertex field float count
        let num_splats = flat_vec.len() / chunk_len;
        let mut splats = Vec::with_capacity(num_splats);

        for i in 0..num_splats {
            let offset = i * chunk_len;
            let slice = &flat_vec[offset..offset + chunk_len];
            splats.push(slice.to_vec());
        }

        Ok(splats)
    }
}
