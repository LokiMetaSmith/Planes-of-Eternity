use candle_core::{Result, Tensor, Module, DType};
use candle_nn::{Embedding, VarBuilder, Linear, LayerNorm};

#[derive(Debug, Clone)]
pub struct DiffusionConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_head: usize,
    pub num_layers: usize,
    pub max_seq_len: usize,
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        Self {
            vocab_size: 512, // Voxel IDs
            d_model: 256,
            n_head: 4,
            num_layers: 4,
            max_seq_len: 1024, // 32x32 chunk slice = 1024 tokens
        }
    }
}

#[derive(Debug)]
struct SelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    n_head: usize,
    d_head: usize,
    scale: f64,
}

impl SelfAttention {
    fn new(d_model: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
        let d_head = d_model / n_head;
        let query = candle_nn::linear(d_model, d_model, vb.pp("query"))?;
        let key = candle_nn::linear(d_model, d_model, vb.pp("key"))?;
        let value = candle_nn::linear(d_model, d_model, vb.pp("value"))?;
        let out = candle_nn::linear(d_model, d_model, vb.pp("out"))?;

        Ok(Self {
            query, key, value, out,
            n_head,
            d_head,
            scale: 1.0 / (d_head as f64).sqrt(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;

        let q = self.query.forward(x)?;
        let k = self.key.forward(x)?;
        let v = self.value.forward(x)?;

        // Reshape for heads: [B, T, n_head, d_head] -> permute -> [B, n_head, T, d_head]
        let q = q.reshape((b, t, self.n_head, self.d_head))?.permute((0, 2, 1, 3))?;
        let k = k.reshape((b, t, self.n_head, self.d_head))?.permute((0, 2, 1, 3))?;
        let v = v.reshape((b, t, self.n_head, self.d_head))?.permute((0, 2, 1, 3))?;

        // Attention scores: Q * K^T -> [B, n_head, T, T]
        // Bidirectional attention: No masking needed for "future" tokens
        let att = (q.matmul(&k.t()?)? * self.scale)?;
        let att = candle_nn::ops::softmax(&att, 3)?;

        // Output: Att * V -> [B, n_head, T, d_head]
        let out = att.matmul(&v)?;

        // Reassemble: [B, T, n_head, d_head] -> [B, T, C]
        let out = out.permute((0, 2, 1, 3))?.reshape((b, t, c))?;

        self.out.forward(&out)
    }
}

#[derive(Debug)]
struct Block {
    sa: SelfAttention,
    ff1: Linear,
    ff2: Linear,
    ln1: LayerNorm,
    ln2: LayerNorm,
}

impl Block {
    fn new(cfg: &DiffusionConfig, vb: VarBuilder) -> Result<Self> {
        let sa = SelfAttention::new(cfg.d_model, cfg.n_head, vb.pp("sa"))?;
        let ff1 = candle_nn::linear(cfg.d_model, cfg.d_model * 4, vb.pp("ff1"))?;
        let ff2 = candle_nn::linear(cfg.d_model * 4, cfg.d_model, vb.pp("ff2"))?;
        let ln1 = candle_nn::layer_norm(cfg.d_model, 1e-5, vb.pp("ln1"))?;
        let ln2 = candle_nn::layer_norm(cfg.d_model, 1e-5, vb.pp("ln2"))?;
        Ok(Self { sa, ff1, ff2, ln1, ln2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = self.ln1.forward(x)?;
        let x = self.sa.forward(&x)?;
        let x = (x + residual)?;

        let residual = &x;
        let x = self.ln2.forward(&x)?;
        let x = self.ff1.forward(&x)?;
        let x = x.relu()?;
        let x = self.ff2.forward(&x)?;
        let x = (x + residual)?;

        Ok(x)
    }
}

#[derive(Debug)]
pub struct DiffusionModel {
    token_embed: Embedding,
    pos_embed: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    head: Linear,
}

impl DiffusionModel {
    pub fn new(cfg: &DiffusionConfig, vb: VarBuilder) -> Result<Self> {
        let token_embed = candle_nn::embedding(cfg.vocab_size, cfg.d_model, vb.pp("token_embed"))?;
        let pos_embed = candle_nn::embedding(cfg.max_seq_len, cfg.d_model, vb.pp("pos_embed"))?;

        let mut blocks = Vec::new();
        for i in 0..cfg.num_layers {
            blocks.push(Block::new(cfg, vb.pp(format!("block_{}", i)))?);
        }

        let ln_f = candle_nn::layer_norm(cfg.d_model, 1e-5, vb.pp("ln_f"))?;
        let head = candle_nn::linear(cfg.d_model, cfg.vocab_size, vb.pp("head"))?;

        Ok(Self {
            token_embed,
            pos_embed,
            blocks,
            ln_f,
            head,
        })
    }

    pub fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let (b, t) = tokens.dims2()?;

        let t_emb = self.token_embed.forward(tokens)?;
        let positions = Tensor::arange(0u32, t as u32, tokens.device())?;
        let p_emb = self.pos_embed.forward(&positions)?.broadcast_as((b, t, self.pos_embed.embeddings().dim(1)?))?;

        let mut x = (t_emb + p_emb)?;

        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        let x = self.ln_f.forward(&x)?;
        self.head.forward(&x)
    }
}

pub struct DiscreteDiffusion {
    pub model: DiffusionModel,
    pub config: DiffusionConfig,
}

impl DiscreteDiffusion {
    pub fn new(config: DiffusionConfig, vb: VarBuilder) -> Result<Self> {
        let model = DiffusionModel::new(&config, vb)?;
        Ok(Self { model, config })
    }

    // Forward process: Corrupt input x0 to xt
    // Simple Masking Strategy: Replace tokens with mask_token (0) with probability p
    pub fn q_sample(&self, x0: &Tensor, mask_prob: f64) -> Result<(Tensor, Tensor)> {
        // x0: [B, T]
        let mask = Tensor::rand(0.0f32, 1.0f32, x0.shape(), x0.device())?;
        let mask_indices = mask.lt(mask_prob)?; // 1 where masked, 0 where kept

        // MASK token index = 0 (Assuming 0 is reserved/Air, or we reserve a special one)
        // Let's assume 0 is "Air/Mask" for now in Voxel world.
        // Actually, we should probably use a special token ID if possible, but 0 works for "Void".
        let mask_token = Tensor::zeros_like(x0)?;

        // x_t = x0 * (1 - mask) + mask_token * mask
        // Candle doesn't have `where` easily exposed for all types, so use arithmetic
        let mask_f = mask_indices.to_dtype(DType::F32)?;
        let x0_f = x0.to_dtype(DType::F32)?;
        let mask_token_f = mask_token.to_dtype(DType::F32)?;

        let x_t_f = ((x0_f * (1.0 - &mask_f))? + (mask_token_f * &mask_f)?)?;
        let x_t = x_t_f.to_dtype(x0.dtype())?;

        Ok((x_t, mask_indices))
    }

    // Reverse process: Denoise xt to x0
    // Predicts logits for x0
    pub fn p_sample(&self, xt: &Tensor) -> Result<Tensor> {
        let logits = self.model.forward(xt)?;
        // Return logits for sampling
        Ok(logits)
    }

    // Generate loop (Ancestral Sampling / Gibbs Sampling / Order Agnostic)
    // For simplicity: One-shot prediction (BERT style) or Iterative
    // Let's implement a simple Iterative Refinement:
    // 1. Start with fully masked/random noise
    // 2. Predict x0
    // 3. Re-mask a smaller portion? Or just take high confidence?
    //
    // For this implementation, we'll expose a "denoise" function that takes a partially masked input
    // and fills in the masks.
    pub fn denoise(&self, x_masked: &Tensor) -> Result<Tensor> {
        let logits = self.p_sample(x_masked)?;
        let pred_tokens = logits.argmax(2)?; // [B, T]
        Ok(pred_tokens)
    }
}
