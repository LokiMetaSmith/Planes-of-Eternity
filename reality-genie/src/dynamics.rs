use candle_core::{Result, Tensor, Module};
use candle_nn::{Embedding, VarBuilder, Linear, LayerNorm};

#[derive(Debug, Clone)]
pub struct DynamicsConfig {
    pub vocab_size: usize,      // VQ-VAE codebook size
    pub num_actions: usize,
    pub d_model: usize,
    pub n_head: usize,
    pub num_layers: usize,
    pub max_seq_len: usize,
}

impl Default for DynamicsConfig {
    fn default() -> Self {
        Self {
            vocab_size: 512,
            num_actions: 8,
            d_model: 256,
            n_head: 4,
            num_layers: 4,
            max_seq_len: 1024,
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

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;

        let q = self.query.forward(x)?;
        let k = self.key.forward(x)?;
        let v = self.value.forward(x)?;

        // Reshape for heads: [B, T, n_head, d_head] -> permute -> [B, n_head, T, d_head]
        let q = q.reshape((b, t, self.n_head, self.d_head))?.permute((0, 2, 1, 3))?;
        let k = k.reshape((b, t, self.n_head, self.d_head))?.permute((0, 2, 1, 3))?;
        let v = v.reshape((b, t, self.n_head, self.d_head))?.permute((0, 2, 1, 3))?;

        // Attention scores: Q * K^T -> [B, n_head, T, T]
        let att = (q.matmul(&k.t()?)? * self.scale)?;

        let att = match mask {
            Some(m) => att.broadcast_add(m)?,
            None => att,
        };

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
    fn new(cfg: &DynamicsConfig, vb: VarBuilder) -> Result<Self> {
        let sa = SelfAttention::new(cfg.d_model, cfg.n_head, vb.pp("sa"))?;
        let ff1 = candle_nn::linear(cfg.d_model, cfg.d_model * 4, vb.pp("ff1"))?;
        let ff2 = candle_nn::linear(cfg.d_model * 4, cfg.d_model, vb.pp("ff2"))?;
        let ln1 = candle_nn::layer_norm(cfg.d_model, 1e-5, vb.pp("ln1"))?;
        let ln2 = candle_nn::layer_norm(cfg.d_model, 1e-5, vb.pp("ln2"))?;
        Ok(Self { sa, ff1, ff2, ln1, ln2 })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = x;
        let x = self.ln1.forward(x)?;
        let x = self.sa.forward(&x, mask)?;
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
pub struct DynamicsModel {
    token_embed: Embedding,
    action_embed: Embedding, // We map actions to same d_model space
    pos_embed: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    head: Linear,
}

impl DynamicsModel {
    pub fn new(cfg: &DynamicsConfig, vb: VarBuilder) -> Result<Self> {
        let token_embed = candle_nn::embedding(cfg.vocab_size, cfg.d_model, vb.pp("token_embed"))?;
        let action_embed = candle_nn::embedding(cfg.num_actions, cfg.d_model, vb.pp("action_embed"))?;
        let pos_embed = candle_nn::embedding(cfg.max_seq_len, cfg.d_model, vb.pp("pos_embed"))?;

        let mut blocks = Vec::new();
        for i in 0..cfg.num_layers {
            blocks.push(Block::new(cfg, vb.pp(format!("block_{}", i)))?);
        }

        let ln_f = candle_nn::layer_norm(cfg.d_model, 1e-5, vb.pp("ln_f"))?;
        let head = candle_nn::linear(cfg.d_model, cfg.vocab_size, vb.pp("head"))?;

        Ok(Self {
            token_embed,
            action_embed,
            pos_embed,
            blocks,
            ln_f,
            head,
        })
    }

    pub fn forward(&self,
                   tokens: &Tensor,
                   _actions: Option<&Tensor>, // Actions injected at specific steps? Or interleaved?
                   // Simplification: We assume input is just a sequence of tokens where actions are already handled
                   // OR we have a separate logic.
                   // Genie: Frame tokens + Action token.
                   // If we pass `tokens` which contains mix of [0..vocab_size) and [0..num_actions), we need to know which is which.
                   // Let's assume `tokens` is [B, T]. We also take `token_types` [B, T] where 0=Frame, 1=Action.
                   token_types: &Tensor
    ) -> Result<Tensor> {
        let (b, t) = tokens.dims2()?;

        // Embeddings
        // We use gather logic? Or just add separate embeddings.
        // If type==0, use token_embed(val). If type==1, use action_embed(val).
        // Since candle doesn't support conditional embedding easily in one op,
        // we can embed all using both and mask.

        let t_emb = self.token_embed.forward(tokens)?;
        let a_emb = self.action_embed.forward(tokens)?; // Note: tokens here must be in range [0, max(vocab, actions)]

        // Create masks from token_types
        // type 0 -> keep t_emb, type 1 -> keep a_emb
        // token_types [B, T]. unsqueeze to [B, T, 1] for broadcasting.
        let tt_expanded = token_types.unsqueeze(2)?;

        let mask_t = tt_expanded.eq(0.0)?.to_dtype(candle_core::DType::F32)?.broadcast_as(t_emb.shape())?;
        let mask_a = tt_expanded.eq(1.0)?.to_dtype(candle_core::DType::F32)?.broadcast_as(a_emb.shape())?;

        let x = ((t_emb * mask_t)? + (a_emb * mask_a)?)?;

        // Positional embedding
        let positions = Tensor::arange(0u32, t as u32, tokens.device())?; // [T]
        let p_emb = self.pos_embed.forward(&positions)?.broadcast_as((b, t, self.pos_embed.embeddings().dim(1)?))?;

        let mut x = (x + p_emb)?;

        // Causal mask
        // mask[i, j] = -inf if j > i
        let mask = self.make_causal_mask(t, x.device())?;

        for block in &self.blocks {
            x = block.forward(&x, Some(&mask))?;
        }

        let x = self.ln_f.forward(&x)?;
        self.head.forward(&x)
    }

    fn make_causal_mask(&self, t: usize, device: &candle_core::Device) -> Result<Tensor> {
        let mask: Vec<f32> = (0..t)
            .flat_map(|i| (0..t).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        Tensor::from_vec(mask, (t, t), device)?.broadcast_as((1, 1, t, t))
    }
}
