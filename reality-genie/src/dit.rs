use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{layer_norm, linear, Embedding, LayerNorm, Linear, VarBuilder, VarMap};

/// Configuration for ActionConditionedDiT
#[derive(Debug, Clone)]
pub struct DiTConfig {
    pub num_actions: usize,
    pub action_dim: usize,
    pub d_model: usize,
    pub n_head: usize,
    pub num_layers: usize,
    pub num_spatial_tokens: usize,
    pub max_temporal_frames: usize,
    pub max_anchor_frames: usize,
    pub dropout: f64,
}

impl Default for DiTConfig {
    fn default() -> Self {
        Self {
            num_actions: 16,
            action_dim: 64,
            d_model: 256,
            n_head: 8,
            num_layers: 4,
            num_spatial_tokens: 64,
            max_temporal_frames: 8,
            max_anchor_frames: 4,
            dropout: 0.0,
        }
    }
}

/// 1. Action conditioning embedder: maps discrete action inputs (e.g. keyboard keys/combos)
/// into continuous token embeddings and projects them into the model's feature dimension.
#[derive(Debug)]
pub struct ActionConditioningEmbedder {
    embedding: Embedding,
    linear_projection: Linear,
}

impl ActionConditioningEmbedder {
    pub fn new(num_actions: usize, action_dim: usize, d_model: usize, vb: VarBuilder) -> Result<Self> {
        let embedding = candle_nn::embedding(num_actions, action_dim, vb.pp("embedding"))?;
        let linear_projection = linear(action_dim, d_model, vb.pp("linear_projection"))?;
        Ok(Self {
            embedding,
            linear_projection,
        })
    }

    pub fn forward(&self, actions: &Tensor) -> Result<Tensor> {
        let emb = self.embedding.forward(actions)?;
        let proj = self.linear_projection.forward(&emb)?;
        Ok(proj)
    }
}

/// 2. Frame anchor buffer: manages sliding-window historical keyframe latents
/// to preserve landmark memory and bound long-horizon trajectory drift.
#[derive(Debug, Clone)]
pub struct FrameAnchorBuffer {
    pub max_window_size: usize,
    pub anchor_interval: usize,
    pub cached_anchors: Vec<Tensor>,
}

impl FrameAnchorBuffer {
    pub fn new(max_window_size: usize, anchor_interval: usize) -> Self {
        Self {
            max_window_size,
            anchor_interval,
            cached_anchors: Vec::new(),
        }
    }

    /// Insert new frame latent tensor of shape [B, S, C] or [B, 1, S, C]
    pub fn push_frame(&mut self, frame: Tensor, step_index: usize) -> Result<()> {
        if step_index % self.anchor_interval == 0 {
            self.cached_anchors.push(frame);
            if self.cached_anchors.len() > self.max_window_size {
                self.cached_anchors.remove(0);
            }
        }
        Ok(())
    }

    /// Concatenate cached anchors into a tensor of shape [B, NumAnchors * S, C]
    pub fn get_anchor_context(&self, device: &Device, b: usize, _s: usize, c: usize, dtype: DType) -> Result<Tensor> {
        if self.cached_anchors.is_empty() {
            // Return dummy zero context [B, 1, C]
            Tensor::zeros((b, 1, c), dtype, device)
        } else {
            // Stack and reshape cached anchors: each [B, S, C] or [B, 1, S, C]
            let mut reshaped = Vec::new();
            for anchor in &self.cached_anchors {
                let a = if anchor.rank() == 4 {
                    anchor.squeeze(1)?
                } else {
                    anchor.clone()
                };
                reshaped.push(a);
            }
            Tensor::cat(&reshaped, 1)
        }
    }

    pub fn clear(&mut self) {
        self.cached_anchors.clear();
    }
}

/// Causal Temporal Attention over [B, T*S, C] flattened sequence with explicit causal mask
#[derive(Debug)]
pub struct CausalTemporalAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    n_head: usize,
    d_head: usize,
    scale: f64,
}

impl CausalTemporalAttention {
    pub fn new(d_model: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
        let d_head = d_model / n_head;
        let query = linear(d_model, d_model, vb.pp("query"))?;
        let key = linear(d_model, d_model, vb.pp("key"))?;
        let value = linear(d_model, d_model, vb.pp("value"))?;
        let out = linear(d_model, d_model, vb.pp("out"))?;

        Ok(Self {
            query,
            key,
            value,
            out,
            n_head,
            d_head,
            scale: 1.0 / (d_head as f64).sqrt(),
        })
    }

    /// Generates a causal temporal attention mask where spatial tokens in frame i
    /// can attend to all spatial tokens in frame j <= i, but not j > i.
    pub fn build_causal_temporal_mask(
        t: usize,
        s: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let total_tokens = t * s;
        let mut mask_data = vec![0.0f32; total_tokens * total_tokens];

        for qi in 0..total_tokens {
            let q_frame = qi / s;
            for ki in 0..total_tokens {
                let k_frame = ki / s;
                if k_frame > q_frame {
                    mask_data[qi * total_tokens + ki] = -1e9f32;
                }
            }
        }

        let mask_tensor = Tensor::from_vec(mask_data, (1, 1, total_tokens, total_tokens), device)?;
        mask_tensor.to_dtype(dtype)
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, seq_len, c) = x.dims3()?;

        let q = self.query.forward(x)?;
        let k = self.key.forward(x)?;
        let v = self.value.forward(x)?;

        let q = q.reshape((b, seq_len, self.n_head, self.d_head))?.permute((0, 2, 1, 3))?.contiguous()?;
        let k = k.reshape((b, seq_len, self.n_head, self.d_head))?.permute((0, 2, 1, 3))?.contiguous()?;
        let v = v.reshape((b, seq_len, self.n_head, self.d_head))?.permute((0, 2, 1, 3))?.contiguous()?;

        let kt = k.t()?.contiguous()?;
        let mut att = (q.matmul(&kt)? * self.scale)?;
        if let Some(m) = mask {
            let m_b = m.broadcast_as(att.shape())?;
            att = (&att + m_b)?;
        }
        let att = candle_nn::ops::softmax(&att, 3)?;
        let out = att.matmul(&v)?;

        let out = out.permute((0, 2, 1, 3))?.contiguous()?.reshape((b, seq_len, c))?;
        self.out.forward(&out)
    }
}

/// Cross Attention for Anchor context
#[derive(Debug)]
pub struct CrossAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    n_head: usize,
    d_head: usize,
    scale: f64,
}

impl CrossAttention {
    pub fn new(d_model: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
        let d_head = d_model / n_head;
        let query = linear(d_model, d_model, vb.pp("query"))?;
        let key = linear(d_model, d_model, vb.pp("key"))?;
        let value = linear(d_model, d_model, vb.pp("value"))?;
        let out = linear(d_model, d_model, vb.pp("out"))?;

        Ok(Self {
            query,
            key,
            value,
            out,
            n_head,
            d_head,
            scale: 1.0 / (d_head as f64).sqrt(),
        })
    }

    pub fn forward(&self, x: &Tensor, context: &Tensor) -> Result<Tensor> {
        let (b, seq_len, c) = x.dims3()?;
        let (_, ctx_len, _) = context.dims3()?;

        let q = self.query.forward(x)?;
        let k = self.key.forward(context)?;
        let v = self.value.forward(context)?;

        let q = q.reshape((b, seq_len, self.n_head, self.d_head))?.permute((0, 2, 1, 3))?.contiguous()?;
        let k = k.reshape((b, ctx_len, self.n_head, self.d_head))?.permute((0, 2, 1, 3))?.contiguous()?;
        let v = v.reshape((b, ctx_len, self.n_head, self.d_head))?.permute((0, 2, 1, 3))?.contiguous()?;

        let kt = k.t()?.contiguous()?;
        let att = (q.matmul(&kt)? * self.scale)?;
        let att = candle_nn::ops::softmax(&att, 3)?;
        let out = att.matmul(&v)?;

        let out = out.permute((0, 2, 1, 3))?.contiguous()?.reshape((b, seq_len, c))?;
        self.out.forward(&out)
    }
}

/// 3. Spatiotemporal DiT block with AdaLN (adaptive layer norm modulation for timestep/action)
#[derive(Debug)]
pub struct SpatiotemporalDitBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm_anchor: LayerNorm,
    temporal_causal_attn: CausalTemporalAttention,
    anchor_cross_attn: CrossAttention,
    ff1: Linear,
    ff2: Linear,
    // AdaLN modulation parameters: scale1, shift1, scale2, shift2
    adaln_proj: Linear,
}

impl SpatiotemporalDitBlock {
    pub fn new(cfg: &DiTConfig, vb: VarBuilder) -> Result<Self> {
        let norm1 = layer_norm(cfg.d_model, 1e-5, vb.pp("norm1"))?;
        let norm2 = layer_norm(cfg.d_model, 1e-5, vb.pp("norm2"))?;
        let norm_anchor = layer_norm(cfg.d_model, 1e-5, vb.pp("norm_anchor"))?;

        let temporal_causal_attn = CausalTemporalAttention::new(cfg.d_model, cfg.n_head, vb.pp("causal_attn"))?;
        let anchor_cross_attn = CrossAttention::new(cfg.d_model, cfg.n_head, vb.pp("anchor_attn"))?;

        let ff1 = linear(cfg.d_model, cfg.d_model * 4, vb.pp("ff1"))?;
        let ff2 = linear(cfg.d_model * 4, cfg.d_model, vb.pp("ff2"))?;

        let adaln_proj = linear(cfg.d_model, cfg.d_model * 4, vb.pp("adaln_proj"))?;

        Ok(Self {
            norm1,
            norm2,
            norm_anchor,
            temporal_causal_attn,
            anchor_cross_attn,
            ff1,
            ff2,
            adaln_proj,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        conditioning_emb: &Tensor,
        anchor_context: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Conditioning project produces (gamma1, beta1, gamma2, beta2)
        let adaln_params = self.adaln_proj.forward(conditioning_emb)?; // [B, T_cond, 4 * d_model]
        let (b, seq_len, c) = x.dims3()?;

        let chunks = adaln_params.chunk(4, adaln_params.rank() - 1)?;

        let expand_param = |p: &Tensor| -> Result<Tensor> {
            if p.rank() == 3 {
                let (_, t_cond, _) = p.dims3()?;
                if t_cond > 1 && seq_len % t_cond == 0 {
                    let s_tokens = seq_len / t_cond;
                    p.unsqueeze(2)?.broadcast_as((b, t_cond, s_tokens, c))?.reshape((b, seq_len, c))
                } else {
                    p.broadcast_as((b, seq_len, c))
                }
            } else {
                p.broadcast_as((b, seq_len, c))
            }
        };

        let gamma1 = expand_param(&chunks[0])?;
        let beta1 = expand_param(&chunks[1])?;
        let gamma2 = expand_param(&chunks[2])?;
        let beta2 = expand_param(&chunks[3])?;

        // 1. Normalized & modulated Causal Temporal Attention
        let norm_x = self.norm1.forward(x)?;
        let modulated_x = (norm_x.broadcast_mul(&(gamma1.affine(1.0, 1.0)?))? + &beta1)?;
        let attn_out = self.temporal_causal_attn.forward(&modulated_x, mask)?;
        let x = (x + attn_out)?;

        // 2. Anchor Cross Attention
        let norm_x2 = self.norm_anchor.forward(&x)?;
        let anchor_out = self.anchor_cross_attn.forward(&norm_x2, anchor_context)?;
        let x = (&x + anchor_out)?;

        // 3. Feed forward
        let norm_x3 = self.norm2.forward(&x)?;
        let modulated_ff = (norm_x3.broadcast_mul(&(gamma2.affine(1.0, 1.0)?))? + &beta2)?;
        let ff_out = self.ff1.forward(&modulated_ff)?.relu()?;
        let ff_out = self.ff2.forward(&ff_out)?;
        let x = (&x + ff_out)?;

        Ok(x)
    }
}

/// Sinusoidal Timestep Embedder
#[derive(Debug)]
pub struct TimestepEmbedder {
    linear1: Linear,
    linear2: Linear,
    frequency_embedding_size: usize,
}

impl TimestepEmbedder {
    pub fn new(d_model: usize, vb: VarBuilder) -> Result<Self> {
        let frequency_embedding_size = d_model / 2;
        let linear1 = linear(frequency_embedding_size, d_model, vb.pp("linear1"))?;
        let linear2 = linear(d_model, d_model, vb.pp("linear2"))?;
        Ok(Self {
            linear1,
            linear2,
            frequency_embedding_size,
        })
    }

    pub fn forward(&self, timesteps: &Tensor) -> Result<Tensor> {
        let half_dim = self.frequency_embedding_size / 2;
        let device = timesteps.device();
        let log_base = (10000.0f64).ln() / (half_dim as f64 - 1.0);
        let mut inv_freqs = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            inv_freqs.push((-log_base * i as f64).exp() as f32);
        }
        let inv_freq_tensor = Tensor::from_vec(inv_freqs, (1, half_dim), device)?;

        let ts_f32 = timesteps.to_dtype(DType::F32)?.reshape((timesteps.elem_count(), 1))?;
        let args = ts_f32.matmul(&inv_freq_tensor)?;
        let sin = args.sin()?;
        let cos = args.cos()?;
        let emb = Tensor::cat(&[&sin, &cos], 1)?;

        let emb = self.linear1.forward(&emb)?.relu()?;
        self.linear2.forward(&emb)
    }
}

/// Action-Conditioned Spatiotemporal Diffusion Transformer (ActionConditionedDiT)
#[derive(Debug)]
pub struct ActionConditionedDiT {
    pub config: DiTConfig,
    action_embedder: ActionConditioningEmbedder,
    time_embedder: TimestepEmbedder,
    latent_proj: Linear,
    blocks: Vec<SpatiotemporalDitBlock>,
    final_norm: LayerNorm,
    out_proj: Linear,
}

impl ActionConditionedDiT {
    pub fn new(cfg: DiTConfig, vb: VarBuilder) -> Result<Self> {
        let action_embedder = ActionConditioningEmbedder::new(
            cfg.num_actions,
            cfg.action_dim,
            cfg.d_model,
            vb.pp("action_embedder"),
        )?;
        let time_embedder = TimestepEmbedder::new(cfg.d_model, vb.pp("time_embedder"))?;
        let latent_proj = linear(cfg.d_model, cfg.d_model, vb.pp("latent_proj"))?;

        let mut blocks = Vec::new();
        for i in 0..cfg.num_layers {
            blocks.push(SpatiotemporalDitBlock::new(&cfg, vb.pp(format!("block_{}", i)))?);
        }

        let final_norm = layer_norm(cfg.d_model, 1e-5, vb.pp("final_norm"))?;
        let out_proj = linear(cfg.d_model, cfg.d_model, vb.pp("out_proj"))?;

        Ok(Self {
            config: cfg,
            action_embedder,
            time_embedder,
            latent_proj,
            blocks,
            final_norm,
            out_proj,
        })
    }

    pub fn default_cpu() -> Result<Self> {
        let device = Device::Cpu;
        let cfg = DiTConfig::default();
        let vars = VarMap::new();
        let vb = VarBuilder::from_varmap(&vars, DType::F32, &device);
        Self::new(cfg, vb)
    }

    /// Forward pass:
    /// - `x`: Spatial-temporal latents tensor of shape [B, T, S, C] or [B, T * S, C]
    /// - `actions`: Action IDs tensor of shape [B] or [B, T]
    /// - `timesteps`: Timestep tensor of shape [B]
    /// - `anchor_context`: Cached keyframe anchor context tensor of shape [B, AnchorTokens, C]
    pub fn forward(
        &self,
        x: &Tensor,
        actions: &Tensor,
        timesteps: &Tensor,
        anchor_context: &Tensor,
    ) -> Result<Tensor> {
        let (b, seq_len, c) = if x.rank() == 4 {
            let (b, t, s, c) = x.dims4()?;
            (b, t * s, c)
        } else {
            x.dims3()?
        };

        let x_flat = if x.rank() == 4 {
            x.reshape((b, seq_len, c))?
        } else {
            x.clone()
        };

        let t_frames = seq_len / self.config.num_spatial_tokens;

        // 1. Embed actions and timesteps
        let act_emb = if actions.rank() == 1 {
            let emb = self.action_embedder.forward(actions)?; // [B, d_model]
            emb.reshape((b, 1, self.config.d_model))?
        } else {
            self.action_embedder.forward(actions)? // [B, T, d_model]
        };

        let time_emb = self.time_embedder.forward(timesteps)?.reshape((b, 1, self.config.d_model))?;
        let time_emb_bc = time_emb.broadcast_as(act_emb.shape())?;
        let conditioning = (act_emb + time_emb_bc)?;

        // 2. Linear projection of input latents
        let mut h = self.latent_proj.forward(&x_flat)?;

        // 3. Build causal temporal mask
        let mask = CausalTemporalAttention::build_causal_temporal_mask(
            t_frames,
            self.config.num_spatial_tokens,
            x.device(),
            x.dtype(),
        )?;

        // 4. Pass through DiT blocks
        for block in &self.blocks {
            h = block.forward(&h, &conditioning, anchor_context, Some(&mask))?;
        }

        // 5. Final norm and projection
        let h = self.final_norm.forward(&h)?;
        let out = self.out_proj.forward(&h)?;

        if x.rank() == 4 {
            out.reshape((b, t_frames, self.config.num_spatial_tokens, c))
        } else {
            Ok(out)
        }
    }
}

/// Real-time Action-Conditioned Trajectory Rollout Sampler
#[derive(Debug)]
pub struct ActionConditionedSampler {
    pub model: ActionConditionedDiT,
    pub num_denoising_steps: usize,
}

impl ActionConditionedSampler {
    pub fn new(model: ActionConditionedDiT, num_denoising_steps: usize) -> Self {
        Self {
            model,
            num_denoising_steps,
        }
    }

    /// Sample/predict next frame latents [B, 1, S, C] given user action and current anchor context
    pub fn sample_next_frame(
        &self,
        b: usize,
        action: u32,
        anchor_buffer: &mut FrameAnchorBuffer,
        step_index: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let s = self.model.config.num_spatial_tokens;
        let c = self.model.config.d_model;
        let dtype = DType::F32;

        let actions_tensor = Tensor::from_vec(vec![action; b], (b,), device)?;
        let anchor_ctx = anchor_buffer.get_anchor_context(device, b, s, c, dtype)?;

        // Initialize with pure Gaussian noise latents for 1 frame: [B, 1, S, C]
        let mut x_t = Tensor::randn(0.0f32, 1.0f32, (b, 1, s, c), device)?;

        for step in (0..self.num_denoising_steps).rev() {
            let timestep = (step * 1000 / self.num_denoising_steps) as u32;
            let timesteps_tensor = Tensor::from_vec(vec![timestep; b], (b,), device)?;

            let pred_noise = self.model.forward(&x_t, &actions_tensor, &timesteps_tensor, &anchor_ctx)?;

            // Denoising step: x_{t-1} = x_t - alpha * pred_noise
            let alpha = 1.0 / (self.num_denoising_steps as f32);
            x_t = (&x_t - (&pred_noise * (alpha as f64))?)?;
        }

        // Cache newly predicted frame into sliding window buffer
        anchor_buffer.push_frame(x_t.clone(), step_index)?;

        Ok(x_t)
    }

    /// Autoregressively rollout a trajectory sequence of frame latents given an action sequence [B, T]
    pub fn rollout_trajectory(
        &self,
        actions: &Tensor,
        anchor_buffer: &mut FrameAnchorBuffer,
        device: &Device,
    ) -> Result<Tensor> {
        let (b, t_frames) = actions.dims2()?;
        let mut frame_list = Vec::with_capacity(t_frames);

        for t_step in 0..t_frames {
            let action_val = actions.get(0)?.get(t_step)?.to_scalar::<u32>()?;
            let frame = self.sample_next_frame(b, action_val, anchor_buffer, t_step, device)?;
            frame_list.push(frame);
        }

        // Concatenate along temporal axis -> [B, T, S, C]
        Tensor::cat(&frame_list, 1)
    }

    pub fn rollout_from_actions_vec(&self, actions: &[u32]) -> Result<Vec<f32>> {
        let device = Device::Cpu;
        let mut anchor_buffer = FrameAnchorBuffer::new(4, 2);
        let t_len = actions.len();
        let actions_tensor = Tensor::from_vec(actions.to_vec(), (1, t_len), &device)?;
        let rollout = self.rollout_trajectory(&actions_tensor, &mut anchor_buffer, &device)?;
        rollout.flatten_all()?.to_vec1::<f32>()
    }
}
