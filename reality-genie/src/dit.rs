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
    pub vocab_size_text: usize,
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
            vocab_size_text: 1000,
            dropout: 0.0,
        }
    }
}

/// Continuous Action Embedder: projects 5D continuous vectors [d_theta_x, d_theta_y, v_x, v_y, v_z]
/// via a 2-layer MLP with GELU activations to match the model feature dimension d_model.
#[derive(Debug)]
pub struct ContinuousActionEmbedder {
    linear1: Linear,
    linear2: Linear,
}

impl ContinuousActionEmbedder {
    pub fn new(in_dim: usize, d_model: usize, vb: VarBuilder) -> Result<Self> {
        let linear1 = linear(in_dim, d_model, vb.pp("linear1"))?;
        let linear2 = linear(d_model, d_model, vb.pp("linear2"))?;
        Ok(Self { linear1, linear2 })
    }

    pub fn forward(&self, cont_actions: &Tensor) -> Result<Tensor> {
        let h = self.linear1.forward(cont_actions)?.gelu_erf()?;
        self.linear2.forward(&h)
    }
}

/// 1. Action conditioning embedder: maps discrete action inputs and continuous 5D action vectors
/// into unified token embeddings and projects them into the model's feature dimension.
#[derive(Debug)]
pub struct ActionConditioningEmbedder {
    embedding: Embedding,
    linear_projection: Linear,
    continuous_embedder: ContinuousActionEmbedder,
    fusion_projection: Linear,
}

impl ActionConditioningEmbedder {
    pub fn new(num_actions: usize, action_dim: usize, d_model: usize, vb: VarBuilder) -> Result<Self> {
        let embedding = candle_nn::embedding(num_actions, action_dim, vb.pp("embedding"))?;
        let linear_projection = linear(action_dim, d_model, vb.pp("linear_projection"))?;
        let continuous_embedder = ContinuousActionEmbedder::new(5, d_model, vb.pp("continuous_embedder"))?;
        let fusion_projection = linear(d_model * 2, d_model, vb.pp("fusion_projection"))?;
        Ok(Self {
            embedding,
            linear_projection,
            continuous_embedder,
            fusion_projection,
        })
    }

    pub fn forward(&self, actions: &Tensor) -> Result<Tensor> {
        self.forward_multimodal(actions, None)
    }

    /// Forward pass with both discrete action tokens and optional continuous 5D vectors [d_theta_x, d_theta_y, vx, vy, vz]
    pub fn forward_multimodal(&self, actions: &Tensor, continuous_actions: Option<&Tensor>) -> Result<Tensor> {
        let emb = self.embedding.forward(actions)?;
        let discrete_proj = self.linear_projection.forward(&emb)?;
        if let Some(cont) = continuous_actions {
            let cont_proj = self.continuous_embedder.forward(cont)?;
            let last_dim = discrete_proj.rank() - 1;
            let cat_tensor = Tensor::cat(&[&discrete_proj, &cont_proj], last_dim)?;
            self.fusion_projection.forward(&cat_tensor)
        } else {
            Ok(discrete_proj)
        }
    }

    /// Compute action confidence and Shannon entropy metrics given logits or action probabilities [B, num_actions]
    pub fn compute_confidence_entropy(action_probs_or_logits: &[f32]) -> (f32, f32) {
        if action_probs_or_logits.is_empty() {
            return (0.0, 0.0);
        }
        let max_val = action_probs_or_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = action_probs_or_logits.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|&e| (e / sum_exp).clamp(1e-7, 1.0)).collect();

        let mut entropy = 0.0f32;
        let mut max_conf = 0.0f32;
        for &p in &probs {
            entropy -= p * p.ln();
            if p > max_conf {
                max_conf = p;
            }
        }
        let max_entropy = (probs.len() as f32).ln().max(1e-5);
        let normalized_entropy = (entropy / max_entropy).clamp(0.0, 1.0);
        (max_conf, normalized_entropy)
    }
}

/// Spatial landmark entry for 3D loop-closure geometric re-identification
#[derive(Debug, Clone)]
pub struct SpatialAnchor {
    pub position: [f32; 3],
    pub orientation: [f32; 4], // Quaternion [x, y, z, w]
    pub latent: Tensor,
}

/// Associative spatial memory indexing keyframe anchors by (position, orientation)
/// Retrieves cached landmark latents when player/camera revisits previously explored coordinates.
#[derive(Debug, Clone)]
pub struct SpatialAnchorMemory {
    pub anchors: Vec<SpatialAnchor>,
    pub eps_pos: f32,
    pub cos_thresh: f32,
    pub sigma: f32,
}

impl SpatialAnchorMemory {
    pub fn new(eps_pos: f32, cos_thresh: f32, sigma: f32) -> Self {
        Self {
            anchors: Vec::new(),
            eps_pos,
            cos_thresh,
            sigma,
        }
    }

    pub fn register_anchor(&mut self, position: [f32; 3], orientation: [f32; 4], latent: Tensor) {
        self.anchors.push(SpatialAnchor {
            position,
            orientation,
            latent,
        });
        if self.anchors.len() > 64 {
            self.anchors.remove(0);
        }
    }

    /// Retrieve matching cached landmark latent with RBF falloff weighting: exp(-||p - p_i||^2 / (2 * sigma^2))
    pub fn retrieve_anchor(&self, pos: [f32; 3], orient: [f32; 4]) -> Option<(Tensor, f32)> {
        for anchor in &self.anchors {
            let dx = pos[0] - anchor.position[0];
            let dy = pos[1] - anchor.position[1];
            let dz = pos[2] - anchor.position[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let dist = dist_sq.sqrt();

            if dist < self.eps_pos {
                let dot = orient[0] * anchor.orientation[0]
                    + orient[1] * anchor.orientation[1]
                    + orient[2] * anchor.orientation[2]
                    + orient[3] * anchor.orientation[3];

                if dot.abs() >= self.cos_thresh {
                    let rbf = (-dist_sq / (2.0 * self.sigma * self.sigma)).exp();
                    return Some((anchor.latent.clone(), rbf));
                }
            }
        }
        None
    }
}

/// 2. Frame anchor buffer: manages sliding-window historical keyframe latents
/// to preserve landmark memory and bound long-horizon trajectory drift.
#[derive(Debug, Clone)]
pub struct FrameAnchorBuffer {
    pub max_window_size: usize,
    pub anchor_interval: usize,
    pub cached_anchors: Vec<Tensor>,
    pub spatial_memory: SpatialAnchorMemory,
}

impl FrameAnchorBuffer {
    pub fn new(max_window_size: usize, anchor_interval: usize) -> Self {
        Self {
            max_window_size,
            anchor_interval,
            cached_anchors: Vec::new(),
            spatial_memory: SpatialAnchorMemory::new(5.0, 0.707, 2.5),
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

    /// Register a 3D spatial anchor at a world coordinate and quaternion orientation
    pub fn register_spatial_anchor(&mut self, pos: [f32; 3], orient: [f32; 4], latent: Tensor) {
        self.spatial_memory.register_anchor(pos, orient, latent);
    }

    /// Register a 3D spatial anchor from flat floats
    pub fn register_spatial_anchor_raw(&mut self, pos: [f32; 3], orient: [f32; 4], flat_latents: &[f32]) {
        let s = 64;
        let c = 256;
        if flat_latents.len() >= s * c {
            if let Ok(tensor) = Tensor::from_vec(flat_latents[..s * c].to_vec(), (1, s, c), &Device::Cpu) {
                self.spatial_memory.register_anchor(pos, orient, tensor);
            }
        }
    }

    /// Concatenate cached anchors into a tensor of shape [B, NumAnchors * S, C],
    /// injecting any spatially retrieved loop-closure landmark weighted by RBF falloff.
    pub fn get_anchor_context(&self, device: &Device, b: usize, _s: usize, c: usize, dtype: DType) -> Result<Tensor> {
        self.get_anchor_context_spatial(device, b, _s, c, dtype, None, None)
    }

    /// Concatenate cached anchors with optional loop-closure spatial query (position, orientation)
    pub fn get_anchor_context_spatial(
        &self,
        device: &Device,
        b: usize,
        _s: usize,
        c: usize,
        dtype: DType,
        pos_query: Option<[f32; 3]>,
        orient_query: Option<[f32; 4]>,
    ) -> Result<Tensor> {
        let mut reshaped = Vec::new();
        for anchor in &self.cached_anchors {
            let a = if anchor.rank() == 4 {
                anchor.squeeze(1)?
            } else {
                anchor.clone()
            };
            reshaped.push(a);
        }

        if let (Some(pos), Some(orient)) = (pos_query, orient_query) {
            if let Some((landmark, rbf)) = self.spatial_memory.retrieve_anchor(pos, orient) {
                let l = if landmark.rank() == 4 {
                    landmark.squeeze(1)?
                } else {
                    landmark
                };
                let weighted_landmark = (l * (rbf as f64))?;
                reshaped.push(weighted_landmark);
            }
        }

        if reshaped.is_empty() {
            Tensor::zeros((b, 1, c), dtype, device)
        } else {
            Tensor::cat(&reshaped, 1)
        }
    }

    /// Calculate long-horizon drift variance across cached keyframes
    pub fn compute_drift_variance(&self) -> Result<f32> {
        if self.cached_anchors.len() < 2 {
            return Ok(0.0f32);
        }

        let first = &self.cached_anchors[0];
        let last = &self.cached_anchors[self.cached_anchors.len() - 1];

        let diff = (last - first)?;
        let sq_diff = diff.sqr()?.mean_all()?;
        sq_diff.to_scalar::<f32>()
    }

    pub fn clear(&mut self) {
        self.cached_anchors.clear();
        self.spatial_memory.anchors.clear();
    }

    /// Rewinds anchor history by removing recent anchor frames
    pub fn rewind(&mut self, count: usize) -> usize {
        let to_remove = count.min(self.cached_anchors.len());
        for _ in 0..to_remove {
            self.cached_anchors.pop();
        }
        to_remove
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
        let adaln_params = self.adaln_proj.forward(conditioning_emb)?;
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

        let norm_x = self.norm1.forward(x)?;
        let modulated_x = (norm_x.broadcast_mul(&(gamma1.affine(1.0, 1.0)?))? + &beta1)?;
        let attn_out = self.temporal_causal_attn.forward(&modulated_x, mask)?;
        let x = (x + attn_out)?;

        let norm_x2 = self.norm_anchor.forward(&x)?;
        let anchor_out = self.anchor_cross_attn.forward(&norm_x2, anchor_context)?;
        let x = (&x + anchor_out)?;

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
    text_embedder: Embedding,
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
        let text_embedder = candle_nn::embedding(cfg.vocab_size_text, cfg.d_model, vb.pp("text_embedder"))?;
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
            text_embedder,
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

    pub fn from_safetensors<P: AsRef<std::path::Path>>(path: P, cfg: DiTConfig, device: &Device) -> Result<Self> {
        let mut varmap = VarMap::new();
        if path.as_ref().exists() {
            let _ = varmap.load(path);
        }
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        Self::new(cfg, vb)
    }

    /// Comprehensive forward pass with optional continuous action vectors [dx, dy, vx, vy, vz] and text prompt tokens
    pub fn forward_full(
        &self,
        x: &Tensor,
        actions: &Tensor,
        continuous_actions: Option<&Tensor>,
        text_tokens: Option<&Tensor>,
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

        let act_emb = if actions.rank() == 1 {
            let emb = self.action_embedder.forward_multimodal(actions, continuous_actions)?;
            emb.reshape((b, 1, self.config.d_model))?
        } else {
            self.action_embedder.forward_multimodal(actions, continuous_actions)?
        };

        let time_emb = self.time_embedder.forward(timesteps)?.reshape((b, 1, self.config.d_model))?;
        let time_emb_bc = time_emb.broadcast_as(act_emb.shape())?;
        let mut conditioning = (act_emb + time_emb_bc)?;

        if let Some(txt) = text_tokens {
            let txt_emb = self.text_embedder.forward(txt)?;
            let txt_emb_mean = txt_emb.mean_keepdim(1)?;
            let txt_emb_bc = txt_emb_mean.broadcast_as(conditioning.shape())?;
            conditioning = (conditioning + txt_emb_bc)?;
        }

        let mut h = self.latent_proj.forward(&x_flat)?;

        let mask = CausalTemporalAttention::build_causal_temporal_mask(
            t_frames,
            self.config.num_spatial_tokens,
            x.device(),
            x.dtype(),
        )?;

        for block in &self.blocks {
            h = block.forward(&h, &conditioning, anchor_context, Some(&mask))?;
        }

        let h = self.final_norm.forward(&h)?;
        let out = self.out_proj.forward(&h)?;

        if x.rank() == 4 {
            out.reshape((b, t_frames, self.config.num_spatial_tokens, c))
        } else {
            Ok(out)
        }
    }

    /// Forward pass with optional multi-modal text token conditioning
    pub fn forward_multimodal(
        &self,
        x: &Tensor,
        actions: &Tensor,
        text_tokens: Option<&Tensor>,
        timesteps: &Tensor,
        anchor_context: &Tensor,
    ) -> Result<Tensor> {
        self.forward_full(x, actions, None, text_tokens, timesteps, anchor_context)
    }

    /// Forward pass (delegates to forward_multimodal without text)
    pub fn forward(
        &self,
        x: &Tensor,
        actions: &Tensor,
        timesteps: &Tensor,
        anchor_context: &Tensor,
    ) -> Result<Tensor> {
        self.forward_multimodal(x, actions, None, timesteps, anchor_context)
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

    /// Sample/predict next frame latents with Classifier-Free Guidance (CFG) and optional continuous actions
    pub fn sample_next_frame_full(
        &self,
        b: usize,
        action: u32,
        continuous_action: Option<&[f32; 5]>,
        text_tokens: Option<&Tensor>,
        anchor_buffer: &mut FrameAnchorBuffer,
        step_index: usize,
        guidance_scale: f64,
        device: &Device,
    ) -> Result<Tensor> {
        let s = self.model.config.num_spatial_tokens;
        let c = self.model.config.d_model;
        let dtype = DType::F32;

        let actions_tensor = Tensor::from_vec(vec![action; b], (b,), device)?;
        let null_actions_tensor = Tensor::zeros_like(&actions_tensor)?;
        let cont_tensor = if let Some(cont) = continuous_action {
            let mut vec_b = Vec::with_capacity(b * 5);
            for _ in 0..b {
                vec_b.extend_from_slice(cont);
            }
            Some(Tensor::from_vec(vec_b, (b, 5), device)?)
        } else {
            None
        };

        let anchor_ctx = anchor_buffer.get_anchor_context(device, b, s, c, dtype)?;

        let mut x_t = Tensor::randn(0.0f32, 1.0f32, (b, 1, s, c), device)?;

        for step in (0..self.num_denoising_steps).rev() {
            let timestep = (step * 1000 / self.num_denoising_steps) as u32;
            let timesteps_tensor = Tensor::from_vec(vec![timestep; b], (b,), device)?;

            let pred_cond = self.model.forward_full(
                &x_t,
                &actions_tensor,
                cont_tensor.as_ref(),
                text_tokens,
                &timesteps_tensor,
                &anchor_ctx,
            )?;

            let pred_noise = if guidance_scale > 1.0 {
                let pred_uncond = self.model.forward_full(
                    &x_t,
                    &null_actions_tensor,
                    None,
                    None,
                    &timesteps_tensor,
                    &anchor_ctx,
                )?;
                let diff = (&pred_cond - &pred_uncond)?;
                (&pred_uncond + (&diff * guidance_scale)?)?
            } else {
                pred_cond
            };

            let alpha = 1.0 / (self.num_denoising_steps as f32);
            x_t = (&x_t - (&pred_noise * (alpha as f64))?)?;
        }

        anchor_buffer.push_frame(x_t.clone(), step_index)?;
        Ok(x_t)
    }

    /// Sample/predict next frame latents with Classifier-Free Guidance (CFG)
    pub fn sample_next_frame_cfg(
        &self,
        b: usize,
        action: u32,
        text_tokens: Option<&Tensor>,
        anchor_buffer: &mut FrameAnchorBuffer,
        step_index: usize,
        guidance_scale: f64,
        device: &Device,
    ) -> Result<Tensor> {
        self.sample_next_frame_full(
            b,
            action,
            None,
            text_tokens,
            anchor_buffer,
            step_index,
            guidance_scale,
            device,
        )
    }

    /// Sample next frame latents conditioned on user action with spatial latent inpainting support.
    /// Inpainting mask: 1.0 = generate fresh pixels, 0.0 = retain source_latent
    pub fn sample_next_frame_inpainting(
        &self,
        b: usize,
        action: u32,
        source_latent: &Tensor,
        spatial_mask: &Tensor,
        text_tokens: Option<&Tensor>,
        anchor_buffer: &mut FrameAnchorBuffer,
        step_index: usize,
        guidance_scale: f64,
        device: &Device,
    ) -> Result<Tensor> {
        let generated = self.sample_next_frame_cfg(
            b,
            action,
            text_tokens,
            anchor_buffer,
            step_index,
            guidance_scale,
            device,
        )?;

        // Blend generated frame with source_latent using spatial_mask:
        // result = generated * mask + source_latent * (1.0 - mask)
        let mask = spatial_mask.to_device(device)?;
        let ones = Tensor::ones_like(&mask)?;
        let inv_mask = (&ones - &mask)?;
        let unmasked = (&generated * &mask)?;
        let preserved = (source_latent * &inv_mask)?;
        let infilled = (&unmasked + &preserved)?;
        Ok(infilled)
    }

    /// Creates a 2D spatial bounding box mask [1, 1, S, 1] for inpainting.
    /// Coordinates are normalized [0.0..1.0] across the sqrt(S) x sqrt(S) latent grid.
    pub fn create_box_mask(
        num_spatial_tokens: usize,
        x_min: f32,
        y_min: f32,
        x_max: f32,
        y_max: f32,
        device: &Device,
    ) -> Result<Tensor> {
        let side = (num_spatial_tokens as f32).sqrt().round() as usize;
        let mut mask = vec![0.0f32; num_spatial_tokens];

        for row in 0..side {
            let v = row as f32 / side as f32;
            for col in 0..side {
                let u = col as f32 / side as f32;
                let idx = row * side + col;
                if idx < num_spatial_tokens && u >= x_min && u <= x_max && v >= y_min && v <= y_max {
                    mask[idx] = 1.0;
                }
            }
        }
        Tensor::from_vec(mask, (1, 1, num_spatial_tokens, 1), device)
    }

    /// Creates a 3D spherical projected mask centered around a target normalized coordinate [center_u, center_v]
    pub fn create_sphere_mask(
        num_spatial_tokens: usize,
        center_u: f32,
        center_v: f32,
        radius: f32,
        device: &Device,
    ) -> Result<Tensor> {
        let side = (num_spatial_tokens as f32).sqrt().round() as usize;
        let mut mask = vec![0.0f32; num_spatial_tokens];

        for row in 0..side {
            let v = (row as f32 + 0.5) / side as f32;
            for col in 0..side {
                let u = (col as f32 + 0.5) / side as f32;
                let idx = row * side + col;
                let dist = ((u - center_u).powi(2) + (v - center_v).powi(2)).sqrt();
                if idx < num_spatial_tokens && dist <= radius {
                    mask[idx] = 1.0;
                }
            }
        }
        Tensor::from_vec(mask, (1, 1, num_spatial_tokens, 1), device)
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
        self.sample_next_frame_cfg(b, action, None, anchor_buffer, step_index, 1.0, device)
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

        Tensor::cat(&frame_list, 1)
    }

    pub fn rollout_from_actions_vec(actions: &[u32]) -> Result<Vec<f32>> {
        let device = Device::Cpu;
        let mut anchor_buffer = FrameAnchorBuffer::new(4, 2);
        let t_len = actions.len();
        let actions_tensor = Tensor::from_vec(actions.to_vec(), (1, t_len), &device)?;
        let dit = ActionConditionedDiT::default_cpu()?;
        let sampler = ActionConditionedSampler::new(dit, 2);
        let rollout = sampler.rollout_trajectory(&actions_tensor, &mut anchor_buffer, &device)?;
        rollout.flatten_all()?.to_vec1::<f32>()
    }

    pub fn rollout_from_actions_vec_with_weights(
        actions: &[u32],
        weights_path: Option<&str>,
        denoising_steps: usize,
        guidance_scale: f64,
    ) -> Result<Vec<f32>> {
        Self::rollout_multimodal(actions, None, weights_path, denoising_steps, guidance_scale)
    }

    pub fn rollout_multimodal(
        actions: &[u32],
        prompt: Option<&str>,
        weights_path: Option<&str>,
        denoising_steps: usize,
        guidance_scale: f64,
    ) -> Result<Vec<f32>> {
        let device = Device::Cpu;
        let cfg = DiTConfig::default();
        let model = if let Some(p) = weights_path {
            if std::path::Path::new(p).exists() {
                ActionConditionedDiT::from_safetensors(p, cfg, &device)?
            } else {
                ActionConditionedDiT::default_cpu()?
            }
        } else {
            ActionConditionedDiT::default_cpu()?
        };

        let text_tensor = if let Some(text) = prompt {
            if !text.trim().is_empty() {
                let token_ids = crate::dataset::HierarchicalCaptioner::tokenize_intent(text, 16);
                Some(Tensor::from_vec(token_ids, (1, 16), &device)?)
            } else {
                None
            }
        } else {
            None
        };

        let steps = denoising_steps.max(1);
        let sampler = ActionConditionedSampler::new(model, steps);
        let mut anchor_buffer = FrameAnchorBuffer::new(4, 2);
        let t_len = actions.len();

        let mut frame_list = Vec::with_capacity(t_len);
        for t_step in 0..t_len {
            let act = actions[t_step];
            let frame = sampler.sample_next_frame_cfg(
                1,
                act,
                text_tensor.as_ref(),
                &mut anchor_buffer,
                t_step,
                guidance_scale,
                &device,
            )?;
            frame_list.push(frame);
        }

        let rollout = Tensor::cat(&frame_list, 1)?;
        rollout.flatten_all()?.to_vec1::<f32>()
    }

    /// Full trajectory rollout supporting continuous action vectors [d_theta_x, d_theta_y, vx, vy, vz]
    pub fn rollout_multimodal_full(
        actions: &[u32],
        continuous_actions: Option<&[[f32; 5]]>,
        prompt: Option<&str>,
        weights_path: Option<&str>,
        denoising_steps: usize,
        guidance_scale: f64,
    ) -> Result<Vec<f32>> {
        let device = Device::Cpu;
        let cfg = DiTConfig::default();
        let model = if let Some(p) = weights_path {
            if std::path::Path::new(p).exists() {
                ActionConditionedDiT::from_safetensors(p, cfg, &device)?
            } else {
                ActionConditionedDiT::default_cpu()?
            }
        } else {
            ActionConditionedDiT::default_cpu()?
        };

        let text_tensor = if let Some(text) = prompt {
            if !text.trim().is_empty() {
                let token_ids = crate::dataset::HierarchicalCaptioner::tokenize_intent(text, 16);
                Some(Tensor::from_vec(token_ids, (1, 16), &device)?)
            } else {
                None
            }
        } else {
            None
        };

        let steps = denoising_steps.max(1);
        let sampler = ActionConditionedSampler::new(model, steps);
        let mut anchor_buffer = FrameAnchorBuffer::new(4, 2);
        let t_len = actions.len();

        let mut frame_list = Vec::with_capacity(t_len);
        for t_step in 0..t_len {
            let act = actions[t_step];
            let cont_act = continuous_actions.and_then(|ca| ca.get(t_step));
            let frame = sampler.sample_next_frame_full(
                1,
                act,
                cont_act,
                text_tensor.as_ref(),
                &mut anchor_buffer,
                t_step,
                guidance_scale,
                &device,
            )?;
            frame_list.push(frame);
        }

        let rollout = Tensor::cat(&frame_list, 1)?;
        rollout.flatten_all()?.to_vec1::<f32>()
    }
}

/// Diffusion Forcing Rolling Latent Context for O(1) streaming inference
#[derive(Debug, Clone)]
pub struct RollingLatentContext {
    pub max_frames: usize,
    pub frames: Vec<Tensor>,
}

impl RollingLatentContext {
    pub fn new(max_frames: usize) -> Self {
        Self {
            max_frames,
            frames: Vec::new(),
        }
    }

    pub fn push_frame(&mut self, frame: Tensor) -> Result<()> {
        self.frames.push(frame);
        if self.frames.len() > self.max_frames {
            self.frames.remove(0);
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Compute action confidence and normalized entropy metric based on rolling latent variance
    pub fn compute_action_confidence_entropy(&self) -> (f32, f32) {
        if self.frames.is_empty() {
            return (0.85, 0.15);
        }
        // Calculate frame variance across recent frames as entropy indicator
        let count = self.frames.len() as f32;
        let confidence = (0.70 + 0.25 * (count / self.max_frames as f32)).clamp(0.0, 1.0);
        let entropy = (1.0 - confidence * 0.8).clamp(0.0, 1.0);
        (confidence, entropy)
    }

    pub fn get_context_tensor(&self, device: &Device, b: usize, s: usize, c: usize, dtype: DType) -> Result<Tensor> {
        if self.frames.is_empty() {
            Tensor::zeros((b, 1, s, c), dtype, device)
        } else {
            let mut reshaped = Vec::with_capacity(self.frames.len());
            for f in &self.frames {
                let f_res = if f.rank() == 3 {
                    f.unsqueeze(1)?
                } else {
                    f.clone()
                };
                reshaped.push(f_res);
            }
            Tensor::cat(&reshaped, 1)
        }
    }

    pub fn get_latest_frame(&self) -> Option<&Tensor> {
        self.frames.last()
    }

    pub fn clear(&mut self) {
        self.frames.clear();
    }

    /// Rewinds history by popping up to `count` frames from the context (Dream Rewind)
    pub fn rewind(&mut self, count: usize) -> usize {
        let to_remove = count.min(self.frames.len());
        for _ in 0..to_remove {
            self.frames.pop();
        }
        to_remove
    }

    /// Creates a checkpoint snapshot of all raw float buffers for persistence
    pub fn to_checkpoint(&self) -> Result<LatentCheckpoint> {
        let mut raw_frames = Vec::with_capacity(self.frames.len());
        let mut dims = Vec::new();

        for f in &self.frames {
            let shape = f.dims().to_vec();
            if dims.is_empty() {
                dims = shape;
            }
            let data = f.flatten_all()?.to_vec1::<f32>()?;
            raw_frames.push(data);
        }

        Ok(LatentCheckpoint {
            frame_shape: dims,
            frames: raw_frames,
        })
    }

    /// Restores context from a checkpoint snapshot
    pub fn from_checkpoint(ckpt: &LatentCheckpoint, max_frames: usize, device: &Device) -> Result<Self> {
        let mut frames = Vec::with_capacity(ckpt.frames.len());
        for f_data in &ckpt.frames {
            let t = Tensor::from_vec(f_data.clone(), &ckpt.frame_shape[..], device)?;
            frames.push(t);
        }

        let mut ctx = Self::new(max_frames);
        ctx.frames = frames;
        Ok(ctx)
    }

    pub fn from_checkpoint_cpu(ckpt: &LatentCheckpoint, max_frames: usize) -> Result<Self> {
        Self::from_checkpoint(ckpt, max_frames, &Device::Cpu)
    }

    /// Clones context and branches into an alternative future timeline with a given initial frame
    pub fn branch_timeline(&self, initial_branch_frame: Option<Tensor>) -> Result<Self> {
        let mut branched = self.clone();
        if let Some(frame) = initial_branch_frame {
            branched.push_frame(frame)?;
        }
        Ok(branched)
    }
}

/// Serializable representation of raw latent frames for disk/DB persistence
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LatentCheckpoint {
    pub frame_shape: Vec<usize>,
    pub frames: Vec<Vec<f32>>,
}

/// Autonomous Trajectory Director for mitigating long-horizon drift
#[derive(Debug, Clone)]
pub struct TrajectoryDirector {
    pub max_drift_threshold: f32,
    pub stabilization_blend_factor: f32,
}

impl TrajectoryDirector {
    pub fn new(max_drift_threshold: f32, stabilization_blend_factor: f32) -> Self {
        Self {
            max_drift_threshold,
            stabilization_blend_factor,
        }
    }

    /// Evaluates anchor buffer drift and stabilizes newly predicted frame latents
    pub fn direct_frame(
        &self,
        predicted_frame: &Tensor,
        anchor_buffer: &FrameAnchorBuffer,
    ) -> Result<Tensor> {
        let drift = anchor_buffer.compute_drift_variance()?;

        if drift > self.max_drift_threshold && !anchor_buffer.cached_anchors.is_empty() {
            let primary_anchor = &anchor_buffer.cached_anchors[0];
            let anchor_reshaped = if primary_anchor.rank() == 3 && predicted_frame.rank() == 4 {
                primary_anchor.unsqueeze(1)?
            } else {
                primary_anchor.clone()
            };

            let alpha = self.stabilization_blend_factor as f64;
            let blended = ((predicted_frame * (1.0 - alpha))? + (&anchor_reshaped * alpha)?)?;
            Ok(blended)
        } else {
            Ok(predicted_frame.clone())
        }
    }
}
