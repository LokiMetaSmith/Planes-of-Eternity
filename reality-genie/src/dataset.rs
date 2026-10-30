use candle_core::{Device, Result, Tensor};

/// Individual trajectory pairing: a sequence of spatial frame latents [T, S, C]
/// paired with millisecond-aligned discrete action tokens [T].
#[derive(Debug, Clone)]
pub struct TrajectoryItem {
    pub latents: Tensor, // Shape: [T, S, C]
    pub actions: Tensor, // Shape: [T]
    pub caption: Option<String>,
}

impl TrajectoryItem {
    pub fn new(latents: Tensor, actions: Tensor, caption: Option<String>) -> Result<Self> {
        let (t_lat, _s, _c) = latents.dims3()?;
        let t_act = actions.dims1()?;
        if t_lat != t_act {
            candle_core::bail!("Trajectory length mismatch: latents T={} vs actions T={}", t_lat, t_act);
        }
        Ok(Self {
            latents,
            actions,
            caption,
        })
    }
}

/// Dataset manager for paired video/frame latents and action trajectories.
#[derive(Debug, Clone)]
pub struct TrajectoryDataset {
    pub items: Vec<TrajectoryItem>,
}

impl TrajectoryDataset {
    pub fn new(items: Vec<TrajectoryItem>) -> Self {
        Self { items }
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<&TrajectoryItem> {
        self.items.get(index)
    }

    /// Generate synthetic paired trajectory dataset for local training and testing.
    pub fn generate_synthetic(
        num_trajectories: usize,
        t_frames: usize,
        num_spatial_tokens: usize,
        d_model: usize,
        num_actions: usize,
        device: &Device,
    ) -> Result<Self> {
        let mut items = Vec::with_capacity(num_trajectories);

        for i in 0..num_trajectories {
            // Generate random latents [T, S, C]
            let latents = Tensor::randn(0.0f32, 1.0f32, (t_frames, num_spatial_tokens, d_model), device)?;

            // Generate synthetic actions [T] in range [0, num_actions - 1]
            let raw_actions: Vec<u32> = (0..t_frames)
                .map(|t_step| ((t_step + i) % num_actions) as u32)
                .collect();
            let actions = Tensor::from_vec(raw_actions, (t_frames,), device)?;

            let item = TrajectoryItem::new(
                latents,
                actions,
                Some(format!("Synthetic trajectory sequence {}", i)),
            )?;
            items.push(item);
        }

        Ok(Self { items })
    }

    /// Create a batched tensor pair ([B, T, S, C], [B, T]) for training step
    pub fn get_batch(&self, indices: &[usize]) -> Result<(Tensor, Tensor)> {
        if indices.is_empty() {
            candle_core::bail!("Cannot get batch with empty indices");
        }

        let mut latents_list = Vec::with_capacity(indices.len());
        let mut actions_list = Vec::with_capacity(indices.len());

        for &idx in indices {
            let item = &self.items[idx % self.items.len()];
            latents_list.push(item.latents.unsqueeze(0)?); // [1, T, S, C]
            actions_list.push(item.actions.unsqueeze(0)?); // [1, T]
        }

        let batched_latents = Tensor::cat(&latents_list, 0)?;
        let batched_actions = Tensor::cat(&actions_list, 0)?;

        Ok((batched_latents, batched_actions))
    }
}
