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

    /// Exports the trajectory dataset to a `.safetensors` file containing aligned
    /// `latents_{i}` and `actions_{i}` tensors.
    pub fn save_to_safetensors<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let mut map = std::collections::HashMap::new();

        for (i, item) in self.items.iter().enumerate() {
            let lat_key = format!("latents_{}", i);
            let act_key = format!("actions_{}", i);
            map.insert(lat_key, item.latents.clone());
            map.insert(act_key, item.actions.clone());
        }

        candle_core::safetensors::save(&map, path)?;
        Ok(())
    }

    /// Loads a trajectory dataset from an exported `.safetensors` file
    pub fn load_from_safetensors<P: AsRef<std::path::Path>>(path: P, device: &Device) -> Result<Self> {
        let tensors = candle_core::safetensors::load(path, device)?;
        let mut items = Vec::new();
        let mut i = 0;

        loop {
            let lat_key = format!("latents_{}", i);
            let act_key = format!("actions_{}", i);

            if let (Some(lat), Some(act)) = (tensors.get(&lat_key), tensors.get(&act_key)) {
                let item = TrajectoryItem::new(lat.clone(), act.clone(), Some(format!("Trajectory {}", i)))?;
                items.push(item);
                i += 1;
            } else {
                break;
            }
        }

        Ok(Self { items })
    }
}

/// Hierarchical Captioning & Intent Mapper for pairing semantic text prompts with action sequences.
/// Maps global goals ("explore the dungeon", "retreat from enemy", "jump across chasm") into
/// hierarchical local action tokens and synthetic prompt descriptions.
#[derive(Debug, Clone)]
pub struct HierarchicalCaptioner;

impl HierarchicalCaptioner {
    /// Maps a semantic intent string into an action sequence [T]
    pub fn intent_to_actions(intent: &str, length: usize) -> Vec<u32> {
        let lower = intent.to_lowercase();
        let base_pattern: Vec<u32> = if lower.contains("jump") || lower.contains("leap") {
            vec![1, 1, 5, 1] // Forward, Forward, Jump, Forward
        } else if lower.contains("retreat") || lower.contains("back") {
            vec![2, 2, 2, 2] // Backward...
        } else if lower.contains("cast") || lower.contains("spell") || lower.contains("fire") {
            vec![1, 6, 6, 1] // Forward, CastSpell, CastSpell, Forward
        } else if lower.contains("left") {
            vec![3, 3, 1, 3] // Left, Left, Forward, Left
        } else if lower.contains("right") {
            vec![4, 4, 1, 4] // Right, Right, Forward, Right
        } else {
            vec![1, 1, 1, 1] // Default Forward exploration
        };

        (0..length)
            .map(|i| base_pattern[i % base_pattern.len()])
            .collect()
    }

    /// Generates a hierarchical structured caption from an action sequence
    pub fn actions_to_hierarchical_caption(actions: &[u32]) -> (String, String) {
        let global_intent = if actions.contains(&5) {
            "Locomotion with vertical obstacle navigation"
        } else if actions.contains(&6) {
            "Combat or environmental spell interaction"
        } else if actions.iter().all(|&a| a == 2) {
            "Defensive rearward retreat"
        } else {
            "Forward exploratory navigation"
        };

        let token_descriptions: Vec<&str> = actions
            .iter()
            .map(|&a| match a {
                1 => "MoveForward",
                2 => "MoveBackward",
                3 => "MoveLeft",
                4 => "MoveRight",
                5 => "Jump",
                6 => "CastSpell",
                _ => "Idle",
            })
            .collect();

        let fine_grained = token_descriptions.join(" -> ");
        (global_intent.to_string(), fine_grained)
    }

    /// Synthesizes tokenized text IDs from an intent string for multimodal DiT conditioning
    pub fn tokenize_intent(intent: &str, max_tokens: usize) -> Vec<u32> {
        intent
            .bytes()
            .take(max_tokens)
            .map(|b| (b % 100) as u32)
            .collect()
    }
}
