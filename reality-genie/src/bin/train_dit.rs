use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use reality_genie::dataset::TrajectoryDataset;
use reality_genie::dit::{ActionConditionedDiT, DiTConfig, FrameAnchorBuffer};

fn compute_collision_boundary_loss(latents: &Tensor) -> Result<Tensor> {
    let s = latents.dim(2)?;
    if s < 2 {
        return Tensor::zeros((), DType::F32, latents.device());
    }
    let s_a = latents.narrow(2, 0, s - 1)?;
    let s_b = latents.narrow(2, 1, s - 1)?;
    let diff = (s_a - s_b)?;
    let dist_sq = diff.sqr()?.sum_keepdim(3)?;
    let penalty = (0.05 - dist_sq)?.relu()?;
    penalty.mean_all()
}

fn main() -> Result<()> {
    let global_start = std::time::Instant::now();
    println!("Starting Action-Conditioned DiT Training Loop with Physics Boundary Penalty Loss...");

    let device = Device::Cpu;

    let num_trajectories = 8;
    let t_frames = 4;
    let num_spatial_tokens = 16;
    let d_model = 64;
    let num_actions = 8;

    println!("Generating synthetic trajectory dataset...");
    let dataset = TrajectoryDataset::generate_synthetic(
        num_trajectories,
        t_frames,
        num_spatial_tokens,
        d_model,
        num_actions,
        &device,
    )?;

    let cfg = DiTConfig {
        num_actions,
        action_dim: 16,
        d_model,
        n_head: 2,
        num_layers: 2,
        num_spatial_tokens,
        max_temporal_frames: t_frames,
        max_anchor_frames: 2,
        vocab_size_text: 1000,
        dropout: 0.0,
    };

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = ActionConditionedDiT::new(cfg, vb)?;

    let params = ParamsAdamW::default();
    let mut opt = AdamW::new(varmap.all_vars(), params)?;

    let epochs = 5;
    let batch_size = 2;

    println!("Beginning Training with Physics Collision Penalty (Epochs: {})...", epochs);

    let anchor_buffer = FrameAnchorBuffer::new(2, 2);

    for epoch in 1..=epochs {
        let epoch_start = std::time::Instant::now();
        let mut epoch_loss = 0.0;
        let num_batches = dataset.len() / batch_size;

        for b_idx in 0..num_batches {
            let indices: Vec<usize> = (b_idx * batch_size..(b_idx + 1) * batch_size).collect();
            let (latents, actions) = dataset.get_batch(&indices)?;

            let (b, t, s, c) = latents.dims4()?;

            let noise = Tensor::randn(0.0f32, 1.0f32, (b, t, s, c), &device)?;
            let timesteps = Tensor::from_vec(vec![500u32; b], (b,), &device)?;

            let noisy_latents = (&latents + &noise)?;
            let anchor_ctx = anchor_buffer.get_anchor_context(&device, b, s, c, DType::F32)?;

            let pred_noise = model.forward(&noisy_latents, &actions, &timesteps, &anchor_ctx)?;

            // 1. MSE Noise Prediction Loss
            let diff = (&pred_noise - &noise)?;
            let mse_loss = diff.sqr()?.mean_all()?;

            // 2. Physics Collision & Boundary Penalty Loss
            let collision_loss = compute_collision_boundary_loss(&latents)?;

            // Total Loss = MSE + 0.1 * Collision Penalty
            let total_loss = (mse_loss + (&collision_loss * 0.1)?)?;

            opt.backward_step(&total_loss)?;
            epoch_loss += total_loss.to_scalar::<f32>()?;
        }

        let avg_loss = if num_batches > 0 {
            epoch_loss / num_batches as f32
        } else {
            0.0
        };

        println!(
            "Epoch {}: Avg Loss: {:.6} (Took: {:.2?})",
            epoch,
            avg_loss,
            epoch_start.elapsed()
        );
    }

    let save_path = "action_dit_model.safetensors";
    println!("Saving Action-Conditioned DiT weights to {}...", save_path);
    varmap.save(save_path)?;

    println!(
        "Action-Conditioned DiT Training Complete in {:.2?}!",
        global_start.elapsed()
    );

    Ok(())
}
