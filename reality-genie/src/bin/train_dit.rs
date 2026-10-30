use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use reality_genie::dataset::TrajectoryDataset;
use reality_genie::dit::{ActionConditionedDiT, DiTConfig, FrameAnchorBuffer};

fn main() -> Result<()> {
    let global_start = std::time::Instant::now();
    println!("Starting Action-Conditioned DiT Training Loop...");

    let device = Device::Cpu;

    // 1. Build synthetic trajectory dataset
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

    // 2. Build DiT model configuration
    let cfg = DiTConfig {
        num_actions,
        action_dim: 16,
        d_model,
        n_head: 2,
        num_layers: 2,
        num_spatial_tokens,
        max_temporal_frames: t_frames,
        max_anchor_frames: 2,
        dropout: 0.0,
    };

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = ActionConditionedDiT::new(cfg, vb)?;

    // 3. Setup AdamW Optimizer
    let params = ParamsAdamW::default();
    let mut opt = AdamW::new(varmap.all_vars(), params)?;

    let epochs = 5;
    let batch_size = 2;

    println!("Beginning Action-Conditioned DiT Training (Epochs: {})...", epochs);

    let anchor_buffer = FrameAnchorBuffer::new(2, 2);

    // 4. Training Loop
    for epoch in 1..=epochs {
        let epoch_start = std::time::Instant::now();
        let mut epoch_loss = 0.0;
        let num_batches = dataset.len() / batch_size;

        for b_idx in 0..num_batches {
            let indices: Vec<usize> = (b_idx * batch_size..(b_idx + 1) * batch_size).collect();
            let (latents, actions) = dataset.get_batch(&indices)?; // [B, T, S, C], [B, T]

            let (b, t, s, c) = latents.dims4()?;

            // Sample random noise and timesteps
            let noise = Tensor::randn(0.0f32, 1.0f32, (b, t, s, c), &device)?;
            let timesteps = Tensor::from_vec(vec![500u32; b], (b,), &device)?;

            // Corrupt latents with noise
            let noisy_latents = (&latents + &noise)?;

            // Get anchor context
            let anchor_ctx = anchor_buffer.get_anchor_context(&device, b, s, c, DType::F32)?;

            // Forward pass
            let pred_noise = model.forward(&noisy_latents, &actions, &timesteps, &anchor_ctx)?;

            // MSE Loss: (pred_noise - noise)^2
            let diff = (&pred_noise - &noise)?;
            let loss = diff.sqr()?.mean_all()?;

            opt.backward_step(&loss)?;
            epoch_loss += loss.to_scalar::<f32>()?;
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

    // 5. Save Model Weights
    let save_path = "action_dit_model.safetensors";
    println!("Saving Action-Conditioned DiT weights to {}...", save_path);
    varmap.save(save_path)?;

    println!(
        "Action-Conditioned DiT Training Complete in {:.2?}!",
        global_start.elapsed()
    );

    Ok(())
}
