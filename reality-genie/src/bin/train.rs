use candle_core::{Device, Tensor, Result, DType};
use candle_nn::{VarBuilder, VarMap, Optimizer, AdamW, ParamsAdamW};
use reality_genie::diffusion::{DiscreteDiffusion, DiffusionConfig};
use std::fs;
use std::path::Path;

fn load_voxel_chunks(dir: &str, device: &Device) -> Result<Vec<Tensor>> {
    let mut tensors = Vec::new();
    let path = Path::new(dir);

    if path.is_dir() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let p = entry.path();
            if p.is_file() && p.extension().and_then(|s| s.to_str()) == Some("npy") {
                let bytes = fs::read(&p)?;

                // Simple parsing for flat uint8 npy file
                if let Ok(data) = npyz::NpyFile::new(&bytes[..]) {
                    if let Ok(flat) = data.into_vec::<u8>() {
                        let t = Tensor::from_vec(
                            flat.into_iter().map(|x| x as u32).collect::<Vec<_>>(),
                            (1, 10 * 10 * 10), // Assuming 10x10x10 resolution from generate_data.py
                            device,
                        )?;
                        tensors.push(t);
                        println!("Loaded {:?}", p);
                    }
                }
            }
        }
    }
    Ok(tensors)
}

fn main() -> Result<()> {
    let global_start = std::time::Instant::now();
    println!("Starting Reality Genie Diffusion Training Loop...");

    let device = Device::Cpu; // Or Device::new_cuda(0)? for GPU
    let synthetic_data_dir = "data/synthetic";

    // 1. Load Data
    println!("Loading voxel chunks from {}...", synthetic_data_dir);
    let mut dataset = load_voxel_chunks(synthetic_data_dir, &device).unwrap_or_else(|_| vec![]);

    // Create some dummy data if we don't have python output yet so the script works
    if dataset.is_empty() {
        println!("No .npy files found in {}. Generating dummy data...", synthetic_data_dir);
        for _ in 0..10 {
            // [Batch=1, Seq=1000]
            let t = Tensor::ones((1, 1000), DType::U32, &device)?;
            dataset.push(t);
        }
    }

    // 2. Build Model
    let cfg = DiffusionConfig {
        vocab_size: 10,       // Max voxel ID expected
        d_model: 64,
        n_head: 4,
        num_layers: 2,
        max_seq_len: 1000,    // 10x10x10 chunk = 1000 tokens
    };

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = DiscreteDiffusion::new(cfg, vb)?;

    // 3. Setup Optimizer
    let params = ParamsAdamW::default();
    let mut opt = AdamW::new(varmap.all_vars(), params)?;

    let epochs = 5;
    let mask_prob = 0.15; // Mask 15% of tokens

    println!("Beginning Training (Epochs: {})...", epochs);

    // 4. Training Loop
    for epoch in 1..=epochs {
        let epoch_start = std::time::Instant::now();
        let mut epoch_loss = 0.0;

        for batch in &dataset {
            // Forward Process: Corrupt input
            let (x_masked, _mask_indices) = model.q_sample(batch, mask_prob)?;

            // Forward Pass
            let logits = model.p_sample(&x_masked)?;

            // Compute Cross Entropy Loss
            // Target is the original unmasked tokens.
            let (b, t, v) = logits.dims3()?;
            let logits_flat = logits.reshape((b * t, v))?;
            let target_flat = batch.reshape(b * t)?;

            let loss = candle_nn::loss::cross_entropy(&logits_flat, &target_flat)?;

            // Backprop
            opt.backward_step(&loss)?;

            epoch_loss += loss.to_scalar::<f32>()?;
        }

        let epoch_duration = epoch_start.elapsed();
        println!("Epoch {}: Avg Loss: {:.4} (Took: {:.2?})", epoch, epoch_loss / dataset.len() as f32, epoch_duration);
    }

    // 5. Save Weights
    let save_path = "diffusion_model.safetensors";
    println!("Saving trained weights to {}...", save_path);
    varmap.save(save_path)?;

    let total_duration = global_start.elapsed();
    println!("Training Complete in {:.2?}!", total_duration);
    Ok(())
}
