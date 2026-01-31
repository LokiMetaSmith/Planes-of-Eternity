use candle_core::{Device, Tensor, Result, DType, Module, IndexOp};
use candle_nn::VarBuilder;
use reality_genie::vqvae::{VqVae, VqVaeConfig};
use reality_genie::lam::{LatentActionModel, LamConfig};
use reality_genie::dynamics::{DynamicsModel, DynamicsConfig};

fn main() -> Result<()> {
    println!("Initializing Genie System...");

    let device = Device::Cpu; // Use CUDA if available in real scenario

    // 1. Setup Models
    let vq_cfg = VqVaeConfig::default();
    let lam_cfg = LamConfig::default();
    let dyn_cfg = DynamicsConfig::default();

    // Create random weights
    let varmap = candle_nn::VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    println!("Building VQ-VAE...");
    let vqvae = VqVae::new(&vq_cfg, vb.pp("vqvae"))?;

    println!("Building LAM...");
    let lam = LatentActionModel::new(&lam_cfg, vb.pp("lam"))?;

    println!("Building Dynamics Model...");
    let dynamics = DynamicsModel::new(&dyn_cfg, vb.pp("dynamics"))?;

    // 2. Mock Inference Loop
    println!("\nStarting Inference Loop:");

    // Generate mock input image [B, C, H, W]
    let batch_size = 1;
    let img_size = 32; // Reduced for speed in demo, assuming VQ-VAE handles it (default config assumes convolutions fit)
    // Note: VQ-VAE config default stride is 2*2 = 4 total downsampling. 32 -> 8.
    let input_image = Tensor::randn(0f32, 1f32, (batch_size, 3, img_size, img_size), &device)?;

    println!("-> Input Image Shape: {:?}", input_image.shape());

    // 3. Tokenize
    println!("-> Tokenizing...");
    let tokens = vqvae.encode_to_indices(&input_image)?; // [B, H', W']
    println!("-> Tokens Shape: {:?}", tokens.shape());

    // 4. Predict Action (Mocking 2 frames)
    // For LAM, we need latents (before quantization or after? Paper uses quantized tokens or latents. Let's use latents)
    // But `encode_to_indices` returns indices.
    // Let's get Z from encoder for LAM
    let z = vqvae.encoder.forward(&input_image)?;
    println!("-> Latents Shape: {:?}", z.shape());

    let action_logits = lam.forward(&z, &z)?; // Predicting action between frame 1 and frame 1 (static) -> should be "no-op"
    let action = action_logits.argmax(1)?;
    println!("-> Predicted Action: {:?}", action.to_vec1::<u32>()?);

    // 5. Dynamics Prediction
    // Input: [Token1, Token2, ..., Action, Token1, ...]
    // Flatten tokens
    let tokens_flat = tokens.flatten_all()?; // [B * H' * W']
    // Convert tokens to sequence
    let tokens_seq = tokens_flat.unsqueeze(0)?; // [1, T]

    // Prepare input for dynamics: Sequence of Frame Tokens + Action Token
    // We need to construct [Frame Tokens, Action Token]
    // Action token needs to be offset or flagged.
    // In our dynamics model, we use `token_types` to distinguish.

    // action is [B] (here [1]) containing the index.
    // We want action_tensor to be [1, 1].
    let action_tensor = action.reshape((1, 1))?;

    // Concatenate tokens and action
    let input_seq = Tensor::cat(&[&tokens_seq, &action_tensor], 1)?; // [1, T+1]

    // Token Types: 0 for frame, 1 for action
    let t_len = tokens_seq.dim(1)?;
    let type_0 = Tensor::zeros((1, t_len), DType::F32, &device)?;
    let type_1 = Tensor::ones((1, 1), DType::F32, &device)?;
    let types = Tensor::cat(&[&type_0, &type_1], 1)?;

    println!("-> Dynamics Input Shape: {:?}", input_seq.shape());

    let next_token_logits = dynamics.forward(&input_seq, None, &types)?;
    println!("-> Dynamics Output Logits: {:?}", next_token_logits.shape());

    // Predict next token (Autoregressive step 1)
    let next_token = next_token_logits.i((0, t_len, ..))?.argmax(0)?; // [Vocab] -> Scalar
    println!("-> Predicted Next Token: {}", next_token.to_scalar::<u32>()?);

    // 6. Detokenize (Reconstruction for demo)
    // We decode the original tokens to show VQ-VAE works
    let recon = vqvae.decode_from_indices(&tokens)?;
    println!("-> Reconstructed Image Shape: {:?}", recon.shape());

    println!("\nGenie Pipeline Verified Successfully.");
    Ok(())
}
