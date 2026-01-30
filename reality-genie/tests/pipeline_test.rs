use candle_core::{Device, Tensor, Result, DType};
use candle_nn::VarBuilder;
use reality_genie::vqvae::{VqVae, VqVaeConfig};
use reality_genie::lam::{LatentActionModel, LamConfig};
use reality_genie::dynamics::{DynamicsModel, DynamicsConfig};

#[test]
fn test_vqvae_shapes() -> Result<()> {
    let device = Device::Cpu;
    let cfg = VqVaeConfig::default();
    let vb = VarBuilder::zeros(DType::F32, &device);
    let vqvae = VqVae::new(&cfg, vb)?;

    let input = Tensor::randn(0f32, 1f32, (1, 3, 32, 32), &device)?;
    let (recon, _, indices) = vqvae.forward(&input)?;

    assert_eq!(recon.shape().dims(), &[1, 3, 32, 32]);
    assert_eq!(indices.shape().dims(), &[1, 8, 8]); // 32 / 4 = 8
    Ok(())
}

#[test]
fn test_lam_shapes() -> Result<()> {
    let device = Device::Cpu;
    let cfg = LamConfig::default();
    let vb = VarBuilder::zeros(DType::F32, &device);
    let lam = LatentActionModel::new(&cfg, vb)?;

    // LAM expects flattened latents for 2 frames.
    // Assuming input_dim in config matches actual latent size.
    // Config default input_dim = 64 * 8 * 8 = 4096.
    // But VQVAE embedding dim is 64. 8x8 spatial.
    // So flatten size is correct.

    // We pass latents [B, 64, 8, 8]
    let z = Tensor::randn(0f32, 1f32, (1, 64, 8, 8), &device)?;

    let logits = lam.forward(&z, &z)?;
    assert_eq!(logits.shape().dims(), &[1, cfg.num_actions]);
    Ok(())
}

#[test]
fn test_dynamics_forward() -> Result<()> {
    let device = Device::Cpu;
    let cfg = DynamicsConfig {
        vocab_size: 100,
        num_actions: 5,
        d_model: 32,
        n_head: 4,
        num_layers: 2,
        max_seq_len: 50,
    };
    let vb = VarBuilder::zeros(DType::F32, &device);
    let model = DynamicsModel::new(&cfg, vb)?;

    // Input sequence of length 10
    let tokens = Tensor::zeros((1, 10), DType::U32, &device)?;

    // Token types: first 9 are frames (0), last 1 is action (1)
    let mut types_vec = vec![0f32; 9];
    types_vec.push(1.0);
    let types = Tensor::from_vec(types_vec, (1, 10), &device)?;

    let logits = model.forward(&tokens, None, &types)?;

    assert_eq!(logits.shape().dims(), &[1, 10, cfg.vocab_size]);
    Ok(())
}
