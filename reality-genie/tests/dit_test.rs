use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use reality_genie::dataset::TrajectoryDataset;
use reality_genie::dit::{
    ActionConditionedDiT, ActionConditionedSampler, ActionConditioningEmbedder,
    CausalTemporalAttention, DiTConfig, FrameAnchorBuffer,
};
use reality_genie::vqvae::LatentVoxelDecoder;

#[test]
fn test_action_conditioning_embedder() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let num_actions = 10;
    let action_dim = 32;
    let d_model = 64;

    let vars = candle_nn::VarMap::new();
    let vb = VarBuilder::from_varmap(&vars, DType::F32, &device);

    let embedder = ActionConditioningEmbedder::new(num_actions, action_dim, d_model, vb)?;

    let actions = Tensor::from_vec(vec![1u32, 5u32], (2,), &device)?;
    let output = embedder.forward(&actions)?;

    assert_eq!(output.dims(), &[2, d_model]);
    Ok(())
}

#[test]
fn test_frame_anchor_buffer() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let max_window_size = 3;
    let anchor_interval = 2;

    let mut buffer = FrameAnchorBuffer::new(max_window_size, anchor_interval);

    let b = 2;
    let s = 16;
    let c = 64;

    let empty_ctx = buffer.get_anchor_context(&device, b, s, c, DType::F32)?;
    assert_eq!(empty_ctx.dims(), &[b, 1, c]);

    for step in 0..10 {
        let frame = Tensor::zeros((b, s, c), DType::F32, &device)?;
        buffer.push_frame(frame, step)?;
    }

    assert_eq!(buffer.cached_anchors.len(), max_window_size);

    let drift_var = buffer.compute_drift_variance()?;
    assert_eq!(drift_var, 0.0f32);

    let anchor_ctx = buffer.get_anchor_context(&device, b, s, c, DType::F32)?;
    assert_eq!(anchor_ctx.dims(), &[b, max_window_size * s, c]);

    buffer.clear();
    assert_eq!(buffer.cached_anchors.len(), 0);

    Ok(())
}

#[test]
fn test_causal_temporal_mask() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let t = 3;
    let s = 4;
    let mask = CausalTemporalAttention::build_causal_temporal_mask(t, s, &device, DType::F32)?;

    assert_eq!(mask.dims(), &[1, 1, t * s, t * s]);

    let mask_vec = mask.flatten_all()?.to_vec1::<f32>()?;
    let idx_0_to_4 = 0 * (t * s) + 4;
    assert_eq!(mask_vec[idx_0_to_4], -1e9f32);

    let idx_4_to_0 = 4 * (t * s) + 0;
    assert_eq!(mask_vec[idx_4_to_0], 0.0f32);

    Ok(())
}

#[test]
fn test_action_conditioned_dit_forward() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let cfg = DiTConfig {
        num_actions: 8,
        action_dim: 16,
        d_model: 64,
        n_head: 2,
        num_layers: 2,
        num_spatial_tokens: 16,
        max_temporal_frames: 4,
        max_anchor_frames: 2,
        vocab_size_text: 100,
        dropout: 0.0,
    };

    let vars = candle_nn::VarMap::new();
    let vb = VarBuilder::from_varmap(&vars, DType::F32, &device);

    let dit = ActionConditionedDiT::new(cfg.clone(), vb)?;

    let b = 2;
    let t = 2;
    let s = cfg.num_spatial_tokens;
    let c = cfg.d_model;

    let x = Tensor::zeros((b, t, s, c), DType::F32, &device)?;
    let actions = Tensor::from_vec(vec![1u32, 3u32], (b,), &device)?;
    let timesteps = Tensor::from_vec(vec![100u32, 200u32], (b,), &device)?;
    let anchor_context = Tensor::zeros((b, s, c), DType::F32, &device)?;

    let output = dit.forward(&x, &actions, &timesteps, &anchor_context)?;

    assert_eq!(output.dims(), &[b, t, s, c]);

    Ok(())
}

#[test]
fn test_multimodal_forward() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let cfg = DiTConfig {
        num_actions: 8,
        action_dim: 16,
        d_model: 64,
        n_head: 2,
        num_layers: 2,
        num_spatial_tokens: 16,
        max_temporal_frames: 4,
        max_anchor_frames: 2,
        vocab_size_text: 100,
        dropout: 0.0,
    };

    let vars = candle_nn::VarMap::new();
    let vb = VarBuilder::from_varmap(&vars, DType::F32, &device);

    let dit = ActionConditionedDiT::new(cfg.clone(), vb)?;

    let b = 2;
    let t = 2;
    let s = cfg.num_spatial_tokens;
    let c = cfg.d_model;

    let x = Tensor::zeros((b, t, s, c), DType::F32, &device)?;
    let actions = Tensor::from_vec(vec![1u32, 3u32], (b,), &device)?;
    let text_tokens = Tensor::from_vec(vec![5u32, 10u32, 15u32, 20u32], (b, 2), &device)?;
    let timesteps = Tensor::from_vec(vec![100u32, 200u32], (b,), &device)?;
    let anchor_context = Tensor::zeros((b, s, c), DType::F32, &device)?;

    let output = dit.forward_multimodal(&x, &actions, Some(&text_tokens), &timesteps, &anchor_context)?;

    assert_eq!(output.dims(), &[b, t, s, c]);

    Ok(())
}

#[test]
fn test_cfg_sampler() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let dit = ActionConditionedDiT::default_cpu()?;
    let sampler = ActionConditionedSampler::new(dit, 2);

    let mut anchor_buffer = FrameAnchorBuffer::new(4, 2);
    let sample = sampler.sample_next_frame_cfg(1, 1, None, &mut anchor_buffer, 0, 2.5, &device)?;

    assert_eq!(sample.dims(), &[1, 1, 64, 256]);

    Ok(())
}

#[test]
fn test_latent_voxel_decoder() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let d_model = 32;
    let vocab_size = 10;

    let vars = candle_nn::VarMap::new();
    let vb = VarBuilder::from_varmap(&vars, DType::F32, &device);

    let decoder = LatentVoxelDecoder::new(d_model, vocab_size, vb)?;

    let latents = Tensor::zeros((2, 4, 16, d_model), DType::F32, &device)?;
    let voxels = decoder.decode_voxels(&latents)?;

    assert_eq!(voxels.dims(), &[2, 4, 16]);

    let splats = decoder.decode_splats(&latents)?;
    assert!(!splats.is_empty());

    Ok(())
}

#[test]
fn test_trajectory_dataset() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let dataset = TrajectoryDataset::generate_synthetic(4, 3, 8, 32, 5, &device)?;

    assert_eq!(dataset.len(), 4);
    assert!(!dataset.is_empty());

    let (batch_latents, batch_actions) = dataset.get_batch(&[0, 2])?;
    assert_eq!(batch_latents.dims(), &[2, 3, 8, 32]);
    assert_eq!(batch_actions.dims(), &[2, 3]);

    Ok(())
}

#[test]
fn test_action_conditioned_sampler_rollout() -> candle_core::Result<()> {
    let dit = ActionConditionedDiT::default_cpu()?;
    let sampler = ActionConditionedSampler::new(dit, 2);

    let actions = vec![1u32, 2u32, 3u32];
    let latents_vec = sampler.rollout_from_actions_vec(&actions)?;

    assert_eq!(latents_vec.len(), 3 * 64 * 256);

    Ok(())
}
