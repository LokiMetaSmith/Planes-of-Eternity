use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use reality_genie::dataset::TrajectoryDataset;
use reality_genie::dit::{
    ActionConditionedDiT, ActionConditionedSampler, ActionConditioningEmbedder,
    CausalTemporalAttention, DiTConfig, FrameAnchorBuffer, RollingLatentContext, TrajectoryDirector,
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
fn test_rolling_latent_context() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let mut context = RollingLatentContext::new(3);

    assert_eq!(context.len(), 0);
    assert!(context.is_empty());

    let empty_tensor = context.get_context_tensor(&device, 1, 4, 16, DType::F32)?;
    assert_eq!(empty_tensor.dims(), &[1, 1, 4, 16]);

    for i in 0..5 {
        let frame = Tensor::full(i as f32, (1, 4, 16), &device)?;
        context.push_frame(frame)?;
    }

    // Should be clamped to max_frames = 3
    assert_eq!(context.len(), 3);

    let ctx_tensor = context.get_context_tensor(&device, 1, 4, 16, DType::F32)?;
    assert_eq!(ctx_tensor.dims(), &[1, 3, 4, 16]);

    context.clear();
    assert_eq!(context.len(), 0);

    Ok(())
}

#[test]
fn test_continuous_action_conditioning() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let num_actions = 8;
    let action_dim = 16;
    let d_model = 64;

    let vars = candle_nn::VarMap::new();
    let vb = VarBuilder::from_varmap(&vars, DType::F32, &device);

    let embedder = ActionConditioningEmbedder::new(num_actions, action_dim, d_model, vb)?;

    let discrete_actions = Tensor::from_vec(vec![2u32, 4u32], (2,), &device)?;
    // Continuous 5D vector: [d_theta_x, d_theta_y, vx, vy, vz]
    let cont_actions = Tensor::from_vec(vec![0.05f32, -0.02f32, 1.2f32, 0.0f32, -0.5f32, 0.0f32, 0.01f32, 0.0f32, 0.0f32, 0.0f32], (2, 5), &device)?;

    let output = embedder.forward_multimodal(&discrete_actions, Some(&cont_actions))?;
    assert_eq!(output.dims(), &[2, d_model]);

    Ok(())
}

#[test]
fn test_spatial_anchor_loop_closure() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let mut buffer = FrameAnchorBuffer::new(4, 2);

    let landmark_pos = [10.0f32, 2.0f32, -5.0f32];
    let landmark_orient = [0.0f32, 0.0f32, 0.0f32, 1.0f32]; // Identity quaternion
    let landmark_latent = Tensor::ones((1, 8, 32), DType::F32, &device)?;

    buffer.register_spatial_anchor(landmark_pos, landmark_orient, landmark_latent);

    // Query 1: Close coordinates with similar orientation -> should hit and return weighted context
    let query_pos_close = [10.2f32, 2.1f32, -5.1f32];
    let ctx_hit = buffer.get_anchor_context_spatial(&device, 1, 8, 32, DType::F32, Some(query_pos_close), Some(landmark_orient))?;
    assert_eq!(ctx_hit.dims(), &[1, 8, 32]);

    // Query 2: Far coordinates -> should not hit, returns dummy zero context
    let query_pos_far = [100.0f32, 20.0f32, -50.0f32];
    let ctx_miss = buffer.get_anchor_context_spatial(&device, 1, 8, 32, DType::F32, Some(query_pos_far), Some(landmark_orient))?;
    assert_eq!(ctx_miss.dims(), &[1, 1, 32]);

    Ok(())
}

#[test]
fn test_box_and_sphere_masks() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let num_tokens = 64; // 8x8

    let box_mask = ActionConditionedSampler::create_box_mask(num_tokens, 0.25, 0.25, 0.75, 0.75, &device)?;
    assert_eq!(box_mask.dims(), &[1, 1, num_tokens, 1]);

    let sphere_mask = ActionConditionedSampler::create_sphere_mask(num_tokens, 0.5, 0.5, 0.3, &device)?;
    assert_eq!(sphere_mask.dims(), &[1, 1, num_tokens, 1]);

    let mask_vec = sphere_mask.flatten_all()?.to_vec1::<f32>()?;
    let active_count: f32 = mask_vec.iter().sum();
    assert!(active_count > 0.0 && active_count < (num_tokens as f32));

    Ok(())
}

#[test]
fn test_action_confidence_entropy_diagnostics() -> candle_core::Result<()> {
    // Test pure categorical distribution
    let uniform_logits = vec![1.0, 1.0, 1.0, 1.0];
    let (conf, entropy) = ActionConditioningEmbedder::compute_confidence_entropy(&uniform_logits);
    assert!((conf - 0.25).abs() < 1e-4);
    assert!((entropy - 1.0).abs() < 1e-3);

    let confident_logits = vec![10.0, 0.0, 0.0, 0.0];
    let (c2, e2) = ActionConditioningEmbedder::compute_confidence_entropy(&confident_logits);
    assert!(c2 > 0.99);
    assert!(e2 < 0.05);

    let mut rolling = RollingLatentContext::new(8);
    let (rc, re) = rolling.compute_action_confidence_entropy();
    assert!(rc > 0.5 && re < 0.5);

    let dummy_frame = Tensor::zeros((1, 4, 16), DType::F32, &Device::Cpu)?;
    rolling.push_frame(dummy_frame)?;
    let (rc2, re2) = rolling.compute_action_confidence_entropy();
    assert!(rc2 > 0.0 && re2 > 0.0);
    Ok(())
}

#[test]
fn test_spatial_latent_inpainting() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let dit = ActionConditionedDiT::default_cpu()?;
    let sampler = ActionConditionedSampler::new(dit, 2);

    let mut anchor_buffer = FrameAnchorBuffer::new(4, 1);
    let s = 64;
    let c = 256;
    let source_latent = Tensor::zeros((1, 1, s, c), DType::F32, &device)?;
    // Mask: 1 for left half tokens (regenerate), 0 for right half tokens (keep source)
    let half_len = (s * c) / 2;
    let mask_data = vec![1.0f32; half_len].into_iter().chain(vec![0.0f32; half_len].into_iter()).collect::<Vec<f32>>();
    let spatial_mask = Tensor::from_vec(mask_data, (1, 1, s, c), &device)?;

    let infilled = sampler.sample_next_frame_inpainting(
        1,
        0,
        &source_latent,
        &spatial_mask,
        None,
        &mut anchor_buffer,
        0,
        1.0,
        &device,
    )?;

    assert_eq!(infilled.dims(), &[1, 1, s, c]);
    Ok(())
}

#[test]
fn test_latent_checkpoint_and_rewind() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let mut context = RollingLatentContext::new(5);

    for i in 0..4 {
        let f = Tensor::full(i as f32, (1, 4, 16), &device)?;
        context.push_frame(f)?;
    }
    assert_eq!(context.len(), 4);

    // Test checkpoint export
    let ckpt = context.to_checkpoint()?;
    assert_eq!(ckpt.frames.len(), 4);
    assert_eq!(ckpt.frame_shape, vec![1, 4, 16]);

    // Test rewind
    let rewound = context.rewind(2);
    assert_eq!(rewound, 2);
    assert_eq!(context.len(), 2);

    // Test restore from checkpoint
    let restored = RollingLatentContext::from_checkpoint(&ckpt, 5, &device)?;
    assert_eq!(restored.len(), 4);

    // Test timeline branching
    let branch_initial = Tensor::full(99.0f32, (1, 4, 16), &device)?;
    let branch = restored.branch_timeline(Some(branch_initial))?;
    assert_eq!(branch.len(), 5);

    Ok(())
}

#[test]
fn test_hierarchical_captioner() {
    use reality_genie::dataset::HierarchicalCaptioner;

    let actions = HierarchicalCaptioner::intent_to_actions("leap over the ditch", 6);
    assert_eq!(actions.len(), 6);
    assert!(actions.contains(&5)); // Jump

    let (global_intent, fine_grained) = HierarchicalCaptioner::actions_to_hierarchical_caption(&actions);
    assert!(global_intent.contains("vertical obstacle navigation"));
    assert!(fine_grained.contains("Jump"));

    let tokens = HierarchicalCaptioner::tokenize_intent("explore", 4);
    assert_eq!(tokens.len(), 4);
}

#[test]
fn test_solidity_collision_query() {
    let decoded_points = vec![
        [0.0f32, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ];

    // Colliding entity
    let (hit, pen) = LatentVoxelDecoder::check_solidity_collision(&decoded_points, [0.1, 0.0, 0.0], 0.5);
    assert!(hit);
    assert!(pen > 0.0);

    // Non-colliding entity
    let (hit_far, _pen_far) = LatentVoxelDecoder::check_solidity_collision(&decoded_points, [10.0, 10.0, 10.0], 0.5);
    assert!(!hit_far);
}

#[test]
fn test_trajectory_dataset_safetensors_roundtrip() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let dataset = TrajectoryDataset::generate_synthetic(2, 3, 4, 16, 4, &device)?;
    let tmp_path = "/tmp/test_trajectory_dataset.safetensors";

    dataset.save_to_safetensors(tmp_path)?;
    assert!(std::path::Path::new(tmp_path).exists());

    let loaded = TrajectoryDataset::load_from_safetensors(tmp_path, &device)?;
    assert_eq!(loaded.len(), 2);

    let (orig_l, orig_a) = dataset.get_batch(&[0])?;
    let (load_l, load_a) = loaded.get_batch(&[0])?;

    assert_eq!(orig_l.dims(), load_l.dims());
    assert_eq!(orig_a.dims(), load_a.dims());

    std::fs::remove_file(tmp_path).ok();
    Ok(())
}

#[test]
fn test_rollout_multimodal() -> candle_core::Result<()> {
    let actions = vec![1u32, 5u32];
    let prompt = "jump over crystal spikes";
    let latents = ActionConditionedSampler::rollout_multimodal(&actions, Some(prompt), None, 1, 1.5)?;

    assert_eq!(latents.len(), 2 * 64 * 256);
    Ok(())
}

#[test]
fn test_trajectory_director_drift_stabilization() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let director = TrajectoryDirector::new(0.01, 0.2);

    let mut anchor_buffer = FrameAnchorBuffer::new(4, 2);
    // Push two identical frames -> drift variance = 0.0
    let f1 = Tensor::zeros((1, 8, 16), DType::F32, &device)?;
    let f2 = Tensor::zeros((1, 8, 16), DType::F32, &device)?;
    anchor_buffer.push_frame(f1, 0)?;
    anchor_buffer.push_frame(f2, 2)?;

    let candidate = Tensor::ones((1, 8, 16), DType::F32, &device)?;
    let stabilized = director.direct_frame(&candidate, &anchor_buffer)?;

    // Drift was 0 <= 0.01, so not drifted, unchanged
    assert_eq!(stabilized.dims(), &[1, 8, 16]);
    let val = stabilized.flatten_all()?.to_vec1::<f32>()?[0];
    assert!((val - 1.0).abs() < 1e-4);

    // Now test with high drift threshold trigger
    let director_strict = TrajectoryDirector::new(0.0000001, 0.5);
    let mut high_drift_buffer = FrameAnchorBuffer::new(4, 1);
    let frame_a = Tensor::zeros((1, 8, 16), DType::F32, &device)?;
    let frame_b = Tensor::full(10.0f32, (1, 8, 16), &device)?;
    high_drift_buffer.push_frame(frame_a, 0)?;
    high_drift_buffer.push_frame(frame_b, 1)?;

    let stabilized_drift = director_strict.direct_frame(&candidate, &high_drift_buffer)?;
    assert_eq!(stabilized_drift.dims(), &[1, 8, 16]);
    // primary anchor is frame_a (zeros), candidate is ones, blended with alpha 0.5 -> 0.5
    let val_drift = stabilized_drift.flatten_all()?.to_vec1::<f32>()?[0];
    assert!((val_drift - 0.5).abs() < 1e-4);

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
    let _dit = ActionConditionedDiT::default_cpu()?;
    let actions = vec![1u32, 2u32, 3u32];
    let latents_vec = ActionConditionedSampler::rollout_from_actions_vec(&actions)?;

    assert_eq!(latents_vec.len(), 3 * 64 * 256);

    Ok(())
}
