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

#[test]
fn test_episodic_memory_graph_recall() {
    let mut memory_graph = reality_genie::dit::EpisodicMemoryGraph::new();
    let latents1 = vec![0.9f32, 0.1f32, 0.0f32, 0.0f32];
    let latents2 = vec![0.0f32, 0.0f32, 0.8f32, 0.2f32];

    let id1 = memory_graph.record_episode("ancient_ruin", [10.0, 0.0, -15.0], &latents1, 1000);
    let _id2 = memory_graph.record_episode("cyber_tower", [12.0, 0.0, -18.0], &latents2, 2000);

    let recalled = memory_graph.recall_episodes(&[0.85, 0.1, 0.0, 0.0], 1);
    assert_eq!(recalled.len(), 1);
    assert_eq!(recalled[0].id, id1);
    assert_eq!(recalled[0].semantic_tag, "ancient_ruin");
}

#[test]
fn test_neural_gi_evaluation() {
    let gi = reality_genie::dit::NeuralGiField::new(8);
    let latents = vec![0.1f32; 16];
    let noon_sample = gi.evaluate_gi([0.0, 1.0, 0.0], 0.25, &latents); // 0.25 = sunrise/day
    assert!(noon_sample.irradiance_rgb[0] > 0.1);
    assert!(noon_sample.ambient_occlusion > 0.0);
    assert!(noon_sample.directional_shadow_factor > 0.0);

    let midnight_sample = gi.evaluate_gi([0.0, 1.0, 0.0], 0.75, &latents); // 0.75 = night
    assert!(midnight_sample.irradiance_rgb[0] < noon_sample.irradiance_rgb[0]);
}

#[test]
fn test_supergenie_upscaling() {
    let upscaler = reality_genie::dit::SuperGenieUpscaler::new(4, 4, 4);
    let coarse_latents = vec![0.2f32; 16]; // 4 tokens with 4 channels
    let upscaled = upscaler.upscale_latents(&coarse_latents, 4);

    assert_eq!(upscaled.len(), 4 * 4 * 4); // 16 tokens with 4 channels = 64 floats
}

#[test]
fn test_neural_fluid_particle_simulation() {
    let mut fluid_sim = reality_genie::dit::NeuralFluidSimulator::new(8);
    let latents = vec![0.5f32; 16];
    fluid_sim.diffuse_velocity_field(&latents, 0.016);
    fluid_sim.emit_particles([0.0, 2.0, 0.0], 10);
    assert_eq!(fluid_sim.particles.len(), 10);

    fluid_sim.step_advection(0.016, 0.0);
    assert_eq!(fluid_sim.particles.len(), 10);
    assert!(fluid_sim.particles[0].position[1] >= 0.0);
}

#[test]
fn test_inverse_dynamics_prediction() {
    let predictor = reality_genie::dit::InverseDynamicsPredictor::new(16);
    let latents = vec![0.1f32; 16];
    let current_pos = [0.0, 0.0, 0.0];
    let target_pos = [0.0, 0.0, -5.0];

    let steps = predictor.infer_action_sequence(&latents, current_pos, target_pos, 10);
    assert!(!steps.is_empty());
    // First action moving towards -Z should be Forward (1)
    assert_eq!(steps[0].discrete_action, 1);
    assert!(steps[0].confidence > 0.5);
}

#[test]
fn test_texture_dit_pbr_synthesis() {
    let synth = reality_genie::dit::TextureDitSynthesizer::new(16);
    let latents = vec![0.3f32; 16];
    let mat = synth.synthesize_material(&latents, 1);

    assert!(mat.roughness >= 0.08 && mat.roughness <= 0.98);
    assert!(mat.metallic >= 0.0 && mat.metallic <= 1.0);
    assert!(mat.normal[2] > 0.0);
    assert!(mat.albedo[0] > 0.0 && mat.albedo[1] > 0.0);
}

#[test]
fn test_world_model_gym_environment() {
    let mut env = reality_genie::dit::WorldModelGymEnv::new(16, 50);
    let obs = env.reset();
    assert_eq!(obs.latent_tokens.len(), 16);
    assert_eq!(obs.step_count, 0);

    let step_res = env.step(1); // Move forward towards target
    assert_eq!(step_res.observation.step_count, 1);
    assert!(step_res.reward > -1.0);
    assert!(!step_res.truncated);
}

#[test]
fn test_world_model_mcts_planning() {
    let mut mcts = reality_genie::dit::WorldModelMcts::new(4, 20);
    mcts.goal_objective = reality_genie::dit::MctsGoalObjective::TargetSeek;
    mcts.target_coords = [0.0, 0.0, -10.0];

    let root_latents = vec![0.1f32; 64];
    let root_pos = [0.0, 0.0, 0.0];
    let candidate_paths = mcts.plan_paths(&root_latents, root_pos);

    assert!(!candidate_paths.is_empty());
    assert!(candidate_paths[0].waypoint_positions.len() <= 4);
    assert!(candidate_paths.iter().any(|p| p.is_optimal));
}

#[test]
fn test_audio_dit_acoustic_co_diffusion() {
    let diffuser = reality_genie::dit::AudioDitCoDiffuser::new(16, 64);
    let visual_latents = vec![0.5f32; 64];
    let acoustic_latents = diffuser.diffuse_acoustic_latent(&visual_latents, 1, 0.4);
    assert_eq!(acoustic_latents.len(), 16);

    let packet = diffuser.decode_to_pcm(&acoustic_latents, 1, 0.4, 1024);
    assert_eq!(packet.left_channel.len(), 1024);
    assert_eq!(packet.right_channel.len(), 1024);
    assert_eq!(packet.sample_rate, 44100);
    assert!(packet.reverb_tail_factor > 0.0);
    assert!(packet.dominant_freq_hz > 50.0);
}

#[test]
fn test_pyramid_resolution_mode_and_sfx_genie() {
    use reality_genie::dit::{ActionConditionedDiT, ActionConditionedSampler, DiTConfig, FrameAnchorBuffer, ResolutionPyramidMode, SfxGenie};
    use candle_core::Device;

    let device = Device::Cpu;
        let config = DiTConfig {
        num_actions: 16,
        action_dim: 16,
        d_model: 32,
        n_head: 4,
        num_layers: 2,
        num_spatial_tokens: 64,
        max_temporal_frames: 8,
        max_anchor_frames: 4,
        vocab_size_text: 16,
        dropout: 0.0,
    };
    let varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let model = ActionConditionedDiT::new(config, vb).unwrap();
    let sampler = ActionConditionedSampler::new(model, 1);
    let mut anchor_buf = FrameAnchorBuffer::new(4, 2);

    // Coarse 16
    let out_coarse = sampler.sample_next_frame_pyramid(
        1, 1, None, None, &mut anchor_buf, 0, 1.0, ResolutionPyramidMode::Coarse16, &device
    ).unwrap();
    assert_eq!(out_coarse.dims(), &[1, 1, 64, 32]);

    // Full 64
    let out_full = sampler.sample_next_frame_pyramid(
        1, 1, None, None, &mut anchor_buf, 0, 1.0, ResolutionPyramidMode::Full64, &device
    ).unwrap();
    assert_eq!(out_full.dims(), &[1, 1, 64, 32]);

    // SfxGenie parametric synthesis
    let sfx = SfxGenie::new(44100);
    let footstep = sfx.synthesize_action_sound(1, 4.0, 0.1, 0.5);
    assert!(!footstep.is_empty());
    assert_eq!(footstep.len(), (44100.0 * 0.1) as usize);

    let jump_sound = sfx.synthesize_action_sound(5, 0.0, 0.2, 0.2);
    assert_eq!(jump_sound.len(), (44100.0 * 0.2) as usize);
}

#[test]
fn test_atmospheric_conditioning_and_tinygenie_quantization() {
    use reality_genie::dit::{AtmosphericConditioningEmbedder, QuantizedLinearInt8, SfxGenie};
    use candle_core::{Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

    let atmos_embedder = AtmosphericConditioningEmbedder::new(32, vb.pp("atmos")).unwrap();
    // c_atmos = [sin(2pi * t), cos(2pi * t), weather_density]
    let atmos_input = Tensor::from_slice(&[0.0f32, 1.0f32, 0.5f32], (1, 3), &device).unwrap();
    let emb = atmos_embedder.forward(&atmos_input).unwrap();
    assert_eq!(emb.dims(), &[1, 32]);

    // Test TinyGenie QuantizedLinearInt8
    let weights = vec![0.5f32, -0.25f32, 1.0f32, -0.75f32, 0.1f32, -0.9f32]; // [2 out, 3 in]
    let qlinear = QuantizedLinearInt8::from_f32_weights(&weights, 2, 3, Some(vec![0.1f32, -0.1f32]));
    assert_eq!(qlinear.qweights.len(), 6);
    assert_eq!(qlinear.scales.len(), 2);

    let input_vec = vec![1.0f32, 0.5f32, -0.5f32];
    let mut output_vec = vec![0.0f32; 2];
    qlinear.forward_dequantized(&input_vec, &mut output_vec);
    assert!(output_vec[0].abs() > 0.0);
    assert!(output_vec[1].abs() > 0.0);

    // Test SfxGenie with atmospheric conditions
    let sfx = SfxGenie::new(44100);
    let rainy_step = sfx.synthesize_action_sound_atmos(1, 4.0, 0.1, 0.5, 0.25, 0.8);
    assert_eq!(rainy_step.len(), (44100.0 * 0.1) as usize);
}

#[test]
fn test_lora_adapter_and_presets() {
    use reality_genie::dit::LoraAdapter;

    let adapter = LoraAdapter::preset_style("Cyberpunk", 64, 64, 4);
    assert_eq!(adapter.name, "Cyberpunk");
    assert_eq!(adapter.rank, 4);
    assert_eq!(adapter.in_features, 64);
    assert_eq!(adapter.out_features, 64);

    let input = vec![1.0f32; 64];
    let mut delta = vec![0.0f32; 64];
    adapter.forward_delta(&input, &mut delta);

    // Delta should be non-zero and bounded
    let sum_abs: f32 = delta.iter().map(|v| v.abs()).sum();
    assert!(sum_abs > 0.001);

    // Test serialization roundtrip
    let json_str = serde_json::to_string(&adapter).unwrap();
    let loaded: LoraAdapter = serde_json::from_str(&json_str).unwrap();
    assert_eq!(loaded.name, "Cyberpunk");
}

#[test]
fn test_multi_agent_co_diffusion_swarm() {
    use reality_genie::dit::{MultiAgentCoDiffusion, SwarmAgentState};

    let mut swarm = MultiAgentCoDiffusion::new(64, 8);
    let agents = vec![
        SwarmAgentState {
            id: 1,
            position: [0.0, 0.0, 0.0],
            velocity: [1.0, 0.0, 0.0],
            intent_action: 1,
            influence_radius: 10.0,
            emotional_valence: 0.8,
            dialogue_intent: 1,
            last_speech_utterance: None,
        },
        SwarmAgentState {
            id: 2,
            position: [2.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 1.0],
            intent_action: 1,
            influence_radius: 10.0,
            emotional_valence: -0.5,
            dialogue_intent: 2,
            last_speech_utterance: None,
        },
    ];
    swarm.set_agents(agents);

    let attn = swarm.compute_swarm_attention_matrix();
    assert_eq!(attn.len(), 2);
    assert_eq!(attn[0].len(), 2);
    // Attentions should sum to ~1.0
    let row0_sum: f32 = attn[0].iter().sum();
    assert!((row0_sum - 1.0).abs() < 1e-4);

    let (drift, cohesion) = swarm.step_swarm(0.05);
    assert!(cohesion > 0.0 && cohesion <= 1.0);
    assert!(drift >= 0.0);
}

#[test]
fn test_fracture_dit_destruction_simulation() {
    use reality_genie::dit::FractureDitSolver;

    let mut solver = FractureDitSolver::new(2.0, 0.5);
    let latents = vec![0.5f32; 32];

    // Trigger below threshold -> 0 shards
    solver.trigger_fracture([0.0, 1.0, 0.0], [0.5, 0.0, 0.0], &latents, 16);
    assert_eq!(solver.shards.len(), 0);

    // Trigger above threshold -> 16 shards
    solver.trigger_fracture([0.0, 1.0, 0.0], [3.0, 2.0, 1.0], &latents, 16);
    assert_eq!(solver.shards.len(), 16);

    // Step physics with floor collision at y = 0.0
    solver.step(0.1, 0.0);
    assert_eq!(solver.shards.len(), 16);
    assert!(solver.shards[0].position[1] >= 0.0);
}

#[test]
fn test_visual_odometry_estimation() {
    use reality_genie::dit::VisualOdometryEstimator;

    let estimator = VisualOdometryEstimator::new(64, 1.0);
    let prev = vec![0.1f32; 64];
    let mut curr = vec![0.1f32; 64];

    // Identical latents -> near 0 delta
    let delta_static = estimator.estimate_camera_delta(&prev, &curr);
    assert_eq!(delta_static.translation_delta, [0.0, 0.0, 0.0]);
    assert!(delta_static.confidence > 0.9);

    // Shift latents (motion)
    for i in 0..64 {
        curr[i] += 0.5;
    }
    let delta_motion = estimator.estimate_camera_delta(&prev, &curr);
    assert!(delta_motion.translation_delta[0] != 0.0);
    assert!(delta_motion.confidence < delta_static.confidence);
}

#[test]
fn test_latent_moe_routing() {
    use reality_genie::dit::{LatentMoeBlock, ExpertDomain};

    let moe = LatentMoeBlock::default();
    let tokens = vec![0.5f32; 16];

    // Terrain and Physics have high logits
    let gate_logits = vec![5.0, 4.0, 1.0, 0.0];
    let (output, selections) = moe.forward_moe(&tokens, &gate_logits);

    assert_eq!(output.len(), 16);
    assert_eq!(selections.len(), 2);
    assert_eq!(selections[0].0, ExpertDomain::Terrain);
    assert_eq!(selections[1].0, ExpertDomain::Physics);
    assert!(selections[0].1 > selections[1].1);
}

#[test]
fn test_agentic_companion_cocreation_and_challenges() {
    use reality_genie::dit::AgenticCompanion;

    let mut companion = AgenticCompanion::new("Samantha", [0.0, 1.0, 0.0]);
    assert_eq!(companion.personality.name, "Samantha");
    assert!(companion.current_dialogue.is_some());

    // Update presence floating near player at [10.0, 2.0, 5.0]
    companion.update_presence([10.0, 2.0, 5.0], 0.1);
    assert!(companion.position[0] > 0.0);

    // Companion proposes co-creation
    let creation = companion.propose_co_creation([10.0, 2.0, 5.0], 1);
    assert!(!creation.structure_name.is_empty());
    assert_eq!(companion.past_co_creations.len(), 1);

    // Companion generates dynamic challenge
    let challenge = companion.generate_dynamic_challenge([10.0, 2.0, 5.0]);
    assert_eq!(companion.active_challenges.len(), 1);
    assert!(!challenge.completed);

    // Player far away -> not completed
    let completed_none = companion.check_challenge_completion([0.0, 0.0, 0.0], 2.0);
    assert!(completed_none.is_none());

    // Player moves to target coordinates -> completed!
    let completed_some = companion.check_challenge_completion(challenge.target_coords, 2.0);
    assert!(completed_some.is_some());
    assert!(companion.active_challenges[0].completed);
    assert_eq!(companion.personality.current_mood, "Celebratory");
}

#[test]
fn test_personalized_companion_generation_and_conversation() {
    use reality_genie::dit::{AgenticCompanion, UserProfileConfig, DynamicNpcProfile};

    let profile = UserProfileConfig {
        companion_name: "Aria".to_string(),
        personality_archetype: "Playful Challenger".to_string(),
        tone_prompt: Some("Witty, provocative, sharp".to_string()),
        curiosity: 0.95,
        playfulness: 0.90,
        empathy: 0.70,
        analytical_depth: 0.65,
        speech_voice_style: "Crisp".to_string(),
        visual_hue_shift: 0.4,
    };

    let mut companion = AgenticCompanion::generate_personalized(profile, [0.0, 2.0, 0.0]);
    assert_eq!(companion.personality.name, "Aria");
    assert_eq!(companion.personality.personality_archetype, "Playful Challenger");
    assert_eq!(companion.personality.current_mood, "Playful Challenger");
    assert!(companion.current_dialogue.as_ref().unwrap().contains("Aria"));

    // Visual splat generation with personalized hue shift
    let splats = companion.generate_avatar_splats(1.0);
    assert!(!splats.is_empty());
    assert_eq!(splats.len() % 14, 0);

    // Converse with user and test conversation memory
    let reply1 = companion.respond_to_user("Hello Aria, who are you?", 1000);
    assert!(reply1.contains("Aria"));
    assert_eq!(companion.conversation_memory.len(), 2); // 1 User turn, 1 Aria turn
    assert_eq!(companion.personality.conversational_turns_count, 1);

    // Positive sentiment interaction
    let reply2 = companion.respond_to_user("I love exploring this awesome world with you!", 2000);
    assert!(!reply2.is_empty());
    assert_eq!(companion.conversation_memory.len(), 4);
    assert_eq!(companion.conversation_memory[2].sentiment_score > 0.0, true);

    // Procedural Dynamic NPC generalization
    let mut npc = DynamicNpcProfile::new_procedural(101, "Caspian", "Void-Weaver", "Fantasy");
    assert_eq!(npc.name, "Caspian");
    assert_eq!(npc.personality.current_mood, "Adventurous");

    let npc_reply = npc.converse("What mysteries lie beyond this ridge?", 3000);
    assert!(npc_reply.contains("Caspian"));
    assert_eq!(npc.memory_log.len(), 2);
}
