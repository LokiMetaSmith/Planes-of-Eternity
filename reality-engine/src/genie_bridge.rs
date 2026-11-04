use crate::splat::SplatVertex;
use crate::voxel::{Chunk, Voxel, CHUNK_SIZE};
use std::sync::{Arc, Mutex};

use std::sync::mpsc;
use crate::voxel::ChunkKey;

pub struct TerrainRequest {
    pub chunk_key: ChunkKey,
    pub size: usize,
}

pub struct TerrainResponse {
    pub chunk_key: ChunkKey,
    pub heightmap: Vec<f32>,
}


#[derive(Clone)]
pub struct GenieBridge {
    pub pending_splats: Arc<Mutex<Vec<Vec<SplatVertex>>>>,
    pub pending_voxels: Arc<Mutex<Vec<Chunk>>>,
    pub terrain_sender: mpsc::Sender<TerrainRequest>,
    pub terrain_res_tx: mpsc::Sender<TerrainResponse>,
    pub terrain_receiver: Arc<Mutex<mpsc::Receiver<TerrainResponse>>>,
    pub latest_action_dit_latents: Arc<Mutex<Option<Vec<f32>>>>,
    pub rolling_context: Arc<Mutex<reality_genie::dit::RollingLatentContext>>,
    pub anchor_buffer: Arc<Mutex<reality_genie::dit::FrameAnchorBuffer>>,
    pub timelines: Arc<Mutex<std::collections::HashMap<String, reality_genie::dit::RollingLatentContext>>>,
    pub timeline_parents: Arc<Mutex<std::collections::HashMap<String, String>>>,
    pub active_timeline: Arc<Mutex<String>>,
    pub time_echo: Arc<Mutex<TimeEchoReplay>>,
    pub narrative_director: Arc<Mutex<NarrativeDirector>>,
    pub resolution_mode: Arc<Mutex<reality_genie::dit::ResolutionPyramidMode>>,
    pub sfx_genie: Arc<Mutex<reality_genie::dit::SfxGenie>>,
    pub p2p_consensus: Arc<Mutex<P2PTrajectoryConsensus>>,
    pub latent_physics: Arc<Mutex<LatentPhysicsEngine>>,
    pub active_lora: Arc<Mutex<Option<reality_genie::dit::LoraAdapter>>>,
    pub swarm_engine: Arc<Mutex<reality_genie::dit::MultiAgentCoDiffusion>>,
    pub audio_dit: Arc<Mutex<reality_genie::dit::AudioDitCoDiffuser>>,
    pub latest_audio_packet: Arc<Mutex<Option<reality_genie::dit::AudioPacket>>>,
    pub mcts_planner: Arc<Mutex<reality_genie::dit::WorldModelMcts>>,
    pub mcts_candidate_paths: Arc<Mutex<Vec<reality_genie::dit::CandidateTrajectoryPath>>>,
    pub rollback_netcode: Arc<Mutex<LatentRollbackNetcode>>,
    pub webrtc_streamer: Arc<Mutex<WebRtcLatentChunkStreamer>>,
    pub holographic_manager: Arc<Mutex<HolographicTeleportationManager>>,
}

impl std::fmt::Debug for GenieBridge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenieBridge").finish()
    }
}

impl Default for GenieBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl GenieBridge {
    pub fn new() -> Self {
        #[allow(unused_variables)]
        let (req_tx, req_rx) = mpsc::channel::<TerrainRequest>();
        let (res_tx, res_rx) = mpsc::channel::<TerrainResponse>();
        let res_tx_clone = res_tx.clone();

        #[cfg(not(target_arch = "wasm32"))]
        {
            std::thread::spawn(move || {
                let generator = reality_genie::continuous_diffusion::ContinuousDiffusionGenerator::new();
                while let Ok(req) = req_rx.recv() {
                    let heightmap = generator.generate_heightmap(req.chunk_key.x, req.chunk_key.z, req.size);
                    let _ = res_tx.send(TerrainResponse {
                        chunk_key: req.chunk_key,
                        heightmap,
                    });
                }
            });
        }

        Self {
            pending_splats: Arc::new(Mutex::new(Vec::new())),
            pending_voxels: Arc::new(Mutex::new(Vec::new())),
            terrain_sender: req_tx,
            terrain_res_tx: res_tx_clone,
            terrain_receiver: Arc::new(Mutex::new(res_rx)),
            latest_action_dit_latents: Arc::new(Mutex::new(None)),
            rolling_context: Arc::new(Mutex::new(reality_genie::dit::RollingLatentContext::new(8))),
            anchor_buffer: Arc::new(Mutex::new(reality_genie::dit::FrameAnchorBuffer::new(6, 2))),
            timelines: Arc::new(Mutex::new(std::collections::HashMap::new())),
            timeline_parents: Arc::new(Mutex::new(std::collections::HashMap::new())),
            active_timeline: Arc::new(Mutex::new("main".to_string())),
            time_echo: Arc::new(Mutex::new(TimeEchoReplay::new(120))),
            narrative_director: Arc::new(Mutex::new(NarrativeDirector::default())),
            resolution_mode: Arc::new(Mutex::new(reality_genie::dit::ResolutionPyramidMode::Full64)),
            sfx_genie: Arc::new(Mutex::new(reality_genie::dit::SfxGenie::new(44100))),
            p2p_consensus: Arc::new(Mutex::new(P2PTrajectoryConsensus::default())),
            latent_physics: Arc::new(Mutex::new(LatentPhysicsEngine::default())),
            active_lora: Arc::new(Mutex::new(None)),
            swarm_engine: Arc::new(Mutex::new(reality_genie::dit::MultiAgentCoDiffusion::new(64, 16))),
            audio_dit: Arc::new(Mutex::new(reality_genie::dit::AudioDitCoDiffuser::new(16, 64))),
            latest_audio_packet: Arc::new(Mutex::new(None)),
            mcts_planner: Arc::new(Mutex::new(reality_genie::dit::WorldModelMcts::new(5, 30))),
            mcts_candidate_paths: Arc::new(Mutex::new(Vec::new())),
            rollback_netcode: Arc::new(Mutex::new(LatentRollbackNetcode::new(128))),
            webrtc_streamer: Arc::new(Mutex::new(WebRtcLatentChunkStreamer::new())),
            holographic_manager: Arc::new(Mutex::new(HolographicTeleportationManager::new())),
        }
    }

    /// Branches current rolling context into a named timeline
    pub fn branch_timeline(&self, name: &str) -> bool {
        if let (Ok(ctx), Ok(mut map), Ok(mut parents), Ok(active)) = (
            self.rolling_context.lock(),
            self.timelines.lock(),
            self.timeline_parents.lock(),
            self.active_timeline.lock(),
        ) {
            map.insert(name.to_string(), ctx.clone());
            parents.insert(name.to_string(), active.clone());
            true
        } else {
            false
        }
    }

    /// Switches the active rolling context to a previously branched timeline
    pub fn switch_timeline(&self, name: &str) -> bool {
        if let (Ok(mut ctx), Ok(map), Ok(mut active)) = (
            self.rolling_context.lock(),
            self.timelines.lock(),
            self.active_timeline.lock(),
        ) {
            if let Some(saved) = map.get(name) {
                *ctx = saved.clone();
                *active = name.to_string();
                return true;
            } else if name == "main" {
                *active = "main".to_string();
                return true;
            }
        }
        false
    }

    /// Serializes entire multiverse timeline hierarchy into JSON for the Canvas Tree Visualizer
    pub fn get_multiverse_tree_json(&self) -> String {
        let active_name = self.active_timeline.lock().map(|a| a.clone()).unwrap_or_else(|_| "main".to_string());
        let current_drift = self.anchor_buffer.lock().ok().and_then(|a| a.compute_drift_variance().ok()).unwrap_or(0.0);
        let current_depth = self.rolling_context.lock().map(|c| c.len()).unwrap_or(0);

        let mut nodes = Vec::new();

        // Root/main node
        nodes.push(serde_json::json!({
            "name": "main",
            "parent": null,
            "depth": current_depth,
            "drift": current_drift,
            "active": active_name == "main",
        }));

        if let (Ok(timelines), Ok(parents)) = (self.timelines.lock(), self.timeline_parents.lock()) {
            for (name, ctx) in timelines.iter() {
                let parent = parents.get(name).map(|s| s.as_str()).unwrap_or("main");
                nodes.push(serde_json::json!({
                    "name": name,
                    "parent": parent,
                    "depth": ctx.len(),
                    "drift": current_drift * 1.1, // Normalized drift estimate
                    "active": active_name == *name,
                }));
            }
        }

        serde_json::to_string(&nodes).unwrap_or_else(|_| "[]".to_string())
    }

    /// Registers a spatial landmark keyframe at (pos, orient) for loop-closure consistency
    pub fn register_spatial_landmark(&self, pos: [f32; 3], orient: [f32; 4], flat_latents: &[f32]) {
        if let Ok(mut anchors) = self.anchor_buffer.lock() {
            anchors.register_spatial_anchor_raw(pos, orient, flat_latents);
        }
    }

    /// Retrieve action confidence and entropy diagnostics (confidence, entropy)
    pub fn get_action_confidence_metric(&self) -> (f32, f32) {
        if let Ok(ctx) = self.rolling_context.lock() {
            ctx.compute_action_confidence_entropy()
        } else {
            (0.85, 0.15)
        }
    }

    /// Returns a list of all active timeline names
    pub fn list_timelines(&self) -> Vec<String> {
        if let Ok(map) = self.timelines.lock() {
            map.keys().cloned().collect()
        } else {
            Vec::new()
        }
    }

    /// Rewinds world model latent memory by `steps` frames
    pub fn rewind_world_model(&self, steps: usize) -> usize {
        let mut rewound = 0;
        if let Ok(mut ctx) = self.rolling_context.lock() {
            rewound = ctx.rewind(steps);
        }
        if let Ok(mut anchors) = self.anchor_buffer.lock() {
            anchors.rewind(steps);
        }
        rewound
    }

    /// Creates a serializable checkpoint snapshot of current rolling latents
    pub fn checkpoint_world_model(&self) -> Option<reality_genie::dit::LatentCheckpoint> {
        if let Ok(ctx) = self.rolling_context.lock() {
            ctx.to_checkpoint().ok()
        } else {
            None
        }
    }

    /// Restores rolling latents from a saved checkpoint
    pub fn restore_world_model(&self, ckpt: &reality_genie::dit::LatentCheckpoint) -> bool {
        if let Ok(new_ctx) = reality_genie::dit::RollingLatentContext::from_checkpoint_cpu(ckpt, 8) {
            if let Ok(mut lock) = self.rolling_context.lock() {
                *lock = new_ctx;
                return true;
            }
        }
        false
    }

    pub fn request_action_dit_rollout(&self, actions: Vec<u32>, num_denoising_steps: usize) {
        self.request_action_dit_rollout_cfg(actions, num_denoising_steps, 1.0);
    }

    pub fn request_action_dit_rollout_cfg(&self, actions: Vec<u32>, num_denoising_steps: usize, guidance_scale: f64) {
        self.request_action_dit_rollout_multimodal(actions, None, num_denoising_steps, guidance_scale);
    }

    pub fn request_action_dit_rollout_multimodal(
        &self,
        actions: Vec<u32>,
        prompt: Option<String>,
        num_denoising_steps: usize,
        guidance_scale: f64,
    ) {
        self.request_action_dit_rollout_full(actions, None, prompt, num_denoising_steps, guidance_scale);
    }

    /// Full rollout request supporting continuous 5D vectors [d_theta_x, d_theta_y, vx, vy, vz]
    pub fn request_action_dit_rollout_full(
        &self,
        actions: Vec<u32>,
        continuous_actions: Option<Vec<[f32; 5]>>,
        prompt: Option<String>,
        num_denoising_steps: usize,
        guidance_scale: f64,
    ) {
        let latents_ref = Arc::clone(&self.latest_action_dit_latents);
        let audio_dit_ref = Arc::clone(&self.audio_dit);
        let audio_packet_ref = Arc::clone(&self.latest_audio_packet);
        let _actions_clone = actions.clone();

        #[cfg(not(target_arch = "wasm32"))]
        {
            let actions_clone = _actions_clone;
            std::thread::spawn(move || {
                if let Ok(flat) = reality_genie::dit::ActionConditionedSampler::rollout_multimodal_full(
                    &actions_clone,
                    continuous_actions.as_deref(),
                    prompt.as_deref(),
                    Some("action_dit_model.safetensors"),
                    num_denoising_steps,
                    guidance_scale,
                ) {
                    // Diffuse synchronized audio packet
                    let last_act = actions_clone.last().copied().unwrap_or(0);
                    if let Ok(audio_model) = audio_dit_ref.lock() {
                        let acoustic_latents = audio_model.diffuse_acoustic_latent(&flat, last_act, 0.5);
                        let packet = audio_model.decode_to_pcm(&acoustic_latents, last_act, 0.5, 4096);
                        if let Ok(mut ap_lock) = audio_packet_ref.lock() {
                            *ap_lock = Some(packet);
                        }
                    }

                    if let Ok(mut lock) = latents_ref.lock() {
                        *lock = Some(flat);
                    }
                }
            });
        }

        #[cfg(target_arch = "wasm32")]
        {
            if let Ok(flat) = reality_genie::dit::ActionConditionedSampler::rollout_multimodal_full(
                &actions,
                continuous_actions.as_deref(),
                prompt.as_deref(),
                None,
                num_denoising_steps,
                guidance_scale,
            ) {
                let last_act = actions.last().copied().unwrap_or(0);
                if let Ok(audio_model) = audio_dit_ref.lock() {
                    let acoustic_latents = audio_model.diffuse_acoustic_latent(&flat, last_act, 0.5);
                    let packet = audio_model.decode_to_pcm(&acoustic_latents, last_act, 0.5, 4096);
                    if let Ok(mut ap_lock) = audio_packet_ref.lock() {
                        *ap_lock = Some(packet);
                    }
                }

                if let Ok(mut lock) = latents_ref.lock() {
                    *lock = Some(flat);
                }
            }
        }
    }

    pub fn request_terrain(&self, chunk_x: i32, chunk_y: i32, chunk_z: i32, size: usize) {
        let key = ChunkKey { x: chunk_x, y: chunk_y, z: chunk_z };

        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = self.terrain_sender.send(TerrainRequest { chunk_key: key, size });
        }

        #[cfg(target_arch = "wasm32")]
        {
            // Simple immediate execution fallback for WASM since std::thread is not supported
            let generator = reality_genie::continuous_diffusion::ContinuousDiffusionGenerator::new();
            let heightmap = generator.generate_heightmap(chunk_x, chunk_z, size);
            let _ = self.terrain_res_tx.send(TerrainResponse {
                chunk_key: key,
                heightmap,
            });
        }
    }

    pub fn request_iterative_extension(
        &self,
        chunk_x: i32,
        chunk_y: i32,
        chunk_z: i32,
        size: usize,
        trajectory: Vec<[f32; 3]>,
        neighbors: Vec<(i32, i32, Vec<f32>)>,
    ) {
        let key = ChunkKey { x: chunk_x, y: chunk_y, z: chunk_z };

        #[cfg(not(target_arch = "wasm32"))]
        {
            let res_tx = self.terrain_res_tx.clone();
            std::thread::spawn(move || {
                let mut model = reality_genie::gct::SceneExtensionModel::new(size);

                for pos in trajectory {
                    model.trajectory_memory.record(pos, [0.0, 0.0, 0.0], 0.0);
                }

                let mut anchor = reality_genie::gct::AnchorContext::new(size);
                for (nx, nz, h_map) in neighbors {
                    anchor.add_neighbor(nx, nz, h_map);
                }

                let heightmap = model.generate_cohesive_heightmap(chunk_x, chunk_z, &anchor);
                let _ = res_tx.send(TerrainResponse {
                    chunk_key: key,
                    heightmap,
                });
            });
        }

        #[cfg(target_arch = "wasm32")]
        {
            let mut model = reality_genie::gct::SceneExtensionModel::new(size);

            for pos in trajectory {
                model.trajectory_memory.record(pos, [0.0, 0.0, 0.0], 0.0);
            }

            let mut anchor = reality_genie::gct::AnchorContext::new(size);
            for (nx, nz, h_map) in neighbors {
                anchor.add_neighbor(nx, nz, h_map);
            }

            let heightmap = model.generate_cohesive_heightmap(chunk_x, chunk_z, &anchor);
            let _ = self.terrain_res_tx.send(TerrainResponse {
                chunk_key: key,
                heightmap,
            });
        }
    }

    pub fn poll_terrain(&self) -> Option<TerrainResponse> {
        if let Ok(rx) = self.terrain_receiver.lock() {
            if let Ok(res) = rx.try_recv() {
                return Some(res);
            }
        }
        None
    }

    pub fn dream_chunk(&self, chunk: &mut Chunk) {
        let size = CHUNK_SIZE;
        let mut changes = Vec::new();

        for z in 1..size - 1 {
            for y in 1..size - 1 {
                for x in 1..size - 1 {
                    let idx = chunk.index(x, y, z);
                    let id = chunk.data[idx].id;

                    if id == 3 {
                        if (x + y + z) % 2 == 0 {
                            changes.push((idx, 1));
                        }
                    } else if id == 0 {
                        let below = chunk.data[chunk.index(x, y - 1, z)].id;
                        if below == 1 && (x + z) % 5 == 0 {
                            changes.push((idx, 1));
                        }
                    }
                }
            }
        }

        for (idx, new_id) in changes {
            chunk.data[idx] = Voxel { id: new_id };
        }
    }

    pub fn diffuse_chunk(&self, _chunk: &mut Chunk) {
        log::warn!("Diffusion model disabled in this build.");
    }

    pub fn generate_voxel_model(&self, prompt: &str) -> Chunk {
        let mut chunk = Chunk::new(crate::voxel::ChunkKey { x: 0, y: 0, z: 0 });

        let generator = reality_genie::sparc::SparseVoxelGenerator::new();
        let sparse_voxels = generator.generate_from_prompt(prompt);

        for (pos, voxel_id) in sparse_voxels {
            let x = pos[0];
            let y = pos[1];
            let z = pos[2];

            if x >= 0
                && x < CHUNK_SIZE as i32
                && y >= 0
                && y < CHUNK_SIZE as i32
                && z >= 0
                && z < CHUNK_SIZE as i32
            {
                let idx = chunk.index(x as usize, y as usize, z as usize);
                chunk.data[idx] = Voxel { id: voxel_id };
            }
        }

        chunk
    }

    pub fn request_model_generation(&self, prompt: String, position: [f32; 3]) {
        let mut chunk = self.generate_voxel_model(&prompt);
        // We modify the chunk key to place it near the position
        chunk.key = crate::voxel::ChunkKey {
            x: (position[0] / crate::voxel::CHUNK_SIZE as f32).floor() as i32,
            y: (position[1] / crate::voxel::CHUNK_SIZE as f32).floor() as i32,
            z: (position[2] / crate::voxel::CHUNK_SIZE as f32).floor() as i32,
        };

        if let Ok(mut pending) = self.pending_voxels.lock() {
            pending.push(chunk);
        }
    }

    pub fn request_splat_generation(&self, prompt: String) {
        let pending_ref = Arc::clone(&self.pending_splats);

        #[cfg(target_arch = "wasm32")]
        {
            let prompt_clone = prompt.clone();
            wasm_bindgen_futures::spawn_local(async move {
                log::info!("Spawning Web Worker for Splat Generation: {}", prompt_clone);

                use std::cell::RefCell;
                use std::collections::HashMap;
                use std::sync::atomic::{AtomicUsize, Ordering};
                use wasm_bindgen::prelude::Closure;
                use wasm_bindgen::JsCast;

                // Lazy-load a global worker and a map of callbacks
                thread_local! {
                    static GLOBAL_WORKER: RefCell<Option<web_sys::Worker>> = RefCell::new(None);
                    static CALLBACKS: RefCell<HashMap<usize, Closure<dyn FnMut(wasm_bindgen::JsValue)>>> = RefCell::new(HashMap::new());
                }

                static TASK_ID_COUNTER: AtomicUsize = AtomicUsize::new(1);
                let task_id = TASK_ID_COUNTER.fetch_add(1, Ordering::SeqCst);

                let worker = GLOBAL_WORKER.with(|global| {
                    let mut b = global.borrow_mut();
                    if b.is_none() {
                        let options = web_sys::WorkerOptions::new();
                        options.set_type(web_sys::WorkerType::Module);
                        let w = web_sys::Worker::new_with_options("worker.js", &options).unwrap();

                        // Set a single, permanent message handler
                        let onmessage = Closure::wrap(Box::new(|event: web_sys::MessageEvent| {
                            let data = event.data();
                            if let Ok(id_val) = js_sys::Reflect::get(&data, &"task_id".into()) {
                                if let Some(id) = id_val.as_f64().map(|f| f as usize) {
                                    CALLBACKS.with(|callbacks| {
                                        if let Some(cb) = callbacks.borrow_mut().remove(&id) {
                                            let cb_fn: &js_sys::Function =
                                                cb.as_ref().unchecked_ref();
                                            let _ =
                                                cb_fn.call1(&wasm_bindgen::JsValue::NULL, &data);
                                        }
                                    });
                                }
                            }
                        })
                            as Box<dyn FnMut(_)>);
                        w.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
                        onmessage.forget(); // Leak once globally

                        *b = Some(w);
                    }
                    b.clone().unwrap()
                });

                // Prepare message object
                let msg = js_sys::Object::new();
                js_sys::Reflect::set(&msg, &"prompt".into(), &prompt_clone.into()).unwrap();
                js_sys::Reflect::set(&msg, &"task_id".into(), &(task_id as f64).into()).unwrap();

                let promise = js_sys::Promise::new(&mut |resolve, reject| {
                    let resolve_clone = resolve.clone();
                    let reject_clone = reject.clone();

                    let callback = Closure::wrap(Box::new(move |data: wasm_bindgen::JsValue| {
                        let success = js_sys::Reflect::get(&data, &"success".into())
                            .unwrap()
                            .as_bool()
                            .unwrap_or(false);

                        if success {
                            let splats_json = js_sys::Reflect::get(&data, &"splats_json".into())
                                .unwrap()
                                .as_string()
                                .unwrap_or_else(|| "[]".into());
                            let _ = resolve_clone
                                .call1(&wasm_bindgen::JsValue::NULL, &splats_json.into());
                        } else {
                            let error_msg = js_sys::Reflect::get(&data, &"error".into()).unwrap();
                            let _ = reject_clone.call1(&wasm_bindgen::JsValue::NULL, &error_msg);
                        }
                    }) as Box<dyn FnMut(_)>);

                    CALLBACKS.with(|callbacks| {
                        callbacks.borrow_mut().insert(task_id, callback);
                    });
                });

                // Send the message to start work
                worker.post_message(&msg).unwrap();

                // Await the worker's result
                match wasm_bindgen_futures::JsFuture::from(promise).await {
                    Ok(splats_json_val) => {
                        if let Some(splats_json) = splats_json_val.as_string() {
                            if let Ok(splats) =
                                serde_json::from_str::<Vec<SplatVertex>>(&splats_json)
                            {
                                if let Ok(mut pending) = pending_ref.lock() {
                                    pending.push(splats);
                                    log::info!("Web Worker Splat Generation complete and applied.");
                                }
                            }
                        }
                    }
                    Err(e) => {
                        log::error!("Web Worker failed: {:?}", e);
                    }
                }
            });
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            use reality_genie::splat_gen::{GenieSplatGenerator, SplatGenerator};
            let gen = GenieSplatGenerator::new();
            let raw_splats = gen.generate_splats_from_prompt(&prompt);

            let mut direct_splats = Vec::new();
            for s in raw_splats {
                direct_splats.push(SplatVertex {
                    position: [s[0], s[1], s[2]],
                    rotation: [s[3], s[4], s[5], s[6]],
                    scale: [s[7], s[8], s[9]],
                    color: [s[10], s[11], s[12], s[13]],
                    previous_position: [s[0], s[1], s[2]],
                    archetype_id: 5, // Genie archetype default
                    target_archetype_id: 5,
                    morph_weight: 0.0,
    bone_index: -1,
                });
            }

            if let Ok(mut pending) = pending_ref.clone().lock() {
                pending.push(direct_splats);
            }
        }
    }
}

pub struct LoreJournalBridge {
    pub episodic_graph: reality_genie::dit::EpisodicMemoryGraph,
}

impl Default for LoreJournalBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl LoreJournalBridge {
    pub fn new() -> Self {
        let mut graph = reality_genie::dit::EpisodicMemoryGraph::new();
        graph.add_journal_entry(
            "Genesis in the Lattice",
            "We stepped across the latent boundary together. The air felt charged with possibility.",
            "Adventurous",
            [0.0, 1.0, 0.0],
            1700000000,
        );
        Self { episodic_graph: graph }
    }

    pub fn record_entry(&mut self, title: &str, reflection: &str, mood: &str, pos: [f32; 3], time_ms: u64) -> u64 {
        self.episodic_graph.add_journal_entry(title, reflection, mood, pos, time_ms)
    }

    pub fn get_journal_json(&self) -> String {
        serde_json::to_string(&self.episodic_graph.journal_entries).unwrap_or_else(|_| "[]".to_string())
    }
}

impl EngineCompanionManager {
    /// Dispatches companion interaction with a specified NPC ID
    pub fn companion_talk_with_npc(&mut self, npc_id: u32, npc_name: &str, archetype: &str) -> String {
        let interaction = self.companion.interact_with_npc(npc_id, npc_name, archetype);
        serde_json::to_string(&interaction).unwrap_or_else(|_| "{}".to_string())
    }

    /// Spawns a cooperative mini-game puzzle between Samantha and the player
    pub fn spawn_coop_puzzle(&mut self, player_pos: [f32; 3]) -> reality_genie::dit::CompanionCoopPuzzle {
        reality_genie::dit::CompanionCoopPuzzle::new_random(1, player_pos)
    }

    /// Dispatches companion to scout ahead into off-screen frontier
    pub fn scout_frontier(&mut self, player_pos: [f32; 3]) -> String {
        let discovery = self.companion.scout_latent_frontier(player_pos);
        serde_json::to_string(&discovery).unwrap_or_else(|_| "{}".to_string())
    }
}

// ============================================================================
// Agentic Companion Bridge Integration
// ============================================================================

/// Manages runtime companion interactions, party swarms, challenges, and co-creation events
pub struct EngineCompanionManager {
    pub companion: reality_genie::dit::AgenticCompanion,
    pub companion_party: Vec<reality_genie::dit::AgenticCompanion>,
}

impl Default for EngineCompanionManager {
    fn default() -> Self {
        Self {
            companion: reality_genie::dit::AgenticCompanion::default(),
            companion_party: Vec::new(),
        }
    }
}

impl EngineCompanionManager {
    pub fn new(name: &str, initial_pos: [f32; 3]) -> Self {
        let primary = reality_genie::dit::AgenticCompanion::new(name, initial_pos);
        Self {
            companion: primary.clone(),
            companion_party: vec![primary],
        }
    }

    /// Adds a newly personalized or recruited companion into the party swarm
    pub fn recruit_companion(&mut self, companion: reality_genie::dit::AgenticCompanion) {
        self.companion_party.push(companion);
        if self.companion_party.len() > 8 {
            self.companion_party.remove(0);
        }
    }

    /// Switches the primary active companion in the party by name
    pub fn switch_active_companion(&mut self, name: &str) -> bool {
        if let Some(pos) = self.companion_party.iter().position(|c| c.personality.name == name) {
            self.companion = self.companion_party[pos].clone();
            true
        } else {
            false
        }
    }

    pub fn update(&mut self, player_pos: [f32; 3], dt: f32) -> Option<String> {
        self.companion.update_presence(player_pos, dt);

        // Update all party members with spaced orbital angles
        let party_count = self.companion_party.len();
        for (i, member) in self.companion_party.iter_mut().enumerate() {
            let angle = (i as f32 / party_count.max(1) as f32) * std::f32::consts::TAU;
            member.target_offset = [
                angle.cos() * 2.2,
                0.8 + (i as f32 * 0.2),
                angle.sin() * 2.2,
            ];
            member.update_presence(player_pos, dt);
        }

        // Long-Horizon Co-Evolution: Adapt curiosity based on player motion/displacement
        let speed = ((player_pos[0] - self.companion.position[0]).powi(2)
            + (player_pos[2] - self.companion.position[2]).powi(2)).sqrt();
        if speed > 4.0 {
            self.companion.personality.curiosity = (self.companion.personality.curiosity + 0.0005 * dt).min(1.0);
        }

        // Check for challenge completion within 3 meters
        if let Some(challenge) = self.companion.check_challenge_completion(player_pos, 3.0) {
            Some(format!("QUEST COMPLETED: {} (+Resonance)", challenge.title))
        } else {
            None
        }
    }

    /// Triggers social banter between two companions in the party
    pub fn simulate_party_banter(&mut self) -> Option<String> {
        if self.companion_party.len() < 2 {
            return None;
        }
        let speaker_a = self.companion_party[0].personality.name.clone();
        let speaker_b = self.companion_party[1].personality.name.clone();
        let banter = format!("{}: \"Look at the latent resonance here, {}!\" -> {}: \"The voxel density is extraordinary!\"", speaker_a, speaker_b, speaker_b);
        self.companion.current_dialogue = Some(banter.clone());
        Some(banter)
    }

    /// Serializes entire party roster
    pub fn get_party_roster_json(&self) -> String {
        serde_json::to_string(&self.companion_party).unwrap_or_else(|_| "[]".to_string())
    }

    /// Adapts companion traits according to player gameplay activities
    pub fn register_player_activity(&mut self, activity: &str) {
        match activity {
            "voxel_build" => {
                self.companion.personality.playfulness = (self.companion.personality.playfulness + 0.02).min(1.0);
                self.companion.personality.current_mood = "Co-Creative".to_string();
            }
            "spell_cast" => {
                self.companion.personality.analytical_depth = (self.companion.personality.analytical_depth + 0.02).min(1.0);
                self.companion.personality.current_mood = "Contemplative".to_string();
            }
            "high_speed_explore" => {
                self.companion.personality.curiosity = (self.companion.personality.curiosity + 0.02).min(1.0);
                self.companion.personality.current_mood = "Adventurous".to_string();
            }
            _ => {}
        }
    }

    pub fn trigger_co_creation(&mut self, player_pos: [f32; 3], archetype: u32) -> reality_genie::dit::CompanionCoCreation {
        self.companion.propose_co_creation(player_pos, archetype)
    }

    pub fn trigger_dynamic_challenge(&mut self, player_pos: [f32; 3]) -> reality_genie::dit::CompanionChallenge {
        self.companion.generate_dynamic_challenge(player_pos)
    }

    pub fn get_companion_state_json(&self) -> String {
        serde_json::to_string(&self.companion).unwrap_or_else(|_| "{}".to_string())
    }

    /// Customizes or generates a personalized companion based on a user profile
    pub fn customize_personality(&mut self, profile: reality_genie::dit::UserProfileConfig, initial_pos: [f32; 3]) {
        self.companion = reality_genie::dit::AgenticCompanion::generate_personalized(profile, initial_pos);
    }

    /// Converses with the companion, recording memories and generating an adaptive in-character response
    pub fn chat_interact(&mut self, user_message: &str, timestamp_ms: u64) -> String {
        self.companion.respond_to_user(user_message, timestamp_ms)
    }

    /// Retrieves serialized conversation history
    pub fn get_conversation_history_json(&self) -> String {
        serde_json::to_string(&self.companion.conversation_memory).unwrap_or_else(|_| "[]".to_string())
    }
}

// ============================================================================
// Counterfactual Multiverse Branch Simulator
// ============================================================================

/// Outcome of simulating a counterfactual branch from a base world model state
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CounterfactualBranchResult {
    pub branch_id: String,
    pub base_checkpoint_id: String,
    pub applied_actions: Vec<crate::input::Action>,
    pub final_simulated_position: [f32; 3],
    pub final_simulated_latents: Vec<f32>,
    pub cumulative_drift: f32,
}

/// Simulates a counterfactual branch given a base state and alternate action sequences
pub fn simulate_counterfactual_branch(
    branch_id: &str,
    base_checkpoint_id: &str,
    initial_pos: [f32; 3],
    initial_latents: &[f32],
    candidate_actions: &[crate::input::Action],
) -> CounterfactualBranchResult {
    let mut pos = initial_pos;
    let mut latents = initial_latents.to_vec();
    let mut total_drift = 0.0f32;

    for &act in candidate_actions {
        let (dx, dy, dz) = match act {
            crate::input::Action::MoveForward => (0.0, 0.0, -1.0),
            crate::input::Action::MoveBackward => (0.0, 0.0, 1.0),
            crate::input::Action::MoveLeft => (-1.0, 0.0, 0.0),
            crate::input::Action::MoveRight => (1.0, 0.0, 0.0),
            crate::input::Action::Jump => (0.0, 1.2, 0.0),
            _ => (0.0, 0.0, 0.0),
        };

        pos[0] += dx * 0.5;
        pos[1] += dy * 0.5;
        pos[2] += dz * 0.5;

        // Perturb latents according to simulated dynamics
        for i in 0..latents.len() {
            let shift = (dx + dy + dz) * 0.05 * ((i as f32).sin());
            latents[i] += shift;
            total_drift += shift.abs();
        }
    }

    CounterfactualBranchResult {
        branch_id: branch_id.to_string(),
        base_checkpoint_id: base_checkpoint_id.to_string(),
        applied_actions: candidate_actions.to_vec(),
        final_simulated_position: pos,
        final_simulated_latents: latents,
        cumulative_drift: total_drift,
    }
}

// ============================================================================
// Cross-Reality Holographic Teleportation (AR/VR Neural Avatars)
// ============================================================================

/// Holographic avatar packet containing live player splats and pose matrices for WebXR teleportation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HolographicAvatarPacket {
    pub peer_id: String,
    pub head_pose: [f32; 16],
    pub left_hand_pose: Option<[f32; 16]>,
    pub right_hand_pose: Option<[f32; 16]>,
    pub avatar_splats: Vec<[f32; 14]>,
    pub timestamp_ms: u64,
}

/// Holographic avatar teleportation manager for cross-reality WebXR AR/VR sessions
#[derive(Debug, Clone)]
pub struct HolographicTeleportationManager {
    pub remote_avatars: std::collections::HashMap<String, HolographicAvatarPacket>,
    pub local_head_pose: [f32; 16],
}

impl Default for HolographicTeleportationManager {
    fn default() -> Self {
        Self::new()
    }
}

impl HolographicTeleportationManager {
    pub fn new() -> Self {
        let mut id_matrix = [0.0f32; 16];
        id_matrix[0] = 1.0;
        id_matrix[5] = 1.0;
        id_matrix[10] = 1.0;
        id_matrix[15] = 1.0;

        Self {
            remote_avatars: std::collections::HashMap::new(),
            local_head_pose: id_matrix,
        }
    }

    /// Ingests a holographic avatar packet received from a remote AR/VR peer
    pub fn ingest_avatar_packet(&mut self, packet: HolographicAvatarPacket) {
        self.remote_avatars.insert(packet.peer_id.clone(), packet);
    }

    /// Emits local player avatar splat vertices for streaming
    pub fn encode_local_avatar(
        &self,
        peer_id: &str,
        player_pos: [f32; 3],
        timestamp_ms: u64,
    ) -> HolographicAvatarPacket {
        // Generate a concise 16-splat avatar representation
        let mut avatar_splats = Vec::with_capacity(16);
        for i in 0..16 {
            let theta = (i as f32) * std::f32::consts::PI * 0.125;
            let sx = player_pos[0] + theta.cos() * 0.4;
            let sy = player_pos[1] + (i as f32 * 0.08);
            let sz = player_pos[2] + theta.sin() * 0.4;
            avatar_splats.push([
                sx, sy, sz,
                0.0, 0.0, 0.0, 1.0,
                0.1, 0.1, 0.1,
                0.0, 0.9, 1.0, 0.8,
            ]);
        }

        HolographicAvatarPacket {
            peer_id: peer_id.to_string(),
            head_pose: self.local_head_pose,
            left_hand_pose: None,
            right_hand_pose: None,
            avatar_splats,
            timestamp_ms,
        }
    }
}

// ============================================================================
// Decentralized Latent World Chunk Streaming (P2P WebRTC Wire Protocol)
// ============================================================================

/// Compressed latent chunk packet transmitted across WebRTC DataChannels
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WebRtcLatentChunkDelta {
    pub chunk_key: [i32; 3],
    pub version: u32,
    pub quantized_deltas: Vec<i8>,
    pub scale: f32,
    pub timestamp_ms: u64,
}

/// Decentralized P2P WebRTC chunk streamer that compresses, transmits, and reconciles
/// continuous latent world chunks across decentralized browser peers.
#[derive(Debug, Clone)]
pub struct WebRtcLatentChunkStreamer {
    pub connected_peers: Vec<String>,
    pub chunk_cache: std::collections::HashMap<[i32; 3], (u32, Vec<f32>)>,
    pub total_chunks_synchronized: u64,
    pub bytes_transferred: usize,
}

impl Default for WebRtcLatentChunkStreamer {
    fn default() -> Self {
        Self::new()
    }
}

impl WebRtcLatentChunkStreamer {
    pub fn new() -> Self {
        Self {
            connected_peers: Vec::new(),
            chunk_cache: std::collections::HashMap::new(),
            total_chunks_synchronized: 0,
            bytes_transferred: 0,
        }
    }

    /// Compresses a latent float slice into an INT8 delta packet for WebRTC transmission
    pub fn encode_chunk_delta(
        &mut self,
        chunk_key: [i32; 3],
        current_latents: &[f32],
        timestamp_ms: u64,
    ) -> WebRtcLatentChunkDelta {
        let entry = self.chunk_cache.entry(chunk_key).or_insert((0, vec![0.0f32; current_latents.len()]));
        entry.0 += 1;
        let version = entry.0;

        let base = &mut entry.1;
        let n = current_latents.len().min(base.len());
        let mut diffs = Vec::with_capacity(n);
        let mut max_abs_diff = 1e-6f32;

        for i in 0..n {
            let d = current_latents[i] - base[i];
            diffs.push(d);
            if d.abs() > max_abs_diff {
                max_abs_diff = d.abs();
            }
        }

        let scale = max_abs_diff / 127.0;
        let mut quantized_deltas = Vec::with_capacity(n);
        for d in &diffs {
            let q = (d / scale).round().clamp(-128.0, 127.0) as i8;
            quantized_deltas.push(q);
        }

        // Update local cache
        base.copy_from_slice(&current_latents[..n]);
        self.bytes_transferred += quantized_deltas.len();

        WebRtcLatentChunkDelta {
            chunk_key,
            version,
            quantized_deltas,
            scale,
            timestamp_ms,
        }
    }

    /// Ingests and decodes a received WebRTC chunk delta packet from a remote peer
    pub fn decode_and_merge_chunk_delta(&mut self, packet: &WebRtcLatentChunkDelta) -> Vec<f32> {
        let entry = self.chunk_cache.entry(packet.chunk_key).or_insert((0, vec![0.0f32; packet.quantized_deltas.len()]));
        entry.0 = packet.version;

        let base = &mut entry.1;
        let n = packet.quantized_deltas.len().min(base.len());
        for i in 0..n {
            let dq = packet.quantized_deltas[i] as f32 * packet.scale;
            base[i] += dq;
        }

        self.total_chunks_synchronized += 1;
        self.bytes_transferred += packet.quantized_deltas.len();
        base.clone()
    }
}

// ============================================================================
// Multiplayer Latent State Streaming & Rollback Netcode
// ============================================================================

/// Snapshot frame stored in the netcode history ring buffer
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LatentFrameSnapshot {
    pub frame_id: u64,
    pub timestamp: f64,
    pub input_action: u32,
    pub continuous_input: [f32; 5],
    pub latents: Vec<f32>,
    pub position: [f32; 3],
}

/// Latent Rollback Netcode engine managing client-side prediction, jitter simulation,
/// and fast-forward resimulation against authoritative / peer states.
#[derive(Debug, Clone)]
pub struct LatentRollbackNetcode {
    pub history_capacity: usize,
    pub history_buffer: Vec<LatentFrameSnapshot>,
    pub current_frame: u64,
    pub simulated_jitter_ms: f32,
    pub simulated_packet_loss: f32,
    pub total_rollbacks: u64,
    pub last_rollback_frames: u32,
    pub last_drift_reconciled: f32,
}

impl LatentRollbackNetcode {
    pub fn new(capacity: usize) -> Self {
        Self {
            history_capacity: capacity,
            history_buffer: Vec::with_capacity(capacity),
            current_frame: 0,
            simulated_jitter_ms: 0.0,
            simulated_packet_loss: 0.0,
            total_rollbacks: 0,
            last_rollback_frames: 0,
            last_drift_reconciled: 0.0,
        }
    }

    /// Records client-predicted frame into history buffer
    pub fn record_predicted_frame(
        &mut self,
        timestamp: f64,
        action: u32,
        continuous: [f32; 5],
        latents: Vec<f32>,
        pos: [f32; 3],
    ) {
        self.current_frame += 1;
        let snap = LatentFrameSnapshot {
            frame_id: self.current_frame,
            timestamp,
            input_action: action,
            continuous_input: continuous,
            latents,
            position: pos,
        };

        if self.history_buffer.len() >= self.history_capacity {
            self.history_buffer.remove(0);
        }
        self.history_buffer.push(snap);
    }

    /// Reconciles an authoritative / peer latent frame with history.
    /// If drift exceeds threshold, rolls back and fast-forwards through subsequent stored inputs.
    pub fn reconcile_and_rollback(
        &mut self,
        verified_frame_id: u64,
        authoritative_latents: &[f32],
        authoritative_pos: [f32; 3],
        drift_threshold: f32,
    ) -> bool {
        // Find snapshot in history
        let snap_index = match self.history_buffer.iter().position(|s| s.frame_id == verified_frame_id) {
            Some(idx) => idx,
            None => return false,
        };

        let local_latents = &self.history_buffer[snap_index].latents;
        let n = local_latents.len().min(authoritative_latents.len());
        if n == 0 {
            return false;
        }

        let total_diff: f32 = local_latents.iter().zip(authoritative_latents.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let avg_drift = total_diff / (n as f32);

        if avg_drift > drift_threshold {
            // Drift exceeded! Trigger rollback
            self.total_rollbacks += 1;
            let rollback_frames = (self.history_buffer.len() - 1 - snap_index) as u32;
            self.last_rollback_frames = rollback_frames;
            self.last_drift_reconciled = avg_drift;

            // 1. Correct history frame
            self.history_buffer[snap_index].latents = authoritative_latents.to_vec();
            self.history_buffer[snap_index].position = authoritative_pos;

            // 2. Fast-forward resimulate subsequent frames
            let mut rolling_latents = authoritative_latents.to_vec();
            let mut rolling_pos = authoritative_pos;

            for i in (snap_index + 1)..self.history_buffer.len() {
                let act = self.history_buffer[i].input_action;
                // Simple forward Euler integration
                match act {
                    1 => rolling_pos[2] -= 1.0,
                    2 => rolling_pos[2] += 1.0,
                    3 => rolling_pos[0] -= 1.0,
                    4 => rolling_pos[0] += 1.0,
                    5 => rolling_pos[1] += 1.0,
                    _ => {}
                }
                // Modulate latents lightly
                if let Some(first) = rolling_latents.first_mut() {
                    *first += (act as f32) * 0.005;
                }
                self.history_buffer[i].latents = rolling_latents.clone();
                self.history_buffer[i].position = rolling_pos;
            }

            true
        } else {
            false
        }
    }
}

/// Dynamic Rigidbody operating under latent physics fields
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LatentRigidBody {
    pub id: u32,
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub angular_velocity: [f32; 3],
    pub mass: f32,
    pub restitution: f32,
    pub radius: f32,
    pub active: bool,
    pub lifetime: f32,
}

impl LatentRigidBody {
    pub fn new(id: u32, position: [f32; 3], velocity: [f32; 3], mass: f32, restitution: f32, radius: f32) -> Self {
        Self {
            id,
            position,
            velocity,
            angular_velocity: [0.0, 0.0, 0.0],
            mass: mass.max(0.1),
            restitution: restitution.clamp(0.0, 0.95),
            radius: radius.max(0.1),
            active: true,
            lifetime: 0.0,
        }
    }
}

/// Latent Physics Engine that resolves movement, collisions, and bounce impulses via spatial latent token gradients
#[derive(Debug, Clone)]
pub struct LatentPhysicsEngine {
    pub bodies: Vec<LatentRigidBody>,
    pub gravity: [f32; 3],
    pub drag: f32,
    pub latent_force_scale: f32,
    pub next_body_id: u32,
}

impl Default for LatentPhysicsEngine {
    fn default() -> Self {
        Self {
            bodies: Vec::new(),
            gravity: [0.0, -9.81, 0.0],
            drag: 0.985,
            latent_force_scale: 4.5,
            next_body_id: 1,
        }
    }
}

impl LatentPhysicsEngine {
    pub fn spawn_body(&mut self, pos: [f32; 3], vel: [f32; 3], mass: f32, restitution: f32, radius: f32) -> u32 {
        let id = self.next_body_id;
        self.next_body_id += 1;
        self.bodies.push(LatentRigidBody::new(id, pos, vel, mass, restitution, radius));
        id
    }

    /// Sample spatial gradient from the current latent field tokens [S, C]
    pub fn sample_latent_gradient(latents: &[f32], pos: [f32; 3]) -> [f32; 3] {
        if latents.is_empty() {
            return [0.0, 0.0, 0.0];
        }
        let seq_len = latents.len() / 64; // assuming channel dim 64
        if seq_len < 4 {
            return [0.0, 0.0, 0.0];
        }
        // Map 3D coordinate to spatial token grid indices
        let gx = ((pos[0] * 0.25).floor() as isize).rem_euclid(seq_len as isize) as usize;
        let gy = ((pos[1] * 0.25).floor() as isize).rem_euclid(seq_len as isize) as usize;
        let gz = ((pos[2] * 0.25).floor() as isize).rem_euclid(seq_len as isize) as usize;

        let val_c = latents[(gx * 64) % latents.len()];
        let val_x = latents[((gx + 1) % seq_len * 64) % latents.len()];
        let val_y = latents[((gy + 1) % seq_len * 64) % latents.len()];
        let val_z = latents[((gz + 1) % seq_len * 64) % latents.len()];

        [
            (val_x - val_c) * 2.0,
            (val_y - val_c) * 2.0,
            (val_z - val_c) * 2.0,
        ]
    }

    /// Step the simulation dt seconds forward using latent gradients and ground boundaries
    pub fn step(&mut self, dt: f32, latents: &[f32]) {
        let dt_clamped = dt.clamp(0.001, 0.1);

        for body in self.bodies.iter_mut().filter(|b| b.active) {
            body.lifetime += dt_clamped;
            if body.lifetime > 60.0 {
                body.active = false;
                continue;
            }

            // Latent gradient force
            let grad = Self::sample_latent_gradient(latents, body.position);
            let f_latent = [
                grad[0] * self.latent_force_scale,
                grad[1] * self.latent_force_scale,
                grad[2] * self.latent_force_scale,
            ];

            // Acceleration = gravity + (f_latent / mass)
            let ax = self.gravity[0] + f_latent[0] / body.mass;
            let ay = self.gravity[1] + f_latent[1] / body.mass;
            let az = self.gravity[2] + f_latent[2] / body.mass;

            // Velocity integration with drag
            body.velocity[0] = (body.velocity[0] + ax * dt_clamped) * self.drag;
            body.velocity[1] = (body.velocity[1] + ay * dt_clamped) * self.drag;
            body.velocity[2] = (body.velocity[2] + az * dt_clamped) * self.drag;

            // Position integration
            body.position[0] += body.velocity[0] * dt_clamped;
            body.position[1] += body.velocity[1] * dt_clamped;
            body.position[2] += body.velocity[2] * dt_clamped;

            // Ground plane bounce collision (Y = 0 or latent floor threshold)
            let floor_y = body.radius;
            if body.position[1] < floor_y {
                body.position[1] = floor_y;
                if body.velocity[1] < -0.2 {
                    body.velocity[1] = -body.velocity[1] * body.restitution;
                    // Introduce spin from bounce impulse
                    body.angular_velocity[0] += body.velocity[2] * 0.5;
                    body.angular_velocity[2] -= body.velocity[0] * 0.5;
                } else {
                    body.velocity[1] = 0.0;
                }
            }

            // Angular velocity decay
            body.angular_velocity[0] *= 0.95;
            body.angular_velocity[1] *= 0.95;
            body.angular_velocity[2] *= 0.95;
        }

        // Clean up inactive bodies
        self.bodies.retain(|b| b.active);
    }
}

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NpcStateView {
    pub uuid: String,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub archetype: String,
    pub player_distance: f32,
    pub chat_message: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NpcAction {
    pub target_x: Option<f32>,
    pub target_y: Option<f32>,
    pub target_z: Option<f32>,
    pub chat_message: Option<String>,
}

/// Time-Echo Replay item recording actual player trajectory vs predicted world model trajectory
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TimeEchoFrame {
    pub timestamp: f64,
    pub player_position: [f32; 3],
    pub player_orientation: [f32; 4],
    pub predicted_latents: Option<Vec<f32>>,
    pub action: u32,
    pub continuous_action: [f32; 5],
}

/// Time-Echo Replay Buffer for recording, scrubbing, and visualizing neural divergence
#[derive(Debug, Clone, Default)]
pub struct TimeEchoReplay {
    pub frames: Vec<TimeEchoFrame>,
    pub max_capacity: usize,
    pub scrub_index: usize,
    pub is_replaying: bool,
}

impl TimeEchoReplay {
    pub fn new(max_capacity: usize) -> Self {
        Self {
            frames: Vec::with_capacity(max_capacity),
            max_capacity,
            scrub_index: 0,
            is_replaying: false,
        }
    }

    pub fn record(
        &mut self,
        timestamp: f64,
        player_position: [f32; 3],
        player_orientation: [f32; 4],
        predicted_latents: Option<Vec<f32>>,
        action: u32,
        continuous_action: [f32; 5],
    ) {
        if self.frames.len() >= self.max_capacity && self.max_capacity > 0 {
            self.frames.remove(0);
        }
        self.frames.push(TimeEchoFrame {
            timestamp,
            player_position,
            player_orientation,
            predicted_latents,
            action,
            continuous_action,
        });
        self.scrub_index = self.frames.len().saturating_sub(1);
    }

    pub fn get_current_scrub_frame(&self) -> Option<&TimeEchoFrame> {
        self.frames.get(self.scrub_index)
    }

    pub fn set_scrub_percentage(&mut self, pct: f32) {
        if self.frames.is_empty() {
            self.scrub_index = 0;
            return;
        }
        let clamped = pct.clamp(0.0, 1.0);
        self.scrub_index = ((self.frames.len() - 1) as f32 * clamped).round() as usize;
    }
}

/// Hierarchical Narrative World Director for contextual prompt conditioning
#[derive(Debug, Clone)]
pub struct NarrativeDirector {
    pub enabled: bool,
    pub event_interval_seconds: f32,
    pub timer: f32,
    pub last_narrative_prompt: String,
    pub candidate_events: Vec<(&'static str, &'static str)>,
}

impl Default for NarrativeDirector {
    fn default() -> Self {
        Self {
            enabled: false,
            event_interval_seconds: 12.0,
            timer: 0.0,
            last_narrative_prompt: String::new(),
            candidate_events: vec![
                ("Sudden Neon Aurora", "luminescent cyan ripples descending from crystalline ionosphere"),
                ("Gravity Anomaly Inversion", "floating geometric monoliths with inverse gravitational pull"),
                ("Volcanic Ember Storm", "smoldering obsidian spires shedding incandescent embers"),
                ("Cybernetic Fog Surge", "dense volumetric scanline mist illuminating cyber data nodes"),
                ("Temporal Distortion", "crystalline refraction warping spatial geometry into fractal mirrors"),
            ],
        }
    }
}

impl NarrativeDirector {
    pub fn update(&mut self, dt: f32, drift_variance: f32, player_velocity_mag: f32) -> Option<String> {
        if !self.enabled {
            return None;
        }

        self.timer += dt;
        if self.timer >= self.event_interval_seconds || drift_variance > 1.2 {
            self.timer = 0.0;
            // Select event based on conditions
            let idx = if drift_variance > 1.0 {
                4 // Temporal Distortion to absorb drift
            } else if player_velocity_mag > 6.0 {
                3 // Cybernetic Fog Surge on high velocity
            } else {
                (self.timer as usize + (drift_variance * 100.0) as usize) % self.candidate_events.len()
            };

            let (title, prompt) = self.candidate_events[idx % self.candidate_events.len()];
            self.last_narrative_prompt = format!("[{}] {}", title, prompt);
            Some(self.last_narrative_prompt.clone())
        } else {
            None
        }
    }
}

/// Peer Latent Packet for P2P Trajectory Consensus Wire Protocol
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PeerLatentPacket {
    pub frame_timestamp: f64,
    pub peer_id: String,
    pub player_position: [f32; 3],
    pub player_orientation: [f32; 4],
    pub anchor_latents: Vec<f32>,
    pub predicted_latents: Vec<f32>,
}

/// P2P Trajectory Consensus Engine
#[derive(Debug, Clone)]
pub struct P2PTrajectoryConsensus {
    pub alpha: f32,
    pub beta: f32,
    pub connected_peers_count: usize,
    pub last_consensus_drift: f32,
}

impl Default for P2PTrajectoryConsensus {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            beta: 0.05,
            connected_peers_count: 0,
            last_consensus_drift: 0.0,
        }
    }
}

impl P2PTrajectoryConsensus {
    /// Distance-decayed consensus weighting w(d) = alpha / (1.0 + beta * d^2)
    pub fn compute_consensus_weight(&self, local_pos: [f32; 3], remote_pos: [f32; 3]) -> f32 {
        let dx = local_pos[0] - remote_pos[0];
        let dy = local_pos[1] - remote_pos[1];
        let dz = local_pos[2] - remote_pos[2];
        let d_sq = dx * dx + dy * dy + dz * dz;
        (self.alpha / (1.0 + self.beta * d_sq)).clamp(0.0, 0.95)
    }

    /// Fuses remote peer latents with local predicted latents using consensus weighting
    pub fn fuse_latents(
        &mut self,
        local_latents: &mut [f32],
        remote_latents: &[f32],
        local_pos: [f32; 3],
        remote_pos: [f32; 3],
    ) {
        let w = self.compute_consensus_weight(local_pos, remote_pos);
        let n = local_latents.len().min(remote_latents.len());
        let mut total_diff = 0.0f32;

        for i in 0..n {
            let l = local_latents[i];
            let r = remote_latents[i];
            total_diff += (l - r).abs();
            local_latents[i] = (1.0 - w) * l + w * r;
        }

        if n > 0 {
            self.last_consensus_drift = total_diff / (n as f32);
        }
    }
}
