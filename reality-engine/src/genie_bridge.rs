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
        }
    }

    /// Branches current rolling context into a named timeline
    pub fn branch_timeline(&self, name: &str) -> bool {
        if let (Ok(ctx), Ok(mut map)) = (self.rolling_context.lock(), self.timelines.lock()) {
            map.insert(name.to_string(), ctx.clone());
            true
        } else {
            false
        }
    }

    /// Switches the active rolling context to a previously branched timeline
    pub fn switch_timeline(&self, name: &str) -> bool {
        if let (Ok(mut ctx), Ok(map)) = (self.rolling_context.lock(), self.timelines.lock()) {
            if let Some(saved) = map.get(name) {
                *ctx = saved.clone();
                return true;
            }
        }
        false
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
        let latents_ref = Arc::clone(&self.latest_action_dit_latents);

        #[cfg(not(target_arch = "wasm32"))]
        {
            std::thread::spawn(move || {
                if let Ok(flat) = reality_genie::dit::ActionConditionedSampler::rollout_multimodal(
                    &actions,
                    prompt.as_deref(),
                    Some("action_dit_model.safetensors"),
                    num_denoising_steps,
                    guidance_scale,
                ) {
                    if let Ok(mut lock) = latents_ref.lock() {
                        *lock = Some(flat);
                    }
                }
            });
        }

        #[cfg(target_arch = "wasm32")]
        {
            if let Ok(flat) = reality_genie::dit::ActionConditionedSampler::rollout_multimodal(
                &actions,
                prompt.as_deref(),
                None,
                num_denoising_steps,
                guidance_scale,
            ) {
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
