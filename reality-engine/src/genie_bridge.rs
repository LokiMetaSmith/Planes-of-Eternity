use crate::voxel::{Chunk, Voxel, CHUNK_SIZE};
use crate::splat::SplatVertex;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct GenieBridge {
    pub pending_splats: Arc<Mutex<Vec<Vec<SplatVertex>>>>,
    pub pending_voxels: Arc<Mutex<Vec<Chunk>>>,
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
        Self {
            pending_splats: Arc::new(Mutex::new(Vec::new())),
            pending_voxels: Arc::new(Mutex::new(Vec::new())),
        }
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

                use wasm_bindgen::JsCast;
                use wasm_bindgen::prelude::Closure;
                use std::cell::RefCell;
                use std::collections::HashMap;
                use std::sync::atomic::{AtomicUsize, Ordering};


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
                                            let cb_fn: &js_sys::Function = cb.as_ref().unchecked_ref();
                                            let _ = cb_fn.call1(&wasm_bindgen::JsValue::NULL, &data);
                                        }
                                    });
                                }
                            }
                        }) as Box<dyn FnMut(_)>);
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
                        let success = js_sys::Reflect::get(&data, &"success".into()).unwrap().as_bool().unwrap_or(false);

                        if success {
                            let splats_json = js_sys::Reflect::get(&data, &"splats_json".into()).unwrap().as_string().unwrap_or_else(|| "[]".into());
                            let _ = resolve_clone.call1(&wasm_bindgen::JsValue::NULL, &splats_json.into());
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
                            if let Ok(splats) = serde_json::from_str::<Vec<SplatVertex>>(&splats_json) {
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
            use reality_genie::splat_gen::{SplatGenerator, DummySplatGenerator};
            let gen = DummySplatGenerator::new();
            let raw_splats = gen.generate_splats_from_prompt(&prompt);

            let mut direct_splats = Vec::new();
            for s in raw_splats {
                direct_splats.push(SplatVertex {
                    position: [s[0], s[1], s[2]],
                    rotation: [s[3], s[4], s[5], s[6]],
                    scale: [s[7], s[8], s[9]],
                    color: [s[10], s[11], s[12], s[13]],
                    previous_position: [s[0], s[1], s[2]],
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
