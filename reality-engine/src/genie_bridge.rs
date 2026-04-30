use crate::voxel::{Chunk, Voxel, CHUNK_SIZE};
use crate::splat::SplatVertex;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct GenieBridge {
    pub pending_splats: Arc<Mutex<Vec<Vec<SplatVertex>>>>,
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

    pub fn request_splat_generation(&self, prompt: String) {
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
            });
        }

        if let Ok(mut pending) = self.pending_splats.lock() {
            pending.push(direct_splats);
        }

        let pending_ref = Arc::clone(&self.pending_splats);

        #[cfg(target_arch = "wasm32")]
        {
            let prompt_clone = prompt.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let encoded_prompt = js_sys::encode_uri_component(&prompt_clone);
                let url = format!("http://localhost:8000/generate?prompt={}", String::from(encoded_prompt));

                let opts = web_sys::RequestInit::new();
                opts.set_method("GET");
                opts.set_mode(web_sys::RequestMode::Cors);

                if let Ok(request) = web_sys::Request::new_with_str_and_init(&url, &opts) {
                    if let Some(window) = web_sys::window() {
                        let fetch_promise = window.fetch_with_request(&request);
                        if let Ok(resp_value) = wasm_bindgen_futures::JsFuture::from(fetch_promise).await {
                            let resp: web_sys::Response = resp_value.into();
                            if resp.ok() {
                                if let Ok(json_promise) = resp.json() {
                                    if let Ok(json_val) = wasm_bindgen_futures::JsFuture::from(json_promise).await {
                                        if let Ok(splats) = serde_json::from_str::<Vec<SplatVertex>>(&js_sys::JSON::stringify(&json_val).map(|s| String::from(s)).unwrap_or_else(|_| String::from("[]"))) {
                                            if let Ok(mut pending) = pending_ref.lock() {
                                                pending.push(splats);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            });
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = pending_ref;
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
