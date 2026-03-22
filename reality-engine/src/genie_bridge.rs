use crate::voxel::{Chunk, Voxel, CHUNK_SIZE};

// Placeholder for logic that would use ML crates when not targeting WASM or when features are enabled
// For now, we stub this out to make the build pass without complex dependency management in this environment.

#[derive(Clone)]
pub struct GenieBridge {
    // diffusion: Option<std::sync::Arc<DiscreteDiffusion>>, // Disabled for build fix
}

impl std::fmt::Debug for GenieBridge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenieBridge")
         .finish()
    }
}

impl Default for GenieBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl GenieBridge {
    pub fn new() -> Self {
        Self {}
    }

    // Simulates an ML "Dream" step
    pub fn dream_chunk(&self, chunk: &mut Chunk) {
        // Legacy "Dream" logic (Heuristic)
        let size = CHUNK_SIZE;
        let mut changes = Vec::new();

        for z in 1..size-1 {
            for y in 1..size-1 {
                for x in 1..size-1 {
                    let idx = chunk.index(x, y, z);
                    let id = chunk.data[idx].id;

                    if id == 3 { // Fire -> Obsidian
                        if (x + y + z) % 2 == 0 {
                            changes.push((idx, 1));
                        }
                    }
                    else if id == 0 { // Air -> Bridge
                        let below = chunk.data[chunk.index(x, y-1, z)].id;
                        if below == 1 && (x+z)%5 == 0 {
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

    /// Dynamically generates a 3D voxel model based on a text prompt
    /// using the ML logic in the reality-genie crate.
    pub fn generate_voxel_model(&self, prompt: &str) -> Chunk {
        // Since we are returning a Chunk without knowing its world key, we can use a dummy key
        let mut chunk = Chunk::new(crate::voxel::ChunkKey { x: 0, y: 0, z: 0 });

        let generator = reality_genie::sparc::SparseVoxelGenerator::new();
        let sparse_voxels = generator.generate_from_prompt(prompt);

        for (pos, voxel_id) in sparse_voxels {
            let x = pos[0];
            let y = pos[1];
            let z = pos[2];

            // Validate bounds to prevent out-of-bounds access on Chunk
            if x >= 0 && x < CHUNK_SIZE as i32 &&
               y >= 0 && y < CHUNK_SIZE as i32 &&
               z >= 0 && z < CHUNK_SIZE as i32 {
                let idx = chunk.index(x as usize, y as usize, z as usize);
                chunk.data[idx] = Voxel { id: voxel_id };
            }
        }

        chunk
    }
}

use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NpcStateView {
    pub uuid: String,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub archetype: String,
    // Provide some context about nearby entities (like player)
    pub player_distance: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NpcAction {
    pub target_x: Option<f32>,
    pub target_y: Option<f32>,
    pub target_z: Option<f32>,
    pub chat_message: Option<String>,
}
