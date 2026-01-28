use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::projector::RealityProjector;
use sha2::{Sha256, Digest};
use std::fmt::Write;

#[derive(Serialize, Deserialize, Debug, Clone, Hash, Eq, PartialEq, Copy)]
pub struct ChunkId {
    pub x: i32,
    pub z: i32,
}

impl ChunkId {
    pub fn from_world_pos(x: f32, z: f32, chunk_size: f32) -> Self {
        let chunk_x = (x / chunk_size).floor() as i32;
        let chunk_z = (z / chunk_size).floor() as i32;
        Self { x: chunk_x, z: chunk_z }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Chunk {
    pub id: ChunkId,
    pub anomalies: Vec<RealityProjector>,
    pub hash: String,
}

impl Chunk {
    pub fn new(id: ChunkId) -> Self {
        Self {
            id,
            anomalies: Vec::new(),
            hash: String::new(),
        }
    }

    pub fn calculate_hash(&mut self) -> String {
        let mut hasher = Sha256::new();
        // Serialize anomalies to JSON to get a consistent byte representation
        let json = serde_json::to_string(&self.anomalies).unwrap_or_default();
        hasher.update(json);
        let result = hasher.finalize();
        let hash_string = hex::encode(result);
        self.hash = hash_string.clone();
        hash_string
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WorldState {
    pub chunks: HashMap<ChunkId, Chunk>,
    pub root_hash: String,
}

impl Default for WorldState {
    fn default() -> Self {
        Self {
            chunks: HashMap::new(),
            root_hash: String::new(),
        }
    }
}

impl WorldState {
    pub fn get_or_create_chunk(&mut self, id: ChunkId) -> &mut Chunk {
        self.chunks.entry(id).or_insert_with(|| Chunk::new(id))
    }

    pub fn add_anomaly(&mut self, projector: RealityProjector) {
        let chunk_size = 10.0; // Match the grid size from lib.rs
        let id = ChunkId::from_world_pos(projector.location.x, projector.location.z, chunk_size);
        let chunk = self.get_or_create_chunk(id);

        chunk.anomalies.push(projector);
        chunk.calculate_hash();
        self.calculate_root_hash();
    }

    pub fn calculate_root_hash(&mut self) {
        let mut hasher = Sha256::new();
        // To ensure deterministic order, we must sort keys
        let mut keys: Vec<&ChunkId> = self.chunks.keys().collect();
        keys.sort_by(|a, b| {
            if a.z != b.z {
                a.z.cmp(&b.z)
            } else {
                a.x.cmp(&b.x)
            }
        });

        for key in keys {
            if let Some(chunk) = self.chunks.get(key) {
                hasher.update(&chunk.hash);
            }
        }
        let result = hasher.finalize();
        self.root_hash = hex::encode(result);
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WorldCommit {
    pub parent_hash: String,
    pub new_hash: String,
    pub delta: Vec<RealityProjector>, // For now, just a list of changed projectors
    pub author: String,
    pub timestamp: u64,
}
