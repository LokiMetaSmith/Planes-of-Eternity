use std::collections::{HashMap, HashSet};
use serde::{Serialize, Deserialize, Serializer, Deserializer};
use crate::projector::RealityProjector;
use sha2::{Sha256, Digest};
use std::fmt::Write;

pub const ANOMALY_GRID_SIZE: f32 = 10.0;

#[derive(Debug, Clone, Hash, Eq, PartialEq, Copy)]
pub struct ChunkId {
    pub x: i32,
    pub z: i32,
}

impl Serialize for ChunkId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let s = format!("{}:{}", self.x, self.z);
        serializer.serialize_str(&s)
    }
}

impl<'de> Deserialize<'de> for ChunkId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 2 {
            return Err(serde::de::Error::custom("Invalid ChunkId format"));
        }
        let x = parts[0].parse().map_err(serde::de::Error::custom)?;
        let z = parts[1].parse().map_err(serde::de::Error::custom)?;
        Ok(ChunkId { x, z })
    }
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
    #[serde(default = "default_stability")]
    pub stability: f32, // 1.0 = Stable, 0.0 = Chaotic
}

fn default_stability() -> f32 {
    1.0
}

impl Chunk {
    pub fn new(id: ChunkId) -> Self {
        Self {
            id,
            anomalies: Vec::new(),
            hash: String::new(),
            stability: 1.0,
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

    pub fn merge(&mut self, other: &Chunk) -> bool {
        let mut changed = false;

        for other_anomaly in &other.anomalies {
            // Find if we have an anomaly with same UUID
            if let Some(existing_idx) = self.anomalies.iter().position(|a| a.uuid == other_anomaly.uuid) {
                 // Check timestamp
                 let existing = &mut self.anomalies[existing_idx];
                 if other_anomaly.last_updated > existing.last_updated {
                     *existing = other_anomaly.clone();
                     changed = true;
                 }
            } else {
                // New anomaly (or tombstone)
                self.anomalies.push(other_anomaly.clone());
                changed = true;
            }
        }

        // Merge stability (take the minimum to represent damage/entropy propagation)
        if other.stability < self.stability {
            self.stability = other.stability;
            changed = true;
        }

        if changed {
            self.calculate_hash();
        }

        changed
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WorldState {
    pub chunks: HashMap<ChunkId, Chunk>,
    pub root_hash: String,
    #[serde(skip, default)]
    pub dirty_chunks: HashSet<ChunkId>,
}

impl Default for WorldState {
    fn default() -> Self {
        Self {
            chunks: HashMap::new(),
            root_hash: String::new(),
            dirty_chunks: HashSet::new(),
        }
    }
}

impl WorldState {
    pub fn get_or_create_chunk(&mut self, id: ChunkId) -> &mut Chunk {
        self.chunks.entry(id).or_insert_with(|| Chunk::new(id))
    }

    pub fn add_anomaly(&mut self, projector: RealityProjector) {
        let id = ChunkId::from_world_pos(projector.location.x, projector.location.z, ANOMALY_GRID_SIZE);
        let chunk = self.get_or_create_chunk(id);

        // Apply stability impact based on archetype
        use crate::reality_types::RealityArchetype;
        let stability_cost = match projector.reality_signature.active_style.archetype {
            RealityArchetype::Horror => 0.2,
            RealityArchetype::SciFi => 0.1,
            RealityArchetype::Fantasy => 0.05,
            RealityArchetype::Void => 0.5,
            RealityArchetype::HyperNature => -0.1, // Healing
            RealityArchetype::Toon => 0.0,
            RealityArchetype::Genie => 0.05,
        };

        chunk.stability = (chunk.stability - stability_cost).clamp(0.0, 1.0);

        chunk.anomalies.push(projector);
        chunk.calculate_hash();
        self.calculate_root_hash();
        self.dirty_chunks.insert(id);
    }

    pub fn remove_anomaly(&mut self, uuid: &str, location: cgmath::Point3<f32>) -> bool {
        let id = ChunkId::from_world_pos(location.x, location.z, ANOMALY_GRID_SIZE);

        if let Some(chunk) = self.chunks.get_mut(&id) {
            if let Some(idx) = chunk.anomalies.iter().position(|a| a.uuid == uuid) {
                chunk.anomalies[idx].deleted = true;
                chunk.anomalies[idx].last_updated = crate::projector::get_current_timestamp();
                chunk.calculate_hash();
                self.calculate_root_hash();
                self.dirty_chunks.insert(id);
                return true;
            }
        }
        false
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

    pub fn merge(&mut self, other: WorldState) -> bool {
        let mut changed = false;
        for (id, other_chunk) in other.chunks {
            let chunk = self.get_or_create_chunk(id);
            if chunk.merge(&other_chunk) {
                changed = true;
                self.dirty_chunks.insert(id);
            }
        }

        if changed {
            self.calculate_root_hash();
        }

        changed
    }

    pub fn merge_chunks(&mut self, chunks: Vec<Chunk>) -> bool {
        let mut changed = false;
        for other_chunk in chunks {
            let id = other_chunk.id;
            let chunk = self.get_or_create_chunk(id);
            if chunk.merge(&other_chunk) {
                changed = true;
                self.dirty_chunks.insert(id);
            }
        }

        if changed {
            self.calculate_root_hash();
        }

        changed
    }

    pub fn extract_dirty_chunks(&mut self) -> Vec<Chunk> {
        let mut dirty = Vec::new();
        for id in self.dirty_chunks.drain() {
            if let Some(chunk) = self.chunks.get(&id) {
                dirty.push(chunk.clone());
            }
        }
        dirty
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reality_types::{RealitySignature, RealityArchetype};
    use cgmath::Point3;

    #[test]
    fn test_chunk_merge() {
        let mut chunk1 = Chunk::new(ChunkId { x: 0, z: 0 });
        let mut chunk2 = Chunk::new(ChunkId { x: 0, z: 0 });

        let sig = RealitySignature::default();
        let proj1 = RealityProjector::new(Point3::new(0.0, 0.0, 0.0), sig.clone());
        let mut sig2 = sig.clone();
        sig2.active_style.archetype = RealityArchetype::Horror;
        let proj2 = RealityProjector::new(Point3::new(1.0, 0.0, 0.0), sig2);

        chunk1.anomalies.push(proj1.clone());
        chunk1.calculate_hash();

        chunk2.anomalies.push(proj1.clone()); // Same anomaly
        chunk2.anomalies.push(proj2.clone()); // New anomaly
        chunk2.calculate_hash();

        // Merge chunk2 into chunk1
        let changed = chunk1.merge(&chunk2);
        assert!(changed);
        assert_eq!(chunk1.anomalies.len(), 2);

        // Merge again should not change
        let changed_again = chunk1.merge(&chunk2);
        assert!(!changed_again);
        assert_eq!(chunk1.anomalies.len(), 2);
    }

    #[test]
    fn test_chunk_id_serialization() {
        let id = ChunkId { x: 5, z: -10 };
        let serialized = serde_json::to_string(&id).unwrap();
        assert_eq!(serialized, "\"5:-10\"");

        let deserialized: ChunkId = serde_json::from_str(&serialized).unwrap();
        assert_eq!(id, deserialized);
    }

    #[test]
    fn test_world_state_map_serialization() {
        let mut world = WorldState::default();
        let id = ChunkId { x: 1, z: 2 };
        world.chunks.insert(id, Chunk::new(id));

        let serialized = serde_json::to_string(&world).unwrap();
        println!("Serialized WorldState: {}", serialized);
        // Should contain "1:2" key
        assert!(serialized.contains("\"1:2\":"));

        let deserialized: WorldState = serde_json::from_str(&serialized).unwrap();
        assert!(deserialized.chunks.contains_key(&id));
    }

    #[test]
    fn test_tombstone_propagation() {
        let mut chunk1 = Chunk::new(ChunkId { x: 0, z: 0 });
        let mut chunk2 = Chunk::new(ChunkId { x: 0, z: 0 });

        let sig = RealitySignature::default();
        let mut proj1 = RealityProjector::new(Point3::new(0.0, 0.0, 0.0), sig.clone());
        proj1.last_updated = 100;

        // Chunk 1 has active anomaly
        chunk1.anomalies.push(proj1.clone());

        // Chunk 2 has deleted anomaly (tombstone) with newer timestamp
        let mut proj1_deleted = proj1.clone();
        proj1_deleted.deleted = true;
        proj1_deleted.last_updated = 200;
        chunk2.anomalies.push(proj1_deleted.clone());

        // Merge Chunk 2 into Chunk 1
        // Chunk 1 should accept the deletion because timestamp is newer
        let changed = chunk1.merge(&chunk2);
        assert!(changed);
        assert_eq!(chunk1.anomalies.len(), 1);
        assert!(chunk1.anomalies[0].deleted);
        assert_eq!(chunk1.anomalies[0].last_updated, 200);

        // Merge Chunk 1 (deleted) into Chunk 2 (deleted) -> No change
        let changed_back = chunk2.merge(&chunk1);
        assert!(!changed_back);

        // Scenario: Resurrection Attempt (Old update arrives later)
        let mut chunk3 = Chunk::new(ChunkId { x: 0, z: 0 });
        let mut proj1_old = proj1.clone();
        proj1_old.last_updated = 150; // Newer than original, but older than deletion
        chunk3.anomalies.push(proj1_old);

        // Merge Chunk 3 into Chunk 1 (which has deleted @ 200)
        let changed_resurrect = chunk1.merge(&chunk3);
        assert!(!changed_resurrect); // Should NOT accept update because 150 < 200
        assert!(chunk1.anomalies[0].deleted);
    }
}
