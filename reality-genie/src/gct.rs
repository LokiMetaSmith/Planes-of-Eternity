use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone)]
pub struct TrajectoryPoint {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub timestamp: f32,
}

#[derive(Debug, Clone)]
pub struct TrajectoryMemory {
    pub path: VecDeque<TrajectoryPoint>,
    pub max_capacity: usize,
}

impl TrajectoryMemory {
    pub fn new(max_capacity: usize) -> Self {
        Self {
            path: VecDeque::with_capacity(max_capacity),
            max_capacity,
        }
    }

    pub fn record(&mut self, position: [f32; 3], velocity: [f32; 3], timestamp: f32) {
        if self.path.len() >= self.max_capacity {
            self.path.pop_front();
        }
        self.path.push_back(TrajectoryPoint {
            position,
            velocity,
            timestamp,
        });
    }

    pub fn get_recent_direction(&self) -> [f32; 3] {
        if self.path.len() < 2 {
            return [0.0, 0.0, 1.0]; // Default direction
        }
        let first = self.path.front().unwrap().position;
        let last = self.path.back().unwrap().position;
        let dx = last[0] - first[0];
        let dy = last[1] - first[1];
        let dz = last[2] - first[2];
        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        if len > 0.001 {
            [dx / len, dy / len, dz / len]
        } else {
            [0.0, 0.0, 1.0]
        }
    }
}

#[derive(Debug, Clone)]
pub struct KVSlot {
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub frame_id: usize,
}

#[derive(Debug, Clone)]
pub struct PagedKVCache {
    pub slots: HashMap<usize, KVSlot>, // Slot ID mapping
    pub max_slots: usize,
    pub page_size: usize,
}

impl PagedKVCache {
    pub fn new(max_slots: usize, page_size: usize) -> Self {
        Self {
            slots: HashMap::new(),
            max_slots,
            page_size,
        }
    }

    pub fn insert(&mut self, slot_id: usize, key: Vec<f32>, value: Vec<f32>, frame_id: usize) {
        if self.slots.len() >= self.max_slots {
            // Evict oldest or lowest slot ID
            let oldest_id = self.slots.keys().cloned().min();
            if let Some(id) = oldest_id {
                self.slots.remove(&id);
            }
        }
        self.slots.insert(slot_id, KVSlot { key, value, frame_id });
    }

    pub fn clear(&mut self) {
        self.slots.clear();
    }
}

#[derive(Debug, Clone)]
pub struct AnchorContext {
    pub neighbor_heights: HashMap<[i32; 2], Vec<f32>>, // Coordinates to heightmap
    pub chunk_size: usize,
}

impl AnchorContext {
    pub fn new(chunk_size: usize) -> Self {
        Self {
            neighbor_heights: HashMap::new(),
            chunk_size,
        }
    }

    pub fn add_neighbor(&mut self, chunk_x: i32, chunk_z: i32, heightmap: Vec<f32>) {
        self.neighbor_heights.insert([chunk_x, chunk_z], heightmap);
    }

    /// Blends edges to prevent boundary seams/drift
    pub fn apply_boundary_constraints(
        &self,
        chunk_x: i32,
        chunk_z: i32,
        heights: &mut [f32],
    ) {
        let size = self.chunk_size;

        // Neighbor Offsets
        let neighbors = [
            (-1, 0),  // Left
            (1, 0),   // Right
            (0, -1),  // Down
            (0, 1),   // Up
        ];

        for (dx, dz) in neighbors {
            let nx = chunk_x + dx;
            let nz = chunk_z + dz;

            if let Some(n_heights) = self.neighbor_heights.get(&[nx, nz]) {
                if n_heights.len() != size * size {
                    continue;
                }

                // Smoothly blend edge voxels
                for idx in 0..size {
                    match (dx, dz) {
                        (-1, 0) => {
                            // Blend Left Edge of current chunk with Right Edge of neighbor
                            // Neighbor index: (size - 1) + idx * size
                            // Current index: 0 + idx * size
                            let n_val = n_heights[(size - 1) + idx * size];
                            let c_val = heights[0 + idx * size];
                            heights[0 + idx * size] = n_val; // Absolute constraint

                            // Blend 1 voxel deep inside
                            let c_val_inner = heights[1 + idx * size];
                            heights[1 + idx * size] = 0.5 * c_val_inner + 0.5 * c_val;
                        }
                        (1, 0) => {
                            // Blend Right Edge of current chunk with Left Edge of neighbor
                            // Neighbor index: 0 + idx * size
                            // Current index: (size - 1) + idx * size
                            let n_val = n_heights[0 + idx * size];
                            let c_val = heights[(size - 1) + idx * size];
                            heights[(size - 1) + idx * size] = n_val;

                            let c_val_inner = heights[(size - 2) + idx * size];
                            heights[(size - 2) + idx * size] = 0.5 * c_val_inner + 0.5 * c_val;
                        }
                        (0, -1) => {
                            // Blend Down Edge of current chunk with Up Edge of neighbor
                            // Neighbor index: idx + (size - 1) * size
                            // Current index: idx + 0 * size
                            let n_val = n_heights[idx + (size - 1) * size];
                            let c_val = heights[idx + 0 * size];
                            heights[idx + 0 * size] = n_val;

                            let c_val_inner = heights[idx + 1 * size];
                            heights[idx + 1 * size] = 0.5 * c_val_inner + 0.5 * c_val;
                        }
                        (0, 1) => {
                            // Blend Up Edge of current chunk with Down Edge of neighbor
                            // Neighbor index: idx + 0 * size
                            // Current index: idx + (size - 1) * size
                            let n_val = n_heights[idx + 0 * size];
                            let c_val = heights[idx + (size - 1) * size];
                            heights[idx + (size - 1) * size] = n_val;

                            let c_val_inner = heights[idx + (size - 2) * size];
                            heights[idx + (size - 2) * size] = 0.5 * c_val_inner + 0.5 * c_val;
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct KeyframeWindow {
    pub keyframes: VecDeque<usize>, // List of keyframe frame_ids
    pub max_size: usize,
}

impl KeyframeWindow {
    pub fn new(max_size: usize) -> Self {
        Self {
            keyframes: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    pub fn push_keyframe(&mut self, frame_id: usize) {
        if self.keyframes.len() >= self.max_size {
            self.keyframes.pop_front();
        }
        self.keyframes.push_back(frame_id);
    }
}

pub struct SceneExtensionModel {
    pub trajectory_memory: TrajectoryMemory,
    pub kv_cache: PagedKVCache,
    pub keyframe_window: KeyframeWindow,
    pub chunk_size: usize,
}

impl Default for SceneExtensionModel {
    fn default() -> Self {
        Self::new(32)
    }
}

impl SceneExtensionModel {
    pub fn new(chunk_size: usize) -> Self {
        Self {
            trajectory_memory: TrajectoryMemory::new(100),
            kv_cache: PagedKVCache::new(128, 16),
            keyframe_window: KeyframeWindow::new(8),
            chunk_size,
        }
    }

    /// Perform a simulated feed-forward generation pass of GCT
    /// It combines procedural terrain with spatial trajectory-alignment and anchor constraints.
    pub fn generate_cohesive_heightmap(
        &mut self,
        chunk_x: i32,
        chunk_z: i32,
        anchor_context: &AnchorContext,
    ) -> Vec<f32> {
        let size = self.chunk_size;
        let mut heights = vec![0.0; size * size];

        // 1. Base Continuous Diffusion (Procedural Noise)
        let cd_generator = crate::continuous_diffusion::ContinuousDiffusionGenerator::new();
        let base_heights = cd_generator.generate_heightmap(chunk_x, chunk_z, size);
        heights.copy_from_slice(&base_heights);

        // 2. Trajectory Modulation
        // If player is moving fast or in a specific direction, warp the terrain slightly to look like a rolling trail/path
        let dir = self.trajectory_memory.get_recent_direction();
        let dx = dir[0];
        let dz = dir[2];

        for z in 0..size {
            for x in 0..size {
                let idx = x + z * size;
                // Add a small path-alignment bias
                let alignment = (x as f32 * dx + z as f32 * dz) * 0.05;
                heights[idx] += alignment;
            }
        }

        // 3. Anchor Context Edge-Blending Constraints (No Drift, seamless boundaries)
        anchor_context.apply_boundary_constraints(chunk_x, chunk_z, &mut heights);

        // 4. Update KV Cache & Keyframes
        let frame_id = (chunk_x.abs() + chunk_z.abs()) as usize;
        self.keyframe_window.push_keyframe(frame_id);

        // Store a simulated key/value attention pair for this generated segment
        let mock_key = vec![chunk_x as f32, chunk_z as f32];
        let mock_val = heights[..14].to_vec(); // store sample heights as features
        self.kv_cache.insert(frame_id % 128, mock_key, mock_val, frame_id);

        heights
    }
}
