use crate::genie_bridge::GenieBridge;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use wgpu;

pub const CHUNK_SIZE: usize = 32;
pub const HISTORY_DEPTH: usize = 16; // Store last 16 states (ticks)

pub type VoxelId = u8;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VoxelVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
    pub ao: f32, // New: Ambient Occlusion (0.0 to 1.0)
}

impl VoxelVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<VoxelVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3, // Pos
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3, // Normal
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3, // Color
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 9]>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32, // AO
                },
            ],
        }
    }
}

pub mod voxel_data_serde {
    use super::Voxel;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(data: &Vec<Voxel>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut bytes = Vec::new();
        if data.is_empty() {
            let s = hex::encode(bytes);
            return serializer.serialize_str(&s);
        }

        let mut current_id = data[0].id;
        let mut count: u16 = 0;

        for voxel in data {
            if voxel.id == current_id && count < u16::MAX {
                count += 1;
            } else {
                // Flush
                bytes.push((count >> 8) as u8);
                bytes.push((count & 0xFF) as u8);
                bytes.push(current_id);

                current_id = voxel.id;
                count = 1;
            }
        }
        // Flush last
        bytes.push((count >> 8) as u8);
        bytes.push((count & 0xFF) as u8);
        bytes.push(current_id);

        let s = hex::encode(bytes);
        serializer.serialize_str(&s)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<Voxel>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        let bytes = hex::decode(s).map_err(serde::de::Error::custom)?;

        let mut data = Vec::new();

        let mut i = 0;
        while i + 2 < bytes.len() {
            let count = ((bytes[i] as u16) << 8) | (bytes[i + 1] as u16);
            let id = bytes[i + 2];
            i += 3;

            // Security Enhancement: Prevent Decompression Bomb (DoS)
            // A maliciously crafted run-length encoded string could specify massive counts,
            // leading to memory exhaustion when pushing to the vector.
            // We limit the total decoded size to the expected maximum volume of a chunk.
            const MAX_VOXELS: usize = crate::voxel::CHUNK_SIZE * crate::voxel::CHUNK_SIZE * crate::voxel::CHUNK_SIZE;
            if data.len() + (count as usize) > MAX_VOXELS {
                return Err(serde::de::Error::custom(format!("Security Warning: Chunk data exceeds maximum allowed volume of {}", MAX_VOXELS)));
            }

            for _ in 0..count {
                data.push(Voxel { id });
            }
        }

        Ok(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn test_chunk_key_serialization() {
        let key = ChunkKey { x: 1, y: -2, z: 3 };
        let serialized = serde_json::to_string(&key).unwrap();
        assert_eq!(serialized, "\"1:-2:3\"");

        let deserialized: ChunkKey = serde_json::from_str(&serialized).unwrap();
        assert_eq!(key, deserialized);
    }

    #[test]
    fn test_chunk_key_map_serialization() {
        let mut map = HashMap::new();
        let key = ChunkKey {
            x: 10,
            y: 20,
            z: 30,
        };
        let chunk = Chunk::new(key);
        map.insert(key, chunk);

        // Serialize map - should work because ChunkKey serializes to string
        let serialized = serde_json::to_string(&map).unwrap();
        assert!(serialized.contains("\"10:20:30\":"));
    }

    #[test]
    fn test_create_lod_majority_rule() {
        let mut chunk = Chunk::new(ChunkKey { x: 0, y: 0, z: 0 });

        // We will test factor=2 downsampling (a 2x2x2 block becomes 1 voxel)
        let factor = 2;
        let new_size = chunk.size / factor;

        // Fill the first 2x2x2 block with specific IDs to test majority rule
        // Block coords: x: 0..2, y: 0..2, z: 0..2
        chunk.set(0, 0, 0, Voxel { id: 1 }); // 1 x ID 1
        chunk.set(1, 0, 0, Voxel { id: 2 }); // 3 x ID 2
        chunk.set(0, 1, 0, Voxel { id: 2 });
        chunk.set(1, 1, 0, Voxel { id: 2 });
        chunk.set(0, 0, 1, Voxel { id: 3 }); // 4 x ID 3 (Majority)
        chunk.set(1, 0, 1, Voxel { id: 3 });
        chunk.set(0, 1, 1, Voxel { id: 3 });
        chunk.set(1, 1, 1, Voxel { id: 3 });

        // Generate LOD
        let lod_chunk = chunk.create_lod(factor);

        // The first voxel of the LOD chunk (representing the 2x2x2 block above) should be 3
        let lod_voxel = lod_chunk.get(0, 0, 0);
        assert_eq!(
            lod_voxel.id, 3,
            "LOD should pick the most frequent ID (majority rule)"
        );
        assert_eq!(
            lod_chunk.size, new_size,
            "LOD chunk size should be reduced by factor"
        );
    }

    #[test]
    fn test_rle_compression() {
        let mut data = Vec::new();
        // 5 voxels of id 1
        for _ in 0..5 {
            data.push(Voxel { id: 1 });
        }
        // 3 voxels of id 2
        for _ in 0..3 {
            data.push(Voxel { id: 2 });
        }
        // 1 voxel of id 1
        data.push(Voxel { id: 1 });

        // Expected RLE: (5, 1), (3, 2), (1, 1)
        // Hex:
        // 5 -> 00 05. ID 1 -> 01. -> 000501
        // 3 -> 00 03. ID 2 -> 02. -> 000302
        // 1 -> 00 01. ID 1 -> 01. -> 000101
        // Total: "000501000302000101"

        let mut buf = Vec::new();
        let mut serializer = serde_json::Serializer::new(&mut buf);
        voxel_data_serde::serialize(&data, &mut serializer).unwrap();

        let s = String::from_utf8(buf).unwrap();
        assert_eq!(s, "\"000501000302000101\"");

        // Deserialize
        let mut deserializer = serde_json::Deserializer::from_str(&s);
        let decoded = voxel_data_serde::deserialize(&mut deserializer).unwrap();

        assert_eq!(data, decoded);
    }

    #[test]
    fn test_chunk_set_out_of_bounds() {
        let key = ChunkKey { x: 0, y: 0, z: 0 };
        let mut chunk = Chunk::new(key);

        // This should safely return without panicking
        chunk.set(CHUNK_SIZE, CHUNK_SIZE + 1, CHUNK_SIZE + 2, Voxel { id: 1 });

        // Ensure no voxel was actually set at valid bounds due to overflow
        let voxel = chunk.get(CHUNK_SIZE - 1, CHUNK_SIZE - 1, CHUNK_SIZE - 1);
        assert_eq!(voxel.id, 0);
    }

    #[test]
    fn test_chunk_index_opt_bounds() {
        let key = ChunkKey { x: 0, y: 0, z: 0 };
        let chunk = Chunk::new(key);

        assert_eq!(chunk.index_opt(-1, 0, 0), None);
        assert_eq!(chunk.index_opt(0, -1, 0), None);
        assert_eq!(chunk.index_opt(0, 0, -1), None);

        assert_eq!(chunk.index_opt(CHUNK_SIZE as i32, 0, 0), None);
        assert_eq!(chunk.index_opt(0, CHUNK_SIZE as i32, 0), None);
        assert_eq!(chunk.index_opt(0, 0, CHUNK_SIZE as i32), None);

        // Valid bounds
        assert!(chunk.index_opt(0, 0, 0).is_some());
        assert!(chunk.index_opt(CHUNK_SIZE as i32 - 1, CHUNK_SIZE as i32 - 1, CHUNK_SIZE as i32 - 1).is_some());
    }

    #[test]
    fn test_voxel_world_set_creates_chunk() {
        let mut world = VoxelWorld::new();
        let target_key = ChunkKey { x: 5, y: -2, z: 10 };

        // Ensure chunk doesn't exist
        assert!(world.get_chunk(target_key).is_none());

        // Calculate world coordinates that map to this chunk
        // A chunk's local bounds are 0..32, world pos = chunk * 32 + local
        let world_x = target_key.x * CHUNK_SIZE as i32 + 5;
        let world_y = target_key.y * CHUNK_SIZE as i32 + 5;
        let world_z = target_key.z * CHUNK_SIZE as i32 + 5;

        // Set a non-air voxel
        world.set_voxel_at(world_x, world_y, world_z, Voxel { id: 1 });

        // Verify chunk was created
        let chunk = world.get_chunk(target_key).expect("Chunk should have been created");

        // Verify voxel was set
        let voxel = chunk.get(5, 5, 5);
        assert_eq!(voxel.id, 1);

        // Setting an air block should NOT create a chunk
        let air_target_key = ChunkKey { x: 6, y: -2, z: 10 };
        let air_world_x = air_target_key.x * CHUNK_SIZE as i32 + 5;
        world.set_voxel_at(air_world_x, world_y, world_z, Voxel { id: 0 });

        assert!(world.get_chunk(air_target_key).is_none(), "Air placement should not create new chunks");
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Voxel {
    pub id: VoxelId,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, Copy)]
pub struct ChunkKey {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl Serialize for ChunkKey {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let s = format!("{}:{}:{}", self.x, self.y, self.z);
        serializer.serialize_str(&s)
    }
}

impl<'de> Deserialize<'de> for ChunkKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 3 {
            return Err(serde::de::Error::custom(
                "Invalid ChunkKey format, expected x:y:z",
            ));
        }
        let x = parts[0].parse().map_err(serde::de::Error::custom)?;
        let y = parts[1].parse().map_err(serde::de::Error::custom)?;
        let z = parts[2].parse().map_err(serde::de::Error::custom)?;
        Ok(ChunkKey { x, y, z })
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Chunk {
    pub key: ChunkKey,
    // Flattened array: x + y*SIZE + z*SIZE*SIZE
    #[serde(with = "voxel_data_serde")]
    pub data: Vec<Voxel>,
    pub size: usize, // Dynamic size for LOD
    #[serde(skip)]
    pub history: VecDeque<Vec<Voxel>>,
}

fn hash(x: i32, y: i32, z: i32) -> f32 {
    let mut n = x.wrapping_mul(374761393) ^ y.wrapping_mul(668265263) ^ z.wrapping_mul(393555907);
    n = (n ^ (n >> 13)).wrapping_mul(1274126177);
    (n as f32) / (i32::MAX as f32)
}

impl Chunk {
    pub fn new(key: ChunkKey) -> Self {
        let size = CHUNK_SIZE; // Default full res
        let vol = size * size * size;
        Self {
            key,
            data: vec![Voxel::default(); vol],
            size,
            history: VecDeque::with_capacity(HISTORY_DEPTH),
        }
    }

    #[inline(always)]
    pub fn index(&self, x: usize, y: usize, z: usize) -> usize {
        x + y * self.size + z * self.size * self.size
    }

    #[inline(always)]
    pub fn get(&self, x: usize, y: usize, z: usize) -> Voxel {
        // Optimization: Use bounds checking that does not suffer from bitwise OR non-power-of-2 overlap trap
        if x >= self.size || y >= self.size || z >= self.size {
            return Voxel::default();
        }
        // Safety: Bounds already checked above, allowing the compiler to elide the bounds check on slice access
        unsafe { *self.data.get_unchecked(self.index(x, y, z)) }
    }

    #[inline(always)]
    pub fn index_opt(&self, x: i32, y: i32, z: i32) -> Option<usize> {
        let s = self.size as i32;
        // Optimization: Single bounds check utilizing unsigned cast. Negative values wrap to high unsigned values
        // which will fail the `>= size` check, turning 6 branches into 3.
        if (x as u32) >= s as u32 || (y as u32) >= s as u32 || (z as u32) >= s as u32 {
            None
        } else {
            Some((x as usize) + (y as usize) * self.size + (z as usize) * self.size * self.size)
        }
    }

    #[inline(always)]
    pub fn set(&mut self, x: usize, y: usize, z: usize, voxel: Voxel) {
        // Optimization: Use bounds checking that does not suffer from bitwise OR non-power-of-2 overlap trap
        if x >= self.size || y >= self.size || z >= self.size {
            return;
        }
        let idx = self.index(x, y, z);
        // Safety: Bounds already checked above
        unsafe {
            *self.data.get_unchecked_mut(idx) = voxel;
        }
    }

    pub fn save_state(&mut self) {
        if self.history.len() >= HISTORY_DEPTH {
            self.history.pop_front();
        }
        self.history.push_back(self.data.clone());
    }

    pub fn revert_state(&mut self) -> bool {
        if let Some(prev_state) = self.history.pop_back() {
            self.data = prev_state;
            return true;
        }
        false
    }

    // Create a lower-resolution version of this chunk (LOD)
    pub fn create_lod(&self, factor: usize) -> Chunk {
        let new_size = self.size / factor;
        if new_size == 0 {
            return self.clone();
        }

        let mut new_data = vec![Voxel::default(); new_size * new_size * new_size];

        for z in 0..new_size {
            for y in 0..new_size {
                for x in 0..new_size {
                    // Downsampling Logic: Voting / Majority Rule
                    // Scan the block of 'factor^3' voxels in the original data
                    // Optimization: Replaced HashMap with a stack-allocated fixed array [u16; 256]
                    // to eliminate hashing overhead and heap allocations inside the hot-path loop.
                    let mut counts = [0u16; 256];

                    for dz in 0..factor {
                        for dy in 0..factor {
                            for dx in 0..factor {
                                let ox = x * factor + dx;
                                let oy = y * factor + dy;
                                let oz = z * factor + dz;

                                let voxel = self.get(ox, oy, oz);
                                if voxel.id != 0 {
                                    counts[voxel.id as usize] += 1;
                                }
                            }
                        }
                    }

                    // Pick most common non-air voxel
                    let mut best_id = 0;
                    let mut max_count = 0;
                    for (id, &count) in counts.iter().enumerate() {
                        if count > max_count {
                            max_count = count;
                            best_id = id as u8;
                        }
                    }

                    // Threshold: only if enough matter exists? Or just take majority?
                    // Taking majority preserves volume better.

                    let idx = x + y * new_size + z * new_size * new_size;
                    new_data[idx] = Voxel { id: best_id };
                }
            }
        }

        Chunk {
            key: self.key,
            data: new_data,
            size: new_size,
            history: VecDeque::new(), // LODs don't need history
        }
    }

    pub fn generate(&mut self, base_archetype: Option<crate::reality_types::RealityArchetype>) {
        // Only valid for full size chunks
        if self.size != CHUNK_SIZE {
            return;
        }

        let wx_base = self.key.x * CHUNK_SIZE as i32;
        let wy_base = self.key.y * CHUNK_SIZE as i32;
        let wz_base = self.key.z * CHUNK_SIZE as i32;

        for z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    let wx = wx_base + x as i32;
                    let wy = wy_base + y as i32;
                    let wz = wz_base + z as i32;

                    let mut voxel = Voxel::default();

                    match base_archetype {
                        Some(crate::reality_types::RealityArchetype::Fantasy) | None => {
                            // Ground
                            if wy == -1 {
                                voxel.id = 5;
                            }

                            // Castle (Procedural)
                            if (-9..=9).contains(&wx)
                                && (-9..=9).contains(&wz)
                                && (0..10).contains(&wy)
                                && (wx.abs() > 8 || wz.abs() > 8 || wy == 0)
                            {
                                voxel.id = 1;
                            }

                            // Bridge
                            if wz == 0 && (11..30).contains(&wx) && wy == 5 {
                                voxel.id = 6;
                            }

                            // Volcano
                            let vx = wx - 40;
                            let vz = wz;
                            // Optimization: Use integer arithmetic and squared distances to eliminate expensive f32 conversions
                            // and float sqrt() calls in this tight chunk generation loop.
                            if (0..=20).contains(&wy) {
                                let dist_sq = vx * vx + vz * vz;
                                let max_dist = 20 - wy;
                                if dist_sq <= max_dist * max_dist {
                                    if dist_sq < 4 {
                                        voxel.id = 2;
                                    } else {
                                        voxel.id = 1;
                                    }
                                }
                            }

                            // Fire Noise
                            if voxel.id == 0 && wy > 0 {
                                let n = hash(wx, wy, wz);
                                if n > 0.995 {
                                    voxel.id = 3;
                                }
                            }
                        }
                        Some(crate::reality_types::RealityArchetype::SciFi) => {
                            // Metal/Stone Ground
                            if wy == -1 {
                                voxel.id = 1;
                            }

                            // Procedural Pillars
                            if wx % 10 == 0 && wz % 10 == 0 && wy >= 0 && wy <= 10 {
                                if wy == 10 {
                                    voxel.id = 4; // Glowing energy top
                                } else {
                                    voxel.id = 1; // Metal pillar
                                }
                            }
                        }
                        Some(crate::reality_types::RealityArchetype::Horror) => {
                            // Organic bumpy terrain
                            let height = (hash(wx, 0, wz).abs() * 5.0) as i32;
                            if wy < height {
                                voxel.id = 1; // Dark Stone
                            }

                            // Scattered lava
                            if wy == height && hash(wx, wy, wz).abs() > 0.8 {
                                voxel.id = 2;
                            }
                        }
                        _ => {
                            // Default Fallback Terrain
                            if wy == -1 {
                                voxel.id = 5; // Grass
                            }
                        }
                    }

                    let idx = self.index(x, y, z);
                    self.data[idx] = voxel;
                }
            }
        }
    }

    pub fn diffuse(&mut self) {
        if self.size != CHUNK_SIZE {
            return;
        } // Only simulate on full res

        let mut next_data = self.data.clone();
        let size = self.size as i32;

        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let idx = self.index(x as usize, y as usize, z as usize);
                    let current_id = self.data[idx].id;

                    if current_id == 2 {
                        // Lava spreads fire
                        let neighbors = [
                            (x + 1, y, z),
                            (x - 1, y, z),
                            (x, y + 1, z),
                            (x, y - 1, z),
                            (x, y, z + 1),
                            (x, y, z - 1),
                        ];
                        for (nx, ny, nz) in neighbors {
                            if let Some(nidx) = self.index_opt(nx, ny, nz) {
                                if self.data[nidx].id == 0
                                    && hash(nx + self.key.x, ny, nz).abs() % 1.0 > 0.9
                                {
                                    next_data[nidx].id = 3;
                                }
                            }
                        }
                    } else if current_id == 3 {
                        // Fire burns out/rises
                        if hash(x + self.key.x, y, z).abs() % 1.0 > 0.8 {
                            next_data[idx].id = 0;
                        }
                        if let Some(up_idx) = self.index_opt(x, y + 1, z) {
                            if self.data[up_idx].id == 0 && hash(x, y, z).abs() % 1.0 > 0.7 {
                                next_data[up_idx].id = 3;
                            }
                        }
                    } else if current_id == 7 {
                        // Acid spreading (liquid physics)
                        let down_idx = self.index_opt(x, y - 1, z);
                        if let Some(d_idx) = down_idx {
                            if self.data[d_idx].id == 0 {
                                // Move down if air
                                next_data[idx].id = 0;
                                next_data[d_idx].id = 7;
                            } else if self.data[d_idx].id != 0 {
                                // Block below is solid, try spreading horizontally
                                let neighbors = [
                                    (x + 1, y, z),
                                    (x - 1, y, z),
                                    (x, y, z + 1),
                                    (x, y, z - 1),
                                ];
                                // Move to one random available neighbor to avoid infinite cloning/flooding
                                // since we don't have volume simulation, we just randomly pick one air neighbor and swap.
                                // Using hash to pick deterministically pseudo-random neighbor
                                let mut available = Vec::new();
                                for (nx, ny, nz) in neighbors {
                                    if let Some(nidx) = self.index_opt(nx, ny, nz) {
                                        if self.data[nidx].id == 0 {
                                            available.push(nidx);
                                        }
                                    }
                                }

                                if !available.is_empty() {
                                    let rand_idx = (hash(x + self.key.x, y, z + self.key.z).abs() * 100.0) as usize % available.len();
                                    let target_idx = available[rand_idx];
                                    next_data[idx].id = 0;
                                    next_data[target_idx].id = 7;
                                }
                            }
                        }
                    } else if current_id == 8 {
                        // Fog randomly dissipates
                        if hash(x + self.key.x, y, z).abs() % 1.0 > 0.95 {
                            next_data[idx].id = 0;
                        }
                    } else if current_id == 9 {
                        // Clouds occasionally spawn rain beneath them
                        let down_idx = self.index_opt(x, y - 1, z);
                        if let Some(d_idx) = down_idx {
                            if self.data[d_idx].id == 0 && hash(x + self.key.x, y, z).abs() % 1.0 > 0.9 {
                                next_data[d_idx].id = 10;
                            }
                        }
                    } else if current_id == 10 {
                        // Rain falls straight down
                        let down_idx = self.index_opt(x, y - 1, z);
                        if let Some(d_idx) = down_idx {
                            if self.data[d_idx].id == 0 {
                                next_data[idx].id = 0;
                                next_data[d_idx].id = 10;
                            } else {
                                // Hit solid block, delete rain
                                next_data[idx].id = 0;
                            }
                        } else {
                            // Hit bottom of chunk, delete rain
                            next_data[idx].id = 0;
                        }
                    }
                }
            }
        }
        self.data = next_data;
    }

    // --- Greedy Meshing Implementation ---
    pub fn generate_mesh(&self) -> (Vec<VoxelVertex>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut index_counter = 0;

        let size = self.size;
        let scale_factor = CHUNK_SIZE as f32 / size as f32; // Scale vertices up if LOD is lower resolution

        let offset_x = self.key.x as f32 * CHUNK_SIZE as f32;
        let offset_y = self.key.y as f32 * CHUNK_SIZE as f32;
        let offset_z = self.key.z as f32 * CHUNK_SIZE as f32;

        fn get_color(id: VoxelId) -> [f32; 3] {
            match id {
                1 => [0.5, 0.5, 0.5], // Stone
                2 => [1.0, 0.3, 0.0], // Lava
                3 => [1.0, 0.8, 0.0], // Fire
                4 => [0.0, 0.0, 1.0], // Water
                5 => [0.0, 0.8, 0.0], // Grass
                6 => [0.6, 0.4, 0.2], // Wood
                7 => [0.2, 1.0, 0.2], // Acid
                8 => [0.8, 0.8, 0.8], // Fog
                9 => [0.9, 0.9, 0.9], // Cloud
                10 => [0.5, 0.5, 1.0], // Rain
                _ => [1.0, 0.0, 1.0],
            }
        }

        let mut mask = [0i16; CHUNK_SIZE * CHUNK_SIZE]; // Max size is 32x32

        // Iterate over 3 axes (0=X, 1=Y, 2=Z)
        for d in 0..3 {
            let u = (d + 1) % 3;
            let v = (d + 2) % 3;

            let mut x = [0; 3];
            let mut q = [0; 3];

            q[d] = 1;

            // Iterate through slices
            for i in -1..(size as i32) {
                x[d] = i;

                // Compute Mask
                let mut n = 0;
                for j in 0..size {
                    for k in 0..size {
                        x[u] = j as i32;
                        x[v] = k as i32;

                        let current = if (i as u32) < size as u32 {
                            self.get(x[0] as usize, x[1] as usize, x[2] as usize).id
                        } else {
                            0
                        };

                        let neighbor = if ((i + 1) as u32) < size as u32 {
                            self.get(
                                (x[0] + q[0]) as usize,
                                (x[1] + q[1]) as usize,
                                (x[2] + q[2]) as usize,
                            )
                            .id
                        } else {
                            0
                        };

                        let mask_val = if current != 0 && neighbor == 0 {
                            current as i16 // Front face
                        } else if current == 0 && neighbor != 0 {
                            -(neighbor as i16) // Back face
                        } else {
                            0
                        };
                        mask[n] = mask_val;
                        n += 1;
                    }
                }

                // Generate Mesh from Mask
                n = 0;
                for j in 0..size {
                    for k in 0..size {
                        if mask[n] != 0 {
                            let type_id = mask[n];
                            let mut width = 1;
                            let mut height = 1;

                            // Greedy expansion width
                            while k + width < size && mask[n + width] == type_id {
                                width += 1;
                            }

                            // Greedy expansion height
                            let mut done = false;
                            while j + height < size {
                                for w in 0..width {
                                    if mask[n + w + height * size] != type_id {
                                        done = true;
                                        break;
                                    }
                                }
                                if done {
                                    break;
                                }
                                height += 1;
                            }

                            // Add Quad
                            x[u] = j as i32;
                            x[v] = k as i32;

                            // Vertices
                            let x1 = x[0] as f32;
                            let y1 = x[1] as f32;
                            let z1 = x[2] as f32;

                            let is_current_face = type_id > 0;

                            let pos_offset = 1.0;

                            let mut p = [x1, y1, z1];
                            p[d] += pos_offset;

                            // Optimization: Instead of creating unit vectors and multiplying every component,
                            // we directly construct the scaled w_vec and h_vec which skips 6 float multiplications per quad.
                            let mut w_vec = [0.0; 3];
                            w_vec[v] = width as f32;

                            let mut h_vec = [0.0; 3];
                            h_vec[u] = height as f32;

                            // Coordinates (Scaled by LOD factor)
                            // Note: offset_x/y/z are world coordinates of chunk corner.
                            // p coordinates are local 0..size. Need to multiply by scale_factor.

                            let apply_scale = |v: [f32; 3]| -> [f32; 3] {
                                [
                                    v[0] * scale_factor + offset_x,
                                    v[1] * scale_factor + offset_y,
                                    v[2] * scale_factor + offset_z,
                                ]
                            };

                            let p0 = apply_scale([p[0], p[1], p[2]]);
                            let p1 =
                                apply_scale([p[0] + w_vec[0], p[1] + w_vec[1], p[2] + w_vec[2]]);
                            let p2 = apply_scale([
                                p[0] + w_vec[0] + h_vec[0],
                                p[1] + w_vec[1] + h_vec[1],
                                p[2] + w_vec[2] + h_vec[2],
                            ]);
                            let p3 =
                                apply_scale([p[0] + h_vec[0], p[1] + h_vec[1], p[2] + h_vec[2]]);

                            let normal = if is_current_face {
                                let mut n = [0.0; 3];
                                n[d] = 1.0;
                                n
                            } else {
                                let mut n = [0.0; 3];
                                n[d] = -1.0;
                                n
                            };

                            let color = get_color(type_id.unsigned_abs() as u8);

                            // Calculate AO
                            // Vertex-based AO using neighbor sampling
                            let ao_layer = if is_current_face { i + 1 } else { i };

                            // Helper to check if a voxel at (du, dv) in the current layer is solid
                            let check_voxel = |u_val: i32, v_val: i32| -> bool {
                                let mut coords = [0i32; 3];
                                coords[d] = ao_layer;
                                coords[u] = u_val;
                                coords[v] = v_val;

                                // Boundary check (Chunk local only)
                                if (coords[0] as u32) >= size as u32
                                    || (coords[1] as u32) >= size as u32
                                    || (coords[2] as u32) >= size as u32
                                {
                                    return false; // Assume empty outside chunk to prevent dark borders
                                }

                                let vox = self.get(
                                    coords[0] as usize,
                                    coords[1] as usize,
                                    coords[2] as usize,
                                );
                                vox.id != 0
                            };

                            let calc_ao_vertex = |u_base: i32,
                                                  v_base: i32,
                                                  u_quad_dir: i32,
                                                  v_quad_dir: i32|
                             -> f32 {
                                // Calculate offsets for quadrants
                                let u_same = if u_quad_dir == 1 { 0 } else { -1 };
                                let v_same = if v_quad_dir == 1 { 0 } else { -1 };
                                let u_opp = if u_quad_dir == 1 { -1 } else { 0 };
                                let v_opp = if v_quad_dir == 1 { -1 } else { 0 };

                                // Check neighbors in the 3 quadrants NOT occupied by the quad
                                let side1 = check_voxel(u_base + u_opp, v_base + v_same);
                                let side2 = check_voxel(u_base + u_same, v_base + v_opp);
                                let corner = check_voxel(u_base + u_opp, v_base + v_opp);

                                if side1 && side2 {
                                    0.0 // Fully occluded corner
                                } else {
                                    let mut occlusion = 0;
                                    if side1 {
                                        occlusion += 1;
                                    }
                                    if side2 {
                                        occlusion += 1;
                                    }
                                    if corner {
                                        occlusion += 1;
                                    }
                                    1.0 - (occlusion as f32 * 0.25)
                                }
                            };

                            // p0: Min U, Min V. Quad (+1, +1)
                            let ao0 = calc_ao_vertex(x[u], x[v], 1, 1);
                            // p1: Min U, Max V. Quad (+1, -1)
                            let ao1 = calc_ao_vertex(x[u], x[v] + width as i32, 1, -1);
                            // p2: Max U, Max V. Quad (-1, -1)
                            let ao2 =
                                calc_ao_vertex(x[u] + height as i32, x[v] + width as i32, -1, -1);
                            // p3: Max U, Min V. Quad (-1, +1)
                            let ao3 = calc_ao_vertex(x[u] + height as i32, x[v], -1, 1);

                            if is_current_face {
                                vertices.push(VoxelVertex {
                                    position: p0,
                                    normal,
                                    color,
                                    ao: ao0,
                                });
                                vertices.push(VoxelVertex {
                                    position: p3,
                                    normal,
                                    color,
                                    ao: ao3,
                                });
                                vertices.push(VoxelVertex {
                                    position: p2,
                                    normal,
                                    color,
                                    ao: ao2,
                                });
                                vertices.push(VoxelVertex {
                                    position: p1,
                                    normal,
                                    color,
                                    ao: ao1,
                                });
                            } else {
                                vertices.push(VoxelVertex {
                                    position: p0,
                                    normal,
                                    color,
                                    ao: ao0,
                                });
                                vertices.push(VoxelVertex {
                                    position: p1,
                                    normal,
                                    color,
                                    ao: ao1,
                                });
                                vertices.push(VoxelVertex {
                                    position: p2,
                                    normal,
                                    color,
                                    ao: ao2,
                                });
                                vertices.push(VoxelVertex {
                                    position: p3,
                                    normal,
                                    color,
                                    ao: ao3,
                                });
                            }

                            indices.push(index_counter);
                            indices.push(index_counter + 1);
                            indices.push(index_counter + 2);
                            indices.push(index_counter + 2);
                            indices.push(index_counter + 3);
                            indices.push(index_counter);
                            index_counter += 4;

                            // Clear Mask
                            for w in 0..width {
                                for h in 0..height {
                                    mask[n + w + h * size] = 0;
                                }
                            }
                        }
                        n += 1; // Increment mask index
                    }
                }
            }
        }

        (vertices, indices)
    }
}

#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct VoxelWorld {
    pub chunks: HashMap<ChunkKey, Chunk>,
    #[serde(skip)]
    pub genie: GenieBridge,
    #[serde(default)]
    pub associations: HashMap<[i32; 3], Vec<[i32; 3]>>,
}

impl VoxelWorld {
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
            genie: GenieBridge::new(),
            associations: HashMap::new(),
        }
    }

    pub fn get_chunk(&self, key: ChunkKey) -> Option<&Chunk> {
        self.chunks.get(&key)
    }

    pub fn get_chunk_mut(&mut self, key: ChunkKey) -> Option<&mut Chunk> {
        self.chunks.get_mut(&key)
    }

    pub fn create_chunk(&mut self, key: ChunkKey) -> &mut Chunk {
        self.chunks.entry(key).or_insert_with(|| Chunk::new(key))
    }

    pub fn get_voxel_at(&self, x: i32, y: i32, z: i32) -> Option<Voxel> {
        let chunk_size = CHUNK_SIZE as i32;
        let cx = x.div_euclid(chunk_size);
        let cy = y.div_euclid(chunk_size);
        let cz = z.div_euclid(chunk_size);

        let lx = x.rem_euclid(chunk_size) as usize;
        let ly = y.rem_euclid(chunk_size) as usize;
        let lz = z.rem_euclid(chunk_size) as usize;

        let key = ChunkKey {
            x: cx,
            y: cy,
            z: cz,
        };

        self.get_chunk(key).map(|chunk| chunk.get(lx, ly, lz))
    }

    pub fn set_voxel_at(&mut self, x: i32, y: i32, z: i32, voxel: Voxel) {
        // Optimization: Replace f32 casting and floor with Euclidean division
        // avoiding int-to-float-to-int conversions.
        let chunk_size = CHUNK_SIZE as i32;
        let cx = x.div_euclid(chunk_size);
        let cy = y.div_euclid(chunk_size);
        let cz = z.div_euclid(chunk_size);

        let lx = x.rem_euclid(chunk_size) as usize;
        let ly = y.rem_euclid(chunk_size) as usize;
        let lz = z.rem_euclid(chunk_size) as usize;

        let key = ChunkKey {
            x: cx,
            y: cy,
            z: cz,
        };

        // Ensure chunk exists (optional, or just ignore if not?)
        // If we want to build bridges into void, we should create chunk.
        // For digging, we only care if it exists.

        if let Some(chunk) = self.get_chunk_mut(key) {
            chunk.set(lx, ly, lz, voxel);
        } else if voxel.id != 0 {
            // Create chunk if placing block
            let chunk = self.create_chunk(key);
            chunk.set(lx, ly, lz, voxel);
        }
    }

    pub fn associate_voxels(&mut self, pos_a: [i32; 3], pos_b: [i32; 3]) {
        self.associations.entry(pos_a).or_default().push(pos_b);
        self.associations.entry(pos_b).or_default().push(pos_a);
    }

    pub fn get_associated_voxels(&self, pos: [i32; 3]) -> Option<&Vec<[i32; 3]>> {
        self.associations.get(&pos)
    }

    pub fn remove_voxel_associations(&mut self, pos: [i32; 3]) {
        if let Some(associated) = self.associations.remove(&pos) {
            for neighbor_pos in associated {
                if let Some(neighbors_list) = self.associations.get_mut(&neighbor_pos) {
                    neighbors_list.retain(|&p| p != pos);
                }
            }
        }
    }

    pub fn save_all_states(&mut self) {
        for chunk in self.chunks.values_mut() {
            chunk.save_state();
        }
    }

    pub fn revert_all_states(&mut self) {
        for chunk in self.chunks.values_mut() {
            chunk.revert_state();
        }
    }

    pub fn update_dynamics(&mut self) {
        for chunk in self.chunks.values_mut() {
            chunk.diffuse();
        }
    }

    pub fn dream(&mut self) {
        for chunk in self.chunks.values_mut() {
            self.genie.dream_chunk(chunk);
        }
    }

    pub fn diffuse_chunk(&mut self) {
        for chunk in self.chunks.values_mut() {
            self.genie.diffuse_chunk(chunk);
        }
    }

    pub fn generate_default_world(&mut self) {
        for x in -2..2 {
            for y in -1..2 {
                for z in -2..2 {
                    let key = ChunkKey { x, y, z };
                    let chunk = self.create_chunk(key);

                    // Pseudo-random archetype based on coordinates
                    let n = (x.wrapping_abs() + z.wrapping_abs()) % 3;
                    let archetype = if n == 0 {
                        Some(crate::reality_types::RealityArchetype::Fantasy)
                    } else if n == 1 {
                        Some(crate::reality_types::RealityArchetype::SciFi)
                    } else {
                        Some(crate::reality_types::RealityArchetype::Horror)
                    };

                    chunk.generate(archetype);
                }
            }
        }
    }

    pub fn ray_cast(
        &self,
        origin: cgmath::Point3<f32>,
        direction: cgmath::Vector3<f32>,
        max_dist: f32,
    ) -> Option<(ChunkKey, usize, usize, usize, [i32; 3])> {
        let start = origin;
        let dir = direction;

        let mut t = 0.0;
        let mut x = start.x.floor() as i32;
        let mut y = start.y.floor() as i32;
        let mut z = start.z.floor() as i32;

        let step_x = if dir.x > 0.0 { 1 } else { -1 };
        let step_y = if dir.y > 0.0 { 1 } else { -1 };
        let step_z = if dir.z > 0.0 { 1 } else { -1 };

        let dx = dir.x.abs();
        let dy = dir.y.abs();
        let dz = dir.z.abs();

        let dt_x = if dx > 0.0 { 1.0 / dx } else { f32::INFINITY };
        let dt_y = if dy > 0.0 { 1.0 / dy } else { f32::INFINITY };
        let dt_z = if dz > 0.0 { 1.0 / dz } else { f32::INFINITY };

        let mut t_next_x = if step_x > 0 {
            (x as f32 + 1.0 - start.x) * dt_x
        } else {
            (start.x - x as f32) * dt_x
        };

        let mut t_next_y = if step_y > 0 {
            (y as f32 + 1.0 - start.y) * dt_y
        } else {
            (start.y - y as f32) * dt_y
        };

        let mut t_next_z = if step_z > 0 {
            (z as f32 + 1.0 - start.z) * dt_z
        } else {
            (start.z - z as f32) * dt_z
        };

        let mut normal = [0, 0, 0];

        let chunk_size = CHUNK_SIZE as i32;

        while t <= max_dist {
            // Check voxel at (x, y, z)
            if let Some(voxel) = self.get_voxel_at(x, y, z) {
                if voxel.id != 0 {
                    let cx = x.div_euclid(chunk_size);
                    let cy = y.div_euclid(chunk_size);
                    let cz = z.div_euclid(chunk_size);

                    let lx = x.rem_euclid(chunk_size) as usize;
                    let ly = y.rem_euclid(chunk_size) as usize;
                    let lz = z.rem_euclid(chunk_size) as usize;

                    return Some((
                        ChunkKey {
                            x: cx,
                            y: cy,
                            z: cz,
                        },
                        lx,
                        ly,
                        lz,
                        normal,
                    ));
                }
            }

            // Step
            if t_next_x < t_next_y {
                if t_next_x < t_next_z {
                    x += step_x;
                    t = t_next_x;
                    t_next_x += dt_x;
                    normal = [-step_x, 0, 0];
                } else {
                    z += step_z;
                    t = t_next_z;
                    t_next_z += dt_z;
                    normal = [0, 0, -step_z];
                }
            } else if t_next_y < t_next_z {
                y += step_y;
                t = t_next_y;
                t_next_y += dt_y;
                normal = [0, -step_y, 0];
            } else {
                z += step_z;
                t = t_next_z;
                t_next_z += dt_z;
                normal = [0, 0, -step_z];
            }
        }

        None
    }
}

#[cfg(test)]
mod raycast_tests {
    use super::*;
    use cgmath::{Point3, Vector3};

    #[test]
    fn test_raycast_hit() {
        let mut world = VoxelWorld::new();
        let key = ChunkKey { x: 0, y: 0, z: 0 };
        let chunk = world.create_chunk(key);
        // Place a block at (5, 5, 5)
        chunk.set(5, 5, 5, Voxel { id: 1 });

        // Ray from (5.5, 10.0, 5.5) pointing down (-Y)
        // Should hit (5, 5, 5) on top face (Normal 0, 1, 0)
        let origin = Point3::new(5.5, 10.0, 5.5);
        let direction = Vector3::new(0.0, -1.0, 0.0);

        let hit = world.ray_cast(origin, direction, 20.0);
        assert!(hit.is_some());
        let (k, x, y, z, n) = hit.unwrap();
        assert_eq!(k, key);
        assert_eq!(x, 5);
        assert_eq!(y, 5);
        assert_eq!(z, 5);
        assert_eq!(n, [0, 1, 0]);
    }

    #[test]
    fn test_get_voxel_at() {
        let mut world = VoxelWorld::new();
        let voxel_type = Voxel { id: 2 };

        // Positive coordinates
        world.set_voxel_at(10, 20, 30, voxel_type);
        assert_eq!(world.get_voxel_at(10, 20, 30).unwrap().id, 2);

        // Negative coordinates
        world.set_voxel_at(-10, -20, -30, voxel_type);
        assert_eq!(world.get_voxel_at(-10, -20, -30).unwrap().id, 2);

        // Across chunk boundaries
        world.set_voxel_at(CHUNK_SIZE as i32, 0, 0, voxel_type);
        assert_eq!(world.get_voxel_at(CHUNK_SIZE as i32, 0, 0).unwrap().id, 2);

        // Empty space should return None because the chunk doesn't exist
        assert_eq!(world.get_voxel_at(0, 50, 0), None);

        // What happens if we create the chunk but not the voxel?
        world.create_chunk(ChunkKey { x: 0, y: 0, z: 0 });
        assert_eq!(world.get_voxel_at(0, 0, 0).unwrap().id, 0); // Voxel::default().id is 0
    }

    #[test]
    fn test_raycast_miss() {
        let mut world = VoxelWorld::new();
        let key = ChunkKey { x: 0, y: 0, z: 0 };
        let chunk = world.create_chunk(key);
        // Place a block at (5, 5, 5)
        chunk.set(5, 5, 5, Voxel { id: 1 });

        // Ray from (0,0,0) pointing away
        let origin = Point3::new(0.0, 0.0, 0.0);
        let direction = Vector3::new(-1.0, 0.0, 0.0);

        let hit = world.ray_cast(origin, direction, 20.0);
        assert!(hit.is_none());
    }
}

#[cfg(test)]
mod ao_tests {
    use super::*;

    #[test]
    fn test_vertex_ao_corner_occlusion() {
        let key = ChunkKey { x: 0, y: 0, z: 0 };
        let mut chunk = Chunk::new(key);

        // Ground Voxel at (1, 0, 1)
        chunk.set(1, 0, 1, Voxel { id: 1 });

        // Neighbors in the layer ABOVE (y=1), which is the empty space adjacent to the Top Face.
        // Occluder 1 (Side 1) at (0, 1, 1)
        chunk.set(0, 1, 1, Voxel { id: 1 });
        // Occluder 2 (Side 2) at (1, 1, 0)
        chunk.set(1, 1, 0, Voxel { id: 1 });

        // Note: We leave the Corner (0, 1, 0) empty for this test to check if sides block it.
        // Logic: if side1 && side2 { 0.0 }
        // So even if corner is empty, AO should be 0.0.

        let (vertices, _) = chunk.generate_mesh();

        // Find the vertex at (1, 1, 1) with Normal +Y
        let mut found = false;
        for v in vertices {
            if (v.position[0] - 1.0).abs() < 0.001
                && (v.position[1] - 1.0).abs() < 0.001
                && (v.position[2] - 1.0).abs() < 0.001
                && (v.normal[1] - 1.0).abs() < 0.001
            {
                assert!(
                    v.ao < 0.001,
                    "Expected AO 0.0 due to sides occlusion, got {}",
                    v.ao
                );
                found = true;
            }
        }
        assert!(found, "Did not find expected vertex p0");
    }

    #[test]
    fn test_vertex_ao_partial_occlusion() {
        let key = ChunkKey { x: 0, y: 0, z: 0 };
        let mut chunk = Chunk::new(key);

        // Ground Voxel at (5, 5, 5)
        chunk.set(5, 5, 5, Voxel { id: 1 });

        // Only 1 Occluder at (4, 6, 5) (Side 1)
        chunk.set(4, 6, 5, Voxel { id: 1 });

        let (vertices, _) = chunk.generate_mesh();

        let mut found = false;
        for v in vertices {
            // Vertex p0 at (5, 6, 5) (Top face corner)
            if (v.position[0] - 5.0).abs() < 0.001
                && (v.position[1] - 6.0).abs() < 0.001
                && (v.position[2] - 5.0).abs() < 0.001
                && (v.normal[1] - 1.0).abs() < 0.001
            {
                // 1 Occluder -> 1.0 - 0.25 = 0.75
                assert!(
                    (v.ao - 0.75).abs() < 0.001,
                    "Expected AO 0.75, got {}",
                    v.ao
                );
                found = true;
            }
        }
        assert!(found, "Did not find expected vertex p0");
    }
}

#[cfg(test)]
mod association_tests {
    use super::*;

    #[test]
    fn test_associate_and_get_voxels() {
        let mut world = VoxelWorld::new();
        let pos1 = [1, 2, 3];
        let pos2 = [4, 5, 6];

        world.associate_voxels(pos1, pos2);

        let associated_with_1 = world.get_associated_voxels(pos1).unwrap();
        assert_eq!(associated_with_1.len(), 1);
        assert_eq!(associated_with_1[0], pos2);

        let associated_with_2 = world.get_associated_voxels(pos2).unwrap();
        assert_eq!(associated_with_2.len(), 1);
        assert_eq!(associated_with_2[0], pos1);
    }

    #[test]
    fn test_remove_voxel_associations() {
        let mut world = VoxelWorld::new();
        let pos1 = [0, 0, 0];
        let pos2 = [1, 0, 0];
        let pos3 = [0, 1, 0];

        world.associate_voxels(pos1, pos2);
        world.associate_voxels(pos1, pos3);

        // pos1 is associated with pos2 and pos3
        assert_eq!(world.get_associated_voxels(pos1).unwrap().len(), 2);
        assert_eq!(world.get_associated_voxels(pos2).unwrap().len(), 1);
        assert_eq!(world.get_associated_voxels(pos3).unwrap().len(), 1);

        world.remove_voxel_associations(pos1);

        // pos1 should have no associations
        assert!(world.get_associated_voxels(pos1).is_none());

        // pos2 and pos3 should no longer be associated with pos1
        assert_eq!(world.get_associated_voxels(pos2).unwrap().len(), 0);
        assert_eq!(world.get_associated_voxels(pos3).unwrap().len(), 0);
    }
}

#[cfg(test)]
mod diffuse_tests {
    use super::*;

    #[test]
    fn test_diffuse_lava() {
        let key = ChunkKey { x: 0, y: 0, z: 0 };
        let mut chunk = Chunk::new(key);
        // Place lava
        chunk.set(10, 10, 10, Voxel { id: 2 });
        // Set hash to something predictable that passes the > 0.9 check
        // We know hash uses world pos, so we just run diffuse and see if it spreads,
        // it's deterministic.
        // Or we can just mock it or run it multiple times to ensure it spreads *eventually* if hash condition is met.
        // Actually, since hash is pseudo-random based on coords, it might not spread immediately.
        // Let's run diffuse 10 times and count fire (id 3).
        for _ in 0..10 {
            chunk.diffuse();
        }

        let mut fire_count = 0;
        let mut lava_count = 0;
        for z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    let id = chunk.get(x, y, z).id;
                    if id == 3 {
                        fire_count += 1;
                    }
                    if id == 2 {
                        lava_count += 1;
                    }
                }
            }
        }
        assert_eq!(lava_count, 1, "Lava should remain 1 block");
        // Fire count could be > 0 if it spread.
    }
}

#[cfg(test)]
mod get_set_tests {
    use super::*;

    #[test]
    fn test_chunk_get_out_of_bounds() {
        let chunk = Chunk::new(ChunkKey { x: 0, y: 0, z: 0 });
        // Should safely return air (0) instead of panicking
        let out_of_bounds = chunk.get(CHUNK_SIZE + 1, CHUNK_SIZE + 1, CHUNK_SIZE + 1);
        assert_eq!(out_of_bounds.id, 0);
    }

    #[test]
    fn test_chunk_set_out_of_bounds() {
        let mut chunk = Chunk::new(ChunkKey { x: 0, y: 0, z: 0 });
        // Should do nothing, safely
        chunk.set(
            CHUNK_SIZE + 1,
            CHUNK_SIZE + 1,
            CHUNK_SIZE + 1,
            Voxel { id: 1 },
        );
        // We know it didn't panic and didn't crash.
    }

    #[test]
    fn test_index_opt_out_of_bounds() {
        let chunk = Chunk::new(ChunkKey { x: 0, y: 0, z: 0 });
        assert_eq!(chunk.index_opt(-1, 0, 0), None);
        assert_eq!(chunk.index_opt(CHUNK_SIZE as i32, 0, 0), None);
    }
}
