use std::collections::{HashMap, VecDeque};
use serde::{Serialize, Deserialize};
use wgpu;
use crate::genie_bridge::GenieBridge;

pub const CHUNK_SIZE: usize = 32;
pub const HISTORY_DEPTH: usize = 16; // Store last 16 states (ticks)

pub type VoxelId = u8;

// Material IDs:
// 0 = Air
// 1 = Stone (Castle)
// 2 = Lava (Volcano core)
// 3 = Fire (Diffusion result)
// 4 = Water (Fluid)
// 5 = Grass (Ground)
// 6 = Wood (Bridges)

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VoxelVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
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
            ],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Voxel {
    pub id: VoxelId,
}

impl Default for Voxel {
    fn default() -> Self {
        Self { id: 0 }
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, Copy, Serialize, Deserialize)]
pub struct ChunkKey {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Chunk {
    pub key: ChunkKey,
    // Flattened array: x + y*SIZE + z*SIZE*SIZE
    pub data: Vec<Voxel>,

    // 4D History: Stores snapshots of 'data'
    #[serde(skip)] // Don't serialize history to save space/time for now
    pub history: VecDeque<Vec<Voxel>>,
}

// Pseudo-random number generator
fn hash(x: i32, y: i32, z: i32) -> f32 {
    let mut n = x.wrapping_mul(374761393) ^ y.wrapping_mul(668265263) ^ z.wrapping_mul(393555907);
    n = (n ^ (n >> 13)).wrapping_mul(1274126177);
    (n as f32) / (std::i32::MAX as f32)
}

impl Chunk {
    pub fn new(key: ChunkKey) -> Self {
        let size = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
        Self {
            key,
            data: vec![Voxel::default(); size],
            history: VecDeque::with_capacity(HISTORY_DEPTH),
        }
    }

    pub fn index(x: usize, y: usize, z: usize) -> usize {
        x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE
    }

    pub fn index_opt(x: i32, y: i32, z: i32) -> Option<usize> {
        if x < 0 || x >= CHUNK_SIZE as i32 ||
           y < 0 || y >= CHUNK_SIZE as i32 ||
           z < 0 || z >= CHUNK_SIZE as i32 {
            None
        } else {
            Some((x as usize) + (y as usize) * CHUNK_SIZE + (z as usize) * CHUNK_SIZE * CHUNK_SIZE)
        }
    }

    pub fn get(&self, x: usize, y: usize, z: usize) -> Voxel {
        if x >= CHUNK_SIZE || y >= CHUNK_SIZE || z >= CHUNK_SIZE {
            return Voxel::default();
        }
        self.data[Self::index(x, y, z)]
    }

    pub fn set(&mut self, x: usize, y: usize, z: usize, voxel: Voxel) {
        if x >= CHUNK_SIZE || y >= CHUNK_SIZE || z >= CHUNK_SIZE {
            return;
        }
        self.data[Self::index(x, y, z)] = voxel;
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

    pub fn generate(&mut self) {
        let wx_base = self.key.x * CHUNK_SIZE as i32;
        let wy_base = self.key.y * CHUNK_SIZE as i32;
        let wz_base = self.key.z * CHUNK_SIZE as i32;

        let offset_x = self.key.x as f32 * CHUNK_SIZE as f32;
        let offset_y = self.key.y as f32 * CHUNK_SIZE as f32;
        let offset_z = self.key.z as f32 * CHUNK_SIZE as f32;

        for z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    let wx = wx_base + x as i32;
                    let wy = wy_base + y as i32;
                    let wz = wz_base + z as i32;

                    let mut voxel = Voxel::default();

                    // Ground Plane
                    if wy == -1 {
                        voxel.id = 5; // Grass
                    }

                    // Castle at 0,0,0 (approx)
                    if wx.abs() < 10 && wz.abs() < 10 {
                        if wy >= 0 && wy < 10 {
                            // Walls
                            if wx.abs() > 8 || wz.abs() > 8 {
                                voxel.id = 1; // Stone
                            } else if wy == 0 {
                                voxel.id = 1; // Floor
                            }
                        }
                    }

                    // Bridge
                    if wz == 0 && wx > 10 && wx < 30 && wy == 5 {
                        voxel.id = 6; // Wood
                    }

                    // Volcano at 40,0,0
                    let vx = wx - 40;
                    let vz = wz;
                    let dist = ((vx*vx + vz*vz) as f32).sqrt();
                    let height = 20.0 - dist;

                    if wy as f32 <= height && wy >= 0 {
                        if dist < 2.0 {
                             voxel.id = 2; // Lava Core
                        } else {
                             voxel.id = 1; // Stone
                        }
                    }

                    // Noise for random scattering
                    if voxel.id == 0 && wy > 0 {
                        let n = hash(wx, wy, wz);
                        if n > 0.995 {
                            voxel.id = 3; // Random Fire
                        }
                    }

                    self.data[Self::index(x, y, z)] = voxel;
                }
            }
        }
    }

    pub fn diffuse(&mut self) {
        let mut next_data = self.data.clone();
        let size = CHUNK_SIZE as i32;

        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let idx = Self::index(x as usize, y as usize, z as usize);
                    let current_id = self.data[idx].id;

                    // Logic
                    if current_id == 2 { // Lava
                         let neighbors = [
                            (x+1, y, z), (x-1, y, z),
                            (x, y+1, z), (x, y-1, z),
                            (x, y, z+1), (x, y, z-1)
                        ];
                        for (nx, ny, nz) in neighbors {
                            if let Some(nidx) = Self::index_opt(nx, ny, nz) {
                                if self.data[nidx].id == 0 {
                                    if hash(nx + self.key.x, ny, nz).abs() % 1.0 > 0.9 {
                                         next_data[nidx].id = 3; // Fire
                                    }
                                }
                            }
                        }
                    } else if current_id == 3 { // Fire
                        if hash(x + self.key.x, y, z).abs() % 1.0 > 0.8 {
                            next_data[idx].id = 0;
                        }
                        if y < size - 1 {
                             if let Some(up_idx) = Self::index_opt(x, y+1, z) {
                                 if self.data[up_idx].id == 0 && hash(x,y,z).abs() % 1.0 > 0.7 {
                                     next_data[up_idx].id = 3;
                                 }
                             }
                        }
                    }
                }
            }
        }
        self.data = next_data;
    }

    pub fn generate_mesh(&self) -> (Vec<VoxelVertex>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut index_counter = 0;

        fn get_color(id: VoxelId) -> [f32; 3] {
            match id {
                1 => [0.5, 0.5, 0.5], // Stone
                2 => [1.0, 0.3, 0.0], // Lava
                3 => [1.0, 0.8, 0.0], // Fire
                4 => [0.0, 0.0, 1.0], // Water
                5 => [0.0, 0.8, 0.0], // Grass
                6 => [0.6, 0.4, 0.2], // Wood
                _ => [1.0, 0.0, 1.0], // Error
            }
        }

        // Helper to add face
        let mut add_face = |pos: [f32; 3], corners: [[f32; 3]; 4], normal: [f32; 3], color: [f32; 3]| {
             for corner in corners {
                 vertices.push(VoxelVertex {
                     position: [pos[0] + corner[0], pos[1] + corner[1], pos[2] + corner[2]],
                     normal,
                     color,
                 });
             }
             indices.push(index_counter);
             indices.push(index_counter + 1);
             indices.push(index_counter + 2);
             indices.push(index_counter + 2);
             indices.push(index_counter + 3);
             indices.push(index_counter);
             index_counter += 4;
        };

        let offset_x = self.key.x as f32 * CHUNK_SIZE as f32;
        let offset_y = self.key.y as f32 * CHUNK_SIZE as f32;
        let offset_z = self.key.z as f32 * CHUNK_SIZE as f32;

        for z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    let id = self.get(x, y, z).id;
                    if id == 0 { continue; }

                    let color = get_color(id);
                    let fx = x as f32 + offset_x;
                    let fy = y as f32 + offset_y;
                    let fz = z as f32 + offset_z;
                    let base_pos = [fx, fy, fz];

                    // Check neighbors
                    // +X
                    if x == CHUNK_SIZE - 1 || self.get(x+1, y, z).id == 0 {
                        add_face(base_pos,
                            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0]],
                            [1.0, 0.0, 0.0], color);
                    }
                    // -X
                    if x == 0 || self.get(x-1, y, z).id == 0 {
                        add_face(base_pos,
                            [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                            [-1.0, 0.0, 0.0], color);
                    }
                    // +Y
                    if y == CHUNK_SIZE - 1 || self.get(x, y+1, z).id == 0 {
                        add_face(base_pos,
                            [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                            [0.0, 1.0, 0.0], color);
                    }
                    // -Y
                    if y == 0 || self.get(x, y-1, z).id == 0 {
                        add_face(base_pos,
                            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                            [0.0, -1.0, 0.0], color);
                    }
                    // +Z
                    if z == CHUNK_SIZE - 1 || self.get(x, y, z+1).id == 0 {
                        add_face(base_pos,
                            [[1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
                            [0.0, 0.0, 1.0], color);
                    }
                    // -Z
                    if z == 0 || self.get(x, y, z-1).id == 0 {
                        add_face(base_pos,
                            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
                            [0.0, 0.0, -1.0], color);
                    }
                }
            }
        }

        (vertices, indices)
    }
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct VoxelWorld {
    pub chunks: HashMap<ChunkKey, Chunk>,
    #[serde(skip)]
    pub genie: GenieBridge,
}

impl VoxelWorld {
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
            genie: GenieBridge::new(),
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
        // Apply Genie logic to all chunks
        for chunk in self.chunks.values_mut() {
            self.genie.dream_chunk(chunk);
        }
    }

    pub fn generate_default_world(&mut self) {
        for x in -2..2 {
            for y in -1..2 {
                for z in -2..2 {
                    let key = ChunkKey { x, y, z };
                    let chunk = self.create_chunk(key);
                    chunk.generate();
                }
            }
        }
    }
}
