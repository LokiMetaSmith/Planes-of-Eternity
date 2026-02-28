use std::collections::{HashMap, VecDeque};
use serde::{Serialize, Deserialize};
use wgpu;
use crate::genie_bridge::GenieBridge;

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
    use serde::{Deserializer, Serializer, Deserialize};

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
            let count = ((bytes[i] as u16) << 8) | (bytes[i+1] as u16);
            let id = bytes[i+2];
            i += 3;

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
        let key = ChunkKey { x: 10, y: 20, z: 30 };
        let chunk = Chunk::new(key);
        map.insert(key, chunk);

        // Serialize map - should work because ChunkKey serializes to string
        let serialized = serde_json::to_string(&map).unwrap();
        assert!(serialized.contains("\"10:20:30\":"));
    }

    #[test]
    fn test_rle_compression() {
        let mut data = Vec::new();
        // 5 voxels of id 1
        for _ in 0..5 { data.push(Voxel { id: 1 }); }
        // 3 voxels of id 2
        for _ in 0..3 { data.push(Voxel { id: 2 }); }
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
            return Err(serde::de::Error::custom("Invalid ChunkKey format, expected x:y:z"));
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
    (n as f32) / (std::i32::MAX as f32)
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

    pub fn index(&self, x: usize, y: usize, z: usize) -> usize {
        x + y * self.size + z * self.size * self.size
    }

    pub fn get(&self, x: usize, y: usize, z: usize) -> Voxel {
        if x >= self.size || y >= self.size || z >= self.size {
            return Voxel::default();
        }
        self.data[self.index(x, y, z)]
    }

    pub fn index_opt(&self, x: i32, y: i32, z: i32) -> Option<usize> {
        let s = self.size as i32;
        if x < 0 || x >= s ||
           y < 0 || y >= s ||
           z < 0 || z >= s {
            None
        } else {
            Some((x as usize) + (y as usize) * self.size + (z as usize) * self.size * self.size)
        }
    }

    pub fn set(&mut self, x: usize, y: usize, z: usize, voxel: Voxel) {
        if x >= self.size || y >= self.size || z >= self.size {
            return;
        }
        let idx = self.index(x, y, z);
        self.data[idx] = voxel;
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
        if new_size == 0 { return self.clone(); }

        let mut new_data = vec![Voxel::default(); new_size * new_size * new_size];

        for z in 0..new_size {
            for y in 0..new_size {
                for x in 0..new_size {
                    // Downsampling Logic: Voting / Majority Rule
                    // Scan the block of 'factor^3' voxels in the original data
                    let mut counts = HashMap::new();

                    for dz in 0..factor {
                        for dy in 0..factor {
                            for dx in 0..factor {
                                let ox = x * factor + dx;
                                let oy = y * factor + dy;
                                let oz = z * factor + dz;

                                let voxel = self.get(ox, oy, oz);
                                if voxel.id != 0 {
                                    *counts.entry(voxel.id).or_insert(0) += 1;
                                }
                            }
                        }
                    }

                    // Pick most common non-air voxel
                    let mut best_id = 0;
                    let mut max_count = 0;
                    for (id, count) in counts {
                        if count > max_count {
                            max_count = count;
                            best_id = id;
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

    pub fn generate(&mut self) {
        // Only valid for full size chunks
        if self.size != CHUNK_SIZE { return; }

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

                    // Ground
                    if wy == -1 { voxel.id = 5; }

                    // Castle (Procedural)
                    if wx.abs() < 10 && wz.abs() < 10 {
                        if wy >= 0 && wy < 10 {
                            if wx.abs() > 8 || wz.abs() > 8 { voxel.id = 1; }
                            else if wy == 0 { voxel.id = 1; }
                        }
                    }

                    // Bridge
                    if wz == 0 && wx > 10 && wx < 30 && wy == 5 { voxel.id = 6; }

                    // Volcano
                    let vx = wx - 40;
                    let vz = wz;
                    let dist = ((vx*vx + vz*vz) as f32).sqrt();
                    let height = 20.0 - dist;
                    if wy as f32 <= height && wy >= 0 {
                        if dist < 2.0 { voxel.id = 2; } else { voxel.id = 1; }
                    }

                    // Fire Noise
                    if voxel.id == 0 && wy > 0 {
                        let n = hash(wx, wy, wz);
                        if n > 0.995 { voxel.id = 3; }
                    }

                    let idx = self.index(x, y, z);
                    self.data[idx] = voxel;
                }
            }
        }
    }

    pub fn diffuse(&mut self) {
        if self.size != CHUNK_SIZE { return; } // Only simulate on full res

        let mut next_data = self.data.clone();
        let size = self.size as i32;

        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let idx = self.index(x as usize, y as usize, z as usize);
                    let current_id = self.data[idx].id;

                    if current_id == 2 { // Lava spreads fire
                         let neighbors = [(x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)];
                        for (nx, ny, nz) in neighbors {
                            if let Some(nidx) = self.index_opt(nx, ny, nz) {
                                if self.data[nidx].id == 0 && hash(nx+self.key.x, ny, nz).abs() % 1.0 > 0.9 {
                                     next_data[nidx].id = 3;
                                }
                            }
                        }
                    } else if current_id == 3 { // Fire burns out/rises
                        if hash(x+self.key.x, y, z).abs() % 1.0 > 0.8 { next_data[idx].id = 0; }
                        if let Some(up_idx) = self.index_opt(x, y+1, z) {
                             if self.data[up_idx].id == 0 && hash(x,y,z).abs() % 1.0 > 0.7 { next_data[up_idx].id = 3; }
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

                        let current = if i >= 0 && i < size as i32 {
                            self.get(x[0] as usize, x[1] as usize, x[2] as usize).id
                        } else { 0 };

                        let neighbor = if i + 1 >= 0 && i + 1 < size as i32 {
                            self.get((x[0] + q[0]) as usize, (x[1] + q[1]) as usize, (x[2] + q[2]) as usize).id
                        } else { 0 };

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
                                if done { break; }
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

                            let pos_offset = if is_current_face { 1.0 } else { 1.0 };

                            let mut p = [x1, y1, z1];
                            p[d] += pos_offset;

                            let mut du_vec = [0.0; 3]; du_vec[u] = 1.0;
                            let mut dv_vec = [0.0; 3]; dv_vec[v] = 1.0;

                            let w_vec = [dv_vec[0] * width as f32, dv_vec[1] * width as f32, dv_vec[2] * width as f32];
                            let h_vec = [du_vec[0] * height as f32, du_vec[1] * height as f32, du_vec[2] * height as f32];

                            // Coordinates (Scaled by LOD factor)
                            // Note: offset_x/y/z are world coordinates of chunk corner.
                            // p coordinates are local 0..size. Need to multiply by scale_factor.

                            let apply_scale = |v: [f32; 3]| -> [f32; 3] {
                                [
                                    v[0] * scale_factor + offset_x,
                                    v[1] * scale_factor + offset_y,
                                    v[2] * scale_factor + offset_z
                                ]
                            };

                            let p0 = apply_scale([p[0], p[1], p[2]]);
                            let p1 = apply_scale([p[0] + w_vec[0], p[1] + w_vec[1], p[2] + w_vec[2]]);
                            let p2 = apply_scale([p[0] + w_vec[0] + h_vec[0], p[1] + w_vec[1] + h_vec[1], p[2] + w_vec[2] + h_vec[2]]);
                            let p3 = apply_scale([p[0] + h_vec[0], p[1] + h_vec[1], p[2] + h_vec[2]]);

                            let normal = if is_current_face {
                                let mut n = [0.0; 3]; n[d] = 1.0; n
                            } else {
                                let mut n = [0.0; 3]; n[d] = -1.0; n
                            };

                            let color = get_color(type_id.abs() as u8);

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
                                if coords[0] < 0 || coords[0] >= size as i32 ||
                                   coords[1] < 0 || coords[1] >= size as i32 ||
                                   coords[2] < 0 || coords[2] >= size as i32 {
                                    return false; // Assume empty outside chunk to prevent dark borders
                                }

                                let vox = self.get(coords[0] as usize, coords[1] as usize, coords[2] as usize);
                                vox.id != 0
                            };

                            let calc_ao_vertex = |u_base: i32, v_base: i32, u_quad_dir: i32, v_quad_dir: i32| -> f32 {
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
                                    if side1 { occlusion += 1; }
                                    if side2 { occlusion += 1; }
                                    if corner { occlusion += 1; }
                                    1.0 - (occlusion as f32 * 0.25)
                                }
                            };

                            // p0: Min U, Min V. Quad (+1, +1)
                            let ao0 = calc_ao_vertex(x[u], x[v], 1, 1);
                            // p1: Min U, Max V. Quad (+1, -1)
                            let ao1 = calc_ao_vertex(x[u], x[v] + width as i32, 1, -1);
                            // p2: Max U, Max V. Quad (-1, -1)
                            let ao2 = calc_ao_vertex(x[u] + height as i32, x[v] + width as i32, -1, -1);
                            // p3: Max U, Min V. Quad (-1, +1)
                            let ao3 = calc_ao_vertex(x[u] + height as i32, x[v], -1, 1);

                            if is_current_face {
                                vertices.push(VoxelVertex { position: p0, normal, color, ao: ao0 });
                                vertices.push(VoxelVertex { position: p3, normal, color, ao: ao3 });
                                vertices.push(VoxelVertex { position: p2, normal, color, ao: ao2 });
                                vertices.push(VoxelVertex { position: p1, normal, color, ao: ao1 });
                            } else {
                                vertices.push(VoxelVertex { position: p0, normal, color, ao: ao0 });
                                vertices.push(VoxelVertex { position: p1, normal, color, ao: ao1 });
                                vertices.push(VoxelVertex { position: p2, normal, color, ao: ao2 });
                                vertices.push(VoxelVertex { position: p3, normal, color, ao: ao3 });
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

    pub fn set_voxel_at(&mut self, x: i32, y: i32, z: i32, voxel: Voxel) {
        let cx = (x as f32 / CHUNK_SIZE as f32).floor() as i32;
        let cy = (y as f32 / CHUNK_SIZE as f32).floor() as i32;
        let cz = (z as f32 / CHUNK_SIZE as f32).floor() as i32;

        let lx = (x - cx * CHUNK_SIZE as i32) as usize;
        let ly = (y - cy * CHUNK_SIZE as i32) as usize;
        let lz = (z - cz * CHUNK_SIZE as i32) as usize;

        let key = ChunkKey { x: cx, y: cy, z: cz };

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
                    chunk.generate();
                }
            }
        }
    }

    pub fn ray_cast(&self, origin: cgmath::Point3<f32>, direction: cgmath::Vector3<f32>, max_dist: f32) -> Option<(ChunkKey, usize, usize, usize, [i32; 3])> {
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

        while t <= max_dist {
            // Check voxel at (x, y, z)
            // Convert world coord to ChunkKey and local index
            let cx = (x as f32 / CHUNK_SIZE as f32).floor() as i32;
            let cy = (y as f32 / CHUNK_SIZE as f32).floor() as i32;
            let cz = (z as f32 / CHUNK_SIZE as f32).floor() as i32;

            let lx = (x - cx * CHUNK_SIZE as i32) as usize;
            let ly = (y - cy * CHUNK_SIZE as i32) as usize;
            let lz = (z - cz * CHUNK_SIZE as i32) as usize;

            if let Some(chunk) = self.get_chunk(ChunkKey { x: cx, y: cy, z: cz }) {
                let voxel = chunk.get(lx, ly, lz);
                if voxel.id != 0 {
                    return Some((ChunkKey { x: cx, y: cy, z: cz }, lx, ly, lz, normal));
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
            } else {
                if t_next_y < t_next_z {
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
            if (v.position[0] - 1.0).abs() < 0.001 &&
               (v.position[1] - 1.0).abs() < 0.001 &&
               (v.position[2] - 1.0).abs() < 0.001 &&
               (v.normal[1] - 1.0).abs() < 0.001 {

                assert!(v.ao < 0.001, "Expected AO 0.0 due to sides occlusion, got {}", v.ao);
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
            if (v.position[0] - 5.0).abs() < 0.001 &&
               (v.position[1] - 6.0).abs() < 0.001 &&
               (v.position[2] - 5.0).abs() < 0.001 &&
               (v.normal[1] - 1.0).abs() < 0.001 {

                // 1 Occluder -> 1.0 - 0.25 = 0.75
                assert!((v.ao - 0.75).abs() < 0.001, "Expected AO 0.75, got {}", v.ao);
                found = true;
            }
        }
        assert!(found, "Did not find expected vertex p0");
    }
}
