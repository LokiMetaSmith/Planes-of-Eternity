use serde::{Deserialize, Serialize};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable, Serialize, Deserialize)]
pub struct SplatVertex {
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub color: [f32; 4],
    pub previous_position: [f32; 3],
    pub archetype_id: u32,
    pub padding: [u32; 2], // 80 bytes total (20 u32s)
}

/// 4D Splat Header: Metadata for the stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Splat4DHeader {
    pub magic: String, // "SP4D"
    pub version: u32,
    pub pos_bound: f32,
    pub scale_bound: f32,
    pub color_bound: f32,
    pub static_count: u32,
    pub dynamic_count: u32,
    pub gop_size: u32,
}

/// Static Section: Loaded once, containing splats that don't change.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Splat4DStaticSection {
    pub splats: Vec<SplatVertex>,
}

/// 4D Splat Keyframe: Stores absolute quantized state for dynamic splats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Splat4DKeyframe {
    pub positions: Vec<[i16; 3]>,
    pub rotations: Vec<[i8; 4]>,
    pub scales: Vec<[i8; 3]>,
    pub colors: Vec<[u8; 4]>,
}

/// 4D Splat Delta: Stores quantized changes for a subset of dynamic splats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Splat4DDelta {
    /// Bitmask indicating which dynamic splats have updates in this frame.
    pub active_mask: Vec<u8>,
    pub motion_deltas: Vec<[i8; 3]>,
    pub rotation_deltas: Vec<[i8; 4]>,
    pub scale_deltas: Vec<[i8; 3]>,
    pub color_deltas: Vec<[i8; 4]>,
}

/// Group of Pictures (GOP) for 4D Splat streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Splat4DGop {
    pub start_frame: u32,
    pub keyframe: Splat4DKeyframe,
    pub delta_frames: Vec<Splat4DDelta>,
}

/// Overarching container for a .splat4d "file".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Splat4DContainer {
    pub header: Splat4DHeader,
    pub static_section: Splat4DStaticSection,
    pub gops: Vec<Splat4DGop>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SortEntry {
    pub distance: f32,
    pub index: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SortUniforms {
    pub camera_pos: [f32; 3],
    pub num_splats: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BitonicUniforms {
    pub j: u32,
    pub k: u32,
    pub padding: [u32; 2],
}

impl SplatVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SplatVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 3]>() + std::mem::size_of::<[f32; 4]>())
                        as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 3]>() * 2 + std::mem::size_of::<[f32; 4]>())
                        as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 3]>() * 2
                        + std::mem::size_of::<[f32; 4]>() * 2)
                        as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 3]>() * 3
                        + std::mem::size_of::<[f32; 4]>() * 2)
                        as wgpu::BufferAddress,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        }
    }
}
