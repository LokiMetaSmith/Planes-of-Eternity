# Hybrid Voxel-Gaussian Splat Architecture

## Overview

This document outlines a proposed architecture for integrating 4D Gaussian Splatting into the Reality Engine. While the engine is fundamentally a 4D Voxel Engine utilizing Greedy Meshing, this proposal explores a hybrid approach: using the existing voxel grid as a spatial logic layer (anchors) and rendering specific materials (like liquids or gasses) or the entire scene using dynamically generated Gaussian Splats.

**Warning:** Proceeding with a full transition to Gaussian Splats will break core engine features such as Greedy Meshing optimizations and requires a complete rewrite of the WebGPU rendering pipeline. This document serves as a theoretical exploration of the concept.

## 1. Voxel-to-Splat Generation Pipeline

The core idea is to decouple the physical representation (voxels) from the visual representation (splats).

### 1.1 Voxel Anchoring
Instead of generating a quad mesh for each solid voxel face, the engine will generate a 3D Gaussian centered at the voxel's coordinate in world space.
- Active voxels (ID != 0) act as "generators" for one or more Gaussians.
- The voxel's integer coordinates `(x, y, z)` within a `Chunk` are transformed into the Gaussian's mean `(μ)`.

### 1.2 Attribute Mapping
Voxel properties defined in `reality-engine/src/voxel.rs` will map to Gaussian properties:
- **Scale (Covariance Matrix):** Initialized to the voxel size (1.0). For cohesive liquids (Water: 4, Acid: 7), the scale could dynamically expand to blend with neighbors, creating a continuous surface instead of blocky cubes.
- **Opacity:** Solid materials (Stone: 1, Wood: 5) have high opacity (alpha ≈ 1.0). Gaseous materials (Fog: 8, Cloud: 9) have low opacity and larger scales.
- **Color:** The existing `get_color(id)` function provides the base color.
- **Spherical Harmonics (SH):** Advanced implementation could replace flat colors with low-degree SH coefficients to simulate view-dependent reflections on materials like Ice or Lava.

## 2. Rust Data Structures

To support splatting in WebGPU, a new vertex layout must be defined to pass Gaussian parameters to the shader.

### 2.1 The `SplatVertex`
A new struct will replace or supplement `VoxelVertex` in `reality-engine/src/voxel.rs`.

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SplatVertex {
    pub position: [f32; 3],  // Mean (μ)
    pub rotation: [f32; 4],  // Quaternion for orientation
    pub scale: [f32; 3],     // S_x, S_y, S_z for the covariance matrix
    pub color: [f32; 4],     // RGBA (Base color and opacity)
    // Optional: pub sh_coeffs: [f32; 9] for Degree 1 Spherical Harmonics
}
```

### 2.2 Buffer Management
The `Chunk::generate_mesh` logic will be replaced by a `generate_splats` method that iterates through the 32x32x32 grid, instantiating `SplatVertex` instances and packing them into a flattened array for the `wgpu::Buffer`.

## 3. Dynamic 4D Updates via Cellular Automata

To achieve the "4D" aspect, the splats must animate over time. The existing voxel physics provide the perfect foundation.

### 3.1 Physics-Driven Motion
The `Chunk::diffuse` method already handles cellular automata logic (e.g., Lava spreading, Rain falling).
- When a voxel's ID changes (e.g., from Air to Rain), a new splat is instantiated.
- As the voxel moves down the grid, the corresponding Gaussian's mean `(μ)` updates, creating physical motion.

### 3.2 Temporal Smoothing
Voxel grids update in discrete steps, which would cause splats to teleport abruptly.
- **History Interpolation:** Utilize the `HISTORY_DEPTH` (last 16 states) to interpolate the `position` and `scale` of splats between simulation ticks.
- By lerping between `state[t-1]` and `state[t]`, the discrete cellular automata logic is smoothed into continuous 4D motion.

## 4. WebGPU Shader Modifications

The WGSL pipeline (`shader_voxel.wgsl`) requires a complete overhaul to support splat rasterization.

### 4.1 Projection and Sorting
Gaussian Splatting requires projecting the 3D ellipsoid into 2D screen space.
- The Vertex Shader calculates the 2D covariance matrix based on the view-projection matrix.
- **Depth Sorting:** Unlike opaque voxels using a depth buffer, translucent splats *must* be sorted from back-to-front before rendering. This requires a bitonic sort implementation in a Compute Shader or sorting on the CPU before uploading the vertex buffer.

### 4.2 Fragment Blending
- The fragment shader calculates the 2D Gaussian density function for the pixel.
- Alpha blending is used to accumulate color based on the opacity and density. Note: the current engine uses `wgpu::BlendState::REPLACE`. This must be changed to `wgpu::BlendState::ALPHA_BLENDING`.

## 5. Challenges and Trade-offs
- **Performance Loss:** Dropping Greedy Meshing drastically increases the number of primitives sent to the GPU. A solid 32x32x32 chunk generates 6 quads with Greedy Meshing, but would generate 32,768 splats.
- **Sorting Overhead:** Sorting thousands of splats per frame per chunk is computationally expensive.
- **Art Style:** Soft, photorealistic splats conflict directly with the discrete, retro-aesthetic of the 4D Voxel Engine and Lambda interaction nets.

## 6. Conclusion
After further architectural review, integrating Gaussian Splats has been **rejected**. Point-cloud rendering paradigms (like Gaussian Splatting) are fundamentally incompatible with Reality Engine's core 4D Voxel architecture. Pursuing this would break critical optimizations like Greedy Meshing, complicate voxel destruction and terrain editing, and cause severe depth sorting conflicts between opaque depth-buffered voxels and translucent splats, resulting in massive VRAM and computational overhead. The engine will remain focused on its highly optimized 4D Voxel pipeline.
