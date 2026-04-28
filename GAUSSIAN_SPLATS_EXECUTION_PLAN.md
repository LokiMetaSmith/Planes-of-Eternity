# Gaussian Splats Execution Plan

This execution plan outlines the steps necessary to prototype and potentially integrate the hybrid Voxel-Gaussian system into Reality Engine.

## Phase 1: Foundation and Data Structures

1.  **Define `SplatVertex`**
    *   **Goal:** Create the Rust struct to represent a Gaussian Splat.
    *   **File:** `reality-engine/src/voxel.rs`
    *   **Action:** Define the `SplatVertex` struct with `position`, `rotation`, `scale`, and `color`. Implement `bytemuck` traits for buffer upload. Create `SplatVertex::desc()` to define the `wgpu::VertexBufferLayout`.

2.  **Implement `generate_splats`**
    *   **Goal:** Convert voxel data into a flat array of `SplatVertex` instances.
    *   **File:** `reality-engine/src/voxel.rs`
    *   **Action:** Add a `generate_splats` method to the `Chunk` struct. This method will iterate over the 3D voxel array and instantiate a `SplatVertex` for every non-air voxel, mapping its ID to color and default scale.

## Phase 2: WebGPU Pipeline Overhaul

3.  **Update WGSL Shader for Splatting**
    *   **Goal:** Write a new vertex and fragment shader to rasterize 2D projections of 3D Gaussians.
    *   **File:** `reality-engine/src/shader_voxel.wgsl` (or a new `shader_splat.wgsl`)
    *   **Action:** Implement the 2D covariance projection logic in the vertex shader. Implement the Gaussian density evaluation and alpha blending in the fragment shader.

4.  **Modify WGPU Render Pipeline**
    *   **Goal:** Update the Rust rendering setup to use the new vertex layout and shader.
    *   **File:** `reality-engine/src/lib.rs`
    *   **Action:** In `State::new`, update the `RenderPipelineDescriptor` to use `SplatVertex::desc()`. Critically, change `wgpu::BlendState::REPLACE` to `wgpu::BlendState::ALPHA_BLENDING` for the splat pipeline to allow translucency.

## Phase 3: Depth Sorting (The Hard Part)

5.  **Implement CPU-side Sorting**
    *   **Goal:** Sort splats back-to-front before rendering to ensure correct alpha blending.
    *   **File:** `reality-engine/src/lib.rs`
    *   **Action:** Before uploading the vertex buffer in `render()`, sort the `SplatVertex` array based on distance to `camera.eye`.
    *   *Note: This will be extremely slow for large numbers of splats but is necessary for the initial prototype.*

6.  **Evaluate Compute Shader Sorting (Optional/Future)**
    *   **Goal:** Move sorting to the GPU using a bitonic sort compute shader to improve performance.
    *   **Action:** If CPU sorting is a bottleneck, research and implement a wgpu compute pass for sorting.

## Phase 4: 4D Dynamics and Interpolation

7.  **Integrate Cellular Automata**
    *   **Goal:** Ensure the splat generation updates when voxel physics change.
    *   **File:** `reality-engine/src/voxel.rs` & `lib.rs`
    *   **Action:** Verify that `Chunk::diffuse` correctly triggers a geometry rebuild (calling `generate_splats`) when voxel states change.

8.  **Implement Temporal Smoothing**
    *   **Goal:** Interpolate splat positions between discrete voxel ticks for smooth 4D motion.
    *   **File:** `reality-engine/src/lib.rs`
    *   **Action:** Modify the `generate_splats` logic to calculate the splat's `position` and `scale` by linearly interpolating between the current state and a previous state stored in the history buffer, based on a sub-tick delta time.

## Phase 5: Testing and Benchmarking

9.  **Write Tests**
    *   **Goal:** Verify data structures and generation logic.
    *   **File:** `reality-engine/tests/host_test.rs`
    *   **Action:** Write unit tests to ensure `SplatVertex` alignment is correct and `generate_splats` produces the expected number of vertices based on chunk contents.

10. **Benchmark Performance**
    *   **Goal:** Compare the performance of the Splat pipeline vs. the existing Greedy Meshing pipeline.
    *   **Action:** Run the game with a large number of chunks. Measure FPS and WebGPU memory usage. Evaluate if the visual improvement justifies the performance cost.
