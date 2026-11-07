# Evaluation of Compute Shader Sorting for Gaussian Splats

## Objective
Evaluate the feasibility, benefits, and potential drawbacks of using a compute shader (specifically Bitonic Sort) for depth-sorting Gaussian splats in the Reality Engine.

## Current State
Currently, `reality-engine/src/lib.rs` sorts splats on the CPU back-to-front before rendering:
```rust
// Sort splats back-to-front
active_splats.sort_by(|a, b| {
    // Distance calculation and sorting
});
```
This is inherently an O(N log N) operation performed on the CPU every frame, which becomes a significant bottleneck as the number of splats (N) increases, especially considering WebAssembly environments.

## Proposed Solution: Bitonic Sort via Compute Shader
Bitonic sort is a parallel sorting algorithm well-suited for GPUs. It operates in stages, repeatedly merging sequences into sorted bitonic sequences.

### Implementation Approach in WGPU
1.  **Buffer Setup:** Instead of uploading a sorted vertex buffer every frame, upload the unsorted splat data and a secondary buffer containing an array of indices `[0, 1, 2, ..., N-1]`.
2.  **Distance Calculation:** A preliminary compute pass calculates the squared distance of each splat to the camera and stores it in a buffer, paired with the original index.
3.  **Bitonic Sort Passes:** A compute shader is dispatched multiple times to perform the bitonic sort. The shader reads the distance buffer and swaps elements. Because bitonic sort requires synchronisation across workgroups for larger arrays, this usually requires multiple command buffer dispatches.
4.  **Indirect Rendering:** The rasterization pipeline (`splat_pipeline`) uses the sorted index buffer to draw the splats in the correct order.

### Feasibility
*   **WGPU Support:** WGPU supports compute shaders and storage buffers, making this entirely feasible.
*   **WebGPU Limitation:** WebGPU limits compute workgroup sizes (typically 256). Sorting more elements requires multi-pass dispatches and careful memory barriers.

### Potential Drawbacks
*   **Complexity:** Implementing an efficient bitonic sort in WGSL is non-trivial compared to a single `.sort_by()` call in Rust.
*   **Power of Two:** Bitonic sort natively operates on power-of-two array sizes. Padding the buffer with "invisible" splats at maximum depth is required if the actual number of splats is not a power of two.
*   **Overhead:** For a very small number of splats, the overhead of dispatching multiple compute passes might exceed the CPU sorting time.

## Conclusion and Recommendation
For the initial prototype, CPU sorting is acceptable. However, if the hybrid Voxel-Gaussian system is adopted and performance drops significantly due to thousands of splats, migrating to a Compute Shader Bitonic Sort is highly recommended and architecturally sound within the WGPU framework.

For now, this evaluation satisfies the exploration requirement.