# Reality Engine - Game Bible

## Overview & Concept
Reality Engine is a high-performance procedural reality generation system built with Rust and WebGPU, targeting the web via WebAssembly. It revolves around the core concept of a malleable, 4D voxel world where different "Reality Archetypes" (genres, aesthetics, logic systems) can be projected into the environment.

Players or entities act as "Projectors," possessing a "Reality Signature" that dictates the visual style and rules of their localized space. When multiple projectors intersect, the engine blends their archetypes based on proximity, signal strength (fidelity), and genre conflict.

The ultimate goal is to provide a sandbox where realities collide, enabling emergent gameplay through the interaction of distinct rule sets and aesthetics.

## Core Mechanics

### 1. 4D Voxel Engine
The foundation of the physical world is a performant voxel engine with advanced rendering features:
*   **Greedy Meshing:** Optimizes rendering by combining adjacent voxels with the same properties into larger, single quads, significantly reducing triangle counts.
*   **Level of Detail (LOD):** Simplifies meshes for distant chunks, ensuring performance across large distances.
*   **Advanced Rendering Techniques:**
    *   **Ambient Occlusion (AO):** Vertex-based AO uses neighbor sampling (hard corner rule: if both side neighbors are solid, the vertex is fully occluded) to add depth.
    *   **Ray-Marched Shadows:** Casts shadows from voxels onto the procedural terrain using a 128x128x128 `R8Uint` 3D texture (`voxel_density_texture`).
    *   **Dynamic Day/Night Cycle:** The light direction and sky colors interpolate based on a global time offset (`RealityUniform.global_offset.z`).
    *   **Triplanar Mapping & Procedural Materials:** Materials (Stone, Lava, Fire, Wood) use a 256x256 procedural texture atlas generated at runtime via Value Noise and FBM. Triplanar mapping textures voxels based on world position and normal without explicit UVs.
    *   **Frustum Culling:** Filters voxel meshes against the camera's view-projection matrix to only draw what is visible.

### 2. Reality Projection & Archetypes
The world is a canvas shaped by "Reality Projectors" (anomalies). Each projector broadcasts a specific archetype.
*   **Reality Archetypes:**
    *   `Void`: Default/Empty.
    *   `Fantasy`: High Fantasy (e.g., Castles, Magic).
    *   `SciFi`: Cyberpunk, high-tech.
    *   `Horror`: Eldritch Horror, distorted reality.
    *   `Toon`: Cartoon logic and visuals.
    *   `HyperNature`: Procedural, overgrown nature.
    *   `Genie`: Generative dream spaces.
*   **Blending:** Realities blend based on `fidelity` (signal strength) and `influence_radius`. The `BlendResult` determines the dominant archetype, blend alpha (bleed), and whether genres conflict. Styles are further customized by `roughness`, `scale`, and `distortion` for generative, Stable Diffusion-like control over the noise fields.

### 3. Visual Lambda System (The "Magic")
Magic and logic are represented as functional programming via the Visual Lambda System. Spells are Lambda Calculus terms cast into the physical world.
*   **Interaction Net / Bubble Notation:**
    *   Variables are visualized as "Ports" with physical, curved "Wires" (Quadratic Bézier curves) connecting back to their binding abstractions.
    *   Abstractions are rendered as translucent, Fresnel-shaded spheres containing their bodies (bubbles).
*   **Force-Directed Layout:** A physics system prevents node overlap and provides a clear 3D tree structure.
*   **Direct Manipulation:** Players can physically drag-and-drop to rewire connections or drag terms into applications.
*   **Dynamic Visual Feedback:**
    *   Hover highlighting for nodes, wires, and subtrees visualizes scope and connectivity.
    *   Smooth animations for Beta Reduction (shrinking arguments) with pulsing/color shifts.
    *   Text labels (via HTML Overlays) display bound variable names and Primitives (`FIRE`, `GROWTH`).
    *   Audio feedback for interactions (Consumption, Hover, Cast).

### 4. Multiplayer & Networking
The engine uses a hybrid P2P architecture for multiplayer synchronization.
*   **Signaling Server (`reality-signal-server`):** A Rust/WebSocket server for peer discovery.
*   **WebRTC Data (PeerJS):** High-frequency state synchronization directly between peers.
*   **Delta Sync:** Optimizes network traffic by broadcasting only modified chunks or diffs.
*   **Conflict Resolution:** Uses a Last-Write-Wins (LWW) strategy with `deleted` tombstones to handle anomaly placement and propagation across clients. Deterministic hashing ensures chunk hashes match regardless of insertion order.

### 5. Persistence
The game state is serialized and persisted locally.
*   **GameState:** Includes versioning for schema migrations.
*   **Voxel World Persistence:** Voxel data uses Run-Length Encoding (RLE) to compress the hexadecimal chunk format, preserving world changes.
*   **Input Config:** User key bindings (including specific actions like Diffusion, Time Reverse, Dream) are saved and survive page reloads.

### 6. AR & WebXR
The engine supports immersive Augmented Reality via WebXR.
*   **`immersive-ar` Session:** Integrates camera tracking via `navigator.xr`.
*   **Passthrough Setup:** The WebGPU frame is cleared with a transparent color (`wgpu::Color::TRANSPARENT`), rendering only the engine content over the device camera feed.
*   **Framebuffer Blitting:** A 'render-to-canvas then blit' strategy uses `WebGl2RenderingContext::blit_framebuffer` to copy the wgpu-rendered canvas to the XR framebuffer.

## Technical Architecture
*   **Language:** Rust.
*   **Graphics API:** WebGPU via `wgpu` crate.
*   **Math:** `cgmath` (Note: Uses column-major memory layout).
*   **Target:** WebAssembly (`wasm32-unknown-unknown`) built with `wasm-pack`.
*   **Shaders:** WGSL (`shader_voxel.wgsl`, `shader_lambda.wgsl`, etc.).

## Future Roadmap & TODOs
*(Sourced from `TODO.md`)*
Currently, all major milestones (Voxel rendering, LOD, shadows, AR, Lambda interactions, Net-sync, persistence) are marked as complete. Future additions could include:
*   Expanding Host Test Coverage for platform-specific rendering logic.
*   Migrating from `wasm-pack` to Trunk if maintenance mode issues arise.
*   Further expansion of the Reality Archetype logic and emergent genre conflicts.
