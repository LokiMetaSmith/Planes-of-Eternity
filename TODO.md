# Reality Engine Tasks

## 4D Voxel Engine
- [x] **Optimize Rendering**: Implement Greedy Meshing to reduce triangle count.
- [x] **LOD System**: Implement Level of Detail for distant chunks (Octree or simplified mesh).
- [x] **Enhance Visuals**: Add Shadows, Textures, and more complex shading (AO, reflections). (AO, Reflections Implemented)
    - [x] **Day/Night Cycle**: Implement dynamic lighting and skybox based on time.
- [x] **Depth Buffer**: Fix current simple depth buffering to ensure proper z-sorting/culling of voxel layers.
- [x] **Fix Voxel Rendering**: Fix winding order bug causing invisible voxels.

## Visual Lambda System
- [x] **Render Graph Edges**: Implement a Line Pipeline in `LambdaRenderer` to visualize connections (edges) between lambda nodes. Currently, only nodes are rendered as spheres.
- [x] **Dynamic Positioning**: Anchor the Lambda Graph to the Player/Camera position (or a specific "Casting" anchor) instead of the fixed `(0, 5, 0)` world coordinate.
- [x] **Visual Collapse**: Implement proper visual feedback for `toggle_collapse` (e.g., scale subtree to 0).
- [x] **Force-Directed Layout**: Improve physics in `LambdaSystem` to prevent node overlap and provide a clearer tree structure.

## Visual Lambda V2 (Refactor & Polish)
- [x] **Variables as Wires**: Visualized variables as "Ports" with physical "Wires" connecting back to their binding Abstractions, matching Interaction Net / Bubble Notation.
- [x] **Bubble Visuals**: Rendered Abstractions as translucent, Fresnel-shaded spheres containing their body.
- [x] **Curved Wires**: Implemented Quadratic Bézier curves for edges, with distinct visual styles for Structure (stiff) vs. Wires (loose, arcing).
- [x] **Animations**: Implemented smooth animations for Beta Reduction (consumption/shrinking of arguments) and visual feedback (pulsing/color shifts).
- [x] **Interaction**: Added hover highlighting for nodes and subtrees to visualize scope and connectivity.
- [x] **Text Labels**: Implement text rendering (via HTML Overlays) to display variable names (`x`, `y`) and Primitives (`FIRE`) on the nodes.
- [x] **Direct Manipulation**: Implement drag-and-drop mechanics to physically rewire connections or drag terms into applications (Visual Programming).
- [x] **Step-by-Step Control**: Add UI or Hotkeys to Pause/Play/Step through reduction animations for better debugging/understanding.
- [x] **Sound Effects**: Add audio feedback for interaction events like "Consumption", "Hover", and "Cast".
- [x] **Layout Persistence**: Save and load the physical graph positions in `GameState`, not just the parsed term string.

## Networking & State
- [x] **Delta Sync**: Optimize `broadcast_world_state` to send only modified chunks or diffs, rather than the entire WorldState JSON.
- [x] **Conflict Resolution**: Implement timestamp-based or CRDT-like conflict resolution for Anomaly placement to handle race conditions better than the current append-only logic.
- [x] **Connection UI**: Add visual indicators for PeerJS connection status (Connecting, Connected, Error) in the UI.

## Persistence
- [x] **Save Versioning**: Add a `version` field to `GameState` struct in `persistence.rs` to handle future schema migrations gracefully.
- [x] **Save Management**: Allow multiple save slots or "World Resets" via UI.
- [x] **Input Config**: Persist user key bindings in `GameState` so they survive page reloads.
    - [x] **Voxel Inputs**: Integrated Voxel controls (Diffusion, Time Reverse, Dream) into configurable InputConfig.
- [x] **Voxel World Persistence**: Save and load voxel data (with RLE compression) to preserve world changes.

## AR / WebXR
- [x] **WebXR Support**: Investigate `navigator.xr` integration for true `immersive-ar` sessions (placing content in real-world coordinates), replacing the current "Magic Window" video background approach.

## Polishing
- [x] **Host Tests**: Refactor `lib.rs` to move platform-agnostic logic into a separate module to allow `cargo test` to run more coverage without WASM flags.
- [x] **Error Handling**: gracefully handle network failures (e.g., Signaling server down) without console spam.
