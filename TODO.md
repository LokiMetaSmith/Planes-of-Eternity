# Reality Engine Tasks

## Visual Lambda System
- [x] **Render Graph Edges**: Implement a Line Pipeline in `LambdaRenderer` to visualize connections (edges) between lambda nodes. Currently, only nodes are rendered as spheres.
- [x] **Dynamic Positioning**: Anchor the Lambda Graph to the Player/Camera position (or a specific "Casting" anchor) instead of the fixed `(0, 5, 0)` world coordinate.
- [x] **Visual Collapse**: Implement proper visual feedback for `toggle_collapse` (e.g., scale subtree to 0).
- [x] **Force-Directed Layout**: Improve physics in `LambdaSystem` to prevent node overlap and provide a clearer tree structure.

## Networking & State
- [x] **Delta Sync**: Optimize `broadcast_world_state` to send only modified chunks or diffs, rather than the entire WorldState JSON.
- [ ] **Conflict Resolution**: Implement timestamp-based or CRDT-like conflict resolution for Anomaly placement to handle race conditions better than the current append-only logic.
- [x] **Connection UI**: Add visual indicators for PeerJS connection status (Connecting, Connected, Error) in the UI.

## Persistence
- [x] **Save Versioning**: Add a `version` field to `GameState` struct in `persistence.rs` to handle future schema migrations gracefully.
- [x] **Save Management**: Allow multiple save slots or "World Resets" via UI.

## AR / WebXR
- [ ] **WebXR Support**: Investigate `navigator.xr` integration for true `immersive-ar` sessions (placing content in real-world coordinates), replacing the current "Magic Window" video background approach.

## Polishing
- [x] **Host Tests**: Refactor `lib.rs` to move platform-agnostic logic into a separate module to allow `cargo test` to run more coverage without WASM flags.
- [ ] **Error Handling**: gracefully handle network failures (e.g., Signaling server down) without console spam.
