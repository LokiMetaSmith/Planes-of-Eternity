# Reality Engine

## Setup Instructions

1.  **Install Rust:** `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
2.  **Install wasm-pack:** `cargo install wasm-pack`
3.  **Build:**
    ```bash
    cd reality-engine
    wasm-pack build --target web
    ```
4.  **Serve:**
    You can use any static file server. For example:
    ```bash
    python3 -m http.server
    ```
    Then open `http://localhost:8000/reality-engine/index.html`.

## Architecture

*   **reality-engine:** The core Rust + WebGPU engine.
*   **reference_nanite_webgpu:** Reference implementation of Nanite logic in TypeScript.
*   **legacy_epic_version:** Archived Unreal Engine 5 source code.

## Tasks

- [x] Handle window resize events
- [x] Implement vertex buffers
- [x] Add camera control
- [x] Implement texture loading
- [x] Fix blending discontinuity at equal strength boundaries
- [x] Implement distance-based reality fading
- [x] Implement per-pixel/vertex reality blending (GPU-side)
- [x] Implement Horror and Toon archetypes and correct arbitrary blending
- [x] Implement Archetype Selector UI
- [x] Implement dynamic lighting with normal calculation
- [x] Implement Void archetype rendering
- [x] Implement Local Persistence (Save/Load to LocalStorage)
- [x] Define "Git-like" Game State Structure (Chunk Hashes, Deltas)
- [x] Implement P2P Architecture (Pollen-based Discovery & WebRTC)
- [x] Implement Merge/Conflict Resolution Logic ("Anomaly" Events)
