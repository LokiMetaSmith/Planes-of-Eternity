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
