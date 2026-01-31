# Reality Engine

A high-performance reality manipulation engine built with Rust and WebGPU, targeting the web via WebAssembly.

## Overview

This project implements a procedural reality generation system where different "Reality Archetypes" (e.g., Fantasy, SciFi) can be projected into the world. The engine handles the blending of these realities based on projector strength and distance, rendering the result using a custom WebGPU pipeline.

## Architecture

*   **`reality-engine/`**: The core Rust + WebGPU engine source code.
    *   **`src/engine.rs`**: Core game logic, platform-agnostic (WorldState, Camera, Projectors).
    *   **`src/lib.rs`**: WGPU rendering pipeline and WASM bindings (`State` struct).
    *   **`tests/host_test.rs`**: Integration tests for verifying logic on the host (non-WASM) environment.
*   **`reference_nanite_webgpu/`**: Reference implementation of Nanite-like logic in TypeScript.
*   **`legacy_epic_version/`**: Archived Unreal Engine 5 source code for reference parity.

## Setup Instructions

1.  **Install Rust:**
    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    ```

2.  **Install wasm-pack:**
    ```bash
    cargo install wasm-pack
    ```

3.  **Build (Web):**
    Navigate to the engine directory and build for the web target:
    ```bash
    cd reality-engine
    wasm-pack build --target web
    ```

4.  **Test (Host):**
    You can run logic tests directly on your machine without a browser:
    ```bash
    cd reality-engine
    cargo test --test host_test
    ```
    This verifies engine initialization, interaction logic, and P2P state synchronization.

5.  **Serve:**
    The engine requires a local server to serve the WASM and assets. You can use Python's built-in HTTP server:
    ```bash
    python3 -m http.server
    ```
    Then open your browser to `http://localhost:8000/reality-engine/index.html`.

## Controls

*   **Camera Movement:** Use `WASD` or Arrow keys to move the camera.
*   **Anomaly Injection:** Click anywhere on the ground (Projected Plane) to move the "Anomaly" projector to that location, blending the reality around it.
*   **Lambda Casting:** Press `F` to cast the current Lambda expression as a reality anomaly.
