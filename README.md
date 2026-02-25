# Reality Engine

A high-performance reality manipulation engine built with Rust and WebGPU, targeting the web via WebAssembly.

## Overview

This project implements a procedural reality generation system where different "Reality Archetypes" (e.g., Fantasy, SciFi) can be projected into the world. The engine handles the blending of these realities based on projector strength and distance, rendering the result using a custom WebGPU pipeline.

## Architecture

*   **`reality-engine/`**: The core Rust + WebGPU engine source code.
    *   **`src/engine.rs`**: Core game logic, platform-agnostic (WorldState, Camera, Projectors).
    *   **`src/lib.rs`**: WGPU rendering pipeline and WASM bindings (`State` struct).
    *   **`tests/host_test.rs`**: Integration tests for verifying logic on the host (non-WASM) environment.
*   **`legacy_epic_version/`**: Archived Unreal Engine 5 source code for reference parity.

## Quick Start

We provide setup scripts to automate the installation of Rust and required tools.

**Windows (PowerShell):**
```powershell
.\setup_dev.ps1
```

**macOS / Linux:**
```bash
chmod +x setup_dev.sh
./setup_dev.sh
```

## Setup Instructions

1.  **Install Rust:**
    *   **macOS / Linux:**
        ```bash
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
        ```
    *   **Windows:**
        Download and run [rustup-init.exe](https://win.rustup.rs/).

2.  **Install wasm-pack:**
    ```bash
    cargo install wasm-pack
    ```
    *Note: `wasm-pack` is currently in maintenance mode following the sunsetting of the `rustwasm` organization. It is still the recommended build tool for this project, but migration to `Trunk` may be considered in the future.*

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

5.  **Run Signaling Server:**
    For multiplayer features to work, the signaling server must be running locally:
    ```bash
    cd reality-signal-server
    cargo run
    ```
    This will start a WebSocket server on `ws://localhost:9000`.

6.  **Serve:**
    The engine requires a local server to serve the WASM and assets. You can use Python's built-in HTTP server:
    ```bash
    # macOS / Linux
    python3 -m http.server

    # Windows (if python3 is not available)
    python -m http.server
    ```
    Then open your browser to `http://localhost:8000/reality-engine/index.html`.

## Troubleshooting

### Windows: "linking with `link.exe` failed"

If you encounter an error like `link: extra operand` or `linking with link.exe failed` during build, it usually means the GNU `link.exe` (often from Git Bash or MinGW) is shadowing the Microsoft Visual Studio linker in your system PATH.

**Solution:**
1.  **Use the Developer Command Prompt:** Open the **"Developer Command Prompt for VS 2019/2022"** (search in Start Menu) and run your build commands from there. This ensures the correct Microsoft build tools are prioritized.
2.  **Install C++ Build Tools:** Ensure you have the "Desktop development with C++" workload installed via the Visual Studio Installer.

## Controls

*   **Camera Movement:** Use `WASD` or Arrow keys to move the camera.
*   **Anomaly Injection:** Click anywhere on the ground (Projected Plane) to move the "Anomaly" projector to that location, blending the reality around it.
*   **Lambda Casting:** Press `F` to cast the current Lambda expression as a reality anomaly.
