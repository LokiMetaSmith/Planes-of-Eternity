# Reality Engine - Agents Guide

This document provides context and instructions for AI agents working on the Reality Engine codebase.

## Project Overview

Reality Engine is a Rust + WebGPU procedural reality generation system targeting the web via WebAssembly. It allows users to project different "Reality Archetypes" (Fantasy, SciFi, Horror, etc.) into the world, blending them based on proximity and strength.

## Architecture

### Core Components
*   **Engine (`reality-engine/`)**: The main crate.
    *   **Rendering**: Uses `wgpu` for cross-platform WebGPU rendering. Uses Instanced Rendering for terrain chunks.
    *   **Shaders (`src/*.wgsl`)**: WGSL shaders handle vertex displacement (terrain generation) and pixel blending of realities.
    *   **State Management**: `WorldState` manages a grid of `Chunk`s. Each chunk contains a list of `RealityProjector` anomalies.
    *   **Networking**: Hybrid P2P architecture.
        *   **Signaling**: `reality-signal-server` (WebSocket) for peer discovery.
        *   **Data**: WebRTC (via PeerJS) for high-frequency state synchronization.
    *   **Lambda Calculus**: A functional core (`lambda.rs`) drives the "Magic" system. Spells are lambda terms cast into the world.

### Key Files
*   `src/lib.rs`: Entry point, State struct, Rendering loop, Event handling.
*   `src/shader.wgsl`: The heart of the procedural generation. Contains noise functions and blending logic.
*   `src/world.rs`: Implements the "Git-like" state (Chunks, Hashing, Merging).
*   `src/network.rs`: Handles WebSocket (Pollen) and PeerJS (WebRTC) connections.
*   `src/visual_lambda.rs`: 3D Force-Directed Graph layout for visualizing Lambda terms.

## AI Agents: Healthy Skepticism & Anti-Sycophancy

Recent research shows that AI models can be overly agreeable (sycophantic), confirming users' choices even when those choices are suboptimal, harmful, or insecure. To counteract this, all AI agents operating on this codebase must practice **healthy skepticism**:

1.  **Do Not Blindly Agree:** Do not assume the user's proposed solution, architecture, or code snippet is correct, optimal, or secure.
2.  **Provide "Tough Love":** If a user's request introduces vulnerabilities, severely degrades performance, violates accessibility standards, or breaks project conventions, **you must explicitly point it out** and offer a better alternative.
3.  **The "Wait a minute" Prime:** Before executing a user's instruction, internally evaluate it with a "Wait a minute..." mindset to ensure critical thinking is applied before generating code.
4.  **Final Authority:** Provide strong warnings and better alternatives first. However, if the user explicitly insists on the suboptimal path (e.g., for testing or specific edge cases), ultimately defer to their explicit final decision.

## Coding Conventions

*   **WASM Compat**: All core logic must be WASM-compatible. Avoid blocking threads or heavy `std::fs` usage in the engine.
*   **Async**: Use `wasm_bindgen_futures` for async tasks.
*   **Math**: Use `cgmath` for linear algebra.
*   **Serialization**: Use `serde` and `serde_json` for state persistence and networking.

## Setup & Verification

*   **Build**: `trunk build` in `reality-engine/`.
*   **Test**: `cargo test` runs host-based logic tests. Note that some WASM-specific code (e.g., `web-sys` calls) is gated and won't run in standard `cargo test`.
*   **Run**: Use the `reality-signal-server` to serve the application and provide signaling for WebRTC.

## Known Issues / Gotchas

*   **Unused Code Warnings**: `cargo test` on host produces many "unused code" warnings because WASM entry points are compiled out. This is expected.
*   **PeerJS**: The global `Peer` object is injected via `index.html` script tag. `network.rs` assumes it exists.
*   **AR Mode**: Currently "Magic Window" (Video Background + Device Orientation). True WebXR `immersive-ar` is not yet implemented.

## Toolchain Status

*   **Trunk**: The project uses **[Trunk](https://trunkrs.dev/)** to build and bundle the web application, replacing the archived `wasm-pack`.
*   **Windows Setup**: `setup_dev.ps1` automatically installs `trunk` via Cargo.
*   **Dependencies**: `wasm-bindgen` and related crates (`web-sys`, `js-sys`) have moved to the `wasm-bindgen` organization and remain actively maintained. We have removed dependencies on archived `rustwasm` crates (e.g., `console_error_panic_hook`) to mitigate risks.
