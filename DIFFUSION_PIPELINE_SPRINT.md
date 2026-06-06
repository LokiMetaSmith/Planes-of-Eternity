# Diffusion Pipeline Sprint Plan

**Sprint Goal:** Replace DummySplatGenerator with a functional, Wasm-compatible local Diffusion Pipeline for generating Gaussian Splats.

## Phase 1: Wasm Infrastructure & Weight Management
- [x] **Task 1.1:** Set up the Web Worker architecture in GenieBridge to offload heavy inference from the main game thread.
- [ ] **Task 1.2:** Implement Wasm-compatible HTTP fetching and IndexedDB/CacheStorage caching for the .safetensors model weights.

## Phase 2: Text Encoding & Conditioning
- [x] **Task 2.1:** Integrate the tokenizers crate and implement a lightweight Text Encoder.
- [ ] **Task 2.2:** Add Cross-Attention layers to the DiscreteDiffusion block to accept text embeddings for conditioning.

## Phase 3: The Inference Loop
- [ ] **Task 3.1:** Implement a proper diffusion scheduler (e.g., DDIM or DDPM).
- [ ] **Task 3.2:** Build the iterative denoising inference loop inside DiscreteDiffusion.

## Phase 4: Decoding & Splat Mapping
- [ ] **Task 4.1:** Pass final discrete latent tokens through VqVae::decode_from_indices.
- [ ] **Task 4.2:** Map the VQ-VAE output channels to physical Gaussian Splat properties (position, scale, rotation, color/opacity) into the SplatVertex format.
