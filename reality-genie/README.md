# Reality Genie

Reality Genie is the local-first AI and ML architecture powering dynamic 3D asset generation (such as Voxel Chunks and Gaussian Splats) for the Reality Engine. It is built entirely in Rust utilizing the `candle` ML framework.

## Architecture

The project consists of several core models:
- **DiscreteDiffusion**: A transformer-based model that denoises and generates discrete token sequences (e.g., 3D Voxel IDs).
- **VqVae**: A Vector Quantized Variational Autoencoder to compress raw visual or 3D data into discrete latent tokens.
- **LatentActionModel**: A model to predict actions between frames in latent space.
- **DynamicsModel**: An autoregressive model to predict future world states and tokens based on current state and action.

## Training the DiscreteDiffusion Model

Because building a massive labeled dataset of 3D objects from scratch is difficult, we utilize a Synthetic Data Generation pipeline. We use pre-trained base models ("Teachers") like OpenAI's Shap-E and Point-E to generate millions of synthetic 3D assets, which we then use to train our custom, lightweight local model.

### 1. Generate Synthetic Data (Python)

First, generate the synthetic training data using the provided Python script. You will need to install the `shap-e` and `point-e` libraries from GitHub.

```bash
# Setup Python environment
pip install numpy torch
pip install git+https://github.com/openai/shap-e.git
pip install git+https://github.com/openai/point-e.git

# Run generation script
cd reality-genie/scripts
python generate_data.py --prompt "a wooden chair" --type voxels --samples 5
```

This will generate `.npy` files containing 3D voxel grids in the `data/synthetic` directory.

### 2. Run the Training Loop (Rust)

Once you have generated the synthetic `.npy` datasets, you can launch the training loop in Rust. The script will automatically load the voxel chunks, apply noise, compute Cross-Entropy loss, and update the model's weights using AdamW.

```bash
cd reality-genie
cargo run --bin train
```

If no data is present, the script will generate dummy tensors to verify the loop.

### 3. Exported Weights

Upon completion of the training loop, the model weights are serialized and saved to `reality-genie/diffusion_model.safetensors`. This file can then be loaded by the main Reality Engine to perform local, sovereign inference.
