import os
import argparse
import numpy as np
import torch
# Make sure to pip install shap-e and point-e before running this script
try:
    from shap_e.diffusion.sample import sample_latents
    from shap_e.diffusion.gaussian_diffusion import diffusion_from_config as shape_diffusion_from_config
    from shap_e.models.download import load_model, load_config
    from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, decode_latent_mesh
except ImportError:
    print("Please install shap-e: pip install git+https://github.com/openai/shap-e.git")

try:
    from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config as pointe_diffusion_from_config
    from point_e.diffusion.sampler import PointCloudSampler
    from point_e.models.download import load_checkpoint
    from point_e.models.configs import MODEL_CONFIGS, model_from_config
except ImportError:
    print("Please install point-e: pip install git+https://github.com/openai/point-e.git")

def generate_voxels_shape(prompt, output_dir, resolution=32, num_samples=1):
    """
    Uses Shap-E to generate a 3D mesh from a prompt, then voxelizes it to a 3D grid.
    Saves the output as a NumPy array (.npy).
    """
    print(f"Generating Shap-E voxels for prompt: '{prompt}'")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Shap-E models
    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    diffusion = shape_diffusion_from_config(load_config('diffusion'))

    batch_size = num_samples
    guidance_scale = 15.0

    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    for i, latent in enumerate(latents):
        # Extract mesh from latent
        mesh = decode_latent_mesh(xm, latent).tri_mesh()

        # Simple voxelization (Bounding box + raycast or sampling)
        # For the sake of the synthetic data pipeline, we sample points on the mesh
        # and bin them into a 3D grid.

        # Get vertices
        verts = np.array(mesh.verts)
        if len(verts) == 0:
            print("Warning: empty mesh generated.")
            continue

        # Normalize vertices to [0, 1] range
        min_v = verts.min(axis=0)
        max_v = verts.max(axis=0)
        extents = max_v - min_v
        if extents.max() > 0:
            verts = (verts - min_v) / extents.max()

        # Voxelize into resolution x resolution x resolution grid
        voxel_grid = np.zeros((resolution, resolution, resolution), dtype=np.uint8)

        # Simple point-to-voxel mapping (you could use more complex triangle intersection)
        indices = np.clip((verts * (resolution - 1)).astype(int), 0, resolution - 1)

        # Set to 1 (representing a generic voxel ID, or could map colors)
        for idx in indices:
            voxel_grid[idx[0], idx[1], idx[2]] = 1 # 1 = Solid Voxel

        # Optional: Apply some morphological closing to fill holes in the voxelized surface

        # Flatten and save
        flattened = voxel_grid.flatten()

        # Sanitize prompt for filename
        safe_prompt = prompt.replace(" ", "_")[:20]
        filename = f"{safe_prompt}_shape_{i}.npy"
        filepath = os.path.join(output_dir, filename)
        np.save(filepath, flattened)
        print(f"Saved Shap-E voxel chunk to {filepath}")


def generate_splats_pointe(prompt, output_dir, num_points=4096, num_samples=1):
    """
    Uses Point-E to generate a 3D point cloud from a prompt.
    Saves the output as a NumPy array containing XYZ and RGB values.
    """
    print(f"Generating Point-E point cloud (for Splats) for prompt: '{prompt}'")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Point-E models
    print("Loading point-e models...")
    base_name = 'base40M-textvec'
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = pointe_diffusion_from_config(DIFFUSION_CONFIGS[base_name])

    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    upsampler_diffusion = pointe_diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

    base_model.load_state_dict(load_checkpoint(base_name, device))
    upsampler_model.load_state_dict(load_checkpoint('upsample', device))

    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, num_points], # upscale
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 0.0],
        model_kwargs_key_filter=('texts', ''), # Do not condition the upsampler with text
    )

    for i in range(num_samples):
        # Sample points
        samples = None
        for x in sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt])):
            samples = x

        pc = sampler.output_to_point_clouds(samples)[0]

        # pc has .coords (N, 3) and .channels (Dict with 'R', 'G', 'B')
        coords = pc.coords
        colors = np.stack([pc.channels['R'], pc.channels['G'], pc.channels['B']], axis=-1)

        # Combine to single array: [x, y, z, r, g, b]
        splat_data = np.concatenate([coords, colors], axis=1)

        safe_prompt = prompt.replace(" ", "_")[:20]
        filename = f"{safe_prompt}_pointe_{i}.npy"
        filepath = os.path.join(output_dir, filename)
        np.save(filepath, splat_data)
        print(f"Saved Point-E splat points to {filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic 3D data for Reality Genie.")
    parser.add_argument("--prompt", type=str, default="a cozy cottage with a chimney", help="Text prompt for generation.")
    parser.add_argument("--output_dir", type=str, default="data/synthetic", help="Output directory.")
    parser.add_argument("--type", type=str, choices=["voxels", "splats", "all"], default="all", help="What type of data to generate.")
    parser.add_argument("--samples", type=int, default=1, help="Number of samples to generate.")
    parser.add_argument("--resolution", type=int, default=10, help="Voxel resolution (chunk size). Default 10 for simplicity.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.type in ["voxels", "all"]:
        generate_voxels_shape(args.prompt, args.output_dir, resolution=args.resolution, num_samples=args.samples)

    if args.type in ["splats", "all"]:
        generate_splats_pointe(args.prompt, args.output_dir, num_samples=args.samples)
