import os
import argparse
import time
from generate_data import generate_voxels_shape, generate_splats_pointe

def generate_prompts_with_llm(theme, num_prompts=5):
    """
    Uses a small local LLM to brainstorm a list of 3D asset prompts based on a theme.
    """
    print(f"\n--- Initializing LLM to generate prompts for theme: '{theme}' ---")
    try:
        from transformers import pipeline
    except ImportError:
        print("Please install transformers: pip install transformers")
        # Fallback to hardcoded prompts if transformers isn't available
        return [f"{theme} themed item {i}" for i in range(num_prompts)]

    # Use a small, fast model for prompt generation.
    # In a real heavy-duty pipeline, you might use LLaMA or Mistral.
    generator = pipeline('text-generation', model='gpt2')

    prompt_context = f"List {num_prompts} short, descriptive text prompts for 3D objects that fit a {theme} theme. Each prompt should be a single short phrase describing a physical object.\n1."

    print("Generating...")
    start_time = time.time()
    outputs = generator(prompt_context, max_new_tokens=50, num_return_sequences=1, pad_token_id=50256, truncation=True)
    generation_time = time.time() - start_time

    text = outputs[0]['generated_text']

    # Simple parsing to extract the list items
    lines = text.replace(prompt_context, "").split('\n')
    prompts = []
    for line in lines:
        cleaned = line.strip("0123456789. -")
        if cleaned and len(cleaned) > 3:
            prompts.append(cleaned)

    # Fallback if parsing fails
    if not prompts:
        prompts = [f"A {theme} prop", f"A {theme} building", f"A {theme} vehicle"]

    # Limit to requested number
    prompts = prompts[:num_prompts]

    print(f"LLM generated {len(prompts)} prompts in {generation_time:.2f} seconds.")
    for p in prompts:
        print(f" - {p}")

    return prompts

def run_automated_pipeline(theme, num_prompts, samples_per_prompt, output_dir, data_type):
    """
    End-to-end automated pipeline: LLM prompts -> Shap-E/Point-E -> Disk
    """
    os.makedirs(output_dir, exist_ok=True)

    print("==================================================")
    print(f"Starting Automated Asset Generation Pipeline")
    print(f"Theme: {theme}")
    print("==================================================")

    total_pipeline_start = time.time()

    # 1. Generate Prompts
    prompts = generate_prompts_with_llm(theme, num_prompts)

    # 2. Generate 3D Assets
    total_assets_generated = 0
    asset_generation_start = time.time()

    for prompt in prompts:
        print(f"\nProcessing prompt: '{prompt}'")

        if data_type in ["voxels", "all"]:
            start = time.time()
            generate_voxels_shape(prompt, output_dir, resolution=16, num_samples=samples_per_prompt)
            duration = time.time() - start
            print(f"[Benchmark] Generated {samples_per_prompt} voxel chunks for '{prompt}' in {duration:.2f} seconds ({duration/samples_per_prompt:.2f}s per sample).")
            total_assets_generated += samples_per_prompt

        if data_type in ["splats", "all"]:
            start = time.time()
            generate_splats_pointe(prompt, output_dir, num_samples=samples_per_prompt)
            duration = time.time() - start
            print(f"[Benchmark] Generated {samples_per_prompt} splat point clouds for '{prompt}' in {duration:.2f} seconds ({duration/samples_per_prompt:.2f}s per sample).")
            total_assets_generated += samples_per_prompt

    total_pipeline_duration = time.time() - total_pipeline_start
    print("\n==================================================")
    print("Pipeline Complete!")
    print(f"Total time elapsed: {total_pipeline_duration:.2f} seconds.")
    print(f"Total assets generated: {total_assets_generated}")
    if total_assets_generated > 0:
        print(f"Average time per asset: {total_pipeline_duration / total_assets_generated:.2f} seconds.")
    print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate synthetic 3D data generation using an LLM to brainstorm prompts.")
    parser.add_argument("--theme", type=str, default="cyberpunk city", help="Thematic seed for the LLM to generate prompts.")
    parser.add_argument("--num_prompts", type=int, default=3, help="Number of unique prompts to generate.")
    parser.add_argument("--samples", type=int, default=1, help="Number of 3D samples to generate per prompt.")
    parser.add_argument("--output_dir", type=str, default="../data/synthetic", help="Output directory.")
    parser.add_argument("--type", type=str, choices=["voxels", "splats", "all"], default="voxels", help="What type of data to generate.")

    args = parser.parse_args()

    run_automated_pipeline(args.theme, args.num_prompts, args.samples, args.output_dir, args.type)
