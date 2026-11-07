#!/usr/bin/env python3
import sys
import os
import argparse
from PIL import Image

def generate_splats(image_path, output_path, scale=0.1):
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found.")
        sys.exit(1)

    try:
        img = Image.open(image_path).convert('RGBA')
    except Exception as e:
        print(f"Error opening image: {e}")
        sys.exit(1)

    width, height = img.size
    pixels = img.load()

    # We will generate a splat for each pixel that has alpha > 0
    splats = []

    # Center the object around 0,0,0
    offset_x = width / 2.0
    offset_y = height / 2.0

    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if a > 0: # Only create splats for visible pixels
                # Map to 3D space: x -> x, y -> -y (since image y goes down), z -> 0 (flat object)
                px = (x - offset_x) * scale
                py = (offset_y - y) * scale
                pz = 0.0

                # Normalize color to 0-1
                cr = r / 255.0
                cg = g / 255.0
                cb = b / 255.0
                ca = a / 255.0

                # Format: px, py, pz, rot_x, rot_y, rot_z, rot_w, sx, sy, sz, cr, cg, cb, ca
                # We use identity rotation and uniform scale based on the pixel scale
                # Output a simple float array we can parse in rust
                splat_str = f"{px},{py},{pz},0,0,0,1,{scale},{scale},{scale},{cr},{cg},{cb},{ca}"
                splats.append(splat_str)

    try:
        with open(output_path, 'w') as f:
            for s in splats:
                f.write(f"{s}\n")
        print(f"Successfully generated {len(splats)} splats to '{output_path}'")
    except Exception as e:
        print(f"Error writing output: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Generate Gaussian Splats from a 2D image.')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('output', help='Output splat file path')
    parser.add_argument('--scale', type=float, default=0.1, help='Scale factor for the generated splats (default: 0.1)')

    args = parser.parse_args()
    generate_splats(args.input, args.output, args.scale)

if __name__ == "__main__":
    main()
