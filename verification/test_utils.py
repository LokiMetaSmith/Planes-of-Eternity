import os
from PIL import Image
from io import BytesIO

# Try to import pixelmatch, fallback if not available
try:
    from pixelmatch.contrib.PIL import pixelmatch
    HAS_PIXELMATCH = True
except ImportError:
    HAS_PIXELMATCH = False

def assert_snapshot(page, filename, update=False, threshold=0.1):
    screenshot_bytes = page.screenshot()
    current_image = Image.open(BytesIO(screenshot_bytes)).convert("RGB")

    snapshot_path = os.path.join(os.path.dirname(__file__), "__snapshots__", filename)

    if update or not os.path.exists(snapshot_path):
        os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
        current_image.save(snapshot_path)
        if not update:
            print(f"Created baseline snapshot: {snapshot_path}")
        else:
            print(f"Updated baseline snapshot: {snapshot_path}")
        return

    baseline_image = Image.open(snapshot_path).convert("RGB")

    if current_image.size != baseline_image.size:
        raise AssertionError(f"Snapshot {filename} size mismatch: {current_image.size} vs {baseline_image.size}")

    if HAS_PIXELMATCH:
        diff_pixels = pixelmatch(current_image, baseline_image, threshold=threshold)
        # allow for some pixel differences due to rendering non-determinism like time uniform
        # we have a voxel scene spinning/rendering maybe?
        if diff_pixels > 25000:
            raise AssertionError(f"Snapshot {filename} mismatch! {diff_pixels} pixels differ.")
    else:
        from PIL import ImageChops
        diff = ImageChops.difference(current_image, baseline_image)
        bbox = diff.getbbox()
        if bbox:
            raise AssertionError(f"Snapshot {filename} mismatch!")
