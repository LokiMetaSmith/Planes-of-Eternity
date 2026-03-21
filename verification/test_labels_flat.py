import pytest
from playwright.sync_api import Page, expect
import time

def test_labels_render_correctly(page: Page, dev_server):
    page.goto("http://localhost:9000/")

    # Wait for WGPU to initialize and the canvas to appear
    canvas = page.locator("canvas")
    expect(canvas).to_be_visible(timeout=10000)

    # Inscribe a lambda term to create some nodes
    page.keyboard.press("i")

    # Handle the prompt dialog
    def handle_dialog(dialog):
        dialog.accept("GROWTH TREE")
    page.once("dialog", handle_dialog)

    # Wait a bit for the engine to parse and create the nodes
    time.sleep(2)

    # Check if node labels are rendering
    # The labels are added to #labels-layer as div.node-label
    labels = page.locator("#labels-layer .node-label")

    # We expect at least the status label ("STEP MODE", "AUTO-RUN", or "PAUSED")
    # And the nodes for "GROWTH TREE"
    expect(labels.first).to_be_visible()

    # Let's count visible labels
    visible_count = 0
    for i in range(labels.count()):
        if labels.nth(i).is_visible():
            visible_count += 1
            print(f"Label found: {labels.nth(i).inner_text()}")

    assert visible_count > 0, "No labels are visible on the screen!"

    # Take a screenshot
    page.screenshot(path="/home/jules/verification/labels_flat_verification.png")
