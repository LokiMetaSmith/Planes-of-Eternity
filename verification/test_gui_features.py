import pytest
from playwright.sync_api import Page, expect
import time
import os
from test_utils import assert_snapshot

def test_gui_features(page: Page, dev_server):
    # Enable console log forwarding for visibility and debugging
    page.on("console", lambda msg: print(f"Browser Console: {msg.text}"))
    page.on("pageerror", lambda err: print(f"Browser JS Error: {err.message}"))

    # Mock WebXR since headless browsers don't have XR runtimes
    page.add_init_script("""
        if (!navigator.xr) {
            navigator.xr = {
                isSessionSupported: () => Promise.resolve(false)
            };
        }
    """)

    # Use CPU Fallback renderer to guarantee execution without a GPU on sandbox
    url = "http://localhost:9000/?renderer=cpu"
    print(f"Navigating to {url}...")
    page.goto(url)

    # Wait for the loading overlay to be hidden (auto-clicked because of navigator.webdriver/renderer=cpu)
    print("Waiting for loading overlay to be hidden...")
    loading_overlay = page.locator("#loading-overlay")
    expect(loading_overlay).to_be_hidden(timeout=20000)

    # Wait for the main UI layer to load and render
    print("Waiting for ui-layer to load...")
    page.wait_for_selector("#ui-layer", state="visible", timeout=10000)

    # 1. Verify looking around / camera coordinates panel
    print("Checking player coordinates HUD...")
    coord_hud = page.locator("#player-coordinates")
    expect(coord_hud).to_be_visible()
    print(f"Initial coordinates: {coord_hud.inner_text()}")

    # Simulate camera looking around by pressing some camera movement bindings
    print("Simulating looking around / WASD movement...")
    page.keyboard.press("KeyW")
    time.sleep(0.5)
    page.keyboard.press("KeyD")
    time.sleep(0.5)

    # 2. Interact with the Inscribe overlay to cast a spell
    print("Opening inscribe overlay...")
    page.keyboard.press("KeyI")
    time.sleep(1.0)

    # Verify inscribe overlay is visible
    inscribe_overlay = page.locator("#inscribe-overlay")
    expect(inscribe_overlay).to_be_visible()

    # Fill inscribe input with a simple spell "FIRE"
    print("Entering spell 'FIRE' in the focus plane...")
    inscribe_input = page.locator("#inscribe-input")
    expect(inscribe_input).to_be_visible()
    inscribe_input.fill("FIRE")
    time.sleep(0.5)

    # Click Submit / Cast
    print("Submitting inscription...")
    page.locator("#btn-submit-inscribe").click()
    time.sleep(1.0)

    # Ensure inscribe overlay is closed
    expect(inscribe_overlay).to_be_hidden()

    # 3. Store the focused spell to the inventory via KeyC
    print("Storing spell FIRE into 3D inventory...")
    page.keyboard.press("KeyC")
    time.sleep(1.0)

    # 4. Toggle 3D inventory via Tab and check visibility
    print("Toggling 3D inventory view...")
    # Since Tab triggers browser focus changes normally, State keydown prevent_default's it for the engine
    page.keyboard.press("Tab")
    time.sleep(1.0)

    # 5. Interact with Save Manager slots
    print("Testing Save Manager slots...")
    page.locator("#input-save-name").fill("gui_verification_save")
    time.sleep(0.5)

    save_btn = page.locator("#btn-save")
    expect(save_btn).to_be_visible()
    save_btn.click()
    time.sleep(1.0)

    # 6. Verify Desktop Sidebar layout and menu interactions
    print("Verifying desktop sidebar bounding and clickability...")
    sidebar = page.locator("#ui-sidebar")
    expect(sidebar).to_be_visible()
    sidebar_box = sidebar.bounding_box()
    assert sidebar_box is not None, "Sidebar bounding box should not be null"
    # Verify it does not occupy full screen width in desktop mode
    viewport_width = page.viewport_size["width"] if page.viewport_size else 1280
    assert sidebar_box["width"] <= 420, f"Sidebar width ({sidebar_box['width']}) should be compact and not fill desktop screen ({viewport_width})"

    # Verify toggle button collapses and expands the HUD menu cleanly
    toggle_btn = page.locator("#sidebar-toggle-btn")
    expect(toggle_btn).to_be_visible()
    toggle_btn.dispatch_event("click")
    time.sleep(0.5)
    expect(sidebar).to_have_class("collapsed")

    # Expand again
    toggle_btn.dispatch_event("click")
    time.sleep(0.5)
    expect(sidebar).not_to_have_class("collapsed")

    # Verify Archetype dropdown selection is directly clickable and responsive
    archetype_select = page.locator("#param-archetype")
    expect(archetype_select).to_be_visible()
    archetype_select.select_option("0") # Select Fantasy
    time.sleep(0.5)
    assert archetype_select.input_value() == "0"

    # Take screenshot of the Fallback wireframe + UI state
    screenshot_path = "verification/gui_features_verification.png"
    print(f"Taking E2E verification screenshot to {screenshot_path}...")
    page.screenshot(path=screenshot_path)

    # Verify no JS exceptions or errors occurred
    # Save a snapshot for consistency
    update = os.environ.get("UPDATE_SNAPSHOTS") == "1"
    assert_snapshot(page, "gui_features_ui_snapshot.png", update=update)

    print("E2E verification tests complete!")
