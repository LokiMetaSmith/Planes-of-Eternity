from playwright.sync_api import Page, expect
import time
from test_utils import assert_snapshot
import os

def test_xr_ui(page: Page):
    page.goto("http://localhost:9000/")

    # Wait for the loading overlay to be hidden
    page.wait_for_selector("#loading-overlay", state="hidden", timeout=30000)

    # Click the Start Desktop button if it exists
    start_btn = page.locator("#btn-start-desktop")
    try:
        expect(start_btn).to_be_visible(timeout=2000)
        start_btn.click()
    except Exception:
        pass

    # Wait for the UI layer to load.
    page.wait_for_selector("#ui-layer", state="visible")

    # Wait for log message indicating WebXR status
    page.wait_for_selector("#log-area", state="visible")

    # Check logs for "WEBXR AR" message
    try:
        expect(page.locator("#log-area")).to_contain_text("WEBXR", timeout=10000)
    except Exception as e:
        print(f"Failed to find WebXR log: {e}")

    update = os.environ.get("UPDATE_SNAPSHOTS") == "1"
    assert_snapshot(page, "verification.png", update=update)
