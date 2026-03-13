from playwright.sync_api import Page, expect
import time
from test_utils import assert_snapshot
import os

def test_network_ui(page: Page):
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

    # Check if the Network Uplink card exists
    panel = page.locator("#network-status-panel")
    expect(panel).to_be_visible()

    # Check text content
    expect(panel).to_contain_text("Network Uplink")
    expect(panel).to_contain_text("SIGNAL:")
    expect(panel).to_contain_text("PEER ID:")
    expect(panel).to_contain_text("PEERS:")

    update = os.environ.get("UPDATE_SNAPSHOTS") == "1"
    assert_snapshot(page, "network_ui.png", update=update)
