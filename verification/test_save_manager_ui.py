from playwright.sync_api import Page, expect
import time
from test_utils import assert_snapshot
import os

def test_save_manager_ui(page: Page):
    # Log any console messages from the browser
    page.on("console", lambda msg: print(f"Browser Console [{msg.type}]: {msg.text}"))
    page.on("pageerror", lambda err: print(f"Browser Error: {err}"))

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

    # Check for Save Manager Panel immediately since HTML structure is mostly static
    save_manager_header = page.get_by_text("Save Manager")
    expect(save_manager_header).to_be_visible()

    # Check inputs and buttons
    expect(page.locator("#input-save-name")).to_be_visible()
    expect(page.locator("#select-save-slot")).to_be_visible()
    expect(page.locator("#btn-save")).to_be_visible()
    expect(page.locator("#btn-load")).to_be_visible()
    expect(page.locator("#btn-delete")).to_be_visible()
    expect(page.locator("#btn-reset")).to_be_visible()

    # Type a save name
    page.locator("#input-save-name").fill("test_save")

    update = os.environ.get("UPDATE_SNAPSHOTS") == "1"
    assert_snapshot(page, "save_manager_ui.png", update=update)
