from playwright.sync_api import Page, expect
import time
from test_utils import assert_snapshot
import os

def test_genie_ui(page: Page):
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

    # Wait for the ui layer to be visible
    page.wait_for_selector("#ui-layer", state="visible")

    # Check for the dropdown
    select = page.locator("#param-archetype")
    expect(select).to_be_visible()

    # Check for the new option
    option = select.locator("option[value='5']")
    expect(option).to_have_text("Genie (Dream)")

    # Select it
    select.select_option("5")

    # Verify it is selected
    expect(select).to_have_value("5")

    # We wait just to make sure the screenshot captures the effect
    time.sleep(2)

    update = os.environ.get("UPDATE_SNAPSHOTS") == "1"
    assert_snapshot(page, "genie_ui.png", update=update)
