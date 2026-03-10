from playwright.sync_api import Page, expect
import time

def test_vaporwave_ui(page: Page):
    page.goto("http://localhost:9000/")

    # Wait for the ui layer to be visible
    page.wait_for_selector("#ui-layer", state="visible")

    # Check for the dropdown
    select = page.locator("#param-archetype")
    expect(select).to_be_visible()

    # Check for the new option
    option = select.locator("option[value='8']")
    expect(option).to_have_text("Vaporwave")

    # Click the Start Desktop button if it exists and is visible
    start_btn = page.locator("#btn-start-desktop")
    if start_btn.is_visible():
        start_btn.click()

    # Select it
    select.select_option("8")

    # Verify it is selected
    expect(select).to_have_value("8")

    # We wait just to make sure the screenshot captures the effect
    time.sleep(2)

    # Take screenshot
    page.screenshot(path="verification/vaporwave_ui.png")
