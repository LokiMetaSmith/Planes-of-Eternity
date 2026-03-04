from playwright.sync_api import Page, expect

def test_save_manager_ui(page: Page):
    # Log any console messages from the browser
    page.on("console", lambda msg: print(f"Browser Console [{msg.type}]: {msg.text}"))
    page.on("pageerror", lambda err: print(f"Browser Error: {err}"))

    # Navigate to the served app
    page.goto("http://localhost:9000/")

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

    # The buttons are covered by the loading overlay if we don't hide it
    # We will forcefully remove the loading overlay so Playwright can interact with UI elements
    # Since WASM isn't loading fully on headless browser due to WebGL limitations.
    page.evaluate("document.getElementById('loading-overlay').style.display = 'none'")

    # Type a save name
    page.locator("#input-save-name").fill("test_save")

    # Take screenshot
    page.screenshot(path="verification/save_manager_ui.png")
