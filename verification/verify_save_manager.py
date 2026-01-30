from playwright.sync_api import sync_playwright, expect

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to the served app
        page.goto("http://localhost:8000")

        # Wait for the page to load and WASM to init (roughly)
        # We check if the canvas is there
        expect(page.locator("canvas#reality-canvas")).to_be_visible()

        # Check for Save Manager Panel
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

        # Click Save (this triggers WASM call)
        page.locator("#btn-save").click()

        # Give it a moment for the list to update (if WASM works)
        page.wait_for_timeout(1000)

        # Take screenshot
        page.screenshot(path="verification/save_manager_ui.png")

        print("Verification script finished successfully.")
        browser.close()

if __name__ == "__main__":
    run()
