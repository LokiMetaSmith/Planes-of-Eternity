from playwright.sync_api import sync_playwright, expect

def test_network_ui():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        # Navigate to the served page. Port 8000 is default.
        # Path is /index.html because we are serving from reality-engine/
        # Note: python -m http.server serves the current directory.
        # So http://localhost:8000/index.html should work.
        page.goto("http://localhost:8000/index.html")

        # Wait for the UI layer to load. It might take a moment for WASM to init,
        # but the HTML structure is static in index.html, so it should be immediate.
        page.wait_for_selector("#ui-layer")

        # Check if the Network Uplink card exists
        panel = page.locator("#network-status-panel")
        expect(panel).to_be_visible()

        # Check text content
        expect(panel).to_contain_text("Network Uplink")
        expect(panel).to_contain_text("SIGNAL:")
        expect(panel).to_contain_text("PEER ID:")
        expect(panel).to_contain_text("PEERS:")

        # Take screenshot
        page.screenshot(path="verification/network_ui.png")

        browser.close()

if __name__ == "__main__":
    test_network_ui()
