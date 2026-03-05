from playwright.sync_api import Page, expect

def test_network_ui(page: Page):
    page.goto("http://localhost:9000/")

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
