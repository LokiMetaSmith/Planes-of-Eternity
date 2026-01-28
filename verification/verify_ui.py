from playwright.sync_api import sync_playwright, expect

def verify_ui(page):
    # Navigate to the page
    # Note: reality-engine/index.html is at /reality-engine/index.html
    page.goto("http://localhost:8000/reality-engine/index.html")

    # Wait for the UI layer to appear
    ui_layer = page.locator("#ui-layer")
    expect(ui_layer).to_be_visible()

    # Check for the Archetype selector
    archetype_select = page.locator("#param-archetype")
    expect(archetype_select).to_be_visible()

    # Verify options
    options = archetype_select.locator("option")
    expect(options).to_have_count(4)

    # Verify default selection (SciFi = 1)
    expect(archetype_select).to_have_value("1")

    # Select Fantasy (0)
    archetype_select.select_option("0")
    expect(archetype_select).to_have_value("0")

    # Take screenshot
    page.screenshot(path="verification/ui_screenshot.png")
    print("Screenshot saved to verification/ui_screenshot.png")

if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            verify_ui(page)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()
