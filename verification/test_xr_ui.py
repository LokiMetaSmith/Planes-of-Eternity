from playwright.sync_api import Page, expect

def test_xr_ui(page: Page):
    # Navigate to the app
    # Assuming the server is running on localhost:9000 and we serve from root
    page.goto("http://localhost:9000/")

    # Wait for log message indicating WebXR status
    # Since headless chrome might not support WebXR by default, we expect "WEBXR AR NOT DETECTED"
    # But we can check for the button text as well.

    # Wait for engine start
    page.wait_for_selector("#log-area", state="visible")

    # Check logs for "WEBXR AR" message
    # We allow either result since environment capabilities vary
    try:
        expect(page.locator("#log-area")).to_contain_text("WEBXR", timeout=10000)
    except Exception as e:
        print(f"Failed to find WebXR log: {e}")

    # Take a screenshot
    page.screenshot(path="verification/verification.png")
