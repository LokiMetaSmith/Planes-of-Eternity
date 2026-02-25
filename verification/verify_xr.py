from playwright.sync_api import sync_playwright, expect

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to the app
        # Assuming the server is running on localhost:8000 and we serve from root
        page.goto("http://localhost:8000/reality-engine/index.html")

        # Wait for log message indicating WebXR status
        # Since headless chrome might not support WebXR by default, we expect "WEBXR AR NOT DETECTED"
        # But we can check for the button text as well.

        # Wait for engine start
        page.wait_for_selector("#log-area", state="visible")

        # Check logs for "WEBXR AR" message
        # We allow either result since environment capabilities vary
        try:
            expect(page.locator("#log-area")).to_contain_text("WEBXR AR", timeout=10000)
            print("WebXR check executed and logged.")
        except Exception as e:
            print(f"Failed to find WebXR log: {e}")

        # Take a screenshot
        page.screenshot(path="verification/verification.png")

        browser.close()

if __name__ == "__main__":
    run()
