from playwright.sync_api import sync_playwright, expect
import time

def verify_cpu_renderer():
    print("Launching browser...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Go to URL with forced CPU renderer
        url = "http://localhost:9000/?renderer=cpu"
        print(f"Navigating to {url}...")
        page.goto(url)

        # Wait for the engine loading overlay to hide
        print("Waiting for loading overlay to be hidden...")
        loading_overlay = page.locator("#loading-overlay")
        loading_overlay.wait_for(state="hidden", timeout=20000)

        print("Waiting for the UI layer to render...")
        page.wait_for_selector("#ui-layer", state="visible")

        # Give the renderer a couple seconds to draw
        print("Waiting for draw...")
        time.sleep(3)

        # Take screenshot of the visual fallback
        screenshot_path = "/home/jules/verification/cpu_renderer_fallback.png"
        print(f"Taking screenshot to {screenshot_path}...")
        page.screenshot(path=screenshot_path)

        print("Closing browser.")
        browser.close()

if __name__ == "__main__":
    verify_cpu_renderer()
