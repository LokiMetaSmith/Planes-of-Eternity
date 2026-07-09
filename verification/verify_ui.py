from playwright.sync_api import sync_playwright
import time

def verify():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Mock WASM globals to avoid initialization crashes if they aren't fully loaded
        page.add_init_script("""
            window.gameClient = {
                get_all_custom_spell_bindings_json: () => '{}',
                get_key_binding: () => '',
                list_saves: () => '[]',
                get_network_status: () => '{}'
            };
        """)

        page.goto("http://localhost:9000")
        time.sleep(2)  # Wait for scripts to run

        # Take screenshot of initial state (should show "REALITY SYNCHRONIZED" and Enter button)
        page.screenshot(path="verification/boot.png")

        # Click Enter
        page.click("#btn-start-engine")
        time.sleep(1)

        # Take screenshot after entering (should show main UI)
        page.screenshot(path="verification/ingame.png")

        # Trigger Pointer Lock Exit to show Resume Overlay
        page.evaluate("document.exitPointerLock()")
        time.sleep(1)
        page.screenshot(path="verification/resume.png")

        browser.close()

if __name__ == "__main__":
    verify()
