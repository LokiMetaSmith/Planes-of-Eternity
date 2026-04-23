from playwright.sync_api import sync_playwright, expect

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--use-gl=angle",
                "--use-angle=gl",
                "--enable-webgl",
                "--ignore-gpu-blocklist",
                "--enable-features=Vulkan",
                "--enable-unsafe-webgpu",
                "--use-vulkan=native",
            ]
        )
        context = browser.new_context(record_video_dir="/home/jules/verification/video")
        page = context.new_page()

        try:
            page.goto("http://localhost:9000/")
            page.wait_for_selector("#loading-overlay", state="hidden", timeout=30000)

            # Click the Start Desktop button if it exists
            start_btn = page.locator("#btn-start-desktop")
            try:
                expect(start_btn).to_be_visible(timeout=2000)
                start_btn.click()
            except Exception:
                pass

            page.wait_for_selector("#ui-layer", state="visible")
            page.wait_for_timeout(500)

            # Hover over disabled DEL button
            del_btn = page.locator("#btn-delete")
            del_btn.hover()
            page.wait_for_timeout(500)

            # Hover over RESET button
            reset_btn = page.locator("#btn-reset")
            reset_btn.hover()
            page.wait_for_timeout(500)

            # Open keybinds and hover over [X]
            page.locator("#btn-keybinds").click()
            page.wait_for_selector("#keybind-overlay", state="visible")
            page.wait_for_timeout(500)

            close_btn = page.locator("#btn-close-keybinds")
            close_btn.hover()
            page.wait_for_timeout(500)

            page.screenshot(path="/home/jules/verification/verification.png")
            page.wait_for_timeout(1000)

        finally:
            context.close()
            browser.close()

if __name__ == "__main__":
    run()
