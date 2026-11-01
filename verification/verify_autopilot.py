import time
from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 1200})
        page = context.new_page()

        page.goto("http://127.0.0.1:8080/?renderer=cpu")
        page.wait_for_selector("#ui-layer", state="visible", timeout=30000)
        time.sleep(3)

        # Scroll sidebar so World Model (DiT) card is visible at the top
        page.evaluate("""() => {
            const wmCard = document.querySelector('.cyber-header[role="heading"]');
            const headers = Array.from(document.querySelectorAll('.cyber-header'));
            const wmHeader = headers.find(h => h.innerText.includes('World Model (DiT)'));
            if (wmHeader && wmHeader.parentElement) {
                wmHeader.parentElement.scrollIntoView({ behavior: 'instant', block: 'start' });
            }
        }""")
        time.sleep(1)

        # Programmatically click autopilot button
        page.evaluate("document.getElementById('btn-wm-autopilot').click()")
        time.sleep(1)

        page.screenshot(path="verification/verification_autopilot.png")
        print("Autopilot screenshot saved to verification/verification_autopilot.png")
        browser.close()

if __name__ == "__main__":
    run()
