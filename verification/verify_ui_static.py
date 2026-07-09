from playwright.sync_api import sync_playwright
import time

def verify():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Use file:// to look at the index.html directly without running WASM
        import os
        path = os.path.abspath("reality-engine/index.html")
        page.goto(f"file://{path}")

        # 1. Loading state (default)
        page.screenshot(path="verification/static_loading.png")

        # 2. Mock "Synchronized" state
        page.evaluate("""() => {
            document.getElementById('loading-text').innerText = "REALITY SYNCHRONIZED";
            document.querySelector('.loading-spinner').style.display = 'none';
            document.getElementById('btn-start-engine').style.display = 'block';
        }""")
        page.screenshot(path="verification/static_sync.png")

        # 3. Mock Resume state
        page.evaluate("""() => {
            document.getElementById('loading-overlay').style.display = 'none';
            document.getElementById('resume-overlay').style.display = 'flex';
        }""")
        page.screenshot(path="verification/static_resume.png")

        browser.close()

if __name__ == "__main__":
    verify()
