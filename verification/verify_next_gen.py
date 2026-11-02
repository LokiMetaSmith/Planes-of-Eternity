import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-web-security",
                "--use-gl=angle",
                "--use-angle=swiftshader",
            ]
        )
        page = await browser.new_page(viewport={"width": 1280, "height": 1024})
        await page.goto("http://localhost:8080/?renderer=cpu", wait_until="networkidle")

        # Wait for ui-layer
        await page.wait_for_selector("#ui-layer", timeout=20000)
        await asyncio.sleep(2)

        # Scroll to newly added controls
        await page.evaluate("""() => {
            const el = document.getElementById('btn-run-rl-benchmark');
            if (el) el.scrollIntoView();
        }""")
        await asyncio.sleep(1)

        await page.screenshot(path="verification/next_gen_world_model_features.png")
        print("Screenshot saved to verification/next_gen_world_model_features.png")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())
