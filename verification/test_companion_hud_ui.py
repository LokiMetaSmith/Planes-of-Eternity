import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-gpu"])
        context = await browser.new_context(viewport={"width": 1280, "height": 900})
        page = await context.new_page()
        page.on("console", lambda msg: print(f"[CONSOLE]: {msg.text}"))
        page.on("pageerror", lambda err: print(f"[PAGEERROR]: {err}"))

        await page.goto("http://127.0.0.1:8080/?renderer=cpu", timeout=15000)
        await page.wait_for_timeout(2000)

        # Toggle UI on if needed (H key or check visibility)
        await page.keyboard.press("KeyH")
        await page.wait_for_timeout(500)

        # Scroll companion card into view or take screenshot of it directly
        companion_card = page.locator("#companion-card")
        if await companion_card.count() > 0:
            await companion_card.scroll_into_view_if_needed()
            await companion_card.screenshot(path="verification/companion_card_element.png")
            print("Captured companion card element screenshot!")

        await page.screenshot(path="verification/companion_page_full.png", full_page=True)
        print("Captured full page screenshot!")

        await browser.close()

asyncio.run(main())
