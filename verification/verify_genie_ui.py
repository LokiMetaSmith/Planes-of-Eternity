from playwright.sync_api import sync_playwright, expect

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto("http://localhost:8080/index.html")

            # Check for the dropdown
            select = page.locator("#param-archetype")
            expect(select).to_be_visible()

            # Check for the new option
            # The option text is "Genie (Dream)" and value is "5"
            # We can check if the option exists
            option = select.locator("option[value='5']")
            expect(option).to_have_text("Genie (Dream)")

            # Select it
            select.select_option("5")

            # Verify it is selected
            expect(select).to_have_value("5")

            print("Verified Genie option exists and is selectable.")

            # Take screenshot
            page.screenshot(path="verification/genie_ui.png")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    run()
