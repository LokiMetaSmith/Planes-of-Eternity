from playwright.sync_api import Page, expect

def test_genie_ui(page: Page):
    page.goto("http://localhost:9000/")

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

    # Take screenshot
    page.screenshot(path="verification/genie_ui.png")
