import pytest
from playwright.sync_api import Page, expect

@pytest.mark.usefixtures("dev_server")
def test_browser_ai_toggle(page: Page):
    """
    Tests that the 'ENABLE BROWSER AI' button correctly toggles state and updates its text.
    """
    page.goto("http://localhost:9000/")

    # Wait for the engine loading overlay to hide
    loading_overlay = page.locator("#loading-overlay")
    loading_overlay.wait_for(state="hidden", timeout=15000)

    # Find the AI toggle button
    btn = page.locator("#btn-toggle-ai")

    # Check initial state
    expect(btn).to_be_visible()
    expect(btn).to_have_text("ENABLE BROWSER AI")

    # Toggle on
    btn.click()
    expect(btn).to_have_text("DISABLE BROWSER AI")

    # Wait a short moment to ensure interval doesn't crash the page immediately
    page.wait_for_timeout(500)

    # Toggle off
    btn.click()
    expect(btn).to_have_text("ENABLE BROWSER AI")
