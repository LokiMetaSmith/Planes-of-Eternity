from playwright.sync_api import Page, expect
import time

def test_inscribe_ui(page: Page):
    page.goto("http://localhost:9000/")

    # Define a variable to track if the dialog was handled properly
    dialog_handled = {"handled": False}

    def handle_dialog(dialog):
        assert dialog.type == "prompt", f"Expected prompt dialog, got {dialog.type}"
        assert "Inscribe Reality" in dialog.message, f"Unexpected message: {dialog.message}"
        dialog.accept("FIRE")
        dialog_handled["handled"] = True

    page.on("dialog", handle_dialog)

    # Wait for the engine to load by verifying the UI layer is visible
    page.wait_for_selector("#ui-layer", state="visible")

    # We must ensure that the WGPU canvas and WASM module is fully initialized
    # "WASM INITIALIZED OK" is in the log-area usually.
    page.wait_for_selector("#log-area")

    # Allow some time for full initialization
    page.wait_for_timeout(2000)

    # Simulate pressing the mapped Inscribe key. Note that we should dispatch a
    # keydown event with the exact `code` that Rust expects (e.g. `KeyI`),
    # Playwright's page.keyboard.press sends `code` based on the key character.
    # For 'i' it sends `KeyI`.
    page.keyboard.press("i")

    # Wait for a short moment to ensure the dialog event had time to fire
    page.wait_for_timeout(1000)

    assert dialog_handled["handled"], "The Inscribe dialog was not triggered by the key press."

    # Look for the resulting HTML overlay labels that the visual lambda system generates
    expect(page.locator("body")).to_contain_text("FIRE", timeout=3000)
