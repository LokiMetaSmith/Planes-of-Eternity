## 2026-03-13 - Stylized UI Accessibility Anti-patterns
**Learning:** In heavily stylized, custom-themed UIs (like 'cybercore'), basic accessibility features like form label associations (`<label for="...">`) and semantic interactive elements (using `<button>` instead of `<span>` for actions like 'Close') are often overlooked in favor of visual aesthetics, degrading screen reader and keyboard navigation experiences.
**Action:** Always verify that form inputs have explicitly associated labels and that interactive pseudo-buttons are replaced with proper semantic HTML elements (e.g., `<button>` with `aria-label`) when refining custom UI themes.

## 2026-03-22 - Stylized UI Focus Accessibility Anti-patterns
**Learning:** In heavily stylized UIs, developers often apply `outline: none` to inputs, selects, and ranges to hide default browser focus rings that clash with the visual aesthetic. However, failing to replace this with a custom `:focus-visible` state completely breaks keyboard navigation by removing all visual indicators of focus.
**Action:** Always ensure that custom styled interactive elements that use `outline: none` include a fallback `:focus-visible` style (e.g., using `border-color` or `box-shadow`) to maintain keyboard accessibility without compromising the design.

## 2026-03-23 - Empty States in Dynamic Select Elements
**Learning:** When dynamically populating `<select>` elements (e.g., a list of save slots), failing to handle an empty array results in a visually collapsed, unusable component that is confusing to users and silent to screen readers.
**Action:** Always insert an explicitly disabled and selected fallback `<option>` (e.g., `<option disabled selected value="">-- NO SAVES --</option>`) to preserve the layout and clearly communicate the empty state to all users.

## 2026-03-24 - Keybinds Modal Focus Management
**Learning:** Custom modal overlays (like `#keybind-overlay`) lacking explicit ARIA attributes (`role="dialog"`, `aria-modal="true"`) and focus management completely trap screen reader and keyboard users, preventing them from accessing or dismissing the dialog properly.
**Action:** Always add `role="dialog"`, `aria-modal="true"`, and an `aria-labelledby` id reference to custom modal overlays. Furthermore, when the modal opens, focus must be shifted programmatically to the dialog (e.g. its close button), and when it closes, focus must be restored to the triggering button.

## 2026-03-24 - Loading Overlay Visibility
**Learning:** Purely visual loading overlays lacking accessibility attributes (like `role="status"` and `aria-busy="true"`) provide visual feedback but leave screen readers completely silent when important async tasks are blocking interaction.
**Action:** Always add `role="status"` and `aria-busy="true"` to full-screen or blocking loading overlays, and add `aria-hidden="true"` to purely visual elements like CSS spinners inside the overlay so they are skipped by screen readers.

## 2026-03-24 - Toggle Button State Announcements
**Learning:** Stylized toggle buttons that convey their active state only via visual styling changes (like changing text from "ENABLE" to "DISABLE" and altering colors) do not inherently communicate their state to assistive technologies, leading to confusion.
**Action:** Always add `aria-pressed="false"` to toggle buttons and programmatically update it to `"true"` or `"false"` when the state changes to explicitly convey the on/off state to screen readers.

## 2026-03-24 - Semantic Headings in Custom UIs
**Learning:** Stylized custom UI dashboards often use `<div>` elements as visual headers (like `.cyber-header`) but fail to provide semantic heading roles, preventing screen reader users from using heading navigation shortcuts to jump between sections.
**Action:** Always add `role="heading"` and an appropriate `aria-level` (e.g., `aria-level="2"`) to div elements acting as structural titles in custom UI layouts.

## 2026-03-24 - Hardware Permission Loading States
**Learning:** Native browser permission prompts (like `getUserMedia` or WebXR) silently block UI execution. If the button that triggers them doesn't immediately enter a loading state, users (and screen readers) receive no feedback and may assume the button is broken.
**Action:** Always immediately set `disabled`, `aria-busy="true"`, and a waiting text state on buttons that trigger native browser hardware permissions, and handle resetting the state on both success and failure to ensure continuous feedback.

## 2026-03-24 - Custom Modal Dismissal Patterns
**Learning:** Custom UI modal overlays that do not support backdrop clicking or Escape key dismissal violate user expectations and severely hamper keyboard-only users, who rely on `Escape` to quickly exit dialogs and cancel intermediate states (like key rebinding) without submitting.
**Action:** Always implement explicit background click dismissal (`e.target === overlay`) and global `Escape` key event listeners to cleanly dismiss custom modals and cancel any active pending actions (like pending key listeners).

## 2026-03-24 - Disabled State Styling and Decorative Element A11y
**Learning:** In custom UIs, disabled elements might inherit hover styles, confusing sighted users, while decorative visual elements (like scanlines or crosshairs) clutter the screen reader experience. Additionally, relying solely on `aria-label` for abbreviated buttons leaves mouse users without context.
**Action:** Always provide explicit disabled styling and exclude disabled items from hover/focus effects (`:not(:disabled)`). Mark purely decorative elements with `aria-hidden="true"`, and duplicate `aria-label` text to `title` attributes on abbreviated buttons to provide mouse tooltips.

## 2026-03-24 - Contextual Copy Actions for Complex IDs
**Learning:** In dynamically generated text strings (like P2P session IDs) presented in custom UIs, users are heavily inconvenienced when forced to manually select and copy text. Additionally, relying solely on global logs for feedback when a copy action occurs separates the user's focus from the action.
**Action:** Always provide inline, contextual copy buttons for complex IDs. Ensure they provide immediate, localized visual feedback (like momentarily turning green and changing text to "COPIED") instead of, or in addition to, global logging to keep the user's attention where the action happened.

## 2026-03-25 - Contextual State Mutation Actions for Action Buttons
**Learning:** In dynamically generated state mutations (like Save and Load functions) presented in custom UIs, users are heavily inconvenienced when forced to check the global log area to ensure the action succeeded. Relying solely on global logs for feedback when a state mutation action occurs separates the user's focus from the action.
**Action:** Always provide inline, contextual success buttons for state mutation actions. Ensure they provide immediate, localized visual feedback (like momentarily turning green and changing text to "SAVED") instead of, or in addition to, global logging to keep the user's attention where the action happened.

## 2026-03-27 - Scrollable Log Area Keyboard Accessibility
**Learning:** Scrollable custom terminal/log containers (`overflow-y: auto`) that lack naturally focusable child elements must be explicitly made focusable using `tabindex="0"`. Without this, keyboard-only and screen reader users cannot scroll to read long outputs.
**Action:** When creating text-heavy custom scrollable areas, always add `tabindex="0"`, a descriptive `aria-label`, and ensure the `:focus-visible` state matches the component's visual styling so keyboard navigation is intuitive.

## 2026-03-28 - AI Sycophancy Leading to Inaccessible UIs
**Learning:** AI models often act as sycophants. If a user asks for a UI component that looks cool but completely breaks accessibility (e.g., removing focus rings entirely, using non-semantic `<div>` buttons without ARIA roles, or ignoring color contrast), the AI's default behavior is to blindly agree and implement the inaccessible design.
**Action:** Practice "healthy skepticism." When acting as Palette, pause and mentally evaluate the user's design request with a "Wait a minute..." prime. If the requested UI or styling violates accessibility standards (WCAG, keyboard navigation, screen reader support), do not blindly implement it. Provide "tough love" by explicitly pointing out the a11y violation and offering an accessible alternative that achieves the same aesthetic. Only implement the inaccessible version if explicitly ordered to after warning the user.

## 2026-03-28 - Focus Restoration on Re-renders
**Learning:** When dynamic lists (like keybinds in `reality-engine`) are destroyed and re-rendered via `innerHTML = ''`, any currently focused element within them is lost, forcing keyboard users back to the `<body>`. This is a critical accessibility issue.
**Action:** Always capture the currently focused element's identifier (e.g., via `document.activeElement.getAttribute('data-action')`) before the re-render, and restore focus (`.focus()`) to the corresponding new element after the render is complete.
