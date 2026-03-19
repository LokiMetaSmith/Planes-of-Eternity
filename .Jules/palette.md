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
