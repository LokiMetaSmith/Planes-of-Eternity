## 2026-03-13 - Stylized UI Accessibility Anti-patterns
**Learning:** In heavily stylized, custom-themed UIs (like 'cybercore'), basic accessibility features like form label associations (`<label for="...">`) and semantic interactive elements (using `<button>` instead of `<span>` for actions like 'Close') are often overlooked in favor of visual aesthetics, degrading screen reader and keyboard navigation experiences.
**Action:** Always verify that form inputs have explicitly associated labels and that interactive pseudo-buttons are replaced with proper semantic HTML elements (e.g., `<button>` with `aria-label`) when refining custom UI themes.

## 2026-03-22 - Stylized UI Focus Accessibility Anti-patterns
**Learning:** In heavily stylized UIs, developers often apply `outline: none` to inputs, selects, and ranges to hide default browser focus rings that clash with the visual aesthetic. However, failing to replace this with a custom `:focus-visible` state completely breaks keyboard navigation by removing all visual indicators of focus.
**Action:** Always ensure that custom styled interactive elements that use `outline: none` include a fallback `:focus-visible` style (e.g., using `border-color` or `box-shadow`) to maintain keyboard accessibility without compromising the design.

## 2026-03-23 - Empty States in Dynamic Select Elements
**Learning:** When dynamically populating `<select>` elements (e.g., a list of save slots), failing to handle an empty array results in a visually collapsed, unusable component that is confusing to users and silent to screen readers.
**Action:** Always insert an explicitly disabled and selected fallback `<option>` (e.g., `<option disabled selected value="">-- NO SAVES --</option>`) to preserve the layout and clearly communicate the empty state to all users.
