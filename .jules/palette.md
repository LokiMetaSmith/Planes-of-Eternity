## 2026-03-16 - [Dynamic aria-label Updates on Keybind Buttons]
**Learning:** Screen readers do not automatically announce visual text changes in interactive elements when focus remains static, leading to situations where users in a "rebinding" state see "PRESS KEY..." visually but still hear the old label via assistive technologies.
**Action:** When creating interactable elements with dynamically shifting visual text (such as keybind buttons entering an active "awaiting input" state), explicitly update the `aria-label` via JavaScript (`setAttribute`) to ensure the audible label correctly matches the semantic meaning of the new visual state.

## 2026-03-23 - [A11y Performance: 60fps DOM Updates]
**Learning:** Overlays that rapidly update the DOM (like 60fps label rendering engines or video feeds) cause screen readers to constantly recalculate the accessibility tree, leading to severe performance degradation and "tree thrashing".
**Action:** Always add `aria-hidden="true"` to containers that act as high-frequency rendering targets or visual-only overlays to completely prune them from the accessibility tree.
