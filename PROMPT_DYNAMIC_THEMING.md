# 🎨 Palette's Design Brief: Dynamic UI Theming for Reality Archetypes

**To: UI/UX Development Agent**
**From: Palette 🎨**

The user has requested a highly immersive feature: the UI should dynamically change its aesthetic to match the player's active Reality Archetype (SciFi, Horror, Vaporwave, Steampunk, etc.).

Since this requires a major design system overhaul, introducing dozens of new design tokens and layout changes, it fell outside my daily boundaries as a micro-UX agent. Your mission is to implement this safely, beautifully, and—most importantly—accessibly.

## 🎯 Objectives
1. **Refactor Hardcoded Styles**: Currently, `reality-engine/index.html` relies on hardcoded hex colors (`#00f0ff`, `#ff2a6d`, `#050505`) to achieve its "Cybercore" look. You must extract these into CSS Variables (e.g., `--primary-color`, `--bg-color`, `--border-glow`) scoped to a `.theme-[archetype]` class or similar structure on the `<body>`.
2. **Design Archetype Themes**: Create distinct, readable palettes for the various archetypes available in the `<select id="param-archetype">`.
   - *Example: Vaporwave should use neon pinks and teals, Steampunk should use brass and sepia, Horror should use dark reds and grim greys.*
3. **Listen for State Changes**: Hook into the `gameClient.set_anomaly_archetype(val)` call (or fetch the player's active dominant archetype from the engine) and update the DOM class/variables dynamically so the UI shifts seamlessly when the reality shifts.

## ♿ Critical Accessibility (A11y) Constraints
As Palette, I will not let a cool aesthetic ruin usability. You **MUST** strictly adhere to the following constraints when designing the new color palettes:

- **WCAG AA Contrast Minimums**: Every text color against its background must have a contrast ratio of at least 4.5:1 for normal text, and 3:1 for large text or UI components (like button borders and focus rings).
- **Focus Indicators**: The `outline` and `box-shadow` used for `:focus-visible` must remain highly visible in *every* theme. Do not sacrifice keyboard navigation for "immersion".
- **Semantic Feedback Colors**: Red/Pink (`#ff2a6d`) is currently used for errors/destructive actions (Delete, Reset, Uplink Failed). Green (`#05ffa1`) is used for success (Saved, Loaded). **These semantic meanings must remain consistent and distinct from the primary theme color** so users don't accidentally click "Delete" because it looks like a standard "Vaporwave" button.
- **Background Legibility**: The `.cyber-card` currently uses `rgba(10, 10, 15, 0.85)` with a blur filter to ensure text is readable over the 3D canvas. If you change the background color of these panels, you must ensure the opacity and text shadow still provide adequate contrast against a moving 3D background.

## ⚠️ Implementation Guidelines
- Keep the DOM structure as intact as possible; focus on CSS variable overrides.
- Ensure the `scanline` effect and any visual-only overlays remain marked with `aria-hidden="true"`.
- Verify your changes using Playwright visual tests, running assertions on different theme states.
- If an archetype's ideal aesthetic inherently violates contrast rules (e.g., "dark grey text on black" for Noir), you must provide a "high contrast" fallback or adjust the palette to satisfy WCAG AA requirements. User experience trumps pure aesthetic commitment.

Good luck, and paint carefully!
