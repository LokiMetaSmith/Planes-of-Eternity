import re

with open("TODO.md", "r") as f:
    content = f.read()

content = content.replace(
    "- [ ] **Custom Hotkeys UI**: Allow players to bind custom lambda spell expressions to specific hotkeys via the UI.",
    "- [x] **Custom Hotkeys UI**: Allow players to bind custom lambda spell expressions to specific hotkeys via the UI."
)

with open("TODO.md", "w") as f:
    f.write(content)
