import re

with open("reality-engine/index.html", "r") as f:
    content = f.read()

# Add Custom Spell Bindings UI to Keybinds Overlay
search_ui = """            <div style="text-align: center; color: #888; font-size: 0.8rem;">
                Click a binding, then press a key to rebind.
            </div>
        </div>
    </div>"""

replace_ui = """            <div style="text-align: center; color: #888; font-size: 0.8rem;">
                Click a binding, then press a key to rebind.
            </div>

            <div class="cyber-header" style="font-size: 1rem; margin-top: 20px; border-bottom: none; margin-bottom: 10px;" role="heading" aria-level="3">CUSTOM SPELLS</div>
            <div id="custom-spell-list" style="margin-bottom: 10px;">
            </div>
            <div style="display: flex; gap: 5px; margin-bottom: 10px;">
                <input type="text" id="custom-spell-key" placeholder="Key (e.g. Digit1)" style="flex: 1; padding: 5px;" aria-label="Key Code">
                <input type="text" id="custom-spell-term" placeholder="Lambda (e.g. FIRE)" style="flex: 2; padding: 5px;" aria-label="Lambda Expression">
                <button id="btn-add-custom-spell" class="cyber-button" style="width: auto; padding: 5px 10px;">ADD</button>
            </div>
            <div style="text-align: center; color: #888; font-size: 0.8rem;">
                Bind keys to custom Lambda expressions.
            </div>
        </div>
    </div>"""

content = content.replace(search_ui, replace_ui)

# Add JavaScript logic for Custom Spell Bindings
search_js = """                function renderKeybinds() {
                    const focusedAction = document.activeElement?.getAttribute('data-action');"""

replace_js = """                const customSpellList = document.getElementById('custom-spell-list');
                const btnAddCustomSpell = document.getElementById('btn-add-custom-spell');
                const customSpellKey = document.getElementById('custom-spell-key');
                const customSpellTerm = document.getElementById('custom-spell-term');

                function renderCustomSpells() {
                    customSpellList.innerHTML = '';
                    try {
                        const json = gameClient.get_all_custom_spell_bindings_json();
                        const spells = JSON.parse(json);
                        for (const [key, term] of Object.entries(spells)) {
                            const row = document.createElement('div');
                            row.style.display = 'flex';
                            row.style.justifyContent = 'space-between';
                            row.style.padding = '8px 0';
                            row.style.borderBottom = '1px solid #333';
                            row.style.color = '#ccc';
                            row.style.fontFamily = 'monospace';

                            const label = document.createElement('span');
                            label.innerText = `${key}: ${term}`;

                            const btnRemove = document.createElement('button');
                            btnRemove.innerText = 'X';
                            btnRemove.style.background = '#440000';
                            btnRemove.style.color = '#fff';
                            btnRemove.style.border = '1px solid #ff0000';
                            btnRemove.style.cursor = 'pointer';
                            btnRemove.onclick = () => {
                                gameClient.remove_custom_spell_binding(key);
                                renderCustomSpells();
                            };

                            row.appendChild(label);
                            row.appendChild(btnRemove);
                            customSpellList.appendChild(row);
                        }
                    } catch (e) {
                        console.error("Failed to parse custom spells", e);
                    }
                }

                btnAddCustomSpell.addEventListener('click', () => {
                    const key = customSpellKey.value.trim();
                    const term = customSpellTerm.value.trim();
                    if (key && term) {
                        gameClient.set_custom_spell_binding(key, term);
                        customSpellKey.value = '';
                        customSpellTerm.value = '';
                        renderCustomSpells();
                    }
                });

                function renderKeybinds() {
                    renderCustomSpells();
                    const focusedAction = document.activeElement?.getAttribute('data-action');"""

content = content.replace(search_js, replace_js)

with open("reality-engine/index.html", "w") as f:
    f.write(content)
