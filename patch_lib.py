import re

with open("reality-engine/src/lib.rs", "r") as f:
    content = f.read()

wasm_methods = """
    pub fn get_key_binding(&self, action_name: js_sys::JsString) -> String {
"""

new_wasm_methods = """
    pub fn set_custom_spell_binding(&self, key_code: js_sys::JsString, spell_str: js_sys::JsString) {
        let mut state = self.state.borrow_mut();
        const MAX_KEY_CODE_LEN: u32 = 64;
        const MAX_SPELL_LEN: u32 = 256;
        if key_code.length() > MAX_KEY_CODE_LEN || spell_str.length() > MAX_SPELL_LEN {
            log::warn!("Security Warning: Key code or spell string exceeded length limit. Rejecting set_custom_spell_binding.");
            return;
        }
        let key_code_str: String = key_code.into();
        let spell_string: String = spell_str.into();
        state.engine.input_config.custom_spell_bindings.insert(key_code_str, spell_string);
        self.save_state(&state);
    }

    pub fn get_custom_spell_binding(&self, key_code: js_sys::JsString) -> String {
        let state = self.state.borrow();
        const MAX_KEY_CODE_LEN: u32 = 64;
        if key_code.length() > MAX_KEY_CODE_LEN {
            return "".to_string();
        }
        let key_code_str: String = key_code.into();
        if let Some(spell) = state.engine.input_config.custom_spell_bindings.get(&key_code_str) {
            return spell.clone();
        }
        "".to_string()
    }

    pub fn remove_custom_spell_binding(&self, key_code: js_sys::JsString) {
        let mut state = self.state.borrow_mut();
        const MAX_KEY_CODE_LEN: u32 = 64;
        if key_code.length() > MAX_KEY_CODE_LEN {
            return;
        }
        let key_code_str: String = key_code.into();
        state.engine.input_config.custom_spell_bindings.remove(&key_code_str);
        self.save_state(&state);
    }

    pub fn get_all_custom_spell_bindings_json(&self) -> String {
        let state = self.state.borrow();
        serde_json::to_string(&state.engine.input_config.custom_spell_bindings).unwrap_or_else(|_| "{}".to_string())
    }

    pub fn get_key_binding(&self, action_name: js_sys::JsString) -> String {
"""

content = content.replace(wasm_methods, new_wasm_methods)

with open("reality-engine/src/lib.rs", "w") as f:
    f.write(content)
