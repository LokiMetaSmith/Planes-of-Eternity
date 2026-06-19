import re

with open("reality-engine/src/lib.rs", "r") as f:
    content = f.read()

new_func = """    pub fn set_anomaly_archetype(&self, archetype_id: i32) {
        let mut state = match self.state.try_borrow_mut() {
            Ok(s) => s,
            Err(_) => return, // Ignore if already borrowed by requestAnimationFrame
        };
        let archetype = match archetype_id {"""

content = re.sub(r'    pub fn set_anomaly_archetype\(&self, archetype_id: i32\) \{\n        let mut state = self\.state\.borrow_mut\(\);\n        let archetype = match archetype_id \{', new_func, content)

with open("reality-engine/src/lib.rs", "w") as f:
    f.write(content)
