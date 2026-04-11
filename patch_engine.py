import re

with open("reality-engine/src/engine.rs", "r") as f:
    content = f.read()

# Add handling for custom_spell_bindings
search = """    pub fn process_keyboard(&mut self, key_code: &str, pressed: bool) {
        // Map raw key to Action
        if let Some(action) = self.input_config.map_key(key_code) {
            // Send engine actions via lambda evaluation for scriptability"""

replace = """    pub fn process_keyboard(&mut self, key_code: &str, pressed: bool) {
        // Check for custom spell bindings first
        if let Some(spell_str) = self.input_config.custom_spell_bindings.get(key_code) {
            if pressed {
                log::info!("Casting Custom Spell from Binding: {}", spell_str);
                if let Some(term) = crate::lambda::parse(spell_str) {
                    self.audio.play_cast();
                    if let Some(anomaly) = self.compile_spell(term) {
                        self.world_state.add_anomaly(anomaly.clone());
                        log::info!("Custom Spell Cast Successfully: {:?}", anomaly.reality_signature.active_style.archetype);

                        let color = match anomaly.reality_signature.active_style.archetype {
                            crate::reality_types::RealityArchetype::SciFi => [0.0, 1.0, 1.0, 1.0],
                            crate::reality_types::RealityArchetype::Horror => [1.0, 0.0, 0.0, 1.0],
                            crate::reality_types::RealityArchetype::Fantasy => [0.0, 1.0, 0.0, 1.0],
                            crate::reality_types::RealityArchetype::Toon => [1.0, 1.0, 0.0, 1.0],
                            crate::reality_types::RealityArchetype::HyperNature => [0.0, 0.8, 0.2, 1.0],
                            crate::reality_types::RealityArchetype::Genie => [1.0, 0.0, 1.0, 1.0],
                            crate::reality_types::RealityArchetype::Void => [0.1, 0.1, 0.1, 1.0],
                            crate::reality_types::RealityArchetype::Glitch => [0.0, 1.0, 0.0, 1.0],
                            crate::reality_types::RealityArchetype::Steampunk => [0.8, 0.5, 0.2, 1.0],
                            crate::reality_types::RealityArchetype::Vaporwave => [1.0, 0.0, 1.0, 1.0],
                            crate::reality_types::RealityArchetype::Noir => [0.5, 0.5, 0.5, 1.0],
                            crate::reality_types::RealityArchetype::CyberSpace => [0.0, 1.0, 1.0, 1.0],
                            crate::reality_types::RealityArchetype::Dream => [0.8, 0.6, 1.0, 1.0],
                            crate::reality_types::RealityArchetype::ObraDinn => [0.9, 0.9, 0.8, 1.0],
                            crate::reality_types::RealityArchetype::SolarPunk => [0.2, 0.9, 0.4, 1.0],
                            crate::reality_types::RealityArchetype::Biopunk => [0.8, 0.2, 0.4, 1.0],
                        };

                        self.spell_effects.push(SpellEffect {
                            position: anomaly.location,
                            color,
                            scale: anomaly.reality_signature.active_style.scale,
                            timer: 0.0,
                            max_time: 1.5,
                        });
                    }
                }
            }
            return;
        }

        // Map raw key to Action
        if let Some(action) = self.input_config.map_key(key_code) {
            // Send engine actions via lambda evaluation for scriptability"""

content = content.replace(search, replace)

with open("reality-engine/src/engine.rs", "w") as f:
    f.write(content)
