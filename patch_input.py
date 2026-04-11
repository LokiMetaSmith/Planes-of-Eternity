import re

with open("reality-engine/src/input.rs", "r") as f:
    content = f.read()

# Add to struct
content = re.sub(
    r'(pub archetype_filters: HashMap<RealityArchetype, RealityArchetype>, // Blacklist -> Replacement)',
    r'\1\n    #[serde(default)]\n    pub custom_spell_bindings: HashMap<String, String>, // KeyCode -> Lambda String',
    content
)

# Add to Default
content = re.sub(
    r'(archetype_filters: HashMap::new\(\),\n        };)',
    r'archetype_filters: HashMap::new(),\n            custom_spell_bindings: HashMap::new(),\n        };',
    content
)

with open("reality-engine/src/input.rs", "w") as f:
    f.write(content)
