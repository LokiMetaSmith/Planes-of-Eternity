use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Action {
    MoveForward,
    MoveBackward,
    MoveLeft,
    MoveRight,
    Jump,
    Descend,
    CastSpell,
    Inscribe,
    ToggleAutoReduce,
    Step,
    TogglePause,
    VoxelDiffusion,
    VoxelTimeReverse,
    VoxelDream,
    VoxelDiffuse,
    DropItem,
    PickupItem,
    // Add more actions as needed
}

impl Action {
    pub fn from_string(s: &str) -> Option<Self> {
        match s {
            "MoveForward" => Some(Action::MoveForward),
            "MoveBackward" => Some(Action::MoveBackward),
            "MoveLeft" => Some(Action::MoveLeft),
            "MoveRight" => Some(Action::MoveRight),
            "Jump" => Some(Action::Jump),
            "Descend" => Some(Action::Descend),
            "CastSpell" => Some(Action::CastSpell),
            "Inscribe" => Some(Action::Inscribe),
            "ToggleAutoReduce" => Some(Action::ToggleAutoReduce),
            "Step" => Some(Action::Step),
            "TogglePause" => Some(Action::TogglePause),
            "VoxelDiffusion" => Some(Action::VoxelDiffusion),
            "VoxelTimeReverse" => Some(Action::VoxelTimeReverse),
            "VoxelDream" => Some(Action::VoxelDream),
            "VoxelDiffuse" => Some(Action::VoxelDiffuse),
            "DropItem" => Some(Action::DropItem),
            "PickupItem" => Some(Action::PickupItem),
            _ => None,
        }
    }
}

impl std::fmt::Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Action::MoveForward => "MoveForward",
            Action::MoveBackward => "MoveBackward",
            Action::MoveLeft => "MoveLeft",
            Action::MoveRight => "MoveRight",
            Action::Jump => "Jump",
            Action::Descend => "Descend",
            Action::CastSpell => "CastSpell",
            Action::Inscribe => "Inscribe",
            Action::ToggleAutoReduce => "ToggleAutoReduce",
            Action::Step => "Step",
            Action::TogglePause => "TogglePause",
            Action::VoxelDiffusion => "VoxelDiffusion",
            Action::VoxelTimeReverse => "VoxelTimeReverse",
            Action::VoxelDream => "VoxelDream",
            Action::VoxelDiffuse => "VoxelDiffuse",
            Action::DropItem => "DropItem",
            Action::PickupItem => "PickupItem",
        };
        write!(f, "{}", s)
    }
}

use crate::reality_types::RealityArchetype;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputConfig {
    pub bindings: HashMap<Action, String>, // Action -> KeyCode (e.g. "KeyW")
    pub reverse_bindings: HashMap<String, Action>, // KeyCode -> Action
    #[serde(default)]
    pub archetype_filters: HashMap<RealityArchetype, RealityArchetype>, // Blacklist -> Replacement
}

impl Default for InputConfig {
    fn default() -> Self {
        let mut bindings = HashMap::new();
        bindings.insert(Action::MoveForward, "KeyW".to_string());
        bindings.insert(Action::MoveBackward, "KeyS".to_string());
        bindings.insert(Action::MoveLeft, "KeyA".to_string());
        bindings.insert(Action::MoveRight, "KeyD".to_string());
        bindings.insert(Action::Jump, "Space".to_string());
        bindings.insert(Action::Descend, "ShiftLeft".to_string());
        bindings.insert(Action::CastSpell, "KeyF".to_string());
        bindings.insert(Action::Inscribe, "KeyI".to_string());
        bindings.insert(Action::ToggleAutoReduce, "KeyR".to_string());
        bindings.insert(Action::Step, "KeyE".to_string());
        bindings.insert(Action::TogglePause, "KeyP".to_string());
        bindings.insert(Action::VoxelDiffusion, "KeyY".to_string());
        bindings.insert(Action::VoxelTimeReverse, "KeyT".to_string());
        bindings.insert(Action::VoxelDream, "KeyG".to_string());
        bindings.insert(Action::VoxelDiffuse, "KeyH".to_string());
        bindings.insert(Action::DropItem, "KeyB".to_string());
        bindings.insert(Action::PickupItem, "KeyQ".to_string());

        let mut config = Self {
            bindings,
            reverse_bindings: HashMap::new(),
            archetype_filters: HashMap::new(),
        };
        config.update_reverse_bindings();
        config
    }
}

impl InputConfig {
    pub fn update_reverse_bindings(&mut self) {
        self.reverse_bindings.clear();
        for (action, key) in &self.bindings {
            self.reverse_bindings.insert(key.clone(), *action);
        }
    }

    pub fn set_binding(&mut self, action: Action, key_code: String) {
        // Remove old key for this action if needed?
        // Or if key is already used, unbind it?
        // Simple overwrite for now.

        // If key is used by another action, remove that binding
        if let Some(existing_action) = self.reverse_bindings.get(&key_code) {
            self.bindings.remove(existing_action);
        }

        self.bindings.insert(action, key_code);
        self.update_reverse_bindings();
    }

    pub fn get_binding(&self, action: Action) -> Option<&String> {
        self.bindings.get(&action)
    }

    pub fn map_key(&self, key_code: &str) -> Option<Action> {
        self.reverse_bindings.get(key_code).cloned()
    }
}
