use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Action {
    MoveForward,
    MoveBackward,
    MoveLeft,
    MoveRight,
    CastSpell,
    // Add more actions as needed
}

impl Action {
    pub fn from_string(s: &str) -> Option<Self> {
        match s {
            "MoveForward" => Some(Action::MoveForward),
            "MoveBackward" => Some(Action::MoveBackward),
            "MoveLeft" => Some(Action::MoveLeft),
            "MoveRight" => Some(Action::MoveRight),
            "CastSpell" => Some(Action::CastSpell),
            _ => None,
        }
    }

    pub fn to_string(&self) -> String {
        match self {
            Action::MoveForward => "MoveForward".to_string(),
            Action::MoveBackward => "MoveBackward".to_string(),
            Action::MoveLeft => "MoveLeft".to_string(),
            Action::MoveRight => "MoveRight".to_string(),
            Action::CastSpell => "CastSpell".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputConfig {
    pub bindings: HashMap<Action, String>, // Action -> KeyCode (e.g. "KeyW")
    pub reverse_bindings: HashMap<String, Action>, // KeyCode -> Action
}

impl Default for InputConfig {
    fn default() -> Self {
        let mut bindings = HashMap::new();
        bindings.insert(Action::MoveForward, "KeyW".to_string());
        bindings.insert(Action::MoveBackward, "KeyS".to_string());
        bindings.insert(Action::MoveLeft, "KeyA".to_string());
        bindings.insert(Action::MoveRight, "KeyD".to_string());
        bindings.insert(Action::CastSpell, "KeyF".to_string());

        let mut config = Self {
            bindings,
            reverse_bindings: HashMap::new(),
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
