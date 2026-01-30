use serde::{Serialize, Deserialize};
use crate::projector::RealityProjector;
use crate::world::WorldState;
use web_sys::Storage;
use log::{info, error};

pub const SAVE_VERSION: u32 = 1;

#[derive(Serialize, Deserialize, Debug)]
pub struct PlayerState {
    pub projector: RealityProjector, // Contains location and self-signature
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GameState {
    pub player: PlayerState,
    pub world: WorldState,
    pub timestamp: u64,
    #[serde(default)]
    pub version: u32,
}

pub fn get_save_key(slot: &str) -> String {
    if slot == "default" || slot.is_empty() {
        "reality_engine_save".to_string()
    } else {
        format!("reality_engine_save_{}", slot)
    }
}

pub fn list_saves() -> Vec<String> {
    let mut saves = Vec::new();
    // Always include default if it exists? Or just list what's there.
    // "default" maps to "reality_engine_save".

    if let Ok(Some(storage)) = get_local_storage() {
        let len = storage.length().unwrap_or(0);
        for i in 0..len {
            if let Ok(Some(key)) = storage.key(i) {
                if key == "reality_engine_save" {
                    saves.push("default".to_string());
                } else if let Some(stripped) = key.strip_prefix("reality_engine_save_") {
                    saves.push(stripped.to_string());
                }
            }
        }
    }
    saves
}

pub fn delete_save(slot: &str) {
    let key = get_save_key(slot);
    if let Ok(Some(storage)) = get_local_storage() {
        if let Err(e) = storage.remove_item(&key) {
            error!("Failed to remove save '{}': {:?}", slot, e);
        } else {
            info!("Deleted save slot: {}", slot);
        }
    }
}

pub fn save_to_local_storage(key: &str, state: &GameState) {
    if let Ok(Some(storage)) = get_local_storage() {
        match serde_json::to_string(state) {
            Ok(json) => {
                if let Err(e) = storage.set_item(key, &json) {
                    error!("Failed to save to local storage: {:?}", e);
                }
            },
            Err(e) => error!("Failed to serialize game state: {:?}", e),
        }
    }
}

pub fn load_from_local_storage(key: &str) -> Option<GameState> {
    if let Ok(Some(storage)) = get_local_storage() {
        match storage.get_item(key) {
            Ok(Some(json)) => {
                match serde_json::from_str::<GameState>(&json) {
                    Ok(state) => {
                        if state.version != SAVE_VERSION {
                            if state.version == 0 {
                                info!("Migrating legacy save (v0) to v{}", SAVE_VERSION);
                            } else {
                                log::warn!("Version mismatch: save is v{}, current is v{}", state.version, SAVE_VERSION);
                            }
                        }
                        info!("Loaded game state from local storage.");
                        Some(state)
                    },
                    Err(e) => {
                        error!("Failed to deserialize game state: {:?}", e);
                        None
                    }
                }
            },
            Ok(None) => {
                info!("No save found for key: {}", key);
                None
            },
            Err(e) => {
                error!("Failed to read from local storage: {:?}", e);
                None
            }
        }
    } else {
        None
    }
}

fn get_local_storage() -> Result<Option<Storage>, web_sys::wasm_bindgen::JsValue> {
    let window = web_sys::window().ok_or("No window found")?;
    window.local_storage()
}
