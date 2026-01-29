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
