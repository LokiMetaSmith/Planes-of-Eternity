use serde::{Serialize, Deserialize};
use crate::projector::RealityProjector;
use web_sys::Storage;
use log::{info, error};

#[derive(Serialize, Deserialize, Debug)]
pub struct GameState {
    pub player_projector: RealityProjector,
    pub anomaly_projector: RealityProjector,
    pub chunk_hash: String, // "Git" root hash
    pub timestamp: u64,
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
                match serde_json::from_str(&json) {
                    Ok(state) => {
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
