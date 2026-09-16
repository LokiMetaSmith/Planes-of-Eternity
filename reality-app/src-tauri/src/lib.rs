use tauri::Manager;
use steamworks::{Client, SingleClient};
use std::sync::Mutex;

// Global Steam Client state
struct SteamClientState {
    client: Mutex<Option<Client>>,
}

#[tauri::command]
fn init_steamworks(app_id: u32, state: tauri::State<'_, SteamClientState>) -> Result<String, String> {
    match Client::init_app(app_id) {
        Ok((client, single_client)) => {
            let mut client_state = state.client.lock().unwrap();
            *client_state = Some(client.clone());

            // Spawn a thread to run callbacks
            std::thread::spawn(move || {
                loop {
                    single_client.run_callbacks();
                    std::thread::sleep(std::time::Duration::from_millis(16));
                }
            });

            Ok(format!("Steamworks initialized for app {}", app_id))
        },
        Err(e) => Err(format!("Failed to initialize Steamworks: {:?}", e)),
    }
}

#[tauri::command]
fn is_steam_running(state: tauri::State<'_, SteamClientState>) -> bool {
    state.client.lock().unwrap().is_some()
}

#[tauri::command]
fn get_steam_id(state: tauri::State<'_, SteamClientState>) -> Result<String, String> {
    if let Some(client) = &*state.client.lock().unwrap() {
        Ok(client.user().steam_id().raw().to_string())
    } else {
        Err("Steamworks not initialized".into())
    }
}

#[tauri::command]
fn get_steam_name(state: tauri::State<'_, SteamClientState>) -> Result<String, String> {
    if let Some(client) = &*state.client.lock().unwrap() {
        Ok(client.friends().name())
    } else {
        Err("Steamworks not initialized".into())
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(SteamClientState {
            client: Mutex::new(None),
        })
        .invoke_handler(tauri::generate_handler![
            init_steamworks,
            is_steam_running,
            get_steam_id,
            get_steam_name
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
