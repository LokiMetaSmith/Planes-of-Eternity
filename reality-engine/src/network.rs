use serde::{Serialize, Deserialize};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{MessageEvent, WebSocket};
use std::cell::RefCell;
use std::rc::Rc;
use log::{info, error};

#[derive(Serialize, Deserialize, Debug)]
pub struct PollenPacket {
    pub peer_id: String,
    pub world_hash: String,
    pub location: [f32; 3], // Approximate location for "proximity"
    pub timestamp: u64,
}

pub struct NetworkManager {
    socket: WebSocket,
    peer_id: String,
}

impl NetworkManager {
    pub fn new(url: &str) -> Result<Rc<RefCell<Self>>, JsValue> {
        let ws = WebSocket::new(url)?;

        // Generate a random Peer ID (or use a stored one?)
        let peer_id = format!("peer_{}", (js_sys::Math::random() * 10000.0) as u32);
        info!("Initializing Network Manager. My Peer ID: {}", peer_id);

        let manager = Rc::new(RefCell::new(Self {
            socket: ws.clone(),
            peer_id: peer_id.clone(),
        }));

        // OnMessage Callback
        let onmessage_callback = Closure::wrap(Box::new(move |e: MessageEvent| {
            if let Ok(txt) = e.data().dyn_into::<js_sys::JsString>() {
                let txt_string: String = txt.into();

                // Parse Pollen
                if let Ok(_packet) = serde_json::from_str::<PollenPacket>(&txt_string) {
                    // Ignore our own pollen
                    // Note: The socket broadcast might echo it back depending on server implementation.
                    // The server implementation we wrote DOES echo to others, but let's be safe.
                }
            }
        }) as Box<dyn FnMut(MessageEvent)>);

        ws.set_onmessage(Some(onmessage_callback.as_ref().unchecked_ref()));
        onmessage_callback.forget();

        // OnOpen Callback
        let onopen_callback = Closure::wrap(Box::new(move |_| {
            info!("Connected to Signaling Server");
        }) as Box<dyn FnMut(JsValue)>);
        ws.set_onopen(Some(onopen_callback.as_ref().unchecked_ref()));
        onopen_callback.forget();

        Ok(manager)
    }

    pub fn pollinate(&self, world_hash: &str, location: [f32; 3]) {
        if self.socket.ready_state() == WebSocket::OPEN {
            let packet = PollenPacket {
                peer_id: self.peer_id.clone(),
                world_hash: world_hash.to_string(),
                location,
                timestamp: js_sys::Date::now() as u64,
            };

            if let Ok(json) = serde_json::to_string(&packet) {
                let _ = self.socket.send_with_str(&json);
            }
        }
    }
}
