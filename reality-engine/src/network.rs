use serde::{Serialize, Deserialize};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{MessageEvent, WebSocket};
use std::cell::RefCell;
use std::rc::Rc;
use std::collections::HashMap;
use log::{info, error};
use crate::world::WorldState;

#[derive(Serialize, Deserialize, Debug)]
pub enum SyncMessage {
    WorldUpdate(WorldState),
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = Peer)]
    type Peer;

    #[wasm_bindgen(constructor)]
    fn new(id: &str) -> Peer;

    #[wasm_bindgen(method)]
    fn on(this: &Peer, event: &str, callback: &Closure<dyn FnMut(JsValue)>);

    #[wasm_bindgen(method)]
    fn connect(this: &Peer, id: &str) -> DataConnection;

    type DataConnection;

    #[wasm_bindgen(method)]
    fn on(this: &DataConnection, event: &str, callback: &Closure<dyn FnMut(JsValue)>);

    #[wasm_bindgen(method)]
    fn send(this: &DataConnection, data: &JsValue);

    #[wasm_bindgen(method, getter)]
    fn peer(this: &DataConnection) -> String;
}

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
    peer: Peer,
    connections: HashMap<String, DataConnection>,
    sync_callback: Option<Box<dyn FnMut(SyncMessage)>>,
}

impl NetworkManager {
    pub fn new(url: &str) -> Result<Rc<RefCell<Self>>, JsValue> {
        let ws = WebSocket::new(url)?;

        // Generate a random Peer ID
        let peer_id = format!("peer_{}", (js_sys::Math::random() * 10000.0) as u32);
        info!("Initializing Network Manager. My Peer ID: {}", peer_id);

        // Initialize PeerJS
        let peer = Peer::new(&peer_id);

        let manager = Rc::new(RefCell::new(Self {
            socket: ws.clone(),
            peer_id: peer_id.clone(),
            peer: peer.clone().unchecked_into(), // Peer is a JsValue wrapper, so clone is cheap (reference)
            connections: HashMap::new(),
            sync_callback: None,
        }));

        // Setup PeerJS Listeners
        {
            let _m = manager.clone();
            let on_open = Closure::wrap(Box::new(move |id: JsValue| {
                if let Some(id_str) = id.as_string() {
                     info!("PeerJS Open: ID confirmed as {}", id_str);
                }
            }) as Box<dyn FnMut(JsValue)>);
            peer.on("open", &on_open);
            on_open.forget();

            let m_conn = manager.clone();
            let on_connection = Closure::wrap(Box::new(move |conn: JsValue| {
                let conn = conn.unchecked_into::<DataConnection>();
                // Handle new connection
                NetworkManager::handle_connection(m_conn.clone(), conn);
            }) as Box<dyn FnMut(JsValue)>);
            peer.on("connection", &on_connection);
            on_connection.forget();

            let on_peer_error = Closure::wrap(Box::new(move |err: JsValue| {
                 error!("PeerJS Error: {:?}", err);
            }) as Box<dyn FnMut(JsValue)>);
            peer.on("error", &on_peer_error);
            on_peer_error.forget();
        }

        // Setup WebSocket Listeners (Pollen Discovery)
        {
            let m_ws = manager.clone();
            let onmessage_callback = Closure::wrap(Box::new(move |e: MessageEvent| {
                if let Ok(txt) = e.data().dyn_into::<js_sys::JsString>() {
                    let txt_string: String = txt.into();

                    // Parse Pollen
                    if let Ok(packet) = serde_json::from_str::<PollenPacket>(&txt_string) {
                         let mut inner = m_ws.borrow_mut();
                         // If it's not us, and we are not connected
                         if packet.peer_id != inner.peer_id {
                             if !inner.connections.contains_key(&packet.peer_id) {
                                 info!("Discovered new peer: {}. Connecting via WebRTC...", packet.peer_id);
                                 let conn = inner.peer.connect(&packet.peer_id);
                                 inner.connections.insert(packet.peer_id.clone(), conn.clone().unchecked_into());

                                 // Setup listeners for this outbound connection
                                 // We must drop the borrow before calling handle_connection (if it borrows)
                                 // But handle_connection is static-ish helpers, let's see.
                                 // It takes Rc<RefCell<Self>>.
                                 drop(inner);

                                 NetworkManager::handle_connection(m_ws.clone(), conn);
                             }
                         }
                    }
                }
            }) as Box<dyn FnMut(MessageEvent)>);

            ws.set_onmessage(Some(onmessage_callback.as_ref().unchecked_ref()));
            onmessage_callback.forget();

            let onopen_callback = Closure::wrap(Box::new(move |_| {
                info!("Connected to Signaling Server");
            }) as Box<dyn FnMut(JsValue)>);
            ws.set_onopen(Some(onopen_callback.as_ref().unchecked_ref()));
            onopen_callback.forget();
        }

        Ok(manager)
    }

    fn handle_connection(manager: Rc<RefCell<Self>>, conn: DataConnection) {
         let peer_id = conn.peer();
         info!("Handling connection with {}", peer_id);

         // Store connection if not already stored (for incoming)
         {
             let mut inner = manager.borrow_mut();
             if !inner.connections.contains_key(&peer_id) {
                 inner.connections.insert(peer_id.clone(), conn.clone().unchecked_into());
             }
         }

         let m_data = manager.clone();
         let pid_data = peer_id.clone();
         let on_data = Closure::wrap(Box::new(move |data: JsValue| {
             // Handle data sync
             if let Some(txt) = data.as_string() {
                 info!("Received WebRTC Data from {}: {}", pid_data, txt);

                 // Try to parse SyncMessage
                 if let Ok(msg) = serde_json::from_str::<SyncMessage>(&txt) {
                     let mut inner = m_data.borrow_mut();
                     if let Some(ref mut callback) = inner.sync_callback {
                         callback(msg);
                     }
                 }
             }
         }) as Box<dyn FnMut(JsValue)>);
         conn.on("data", &on_data);
         on_data.forget();

         let pid_open = peer_id.clone();
         let c_open: DataConnection = conn.clone().unchecked_into();
         let on_open = Closure::wrap(Box::new(move |_| {
             info!("WebRTC Connection established with {}", pid_open);
             // Send Hello
             c_open.send(&JsValue::from_str("HELLO_FROM_RUST"));
         }) as Box<dyn FnMut(JsValue)>);
         conn.on("open", &on_open);
         on_open.forget();

         let pid_close = peer_id.clone();
         let m_close = manager.clone();
         let on_close = Closure::wrap(Box::new(move |_| {
             info!("WebRTC Connection closed with {}", pid_close);
             let mut inner = m_close.borrow_mut();
             inner.connections.remove(&pid_close);
         }) as Box<dyn FnMut(JsValue)>);
         conn.on("close", &on_close);
         on_close.forget();

         let pid_error = peer_id.clone();
         let m_error = manager.clone();
         let on_error = Closure::wrap(Box::new(move |err: JsValue| {
             error!("WebRTC Connection error with {}: {:?}", pid_error, err);
             let mut inner = m_error.borrow_mut();
             inner.connections.remove(&pid_error);
         }) as Box<dyn FnMut(JsValue)>);
         conn.on("error", &on_error);
         on_error.forget();
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

    pub fn set_sync_callback<F>(&mut self, callback: F)
    where
        F: FnMut(SyncMessage) + 'static,
    {
        self.sync_callback = Some(Box::new(callback));
    }

    pub fn broadcast_world_state(&self, state: &WorldState) {
        let msg = SyncMessage::WorldUpdate(state.clone());
        if let Ok(json) = serde_json::to_string(&msg) {
            let js_val = JsValue::from_str(&json);
            for conn in self.connections.values() {
                conn.send(&js_val);
            }
        }
    }
}
