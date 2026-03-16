use futures::{SinkExt, StreamExt};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use warp::Filter;

/// Our global unique user id counter.
static NEXT_USER_ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);

/// Our state of currently connected users.
///
/// - Key is their id
/// - Value is a sender of `warp::ws::Message`
type Users =
    Arc<Mutex<HashMap<usize, mpsc::Sender<std::result::Result<warp::ws::Message, warp::Error>>>>>;

#[tokio::main]
async fn main() {
    // Keep track of all connected users, key is usize, value
    // is a websocket sender.
    let users = Users::default();

    // Turn our "state" into a new Filter...
    let users = warp::any().map(move || users.clone());

    // GET /ws -> websocket upgrade
    let chat = warp::path("ws")
        // The `ws()` filter will prepare Websocket handshake...
        // Security Enhancement: Prevent DoS by limiting message and frame size to 8KB
        .and(warp::ws())
        .map(|ws: warp::ws::Ws| ws.max_message_size(8192).max_frame_size(8192))
        .and(users)
        .map(|ws: warp::ws::Ws, users| {
            // This will call our function if the handshake succeeds.
            ws.on_upgrade(move |socket| user_connected(socket, users))
        });

    // Serve static files from the parent directory (repo root)
    let static_files = warp::fs::dir("../reality-engine/dist");

    // Redirect root to the game's index.html
    let root_redirect =
        warp::path::end().map(|| warp::redirect(warp::http::Uri::from_static("/index.html")));

    let routes = chat.or(root_redirect).or(static_files);

    println!("Signaling server and static file server running on http://localhost:9000/");

    // No need for CORS as the frontend is served by the same origin
    warp::serve(routes).run(([127, 0, 0, 1], 9000)).await;
}

async fn user_connected(ws: warp::ws::WebSocket, users: Users) {
    // Use a counter to assign a new unique ID for this user.
    let my_id = NEXT_USER_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // Split the socket into a sender and receiver of messages.
    let (mut user_ws_tx, mut user_ws_rx) = ws.split();

    // Security Enhancement: Use a bounded channel to prevent memory exhaustion DoS
    // from slow clients accumulating too many messages in memory.
    let (tx, rx) = mpsc::channel::<std::result::Result<warp::ws::Message, warp::Error>>(100);
    let mut rx = tokio_stream::wrappers::ReceiverStream::new(rx);

    tokio::task::spawn(async move {
        while let Some(message) = rx.next().await {
            user_ws_tx
                .send(message.unwrap_or_else(|_e| {
                    // Convert error to string or close?
                    warp::ws::Message::close()
                }))
                .await
                .unwrap_or_else(|e| {
                    eprintln!("websocket send error: {}", e);
                });
        }
    });

    // Save the sender in our list of connected users.
    users.lock().unwrap().insert(my_id, tx);

    println!("User {} connected", my_id);

    // Every time the user sends a message, broadcast it to
    // all other users...
    while let Some(result) = user_ws_rx.next().await {
        let msg = match result {
            Ok(msg) => msg,
            Err(e) => {
                eprintln!("websocket error(uid={}): {}", my_id, e);
                break;
            }
        };

        user_message(my_id, msg, &users).await;
    }

    // user_ws_rx stream will keep processing as long as the user stays
    // connected. Once they disconnect, then...
    user_disconnected(my_id, &users).await;
}

async fn user_message(my_id: usize, msg: warp::ws::Message, users: &Users) {
    // Skip any non-Text messages...
    let msg = if let Ok(s) = msg.to_str() {
        s
    } else {
        return;
    };

    // Security Enhancement: Prevent DoS by limiting message length to 8KB
    const MAX_MESSAGE_LEN: usize = 8192;
    if msg.len() > MAX_MESSAGE_LEN {
        eprintln!(
            "Security Warning: User {} sent a message exceeding the maximum length limit ({} bytes). Dropping message.",
            my_id,
            msg.len()
        );
        return;
    }

    let new_msg = format!("{}", msg); // Just echo/relay the pollen

    // New message from this user, send it to everyone else (except same uid)...
    for (&uid, tx) in users.lock().unwrap().iter() {
        if my_id != uid {
            if let Err(e) = tx.try_send(Ok(warp::ws::Message::text(new_msg.clone()))) {
                // If the client is too slow or disconnected, drop the message.
                // This prevents memory exhaustion on the server.
                eprintln!("Failed to send message to user {}: {}", uid, e);
            }
        }
    }
}

async fn user_disconnected(my_id: usize, users: &Users) {
    println!("User {} disconnected", my_id);
    // Stream closed up, so remove from the user list
    users.lock().unwrap().remove(&my_id);
}
