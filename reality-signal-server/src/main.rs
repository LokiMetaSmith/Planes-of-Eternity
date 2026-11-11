use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use warp::Filter;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CrashReport {
    pub message: String,
    pub stack: String,
    pub user_agent: String,
    pub uptime: f64,
}

/// Server configuration options configurable via CLI arguments or environment variables
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: IpAddr,
    pub port: u16,
    pub static_dir: PathBuf,
    pub max_connections: usize,
    pub max_messages_per_second: usize,
    pub max_message_size_bytes: usize,
    pub crash_log_path: PathBuf,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
            port: 9000,
            static_dir: PathBuf::from("../reality-engine/dist"),
            max_connections: 1000,
            max_messages_per_second: 50,
            max_message_size_bytes: 8192,
            crash_log_path: PathBuf::from("crash_logs.txt"),
        }
    }
}

impl ServerConfig {
    /// Parses CLI arguments and environment variables with fallbacks to defaults
    pub fn from_args_and_env<I, T>(args: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        let mut config = Self::default();

        // 1. Check environment variables
        if let Ok(host_env) = std::env::var("REALITY_HOST") {
            config.host = host_env
                .parse::<IpAddr>()
                .map_err(|e| format!("Invalid REALITY_HOST '{}': {}", host_env, e))?;
        }
        if let Ok(port_env) = std::env::var("REALITY_PORT") {
            config.port = port_env
                .parse::<u16>()
                .map_err(|e| format!("Invalid REALITY_PORT '{}': {}", port_env, e))?;
        }
        if let Ok(static_env) = std::env::var("REALITY_STATIC_DIR") {
            config.static_dir = PathBuf::from(static_env);
        }
        if let Ok(max_conn_env) = std::env::var("REALITY_MAX_CONNECTIONS") {
            config.max_connections = max_conn_env
                .parse::<usize>()
                .map_err(|e| format!("Invalid REALITY_MAX_CONNECTIONS: {}", e))?;
        }
        if let Ok(rate_env) = std::env::var("REALITY_MAX_RATE") {
            config.max_messages_per_second = rate_env
                .parse::<usize>()
                .map_err(|e| format!("Invalid REALITY_MAX_RATE: {}", e))?;
        }

        // 2. Parse CLI args (e.g. --host, --port, --static-dir, --max-connections, --rate-limit)
        let args_vec: Vec<String> = args.into_iter().map(|a| a.into()).collect();
        let mut i = 1; // skip binary name if present
        while i < args_vec.len() {
            match args_vec[i].as_str() {
                "--host" => {
                    if i + 1 < args_vec.len() {
                        config.host = args_vec[i + 1]
                            .parse::<IpAddr>()
                            .map_err(|e| format!("Invalid --host: {}", e))?;
                        i += 2;
                    } else {
                        return Err("Missing value for --host".to_string());
                    }
                }
                "--port" | "-p" => {
                    if i + 1 < args_vec.len() {
                        config.port = args_vec[i + 1]
                            .parse::<u16>()
                            .map_err(|e| format!("Invalid --port: {}", e))?;
                        i += 2;
                    } else {
                        return Err("Missing value for --port".to_string());
                    }
                }
                "--static-dir" => {
                    if i + 1 < args_vec.len() {
                        config.static_dir = PathBuf::from(&args_vec[i + 1]);
                        i += 2;
                    } else {
                        return Err("Missing value for --static-dir".to_string());
                    }
                }
                "--max-connections" => {
                    if i + 1 < args_vec.len() {
                        config.max_connections = args_vec[i + 1]
                            .parse::<usize>()
                            .map_err(|e| format!("Invalid --max-connections: {}", e))?;
                        i += 2;
                    } else {
                        return Err("Missing value for --max-connections".to_string());
                    }
                }
                "--rate-limit" => {
                    if i + 1 < args_vec.len() {
                        config.max_messages_per_second = args_vec[i + 1]
                            .parse::<usize>()
                            .map_err(|e| format!("Invalid --rate-limit: {}", e))?;
                        i += 2;
                    } else {
                        return Err("Missing value for --rate-limit".to_string());
                    }
                }
                "--help" | "-h" => {
                    return Err("Usage: reality-signal-server [--host <IP>] [--port <PORT>] [--static-dir <PATH>] [--max-connections <NUM>] [--rate-limit <NUM>]".to_string());
                }
                unknown => {
                    return Err(format!("Unknown argument: {}", unknown));
                }
            }
        }

        Ok(config)
    }

    pub fn socket_addr(&self) -> SocketAddr {
        SocketAddr::new(self.host, self.port)
    }
}

/// Our global unique user id counter.
static NEXT_USER_ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);

/// Our state of currently connected users.
pub type Users =
    Arc<Mutex<HashMap<usize, mpsc::Sender<std::result::Result<warp::ws::Message, warp::Error>>>>>;

/// Constructs all warp routes configured with security headers and config
pub fn build_routes(
    users: Users,
    config: ServerConfig,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    let users_filter = warp::any().map(move || users.clone());
    let cfg_clone1 = config.clone();
    let cfg_clone2 = config.clone();
    let cfg_filter1 = warp::any().map(move || cfg_clone1.clone());
    let cfg_filter2 = warp::any().map(move || cfg_clone2.clone());

    // GET /ws -> websocket upgrade
    let chat = warp::path("ws")
        .and(warp::ws())
        .and(cfg_filter1)
        .map(|ws: warp::ws::Ws, cfg: ServerConfig| {
            ws.max_message_size(cfg.max_message_size_bytes)
                .max_frame_size(cfg.max_message_size_bytes)
        })
        .and(users_filter)
        .and(cfg_filter2)
        .map(|ws: warp::ws::Ws, users, cfg: ServerConfig| {
            ws.on_upgrade(move |socket| user_connected(socket, users, cfg))
        });

    // Serve static files
    let static_files = warp::fs::dir(config.static_dir.clone());

    // POST /api/crash-report
    let crash_log_path = config.crash_log_path.clone();
    let crash_report = warp::path!("api" / "crash-report")
        .and(warp::post())
        .and(warp::body::content_length_limit(1024 * 64)) // 64KB limit
        .and(warp::body::json())
        .map(move |report: CrashReport| {
            let timestamp = chrono::Utc::now().to_rfc3339();
            let log_entry = format!(
                "[{}] Crash Report:\nMessage: {}\nUptime: {:.2}ms\nUser Agent: {}\nStack:\n{}\n----------------------------------------\n",
                timestamp, report.message, report.uptime, report.user_agent, report.stack
            );

            println!("{}", log_entry);

            if let Ok(mut file) = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&crash_log_path)
            {
                if let Err(e) = writeln!(file, "{}", log_entry) {
                    eprintln!("Failed to write to crash log: {}", e);
                }
            } else {
                eprintln!("Failed to open crash log: {:?}", crash_log_path);
            }

            warp::reply::json(&serde_json::json!({ "status": "success" }))
        });

    // Redirect root to index.html
    let root_redirect =
        warp::path::end().map(|| warp::redirect(warp::http::Uri::from_static("/index.html")));

    let csp_connect = format!(
        "default-src 'self'; script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval' 'unsafe-eval' https://unpkg.com; style-src 'self' 'unsafe-inline' https://unpkg.com; connect-src 'self' ws://localhost:{} wss://localhost:{} http://localhost:{} https://localhost:{} ws://{}:{} wss://{}:{} http://{}:{} https://{}:{} https://0.peerjs.com wss://0.peerjs.com https://unpkg.com; img-src 'self' data: blob:; media-src 'self' blob:;",
        config.port, config.port, config.port, config.port,
        config.host, config.port, config.host, config.port, config.host, config.port, config.host, config.port,
    );

    chat.or(crash_report).or(root_redirect).or(static_files)
        .with(warp::reply::with::header("X-Frame-Options", "DENY"))
        .with(warp::reply::with::header("X-Content-Type-Options", "nosniff"))
        .with(warp::reply::with::header("Referrer-Policy", "strict-origin-when-cross-origin"))
        .with(warp::reply::with::header("Content-Security-Policy", csp_connect))
        .with(warp::reply::with::header("Strict-Transport-Security", "max-age=31536000; includeSubDomains"))
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config = match ServerConfig::from_args_and_env(args) {
        Ok(cfg) => cfg,
        Err(msg) => {
            eprintln!("{}", msg);
            std::process::exit(1);
        }
    };

    let users = Users::default();
    let socket_addr = config.socket_addr();
    let routes = build_routes(users, config.clone());

    println!(
        "Signaling server and static file server running on http://{}/ (Static dir: {:?})",
        socket_addr, config.static_dir
    );

    warp::serve(routes)
        .bind(socket_addr)
        .await
        .graceful(async {
            tokio::signal::ctrl_c().await.ok();
            println!("\nShutdown signal received, shutting down gracefully...");
        })
        .run()
        .await;

    println!("Server shut down cleanly.");
}

async fn user_connected(ws: warp::ws::WebSocket, users: Users, config: ServerConfig) {
    let my_id = NEXT_USER_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // DoS Protection: Limit maximum simultaneous connections
    if users.lock().unwrap_or_else(|e| e.into_inner()).len() >= config.max_connections {
        eprintln!(
            "Security Warning: Max connections ({}) reached. Rejecting user {}.",
            config.max_connections, my_id
        );
        let (mut user_ws_tx, _) = ws.split();
        let _ = user_ws_tx.send(warp::ws::Message::close()).await;
        return;
    }

    let (mut user_ws_tx, mut user_ws_rx) = ws.split();
    let (tx, rx) = mpsc::channel::<std::result::Result<warp::ws::Message, warp::Error>>(100);
    let mut rx = tokio_stream::wrappers::ReceiverStream::new(rx);

    tokio::task::spawn(async move {
        while let Some(message) = rx.next().await {
            user_ws_tx
                .send(message.unwrap_or_else(|_e| warp::ws::Message::close()))
                .await
                .unwrap_or_else(|e| {
                    eprintln!("websocket send error: {}", e);
                });
        }
    });

    users.lock().unwrap_or_else(|e| e.into_inner()).insert(my_id, tx);
    println!("User {} connected", my_id);

    let mut graceful_exit = false;
    let mut message_count = 0;
    let mut last_reset = std::time::Instant::now();

    while let Some(result) = user_ws_rx.next().await {
        if last_reset.elapsed().as_secs() >= 1 {
            message_count = 0;
            last_reset = std::time::Instant::now();
        }

        message_count += 1;
        if message_count > config.max_messages_per_second {
            eprintln!(
                "Security Warning: User {} exceeded rate limit ({} msgs/sec). Disconnecting.",
                my_id, config.max_messages_per_second
            );
            break;
        }

        let msg = match result {
            Ok(msg) => msg,
            Err(e) => {
                eprintln!("websocket error(uid={}): {}", my_id, e);
                break;
            }
        };

        if let Ok(text) = msg.to_str() {
            if text == "__GRACEFUL_EXIT__" {
                graceful_exit = true;
                break;
            }
        }

        user_message(my_id, msg, &users, config.max_message_size_bytes).await;
    }

    user_disconnected(my_id, &users, graceful_exit).await;
}

async fn user_message(my_id: usize, msg: warp::ws::Message, users: &Users, max_len: usize) {
    let msg_str = if let Ok(s) = msg.to_str() {
        s
    } else {
        return;
    };

    if msg_str.len() > max_len {
        eprintln!(
            "Security Warning: User {} sent a message exceeding maximum length limit ({} bytes). Dropping.",
            my_id,
            msg_str.len()
        );
        return;
    }

    let new_msg = msg_str.to_string();

    for (&uid, tx) in users.lock().unwrap_or_else(|e| e.into_inner()).iter() {
        if my_id != uid {
            if let Err(e) = tx.try_send(Ok(warp::ws::Message::text(new_msg.clone()))) {
                eprintln!("Failed to send message to user {}: {}", uid, e);
            }
        }
    }
}

async fn user_disconnected(my_id: usize, users: &Users, graceful_exit: bool) {
    if graceful_exit {
        println!("User {} disconnected (Graceful Exit)", my_id);
    } else {
        println!("User {} disconnected (Unexpected Drop/Crash)", my_id);
    }
    users.lock().unwrap_or_else(|e| e.into_inner()).remove(&my_id);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_config_defaults() {
        let config = ServerConfig::default();
        assert_eq!(config.port, 9000);
        assert_eq!(config.max_connections, 1000);
        assert_eq!(config.max_messages_per_second, 50);
        assert_eq!(config.max_message_size_bytes, 8192);
        assert_eq!(config.socket_addr().port(), 9000);
    }

    #[test]
    fn test_server_config_cli_parsing() {
        let args = vec![
            "reality-signal-server".to_string(),
            "--host".to_string(),
            "0.0.0.0".to_string(),
            "--port".to_string(),
            "8080".to_string(),
            "--max-connections".to_string(),
            "250".to_string(),
            "--rate-limit".to_string(),
            "100".to_string(),
            "--static-dir".to_string(),
            "/tmp/static".to_string(),
        ];
        let config = ServerConfig::from_args_and_env(args).expect("Failed to parse config");
        assert_eq!(config.host, IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)));
        assert_eq!(config.port, 8080);
        assert_eq!(config.max_connections, 250);
        assert_eq!(config.max_messages_per_second, 100);
        assert_eq!(config.static_dir, PathBuf::from("/tmp/static"));
    }

    #[test]
    fn test_server_config_invalid_arg() {
        let args = vec!["reality-signal-server".to_string(), "--invalid".to_string()];
        let err = ServerConfig::from_args_and_env(args).unwrap_err();
        assert!(err.contains("Unknown argument"));
    }

    #[tokio::test]
    async fn test_crash_report_route() {
        let users = Users::default();
        let mut config = ServerConfig::default();
        config.crash_log_path = std::env::temp_dir().join(format!("test_crash_logs_{}.txt", std::process::id()));
        let routes = build_routes(users, config);

        let report = CrashReport {
            message: "Test crash message".to_string(),
            stack: "Error at line 42".to_string(),
            user_agent: "HeadlessChrome/1.0".to_string(),
            uptime: 123.45,
        };

        let response = warp::test::request()
            .method("POST")
            .path("/api/crash-report")
            .json(&report)
            .reply(&routes)
            .await;

        assert_eq!(response.status(), 200);
        assert!(response.headers().contains_key("x-frame-options"));
        assert!(response.headers().contains_key("content-security-policy"));
    }

    #[tokio::test]
    async fn test_root_redirect_route() {
        let users = Users::default();
        let config = ServerConfig::default();
        let routes = build_routes(users, config);

        let response = warp::test::request()
            .method("GET")
            .path("/")
            .reply(&routes)
            .await;

        assert_eq!(response.status(), 301);
        assert_eq!(response.headers().get("location").unwrap(), "/index.html");
    }
}
