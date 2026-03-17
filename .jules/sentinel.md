## 2026-03-14 - Missing WebSocket Message Length Limit
**Vulnerability:** The signaling server blindly relays any text message received via WebSocket to all other users without checking its size.
**Learning:** Unrestricted message sizes can be exploited to cause Denial of Service (DoS) through memory exhaustion and excessive bandwidth usage, impacting the server and all connected clients.
**Prevention:** Always implement input validation, including maximum length limits, on all user-supplied data, particularly in broadcast mechanisms.

## 2024-05-24 - Overly Permissive CORS Configuration
**Vulnerability:** The signaling server configured CORS with `allow_any_origin()`, allowing any website to make cross-origin requests to the server.
**Learning:** `warp::cors().allow_any_origin()` is dangerous and should be avoided unless absolutely necessary for public APIs. Since the frontend and API are served from the same origin, CORS middleware isn't even needed, or it should be strictly restricted.
**Prevention:** Avoid `allow_any_origin()`. Use specific allowed origins or rely on the default same-origin policy by omitting CORS configuration if frontend and backend are hosted together.

## 2024-05-24 - Weak random number generation for Peer IDs
**Vulnerability:** The random peer ID generation in `reality-engine/src/network.rs` uses `js_sys::Math::random()`, which is a weak pseudo-random number generator that is not cryptographically secure, and the implementation only multiplies by 10,000, creating a very small keyspace (10,000 possible IDs) which is highly vulnerable to collisions and brute-forcing.
**Learning:** Weak randomness and small keyspaces can lead to predictable IDs, causing potential session hijacking or denial of service by ID collision. The `rand` or `getrandom` crates provide secure, robust random generation in WebAssembly.
**Prevention:** Use cryptographically secure random number generators (e.g., `getrandom` or the `rand` crate with `OsRng` / Web Crypto API) and larger random values (e.g., UUIDs or a 64-bit/128-bit random number) for session or peer IDs.

## 2024-05-24 - Unbounded WebSocket Sender Channel
**Vulnerability:** The signaling server originally used an unbounded `mpsc::unbounded_channel()` for sending messages to clients. If a client connection is slow or a user intentionally avoids reading from their socket, messages will queue up indefinitely in the server's memory.
**Learning:** Unbounded queues on network connections are a classic vector for memory-exhaustion Denial of Service (DoS) attacks.
**Prevention:** Always use bounded channels (like `tokio::sync::mpsc::channel(capacity)`) for buffering outgoing network traffic, and handle the backpressure gracefully (e.g., dropping messages or disconnecting the client if the queue fills up).

## 2026-03-14 - Unbounded Connections and Missing Rate Limiting in WebSocket Server
**Vulnerability:** The signaling server allowed an unlimited number of concurrent WebSocket connections and placed no limit on the rate at which a connected user could send messages.
**Learning:** These unconstrained resource limits provide a wide-open vector for Denial of Service (DoS) attacks. An attacker can exhaust server memory/file descriptors by opening thousands of connections or exhaust network bandwidth and CPU by rapidly spamming messages to trigger broadcasts to all peers.
**Prevention:** Always implement hard limits on concurrent connections (`max_connections`) and enforce per-connection rate limits (e.g., maximum messages per second) on endpoints that process user input or broadcast data.
