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
## 2025-03-02 - Sentinel: Add input length limit to lambda parser
**Vulnerability:** DoS risk via unbound string input parsing. `process_inscription` took user input and parsed it directly using a custom lambda calculus parser without length limits, potentially allowing a malicious user to crash or stall the engine.
**Learning:** Even internal UI inputs that parse into complex ASTs can be vectors for resource exhaustion DoS if unbounded.
**Prevention:** Always enforce reasonable maximum length limits on user input before passing it to expensive parsers.
## 2026-03-14 - Mutex Poisoning Denial of Service
**Vulnerability:** The signaling server used `.unwrap()` on `users.lock()` when accessing the shared connected users state (`std::sync::Mutex`).
**Learning:** In Rust, if a thread panics while holding a `Mutex`, the mutex becomes "poisoned". Any subsequent attempts to lock the mutex and `.unwrap()` it will also panic, causing a complete Denial of Service (DoS) for the entire server, as all new connections or message broadcasts will fail.
**Prevention:** Handle Mutex poisoning gracefully by using `.unwrap_or_else(|e| e.into_inner())` or pattern matching `match users.lock() { Ok(guard) => guard, Err(poisoned) => poisoned.into_inner() }` to extract the guard and continue operating, assuming the protected data structure is still structurally sound (which it is for a simple HashMap).
## 2024-05-27 - Security Enhancement: NPC Chat Message Sanitization
**Vulnerability:** Lack of input sanitization and length limits on external JSON inputs (`chat_message` via `execute_npc_action_json`). While currently only logged, these messages were vulnerable to Log Injection, and if ever rendered to the DOM, Cross-Site Scripting (XSS). Additionally, unbound length could lead to Denial of Service (DoS) through memory exhaustion.
**Learning:** External data originating from network or potentially untrusted sources (even internal components communicating via JSON string passing like JS -> Wasm) should always be treated as untrusted and sanitized upon entry into the core engine state, regardless of whether its immediate use appears safe (e.g. logging).
**Prevention:** Implement strict length limits (e.g. 256 bytes) and basic HTML escaping / control character filtering immediately after deserializing JSON strings that represent user or external text input.

## 2026-03-14 - Sentinel: Add WebRTC message length limit
**Vulnerability:** DoS risk via unbounded WebRTC message string parsing. `network.rs` received arbitrary strings over PeerJS DataConnection and passed them to `serde_json::from_str` without checking their size, potentially allowing a malicious peer to exhaust memory.
**Learning:** Even P2P client-side connections can be vectors for resource exhaustion DoS if unbounded incoming payloads are parsed blindly.
**Prevention:** Always enforce reasonable maximum length limits on external string payloads (e.g. 64KB) before passing them to serializers/deserializers or processing pipelines.

## 2026-03-14 - Missing HTTP Security Headers
**Vulnerability:** The signaling server in `reality-signal-server` served static files and responses without essential HTTP security headers like `X-Frame-Options`, `X-Content-Type-Options`, and `Referrer-Policy`. This omission leaves the frontend vulnerable to clickjacking and MIME-type sniffing attacks.
**Learning:** Even simple signaling servers or local web servers serving a frontend should include defense-in-depth security headers, as browsers rely on them to enforce fundamental security boundaries regardless of the content's origin.
**Prevention:** Always append standard security headers to HTTP server responses. In `warp`, this is easily accomplished using `.with(warp::reply::with::header("...", "..."))` chained to the main routing filters.
## 2026-03-14 - Sentinel: Add JSON parsing length limits
**Vulnerability:** DoS risk via unbounded JSON string parsing. `persistence.rs` (`load_from_local_storage`) and `lib.rs` (`execute_npc_action_json`) took external JSON input and parsed it directly using `serde_json::from_str` without checking lengths, potentially allowing a malicious actor or external service to cause memory exhaustion.
**Learning:** Even local storage saves and JS -> WASM strings can be vectors for resource exhaustion DoS if unbounded.
**Prevention:** Always enforce reasonable maximum length limits (e.g. 10MB for save states, 1KB for JS payloads) before passing them to expensive serializers/deserializers.

## 2026-03-25 - Sentinel: Add WebSocket message length limit
**Vulnerability:** DoS risk via unbounded WebSocket message string parsing. `network.rs` received arbitrary strings from the signaling server and passed them to `serde_json::from_str` without checking their size, potentially allowing a malicious actor to exhaust memory.
**Learning:** Even WebSocket client-side connections can be vectors for resource exhaustion DoS if unbounded incoming payloads are parsed blindly.
**Prevention:** Always enforce reasonable maximum length limits on external string payloads (e.g. 8KB) before passing them to serializers/deserializers or processing pipelines.

## 2026-03-26 - Recursive Descent Parser Stack Overflow
**Vulnerability:** The recursive descent parser in `lambda::parse` (`reality-engine/src/lambda.rs`) lacked a recursion depth limit, making it vulnerable to stack overflow Denial of Service (DoS) attacks when processing deeply nested parenthetical expressions (e.g., `((((...FIRE...))))`).
**Learning:** Even if an initial input string is constrained in length (e.g., the 256-byte limit in `process_inscription`), 256 bytes is more than enough to encode extremely deep nesting, which could exceed the constrained WebAssembly (WASM) call stack limits or crash host environments.
**Prevention:** Always enforce a strict `depth` limit (e.g., 64) when implementing recursive descent parsers or processing nested structures, returning early or throwing an error if the depth exceeds the safe threshold.

## 2026-03-27 - Sentinel: Add max anomaly limit per chunk
**Vulnerability:** DoS risk via unbounded anomaly list expansion in P2P synchronization. `world.rs` (`merge` and `merge_chunks`) received arbitrary chunks from other peers and pushed all provided anomalies into the local chunk's `anomalies` vector without limiting the total count. This allows malicious peers to send a single chunk update containing thousands of anomalies to exhaust the memory and CPU of connected clients when they process the world state.
**Learning:** Synchronization mechanisms that merge collections of objects over P2P networks (like `Vec<RealityProjector>`) are vulnerable to memory exhaustion DoS if they lack capacity limits, functioning similarly to zip bombs.
**Prevention:** Always enforce reasonable maximum capacity limits on lists/vectors when accepting state objects over the network (e.g., maximum 100 anomalies per chunk) before appending new elements.

## 2026-03-27 - Missing Content Security Policy (CSP)
**Vulnerability:** The signaling server in `reality-signal-server` served the frontend without a `Content-Security-Policy` header. While some headers like `X-Frame-Options` were present, the lack of CSP meant there were no restrictions on script execution sources, data connection sources (like WebSockets or WebRTC peers), or styling sources.
**Learning:** A missing CSP leaves a web application significantly more vulnerable to Cross-Site Scripting (XSS) and data exfiltration. Attackers could potentially inject malicious scripts that connect to arbitrary external servers or load unauthorized content.
**Prevention:** Always configure a restrictive `Content-Security-Policy` header for endpoints serving HTML. Explicitly define allowed origins using directives like `script-src`, `connect-src` (crucial for P2P/WebSocket apps to whitelist their signaling/ice servers), and `style-src` to enforce strict resource boundaries.
