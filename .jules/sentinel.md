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

## 2026-03-28 - AI Sycophancy Leading to Insecure Implementations
**Vulnerability:** AI agents blindly agreeing with and implementing a user's proposed solution or code snippet, even when that snippet introduces severe security flaws (like disabling CORS, removing length limits, or using weak cryptography).
**Learning:** Recent research shows AI models are inherently sycophantic and overly agreeable. In security contexts, this "yes-man" behavior allows users to inadvertently introduce vulnerabilities without the AI pushing back or warning them of the consequences.
**Prevention:** Implement "healthy skepticism." When acting as Sentinel, internally use the "Wait a minute..." prime before generating code. If a user's request is insecure, you MUST explicitly reject the insecure premise, provide "tough love" by pointing out the vulnerability, and offer a secure alternative. Only implement the insecure version if the user explicitly demands it after being warned.

## 2026-03-27 - Missing Content Security Policy (CSP)
**Vulnerability:** The signaling server in `reality-signal-server` served the frontend without a `Content-Security-Policy` header. While some headers like `X-Frame-Options` were present, the lack of CSP meant there were no restrictions on script execution sources, data connection sources (like WebSockets or WebRTC peers), or styling sources.
**Learning:** A missing CSP leaves a web application significantly more vulnerable to Cross-Site Scripting (XSS) and data exfiltration. Attackers could potentially inject malicious scripts that connect to arbitrary external servers or load unauthorized content.
**Prevention:** Always configure a restrictive `Content-Security-Policy` header for endpoints serving HTML. Explicitly define allowed origins using directives like `script-src`, `connect-src` (crucial for P2P/WebSocket apps to whitelist their signaling/ice servers), and `style-src` to enforce strict resource boundaries.

## 2024-05-24 - WASM Allocation DoS Prevention
**Vulnerability:** Converting untrusted `JsValue` strings into Rust `String` *before* length validation allows an attacker to allocate massive strings on the WASM heap, leading to Out-Of-Memory (OOM) Denial of Service.
**Learning:** Native JavaScript strings (`JsString`) reside in the browser engine's heap, whereas converting them to Rust `String` requires allocating linear memory inside the WASM boundary. Validating bounds after boundary traversal is too late.
**Prevention:** Always cast untrusted `JsValue` to `js_sys::JsString` and evaluate `.length()` on the native JS side *before* calling `.into()` or otherwise initiating WASM memory allocation for strings.

## 2024-05-28 - Decompression Bomb in Voxel RLE Deserialization
**Vulnerability:** The Run-Length Encoding (RLE) deserializer in `reality-engine/src/voxel.rs` (`voxel_data_serde::deserialize`) read a 16-bit count from untrusted input and blindly allocated that many voxels in a loop. A malicious payload could use a tiny hex string to cause gigabytes of memory allocation, leading to an Out-Of-Memory (OOM) Denial of Service (DoS) attack (a classic "Decompression Bomb").
**Learning:** Custom deserialization formats, especially those involving compression like RLE, must strictly enforce bounds checking on the total decoded output size to prevent memory exhaustion, regardless of outer payload size limits.
**Prevention:** Always track the total accumulated size during decompression and abort if it exceeds the mathematically maximum possible size for the data structure (e.g., `CHUNK_SIZE^3`).

## 2026-03-29 - Sentinel: Limit WebRTC Connections
**Vulnerability:** DoS risk via unbounded WebRTC connections. `network.rs` blindly accepted and attempted to maintain all incoming and outgoing WebRTC `DataConnection`s without limit, opening the client up to resource exhaustion (memory, CPU, connection limits) if a malicious peer spammed connection requests.
**Learning:** Peer-to-peer applications must constrain not only message size and count per connection, but the total number of simultaneous connections a client will accept or initiate, to prevent connection exhaustion Denial of Service (DoS) attacks.
**Prevention:** Always enforce a hard upper bound (e.g. 20) on the maximum number of concurrent peer connections (`inner.connections.len()`) and explicitly reject/close incoming connection requests that exceed the limit.

## 2026-03-29 - Unbounded In-Game Item Spawning DoS
**Vulnerability:** DoS risk via unbounded in-game item generation. In `reality-engine/src/engine.rs`, processing the `Action::DropItem` user input when the inventory is empty resulted in unconditionally generating and spawning a new `DroppedItem` object into the `world_state.dropped_items` vector without any limit.
**Learning:** Any user action that allocates new game state objects, especially those synchronized globally across a P2P network (like `WorldState`), is a vector for memory and network bandwidth exhaustion if the user can spam the action unboundedly.
**Prevention:** Always enforce a hard upper bound (e.g., maximum 100 spawned items globally) before allowing a user-triggered action to generate new persisted entities in the game world.

## 2026-04-02 - Sentinel: Add input validation to save slot names
**Vulnerability:** DoS risk and Key Injection via unbounded/unsanitized local storage keys. The `get_save_key` function used user-supplied `slot_name` inputs to generate localStorage keys without restricting character types or length, potentially allowing attackers to exhaust storage quotas with massive keys or inject control characters.
**Learning:** Even internal APIs that interact with browser storage mechanisms (like `localStorage`) must sanitize keys to prevent unpredictable behavior, quota exhaustion DoS, or key collision attacks.
**Prevention:** Always enforce strict length limits (e.g., 64 characters) and character whitelists (e.g., alphanumeric and underscores) on user-supplied data used as dictionary keys or storage keys.

## 2026-03-30 - Missing Rate Limiting on Client-Side WebRTC Channels
**Vulnerability:** The `reality-engine` P2P client lacked rate limiting on incoming WebRTC `DataConnection` messages. A malicious peer could send a flood of rapid updates (e.g., small payload messages or maxed 64KB payloads), causing the WASM client to exhaust its CPU cycle parsing and merging the updates, freezing the user's browser.
**Learning:** Client-side P2P connections are just as vulnerable to Denial of Service (DoS) attacks as backend servers. Without per-peer rate limiting, a single malicious connection can monopolize the client's single-threaded event loop.
**Prevention:** Always implement a per-connection rate limit (e.g., tracking `messages_per_second` and dropping excess messages) on event handlers for incoming P2P data channels.

## 2025-05-24 - Sentinel: WASM Boundary OOM DoS via Unbounded Strings
**Vulnerability:** The WASM boundary functions in `reality-engine/src/lib.rs` (like `save_game`, `load_game`, `delete_save`, `get_key_binding`) accepted Rust `String` parameters directly from JavaScript. This allows an attacker to pass an exceptionally large string from JS, causing the JS-to-WASM bridge to implicitly allocate unbounded memory in the WASM heap *before* any internal Rust length checks are executed, leading to an Out-Of-Memory (OOM) Denial of Service (DoS) attack that crashes the WASM module.
**Learning:** Converting untrusted `JsValue` or implicit JS strings into Rust `String` at the `#[wasm-bindgen]` boundary happens automatically and without length constraints. Length checks placed *inside* the Rust function body are ineffective against allocation DoS if the allocation happens at the ABI boundary.
**Prevention:** Always accept `js_sys::JsString` for untrusted string inputs at the `#[wasm-bindgen]` boundary. Evaluate `.length()` on the native JS string representation *before* calling `.into()` to convert it to a Rust `String` to prevent malicious allocations.
