## 2026-03-14 - Missing WebSocket Message Length Limit
**Vulnerability:** The signaling server blindly relays any text message received via WebSocket to all other users without checking its size.
**Learning:** Unrestricted message sizes can be exploited to cause Denial of Service (DoS) through memory exhaustion and excessive bandwidth usage, impacting the server and all connected clients.
**Prevention:** Always implement input validation, including maximum length limits, on all user-supplied data, particularly in broadcast mechanisms.

## 2024-05-24 - Overly Permissive CORS Configuration
**Vulnerability:** The signaling server configured CORS with `allow_any_origin()`, allowing any website to make cross-origin requests to the server.
**Learning:** `warp::cors().allow_any_origin()` is dangerous and should be avoided unless absolutely necessary for public APIs. Since the frontend and API are served from the same origin, CORS middleware isn't even needed, or it should be strictly restricted.
**Prevention:** Avoid `allow_any_origin()`. Use specific allowed origins or rely on the default same-origin policy by omitting CORS configuration if frontend and backend are hosted together.
