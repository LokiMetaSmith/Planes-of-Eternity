## 2024-05-24 - Overly Permissive CORS Configuration
**Vulnerability:** The signaling server configured CORS with `allow_any_origin()`, allowing any website to make cross-origin requests to the server.
**Learning:** `warp::cors().allow_any_origin()` is dangerous and should be avoided unless absolutely necessary for public APIs. Since the frontend and API are served from the same origin, CORS middleware isn't even needed, or it should be strictly restricted.
**Prevention:** Avoid `allow_any_origin()`. Use specific allowed origins or rely on the default same-origin policy by omitting CORS configuration if frontend and backend are hosted together.
