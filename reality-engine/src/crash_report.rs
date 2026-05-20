use serde::Serialize;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, RequestMode};

#[derive(Serialize)]
struct CrashReport {
    message: String,
    stack: String,
    user_agent: String,
    uptime: f64,
}

pub fn install_crash_hook() {
    std::panic::set_hook(Box::new(|info| {
        // Log to console first
        console_error_panic_hook::hook(info);

        let window = match web_sys::window() {
            Some(w) => w,
            None => return,
        };

        let message = info.to_string();

        let stack = js_sys::Reflect::get(&js_sys::Error::new(&message), &JsValue::from_str("stack"))
            .ok()
            .and_then(|v| v.as_string())
            .unwrap_or_else(|| "Unknown stack".to_string());

        let navigator = window.navigator();
        let user_agent = navigator.user_agent().unwrap_or_else(|_| "Unknown UA".to_string());

        let uptime = if let Some(perf) = window.performance() {
            perf.now()
        } else {
            0.0
        };

        let report = CrashReport {
            message,
            stack,
            user_agent,
            uptime,
        };

        // Create the UI overlay
        if let Some(document) = window.document() {
            if let Ok(div) = document.create_element("div") {
                div.set_attribute(
                    "style",
                    "position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background-color: rgba(0, 0, 0, 0.9); color: white; padding: 2rem; border-radius: 8px; z-index: 9999; text-align: center; font-family: monospace; border: 1px solid #ff4444; min-width: 300px;",
                ).unwrap_or(());

                div.set_inner_html(
                    "<h2>Engine Error</h2>\
                    <p>The Reality Engine has encountered a critical error and lost connection.</p>\
                    <p style=\"color: #ffaaaa; font-size: 0.9em;\">A crash report is being sent to the server.</p>\
                    <button onclick=\"location.reload()\" style=\"margin-top: 1rem; padding: 0.5rem 1rem; background: #444; color: white; border: none; cursor: pointer;\">Reload Page</button>"
                );

                if let Some(body) = document.body() {
                    let _ = body.append_child(&div);
                }
            }
        }

        // Send the report asynchronously
        let json = match serde_json::to_string(&report) {
            Ok(j) => j,
            Err(_) => return,
        };

        let opts = RequestInit::new();
        opts.set_method("POST");
        opts.set_mode(RequestMode::Cors);
        opts.set_body(&JsValue::from_str(&json));

        let url = "/api/crash-report";
        if let Ok(request) = Request::new_with_str_and_init(url, &opts) {
            let headers = request.headers();
            let _ = headers.set("Content-Type", "application/json");

            wasm_bindgen_futures::spawn_local(async move {
                let window = web_sys::window().unwrap();
                let _ = JsFuture::from(window.fetch_with_request(&request)).await;
            });
        }
    }));
}
