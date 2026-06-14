#[cfg(target_arch = "wasm32")]
pub fn test(r: web_sys::Response) {
    let r_clone: Result<web_sys::Response, wasm_bindgen::JsValue> = r.clone();
}
