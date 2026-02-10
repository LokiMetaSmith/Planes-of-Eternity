use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
// use web_sys::{XrSystem, XrSession, XrSessionMode};

#[wasm_bindgen]
pub async fn is_ar_supported() -> Result<JsValue, JsValue> {
    let window = web_sys::window().unwrap();
    let navigator = window.navigator();

    // Dynamic access to navigator.xr to avoid feature flag issues
    let xr = js_sys::Reflect::get(&navigator, &JsValue::from_str("xr"))?;

    if xr.is_undefined() {
        return Ok(JsValue::from_bool(false));
    }

    // xr.isSessionSupported('immersive-ar')
    let is_session_supported = js_sys::Reflect::get(&xr, &JsValue::from_str("isSessionSupported"))?;
    let is_session_supported_fn = is_session_supported.dyn_into::<js_sys::Function>()?;

    let args = js_sys::Array::new();
    args.push(&JsValue::from_str("immersive-ar"));

    let promise = is_session_supported_fn.apply(&xr, &args)?;
    wasm_bindgen_futures::JsFuture::from(js_sys::Promise::from(promise)).await
}

#[wasm_bindgen]
pub async fn request_ar_session() -> Result<JsValue, JsValue> {
    let window = web_sys::window().unwrap();
    let navigator = window.navigator();
    let xr = js_sys::Reflect::get(&navigator, &JsValue::from_str("xr"))?;

    if xr.is_undefined() {
        return Err(JsValue::from_str("WebXR not supported"));
    }

    // xr.requestSession('immersive-ar')
    let request_session = js_sys::Reflect::get(&xr, &JsValue::from_str("requestSession"))?;
    let request_session_fn = request_session.dyn_into::<js_sys::Function>()?;

    let args = js_sys::Array::new();
    args.push(&JsValue::from_str("immersive-ar"));

    let promise = request_session_fn.apply(&xr, &args)?;
    wasm_bindgen_futures::JsFuture::from(js_sys::Promise::from(promise)).await
}
