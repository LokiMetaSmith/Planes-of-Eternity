use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{XrSession, XrSessionMode};

#[wasm_bindgen(inline_js = "
export function request_session_with_overlay(xr, mode, dom_id) {
    let init = {};
    if (dom_id) {
        let elem = document.getElementById(dom_id);
        if (elem) {
            init = {
                optionalFeatures: ['dom-overlay'],
                domOverlay: { root: elem }
            };
        }
    }
    return xr.requestSession(mode, init);
}
")]
extern "C" {
    #[wasm_bindgen(catch)]
    pub async fn request_session_with_overlay(
        xr: &web_sys::XrSystem,
        mode: &str,
        dom_id: Option<&str>,
    ) -> Result<wasm_bindgen::JsValue, wasm_bindgen::JsValue>;
}

#[wasm_bindgen]
pub async fn is_ar_supported() -> Result<bool, JsValue> {
    let window = web_sys::window().unwrap();
    let navigator = window.navigator();

    let xr_val = js_sys::Reflect::get(&navigator, &JsValue::from_str("xr"));
    let xr = match xr_val {
        Ok(val) => {
            if val.is_undefined() {
                return Ok(false);
            }
            val.unchecked_into::<web_sys::XrSystem>()
        }
        Err(_) => return Ok(false),
    };

    let promise = xr.is_session_supported(XrSessionMode::ImmersiveAr);
    let result = wasm_bindgen_futures::JsFuture::from(promise).await?;

    Ok(result.as_bool().unwrap_or(false))
}

#[wasm_bindgen]
pub async fn request_ar_session() -> Result<XrSession, JsValue> {
    let window = web_sys::window().unwrap();
    let navigator = window.navigator();
    let xr_val = js_sys::Reflect::get(&navigator, &JsValue::from_str("xr"));
    let xr = match xr_val {
        Ok(val) => {
            if val.is_undefined() {
                return Err(JsValue::from_str("WebXR not supported"));
            }
            val.unchecked_into::<web_sys::XrSystem>()
        }
        Err(_) => return Err(JsValue::from_str("WebXR not supported")),
    };

    let session = request_session_with_overlay(&xr, "immersive-ar", Some("ui-layer")).await?;
    Ok(session.unchecked_into::<XrSession>())
}

#[wasm_bindgen]
pub async fn is_vr_supported() -> Result<bool, JsValue> {
    let window = web_sys::window().unwrap();
    let navigator = window.navigator();

    let xr_val = js_sys::Reflect::get(&navigator, &JsValue::from_str("xr"));
    let xr = match xr_val {
        Ok(val) => {
            if val.is_undefined() {
                return Ok(false);
            }
            val.unchecked_into::<web_sys::XrSystem>()
        }
        Err(_) => return Ok(false),
    };

    let promise = xr.is_session_supported(XrSessionMode::ImmersiveVr);
    let result = wasm_bindgen_futures::JsFuture::from(promise).await?;

    Ok(result.as_bool().unwrap_or(false))
}

#[wasm_bindgen]
pub async fn request_vr_session() -> Result<XrSession, JsValue> {
    let window = web_sys::window().unwrap();
    let navigator = window.navigator();
    let xr_val = js_sys::Reflect::get(&navigator, &JsValue::from_str("xr"));
    let xr = match xr_val {
        Ok(val) => {
            if val.is_undefined() {
                return Err(JsValue::from_str("WebXR not supported"));
            }
            val.unchecked_into::<web_sys::XrSystem>()
        }
        Err(_) => return Err(JsValue::from_str("WebXR not supported")),
    };

    let session = request_session_with_overlay(&xr, "immersive-vr", Some("ui-layer")).await?;
    Ok(session.unchecked_into::<XrSession>())
}
