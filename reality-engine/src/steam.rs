use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::window;
use js_sys::Promise;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = ["window", "__TAURI__", "core"], catch)]
    fn invoke(cmd: &str, args: JsValue) -> Result<Promise, JsValue>;
}

pub struct SteamClient;

impl SteamClient {
    pub async fn init(app_id: u32) -> Result<String, JsValue> {
        let args = js_sys::Object::new();
        js_sys::Reflect::set(&args, &JsValue::from_str("appId"), &JsValue::from_f64(app_id as f64))?;

        match invoke("init_steamworks", args.into()) {
            Ok(promise) => {
                let result = JsFuture::from(promise).await?;
                Ok(result.as_string().unwrap_or_default())
            }
            Err(e) => Err(e),
        }
    }

    pub async fn is_running() -> Result<bool, JsValue> {
        match invoke("is_steam_running", JsValue::UNDEFINED) {
            Ok(promise) => {
                let result = JsFuture::from(promise).await?;
                Ok(result.as_bool().unwrap_or(false))
            }
            Err(e) => Err(e),
        }
    }

    pub async fn get_id() -> Result<String, JsValue> {
        match invoke("get_steam_id", JsValue::UNDEFINED) {
            Ok(promise) => {
                let result = JsFuture::from(promise).await?;
                Ok(result.as_string().unwrap_or_default())
            }
            Err(e) => Err(e),
        }
    }

    pub async fn get_name() -> Result<String, JsValue> {
        match invoke("get_steam_name", JsValue::UNDEFINED) {
            Ok(promise) => {
                let result = JsFuture::from(promise).await?;
                Ok(result.as_string().unwrap_or_default())
            }
            Err(e) => Err(e),
        }
    }
}
