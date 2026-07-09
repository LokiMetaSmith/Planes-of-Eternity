use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{IdbDatabase, IdbFactory, IdbOpenDbRequest, IdbTransactionMode, IdbRequest};
use js_sys::Promise;
use wasm_bindgen_futures::JsFuture;

const DB_NAME: &str = "RealityEngineDB";
const STORE_NAME: &str = "saves";
const DB_VERSION: u32 = 1;

pub struct IndexedDb {
    db: IdbDatabase,
}

impl IndexedDb {
    pub async fn new() -> Result<Self, JsValue> {
        let window = web_sys::window().ok_or("No window found")?;
        let factory: IdbFactory = window.indexed_db()?.ok_or("IndexedDB not supported")?;
        let request: IdbOpenDbRequest = factory.open_with_u32(DB_NAME, DB_VERSION)?;

        // Handle upgrade separately as it cannot be part of the general success/error promise
        let on_upgrade = Closure::wrap(Box::new(move |event: web_sys::IdbVersionChangeEvent| {
            let target = event.target().unwrap();
            let request = target.dyn_into::<IdbOpenDbRequest>().unwrap();
            let db = request.result().unwrap().dyn_into::<IdbDatabase>().unwrap();
            if !db.object_store_names().contains(STORE_NAME) {
                db.create_object_store(STORE_NAME).unwrap();
            }
        }) as Box<dyn FnMut(_)>);
        request.set_onupgradeneeded(Some(on_upgrade.as_ref().unchecked_ref()));

        let db_js = Self::wait_for_request(request.into()).await?;
        let db: IdbDatabase = db_js.dyn_into()?;

        on_upgrade.forget();

        Ok(Self { db })
    }

    async fn wait_for_request(request: IdbRequest) -> Result<JsValue, JsValue> {
        let promise = Promise::new(&mut |resolve, reject| {
            let on_success = Closure::once(move |event: web_sys::Event| {
                let target = event.target().unwrap();
                let request = target.dyn_into::<IdbRequest>().unwrap();
                let result = request.result().unwrap();
                resolve.call1(&JsValue::NULL, &result).unwrap();
            });
            let on_error = Closure::once(move |event: web_sys::Event| {
                reject.call1(&JsValue::NULL, &event).unwrap();
            });

            request.set_onsuccess(Some(on_success.as_ref().unchecked_ref()));
            request.set_onerror(Some(on_error.as_ref().unchecked_ref()));

            on_success.forget();
            on_error.forget();
        });

        JsFuture::from(promise).await
    }

    pub async fn save(&self, key: &str, value: &str) -> Result<(), JsValue> {
        let transaction = self.db.transaction_with_str_and_mode(STORE_NAME, IdbTransactionMode::Readwrite)?;
        let store = transaction.object_store(STORE_NAME)?;
        let request = store.put_with_key(&JsValue::from_str(value), &JsValue::from_str(key))?;
        Self::wait_for_request(request).await?;
        Ok(())
    }

    pub async fn load(&self, key: &str) -> Result<Option<String>, JsValue> {
        let transaction = self.db.transaction_with_str(STORE_NAME)?;
        let store = transaction.object_store(STORE_NAME)?;
        let request = store.get(&JsValue::from_str(key))?;

        let result = Self::wait_for_request(request).await?;
        if result.is_null() || result.is_undefined() {
            Ok(None)
        } else {
            Ok(Some(result.as_string().unwrap_or_default()))
        }
    }

    pub async fn delete(&self, key: &str) -> Result<(), JsValue> {
        let transaction = self.db.transaction_with_str_and_mode(STORE_NAME, IdbTransactionMode::Readwrite)?;
        let store = transaction.object_store(STORE_NAME)?;
        let request = store.delete(&JsValue::from_str(key))?;
        Self::wait_for_request(request).await?;
        Ok(())
    }

    pub async fn list_keys(&self) -> Result<Vec<String>, JsValue> {
        let transaction = self.db.transaction_with_str(STORE_NAME)?;
        let store = transaction.object_store(STORE_NAME)?;
        let request = store.get_all_keys()?;

        let result = Self::wait_for_request(request).await?;
        let array = result.dyn_into::<js_sys::Array>()?;
        let mut keys = Vec::new();
        for i in 0..array.length() {
            if let Some(key) = array.get(i).as_string() {
                keys.push(key);
            }
        }
        Ok(keys)
    }
}
