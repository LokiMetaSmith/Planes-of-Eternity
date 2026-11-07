use anyhow::{anyhow, Result};

#[cfg(target_arch = "wasm32")]
pub async fn fetch_safetensors(url: &str) -> Result<Vec<u8>> {
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    let global = js_sys::global();
    let cache_storage = if let Some(window) = global.dyn_ref::<web_sys::Window>() {
        window.caches().map_err(|_| anyhow!("Failed to get caches from Window"))?
    } else if let Some(worker) = global.dyn_ref::<web_sys::WorkerGlobalScope>() {
        worker.caches().map_err(|_| anyhow!("Failed to get caches from WorkerGlobalScope"))?
    } else {
        return Err(anyhow!("Could not find Window or WorkerGlobalScope"));
    };

    let cache_name = "reality-genie-models";

    let cache_promise = cache_storage.open(cache_name);
    let cache_val = JsFuture::from(cache_promise).await.map_err(|_| anyhow!("Failed to open cache"))?;
    let cache: web_sys::Cache = cache_val.unchecked_into();

    let request = web_sys::Request::new_with_str(url).map_err(|_| anyhow!("Failed to create request"))?;

    let match_promise = cache.match_with_request(&request);
    let match_val = JsFuture::from(match_promise).await.map_err(|_| anyhow!("Failed to check cache"))?;

    let response: web_sys::Response = if match_val.is_truthy() {
        match_val.unchecked_into()
    } else {
        let fetch_promise = if let Some(window) = global.dyn_ref::<web_sys::Window>() {
            window.fetch_with_request(&request)
        } else if let Some(worker) = global.dyn_ref::<web_sys::WorkerGlobalScope>() {
            worker.fetch_with_request(&request)
        } else {
            return Err(anyhow!("Cannot fetch"));
        };

        let fetch_val = JsFuture::from(fetch_promise).await.map_err(|_| anyhow!("Network fetch failed"))?;
        let resp: web_sys::Response = fetch_val.unchecked_into();

        if !resp.ok() {
            return Err(anyhow!("Failed to fetch weights: {}", resp.status()));
        }

        let resp_clone = resp.clone().map_err(|_| anyhow!("Failed to clone response"))?;
        let put_promise = cache.put_with_request(&request, &resp_clone);
        let _ = JsFuture::from(put_promise).await.map_err(|_| anyhow!("Failed to put in cache"))?;

        resp
    };

    let buffer_promise = response.array_buffer().map_err(|_| anyhow!("Failed to get array buffer promise"))?;
    let buffer_val = JsFuture::from(buffer_promise).await.map_err(|_| anyhow!("Failed to get array buffer"))?;
    let buffer: js_sys::ArrayBuffer = buffer_val.unchecked_into();
    let uint8_array = js_sys::Uint8Array::new(&buffer);

    let mut vec = vec![0; uint8_array.length() as usize];
    uint8_array.copy_to(&mut vec);

    Ok(vec)
}

#[cfg(not(target_arch = "wasm32"))]
pub async fn fetch_safetensors(url: &str) -> Result<Vec<u8>> {
    // For native, we could just read from disk or use reqwest if desired.
    // As a placeholder, we just read from a local file relative to root.
    // Or we just return an error if this is only supposed to be Wasm.
    // Let's implement a dummy fallback for host test compatibility.
    let path = std::path::Path::new(url).file_name().unwrap_or_default();
    if std::path::Path::new(path).exists() {
        let data = std::fs::read(path)?;
        Ok(data)
    } else {
        Err(anyhow!("File not found natively: {:?}", path))
    }
}
