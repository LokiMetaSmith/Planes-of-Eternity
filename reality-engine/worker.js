import init, { generate_splats_worker_entry } from './reality_engine.js';

onmessage = async (event) => {
    const { prompt, task_id } = event.data;

    try {
        await init();
        const splats_json = generate_splats_worker_entry(prompt);
        postMessage({ success: true, prompt, task_id, splats_json });
    } catch (err) {
        postMessage({ success: false, prompt, task_id, error: err.toString() });
    }
};
