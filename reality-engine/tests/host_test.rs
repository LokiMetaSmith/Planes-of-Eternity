use reality_engine::engine::Engine;
use cgmath::{Point3, Vector3};

#[test]
fn test_engine_initialization() {
    let engine = Engine::new(800, 600, None);
    assert_eq!(engine.width, 800);
    assert_eq!(engine.height, 600);
    assert!(engine.world_state.chunks.is_empty());
}

#[test]
fn test_engine_update() {
    let mut engine = Engine::new(800, 600, None);
    engine.update(0.1);
    assert!(engine.time > 0.0);
}

#[test]
fn test_click_places_anomaly() {
    let mut engine = Engine::new(800, 600, None);

    // Clear lambda nodes to avoid obstruction
    engine.lambda_system.nodes.clear();

    // Camera at (0, 10, 0) looking down (0, 0, 0)
    engine.camera.eye = Point3::new(0.0, 10.0, 0.0);
    engine.camera.target = Point3::new(0.0, 0.0, 0.0);

    // Y-up system. Looking from (0,10,0) to (0,0,0) is looking down -Y.
    // We can use -Z as Up for the camera orientation calculation in this specific top-down view.
    engine.camera.up = Vector3::new(0.0, 0.0, -1.0);

    // Click at center of screen (0, 0)
    // Should cast ray straight down to (0, 0, 0)
    let changed = engine.process_click(0.0, 0.0);

    assert!(changed, "Click should have placed an anomaly");

    // Verify anomaly count (iterating chunks)
    let count = engine.world_state.chunks.values()
        .map(|c| c.anomalies.len())
        .sum::<usize>();

    assert_eq!(count, 1, "There should be 1 anomaly in the world");
}

#[test]
fn test_p2p_merge_logic() {
    // Simulate Peer A
    let mut engine_a = Engine::new(800, 600, None);

    // Clear lambda nodes to avoid obstruction
    engine_a.lambda_system.nodes.clear();

    engine_a.camera.eye = Point3::new(0.0, 10.0, 0.0);
    engine_a.camera.target = Point3::new(0.0, 0.0, 0.0);
    engine_a.camera.up = Vector3::new(0.0, 0.0, -1.0);

    // Peer A places an anomaly
    engine_a.process_click(0.0, 0.0);
    assert!(!engine_a.world_state.chunks.is_empty());

    // Simulate Peer B (starts empty)
    let mut engine_b = Engine::new(800, 600, None);
    assert!(engine_b.world_state.chunks.is_empty());

    // Peer B receives WorldUpdate from Peer A
    // (In reality this goes over network and is deserialized, here we just clone)
    let remote_world_state = engine_a.world_state.clone();

    // Merge
    let merged = engine_b.world_state.merge(remote_world_state);

    assert!(merged, "Merge should return true indicating new data was integrated");

    // Verify Peer B has the anomaly
    let count_b = engine_b.world_state.chunks.values()
        .map(|c| c.anomalies.len())
        .sum::<usize>();

    assert_eq!(count_b, 1, "Peer B should have synchronized the anomaly from Peer A");
    assert_eq!(engine_b.world_state.root_hash, engine_a.world_state.root_hash, "World hashes should match after sync");
}
