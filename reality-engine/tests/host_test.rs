use reality_engine::engine::Engine;
use reality_engine::input::{Action, InputConfig};
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

#[test]
fn test_input_rebinding() {
    let mut engine = Engine::new(800, 600, None);

    // Set up a valid spell term so casting actually does something
    let term = reality_engine::lambda::parse("FIRE").unwrap();
    engine.lambda_system.set_term(term);

    // Default binding: CastSpell -> KeyF
    assert_eq!(engine.input_config.get_binding(Action::CastSpell).unwrap(), "KeyF");

    // Simulate pressing KeyF -> Should cast spell
    // (We check logs or side effect? Anomaly count)
    // Clear initial anomalies
    engine.world_state.chunks.clear();

    engine.process_keyboard("KeyF", true);

    // Verify anomaly added (Cast Spell adds an anomaly)
    let count_f = engine.world_state.chunks.values()
        .map(|c| c.anomalies.len())
        .sum::<usize>();
    assert_eq!(count_f, 1, "Pressing KeyF should cast spell by default");

    // Rebind CastSpell to KeyG
    engine.input_config.set_binding(Action::CastSpell, "KeyG".to_string());
    assert_eq!(engine.input_config.get_binding(Action::CastSpell).unwrap(), "KeyG");

    // Press KeyF -> Should do nothing now
    engine.process_keyboard("KeyF", false); // Release
    engine.process_keyboard("KeyF", true);  // Press

    let count_f_2 = engine.world_state.chunks.values()
        .map(|c| c.anomalies.len())
        .sum::<usize>();
    assert_eq!(count_f_2, 1, "Pressing KeyF should NOT cast spell after rebinding");

    // Press KeyG -> Should cast spell
    engine.process_keyboard("KeyG", true);

    let count_g = engine.world_state.chunks.values()
        .map(|c| c.anomalies.len())
        .sum::<usize>();
    assert_eq!(count_g, 2, "Pressing KeyG should cast spell after rebinding");
}

#[test]
fn test_voxel_input_configuration() {
    let mut engine = Engine::new(800, 600, None);

    // Verify default bindings
    assert_eq!(engine.input_config.get_binding(Action::VoxelDiffusion).unwrap(), "KeyY");
    assert_eq!(engine.input_config.get_binding(Action::VoxelTimeReverse).unwrap(), "KeyT");
    assert_eq!(engine.input_config.get_binding(Action::VoxelDream).unwrap(), "KeyG");

    // Verify reverse mapping
    assert_eq!(engine.input_config.map_key("KeyY"), Some(Action::VoxelDiffusion));

    // Rebind Diffusion to KeyH
    engine.input_config.set_binding(Action::VoxelDiffusion, "KeyH".to_string());

    // Verify new binding
    assert_eq!(engine.input_config.get_binding(Action::VoxelDiffusion).unwrap(), "KeyH");
    assert_eq!(engine.input_config.map_key("KeyH"), Some(Action::VoxelDiffusion));

    // Verify old key is unmapped (or at least not mapped to Diffusion)
    // Note: implementation of set_binding might not remove the old key from reverse_bindings if we don't clear it explicitly,
    // but update_reverse_bindings clears everything and rebuilds.
    // However, if multiple keys mapped to same action? No, HashMap<Action, String>. One key per action.
    // What if multiple Actions mapped to same key? HashMap<String, Action> implies one Action per key.
    // So "KeyY" should now be unmapped or map to nothing.
    assert_eq!(engine.input_config.map_key("KeyY"), None);
}

#[test]
fn test_merge_conflict_resolution() {
    use reality_engine::projector::RealityProjector;
    use reality_engine::reality_types::{RealitySignature, RealityArchetype};
    use reality_engine::world::{Chunk, ChunkId};
    use cgmath::Point3;

    let mut chunk1 = Chunk::new(ChunkId { x: 0, z: 0 });
    let mut chunk2 = Chunk::new(ChunkId { x: 0, z: 0 });

    let sig = RealitySignature::default();

    // Create base projector
    let mut proj1 = RealityProjector::new(Point3::new(0.0, 0.0, 0.0), sig.clone());
    proj1.last_updated = 1000;

    // Add to chunk1
    chunk1.anomalies.push(proj1.clone());

    // Create conflicting projector (same UUID, newer timestamp, different location)
    let mut proj1_update = proj1.clone();
    proj1_update.location = Point3::new(10.0, 10.0, 10.0);
    proj1_update.last_updated = 2000;

    // Add to chunk2
    chunk2.anomalies.push(proj1_update.clone());

    // Create non-conflicting projector (different UUID)
    let proj2 = RealityProjector::new(Point3::new(5.0, 5.0, 5.0), sig.clone());
    chunk2.anomalies.push(proj2.clone());

    // Merge chunk2 into chunk1
    let changed = chunk1.merge(&chunk2);

    assert!(changed, "Merge should happen");
    assert_eq!(chunk1.anomalies.len(), 2, "Should have 2 anomalies (1 updated, 1 new)");

    // Verify update
    let updated_proj = chunk1.anomalies.iter().find(|a| a.uuid == proj1.uuid).expect("Proj1 should exist");
    assert_eq!(updated_proj.last_updated, 2000);
    assert_eq!(updated_proj.location.x, 10.0);

    // Verify new
    let new_proj = chunk1.anomalies.iter().find(|a| a.uuid == proj2.uuid).expect("Proj2 should exist");
    assert_eq!(new_proj.location.x, 5.0);

    // Test reverse merge (older into newer)
    let mut chunk3 = Chunk::new(ChunkId { x: 0, z: 0 });
    let mut proj1_old = proj1.clone();
    proj1_old.last_updated = 500;
    chunk3.anomalies.push(proj1_old);

    let changed_reverse = chunk1.merge(&chunk3);
    assert!(!changed_reverse, "Merging older data should not change anything");

    let current_proj = chunk1.anomalies.iter().find(|a| a.uuid == proj1.uuid).unwrap();
    assert_eq!(current_proj.last_updated, 2000, "Should keep newer version");
}

#[test]
fn test_get_node_labels() {
    let mut engine = Engine::new(800, 600, None);

    // Setup Lambda system with a known term
    // (\x.x) y
    let term = reality_engine::lambda::parse("(\\x.x) y").unwrap();
    engine.lambda_system.set_term(term);

    // Set camera to look at the expected node position
    // Engine update places nodes at camera.eye + forward * 8.0
    engine.camera.eye = Point3::new(0.0, 5.0, 10.0);
    engine.camera.target = Point3::new(0.0, 5.0, 0.0); // Forward is (0, 0, -1) roughly
    engine.camera.up = Vector3::new(0.0, 1.0, 0.0);

    // Update to calculate node positions
    engine.update(0.1);

    // Get labels
    let labels = engine.get_node_labels();

    println!("Labels found: {:?}", labels);

    assert!(!labels.is_empty(), "Should have labels");

    // Check for "λx"
    assert!(labels.iter().any(|l| l.text == "λx"), "Should find Abs label");
    // Check for "y"
    assert!(labels.iter().any(|l| l.text == "y"), "Should find free var label");

    // Check visibility logic (behind camera)
    // Move camera so nodes are behind.
    // Nodes are at ~ (0, 5, 2) (Eye=10, forward=-Z, 8 units => 2)
    // Move eye to (0, 5, 20) looking at (0, 5, 30). Nodes at 2 are behind.
    engine.camera.eye = Point3::new(0.0, 5.0, 20.0);
    engine.camera.target = Point3::new(0.0, 5.0, 30.0);

    // Do NOT call update(), so nodes stay at old position
    let labels_hidden = engine.get_node_labels();
    // Filter out UI status labels
    let lambda_labels: Vec<_> = labels_hidden.iter()
        .filter(|l| l.text != "STEP MODE" && l.text != "AUTO-RUN" && l.text != "PAUSED")
        .collect();
    assert!(lambda_labels.is_empty(), "Labels should be culled if behind camera");
}

#[test]
fn test_lambda_layout_persistence() {
    use reality_engine::persistence::{GameState, PlayerState};
    use reality_engine::world::WorldState;
    use reality_engine::projector::RealityProjector;
    use reality_engine::reality_types::{RealitySignature, RealityArchetype};
    use cgmath::Point3;

    // 1. Create a mock GameState with custom Lambda data
    let mut custom_layout = Vec::new();
    custom_layout.push([10.0, 20.0, 30.0]); // Specific position for root

    let game_state = GameState {
        player: PlayerState {
            projector: RealityProjector::new(Point3::new(0.0, 0.0, 0.0), RealitySignature::default()),
        },
        world: WorldState::default(),
        lambda_source: "WATER".to_string(), // Different from default "FIRE"
        lambda_layout: custom_layout,
        input_config: InputConfig::default(), // Default input config
        timestamp: 0,
        version: 1,
    };

    // 2. Initialize Engine
    let engine = Engine::new(800, 600, Some(game_state));

    // 3. Verify Term
    // Should be WATER (Primitive::Water)
    if let Some(root) = &engine.lambda_system.root_term {
        // format! of term
        assert_eq!(format!("{:?}", root), "Prim(Water)", "Should have loaded WATER term");
    } else {
        panic!("Root term should not be None");
    }

    // 4. Verify Layout
    assert!(!engine.lambda_system.nodes.is_empty(), "Nodes should not be empty");
    let root_pos = engine.lambda_system.nodes[0].position;
    assert_eq!(root_pos.x, 10.0, "X Position mismatch");
    assert_eq!(root_pos.y, 20.0, "Y Position mismatch");
    assert_eq!(root_pos.z, 30.0, "Z Position mismatch");
}
