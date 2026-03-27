use cgmath::{Point3, Vector3};
use reality_engine::engine::Engine;
use reality_engine::input::{Action, InputConfig};

#[derive(Debug, PartialEq)]
pub struct LabelInfo {
    pub text: String,
    pub x: f32,
    pub y: f32,
    pub color: String,
}

pub fn parse_flat_labels(bytes: &[u8]) -> Vec<LabelInfo> {
    let mut labels = Vec::new();
    if bytes.len() < 4 {
        return labels;
    }

    let count = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let mut offset = 4;

    for _ in 0..count {
        if offset + 12 > bytes.len() {
            break;
        }

        let x = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());

        let y = f32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap());

        let r = bytes[offset + 8];
        let g = bytes[offset + 9];
        let b = bytes[offset + 10];
        let _a = bytes[offset + 11];
        let color = format!("#{:02x}{:02x}{:02x}", r, g, b);
        offset += 12;

        if offset + 2 > bytes.len() {
            break;
        }
        let text_len = u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap()) as usize;
        offset += 2;

        if offset + text_len > bytes.len() {
            break;
        }
        let text = String::from_utf8_lossy(&bytes[offset..offset + text_len]).into_owned();
        offset += text_len;

        labels.push(LabelInfo { text, x, y, color });
    }

    labels
}

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
    engine.update(0.1, None);
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
    let count = engine
        .world_state
        .chunks
        .values()
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

    assert!(
        merged,
        "Merge should return true indicating new data was integrated"
    );

    // Verify Peer B has the anomaly
    let count_b = engine_b
        .world_state
        .chunks
        .values()
        .map(|c| c.anomalies.len())
        .sum::<usize>();

    assert_eq!(
        count_b, 1,
        "Peer B should have synchronized the anomaly from Peer A"
    );
    assert_eq!(
        engine_b.world_state.root_hash, engine_a.world_state.root_hash,
        "World hashes should match after sync"
    );
}

#[test]
fn test_input_rebinding() {
    let mut engine = Engine::new(800, 600, None);

    // Set up a valid spell term so casting actually does something
    let term = reality_engine::lambda::parse("FIRE").unwrap();
    engine.lambda_system.set_term(term);

    // Default binding: CastSpell -> KeyF
    assert_eq!(
        engine.input_config.get_binding(Action::CastSpell).unwrap(),
        "KeyF"
    );

    // Simulate pressing KeyF -> Should cast spell
    // (We check logs or side effect? Anomaly count)
    // Clear initial anomalies
    engine.world_state.chunks.clear();

    engine.process_keyboard("KeyF", true);

    // Verify anomaly added (Cast Spell adds an anomaly)
    let count_f = engine
        .world_state
        .chunks
        .values()
        .map(|c| c.anomalies.len())
        .sum::<usize>();
    assert_eq!(count_f, 1, "Pressing KeyF should cast spell by default");

    // Rebind CastSpell to KeyG
    engine
        .input_config
        .set_binding(Action::CastSpell, "KeyG".to_string());
    assert_eq!(
        engine.input_config.get_binding(Action::CastSpell).unwrap(),
        "KeyG"
    );

    // Press KeyF -> Should do nothing now
    engine.process_keyboard("KeyF", false); // Release
    engine.process_keyboard("KeyF", true); // Press

    let count_f_2 = engine
        .world_state
        .chunks
        .values()
        .map(|c| c.anomalies.len())
        .sum::<usize>();
    assert_eq!(
        count_f_2, 1,
        "Pressing KeyF should NOT cast spell after rebinding"
    );

    // Press KeyG -> Should cast spell
    engine.process_keyboard("KeyG", true);

    let count_g = engine
        .world_state
        .chunks
        .values()
        .map(|c| c.anomalies.len())
        .sum::<usize>();
    assert_eq!(
        count_g, 2,
        "Pressing KeyG should cast spell after rebinding"
    );
}

#[test]
fn test_voxel_input_configuration() {
    let mut engine = Engine::new(800, 600, None);

    // Verify default bindings
    assert_eq!(
        engine
            .input_config
            .get_binding(Action::VoxelDiffusion)
            .unwrap(),
        "KeyY"
    );
    assert_eq!(
        engine
            .input_config
            .get_binding(Action::VoxelTimeReverse)
            .unwrap(),
        "KeyT"
    );
    assert_eq!(
        engine.input_config.get_binding(Action::VoxelDream).unwrap(),
        "KeyG"
    );

    // Verify reverse mapping
    assert_eq!(
        engine.input_config.map_key("KeyY"),
        Some(Action::VoxelDiffusion)
    );

    // Rebind Diffusion to KeyH
    engine
        .input_config
        .set_binding(Action::VoxelDiffusion, "KeyH".to_string());

    // Verify new binding
    assert_eq!(
        engine
            .input_config
            .get_binding(Action::VoxelDiffusion)
            .unwrap(),
        "KeyH"
    );
    assert_eq!(
        engine.input_config.map_key("KeyH"),
        Some(Action::VoxelDiffusion)
    );

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
    use cgmath::Point3;
    use reality_engine::projector::RealityProjector;
    use reality_engine::reality_types::RealitySignature;
    use reality_engine::world::{Chunk, ChunkId};

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
    assert_eq!(
        chunk1.anomalies.len(),
        2,
        "Should have 2 anomalies (1 updated, 1 new)"
    );

    // Verify update
    let updated_proj = chunk1
        .anomalies
        .iter()
        .find(|a| a.uuid == proj1.uuid)
        .expect("Proj1 should exist");
    assert_eq!(updated_proj.last_updated, 2000);
    assert_eq!(updated_proj.location.x, 10.0);

    // Verify new
    let new_proj = chunk1
        .anomalies
        .iter()
        .find(|a| a.uuid == proj2.uuid)
        .expect("Proj2 should exist");
    assert_eq!(new_proj.location.x, 5.0);

    // Test reverse merge (older into newer)
    let mut chunk3 = Chunk::new(ChunkId { x: 0, z: 0 });
    let mut proj1_old = proj1.clone();
    proj1_old.last_updated = 500;
    chunk3.anomalies.push(proj1_old);

    let changed_reverse = chunk1.merge(&chunk3);
    assert!(
        !changed_reverse,
        "Merging older data should not change anything"
    );

    let current_proj = chunk1
        .anomalies
        .iter()
        .find(|a| a.uuid == proj1.uuid)
        .unwrap();
    assert_eq!(current_proj.last_updated, 2000, "Should keep newer version");
}

#[test]
fn test_npc_evolution_and_movement() {
    use cgmath::Point3;
    use reality_engine::projector::RealityProjector;
    use reality_engine::reality_types::{RealityArchetype, RealitySignature};

    let mut engine = Engine::new(800, 600, None);

    // Clear out any default spawned NPCs and anomalies
    engine.world_state.npcs.clear();
    engine.world_state.chunks.clear();

    // Spawn an NPC preferring SciFi
    let mut sig_npc = RealitySignature::default();
    sig_npc.active_style.archetype = RealityArchetype::SciFi;
    sig_npc.fidelity = 100.0;

    let npc_start_pos = Point3::new(0.0, 1.0, 0.0);
    let mut npc = RealityProjector::new(npc_start_pos, sig_npc.clone());
    npc.behavior = Some(reality_engine::projector::NpcBehavior {
        preferred_archetype: RealityArchetype::SciFi,
        energy: 100.0,
        mutation_progress: 0.0,
    });
    engine.world_state.npcs.push(npc);

    // Add a strong Horror anomaly at the NPC's location to force mutation
    let mut sig_anomaly = RealitySignature::default();
    sig_anomaly.active_style.archetype = RealityArchetype::Horror;
    sig_anomaly.fidelity = 500.0; // Very strong
    let anomaly = RealityProjector::new(Point3::new(0.0, 0.0, 0.0), sig_anomaly);
    engine.world_state.add_anomaly(anomaly);

    // Ensure the anomaly chunk hash is updated
    engine.world_state.calculate_root_hash();

    // The dominant archetype at (0,0,0) should now be Horror
    let current_arch = engine
        .world_state
        .get_dominant_archetype_at(npc_start_pos)
        .unwrap();
    assert_eq!(
        current_arch,
        RealityArchetype::Horror,
        "Dominant archetype should be Horror"
    );

    // We need to run enough update ticks so mutation_progress hits 100.
    // 15.0 mutation per second. We need 100 / 15.0 = 6.66 seconds minimum.
    // We update with dt=1.0 ten times (10 seconds total).
    for _ in 0..10 {
        engine.update(1.0, None);
        // Force them back to center so they don't leave the anomaly chunk during their agitated wandering
        engine.world_state.npcs[0].location = npc_start_pos;
    }

    // Now verify the state
    let updated_npc = &engine.world_state.npcs[0];
    let behavior = updated_npc
        .behavior
        .as_ref()
        .expect("NPC should have behavior");

    // Verify evolution
    assert_eq!(
        behavior.preferred_archetype,
        RealityArchetype::Horror,
        "NPC should have evolved to prefer Horror"
    );
    assert_eq!(
        updated_npc.reality_signature.active_style.archetype,
        RealityArchetype::Horror,
        "NPC reality projection should have changed to Horror"
    );

    // Verify energy and mutation reset
    assert!(
        behavior.mutation_progress < 15.0,
        "Mutation progress should be very low after reset"
    );
    // After evolution, energy is set to 50.0, but on subsequent ticks of thriving it increases by 5.0 per tick.
    // So it might be 50.0 + 5.0 * remaining_ticks. We just assert it is >= 50.0
    assert!(
        behavior.energy >= 50.0,
        "Energy should have reset to at least 50.0 after evolution"
    );
}

#[test]
fn test_get_node_labels_benchmark() {
    let mut engine = Engine::new(800, 600, None);

    // Generate a very large lambda term to stress test labels
    // e.g., a long sequence of applications
    let mut term_str = String::from("FIRE");
    for _ in 0..50 {
        term_str = format!("(GROWTH {})", term_str);
    }

    let term = reality_engine::lambda::parse(&term_str).unwrap();
    engine.lambda_system.set_term(term);

    // Update to calculate node positions
    engine.update(0.1, None);
    engine.update(0.1, None);
    engine.update(0.1, None);

    // Warmup
    let _ = engine.get_node_labels_json();
    let _ = engine.get_node_labels_flat();

    // Benchmark JSON
    let start_json = std::time::Instant::now();
    for _ in 0..100 {
        let _json_output = engine.get_node_labels_json();
    }
    let duration_json = start_json.elapsed();

    // Benchmark Flat Buffer
    let start_flat = std::time::Instant::now();
    for _ in 0..100 {
        let _flat_output = engine.get_node_labels_flat();
    }
    let duration_flat = start_flat.elapsed();

    println!(
        "JSON Label Generation (100 iterations): {:?}",
        duration_json
    );
    println!(
        "Flat Buffer Label Generation (100 iterations): {:?}",
        duration_flat
    );

    // Flat buffer approach should be faster than JSON serialization
    assert!(
        duration_flat < duration_json,
        "Flat buffer approach should be faster than JSON serialization"
    );
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
    engine.update(0.1, None);
    engine.update(0.1, None);

    // Get labels
    let labels = parse_flat_labels(&engine.get_node_labels_flat());

    println!("Labels found: {:?}", labels);

    assert!(!labels.is_empty(), "Should have labels");

    // Check for "λx"
    assert!(
        labels.iter().any(|l| l.text == "λx"),
        "Should find Abs label"
    );
    // Check for "y"
    assert!(
        labels.iter().any(|l| l.text == "y"),
        "Should find free var label"
    );

    // Check visibility logic (behind camera)
    // Move camera so nodes are behind.
    // Nodes are at ~ (0, 5, 2) (Eye=10, forward=-Z, 8 units => 2)
    // Move eye to (0, 5, 20) looking at (0, 5, 30). Nodes at 2 are behind.
    engine.camera.eye = Point3::new(0.0, 5.0, 20.0);
    engine.camera.target = Point3::new(0.0, 5.0, 30.0);

    // Do NOT call update(), so nodes stay at old position
    let labels_hidden = parse_flat_labels(&engine.get_node_labels_flat());
    // Filter out UI status labels
    let lambda_labels: Vec<_> = labels_hidden
        .iter()
        .filter(|l| l.text != "STEP MODE" && l.text != "AUTO-RUN" && l.text != "PAUSED")
        .collect();
    assert!(
        lambda_labels.is_empty(),
        "Labels should be culled if behind camera"
    );
}

#[test]
fn test_lambda_layout_persistence() {
    use cgmath::Point3;
    use reality_engine::persistence::{GameState, PlayerState};
    use reality_engine::projector::RealityProjector;
    use reality_engine::reality_types::RealitySignature;
    use reality_engine::world::WorldState;

    // 1. Create a mock GameState with custom Lambda data
    let custom_layout = vec![[10.0, 20.0, 30.0]]; // Specific position for root

    let game_state = GameState {
        player: PlayerState {
            projector: RealityProjector::new(
                Point3::new(0.0, 0.0, 0.0),
                RealitySignature::default(),
            ),
        },
        world: WorldState::default(),
        lambda_source: "WATER".to_string(), // Different from default "FIRE"
        lambda_layout: custom_layout,
        input_config: InputConfig::default(), // Default input config
        voxel_world: None,
        timestamp: 0,
        version: 1,
    };

    // 2. Initialize Engine
    let engine = Engine::new(800, 600, Some(game_state));

    // 3. Verify Term
    // Should be WATER (Primitive::Water)
    if let Some(root) = &engine.lambda_system.root_term {
        // format! of term
        assert_eq!(
            format!("{:?}", root),
            "Prim(Water)",
            "Should have loaded WATER term"
        );
    } else {
        panic!("Root term should not be None");
    }

    // 4. Verify Layout
    assert!(
        !engine.lambda_system.nodes.is_empty(),
        "Nodes should not be empty"
    );
    let root_pos = engine.lambda_system.nodes[0].position;
    assert_eq!(root_pos.x, 10.0, "X Position mismatch");
    assert_eq!(root_pos.y, 20.0, "Y Position mismatch");
    assert_eq!(root_pos.z, 30.0, "Z Position mismatch");
}

#[test]
fn test_bound_variable_labels() {
    let mut engine = Engine::new(800, 600, None);

    // (\x.x) y
    // The inner 'x' is a Port.
    let term = reality_engine::lambda::parse("(\\x.x) y").unwrap();
    engine.lambda_system.set_term(term);

    // Update layout
    engine.update(0.1, None);

    // Position camera to see everything
    engine.camera.eye = Point3::new(0.0, 5.0, 20.0);
    engine.camera.target = Point3::new(0.0, 5.0, 0.0);
    engine.camera.up = Vector3::new(0.0, 1.0, 0.0);

    let labels = parse_flat_labels(&engine.get_node_labels_flat());
    println!("Labels: {:?}", labels);

    // We expect:
    // 1. "λx" (Abs)
    // 2. "y" (Free Var)
    // 3. "x" (Bound Var / Port) <-- This is what we are testing for

    let has_abs_x = labels.iter().any(|l| l.text == "λx");
    let has_free_y = labels.iter().any(|l| l.text == "y");

    // Count how many "x" labels we have.
    // One from Abs "λx" (text is "λx")
    // One from Port "x" (text should be "x")
    let has_port_x = labels.iter().any(|l| l.text == "x");

    assert!(has_abs_x, "Missing Abs label");
    assert!(has_free_y, "Missing Free Var label");
    assert!(has_port_x, "Missing Bound Var (Port) label");
}
