use reality_engine::genie_bridge::LatentPhysicsEngine;

#[test]
fn test_ragdoll_ik_constraints() {
    let mut engine = LatentPhysicsEngine::default();
    engine.spawn_ragdoll([0.0, 5.0, 0.0]);

    assert_eq!(engine.ragdolls.len(), 1);
    assert_eq!(engine.bodies.len(), 14);

    let latents = vec![0.0; 256];

    for _ in 0..10 {
        engine.step(0.016, &latents);
    }

    let ragdoll = &engine.ragdolls[0];
    for &(id_a, id_b, target_dist) in &ragdoll.constraints {
        let body_a = engine.bodies.iter().find(|b| b.id == id_a).unwrap();
        let body_b = engine.bodies.iter().find(|b| b.id == id_b).unwrap();

        let dx = body_b.position[0] - body_a.position[0];
        let dy = body_b.position[1] - body_a.position[1];
        let dz = body_b.position[2] - body_a.position[2];
        let current_dist = (dx * dx + dy * dy + dz * dz).sqrt();

        // Assert distance satisfies constraint to within a small tolerance
        assert!((current_dist - target_dist).abs() < 0.05);
    }
}
