use reality_engine::genie_bridge::EngineCompanionManager;
use reality_genie::dit::EpisodicMemoryGraph;

#[test]
fn test_long_horizon_personality_co_evolution() {
    let mut manager = EngineCompanionManager::new("Samantha", [0.0, 0.0, 0.0]);
    let mut memory_graph = EpisodicMemoryGraph::new();

    // Baseline traits
    let initial_curiosity = manager.companion.personality.curiosity;

    // Simulate extensive exploratory gameplay
    for i in 0..10 {
        memory_graph.add_journal_entry(
            &format!("Discovery {}", i),
            "We found a new wonder in the latent frontier!",
            "Adventurous",
            [10.0, 0.0, 10.0],
            i,
        );
    }

    manager.companion.co_evolve_personality(&memory_graph);

    // Traits should have evolved
    assert!(manager.companion.personality.curiosity > initial_curiosity);
    assert!(manager.companion.personality.relationship_depth > 0.0);
}
