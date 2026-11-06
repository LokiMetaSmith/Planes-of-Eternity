use reality_engine::genie_bridge::EngineCompanionManager;
use reality_genie::dit::{AgenticCompanion, MultiAgentCoDiffusion};

#[test]
fn test_swarm_party_movement() {
    let mut manager = EngineCompanionManager::new("Samantha", [0.0, 0.0, 0.0]);
    manager.recruit_companion(AgenticCompanion::new("Leo", [1.0, 0.0, 1.0]));
    manager.recruit_companion(AgenticCompanion::new("Aria", [-1.0, 0.0, -1.0]));

    let mut swarm_engine = MultiAgentCoDiffusion::new(64, 16);

    let player_pos = [10.0, 0.0, 10.0];
    let dt = 0.1;

    let mut banter_triggered = false;
    for _ in 0..400 {
        if let Some(msg) = manager.update(player_pos, dt, &mut swarm_engine) {
            if msg.contains("Look at the latent resonance here") {
                banter_triggered = true;
                break;
            }
        }
    }

    assert_eq!(manager.companion_party.len(), 3);
    for member in &manager.companion_party {
        // Assert they moved towards the player
        assert!(member.position[0] > 0.0);
        assert!(member.position[2] > 0.0);
    }

    // Assert emergent dialogue was triggered by the update loop
    assert!(banter_triggered, "Expected autonomous party banter to trigger");
}
