use reality_engine::engine::{Engine, PhysicsState};
use reality_engine::input::Action;
use reality_engine::lambda::{parse, Primitive};

#[test]
fn test_gravity_and_fly_spells() {
    // 1. Initialize engine
    let mut engine = Engine::new(800, 600, None);

    // Engine starts in Flying state by default after reset/init
    assert_eq!(engine.physics_state, PhysicsState::Flying);

    // 2. Set player position high up
    engine.camera.eye = cgmath::Point3::new(0.0, 10.0, 0.0);
    let start_y = engine.camera.eye.y;

    // 3. Simulate a few frames in Fly mode (no input)
    engine.update(0.1, None);
    engine.update(0.1, None);

    // Player should not fall
    assert_eq!(
        engine.camera.eye.y, start_y,
        "Player should not fall in Flying state"
    );

    // 4. Apply GRAVITY spell
    engine.process_inscription("GRAVITY"); // Parses to Term::Prim(Gravity)
                                           // Send CastSpell action to execute the spell
    engine
        .camera_controller
        .process_action(Action::CastSpell, true);
    engine.process_keyboard("KeyF", true); // Trigger the cast

    // Verify state changed
    assert_eq!(engine.physics_state, PhysicsState::Gravity);

    // 5. Simulate frames to let gravity act
    engine.update(0.1, None);
    engine.update(0.1, None);

    // Player should have fallen
    assert!(
        engine.camera.eye.y < start_y,
        "Player should fall when Gravity is active"
    );
    let current_y = engine.camera.eye.y;

    // 6. Apply LEVITATE spell (Fly composite base)
    engine.process_inscription("LEVITATE");
    engine.process_keyboard("KeyF", true);

    // Verify state changed
    assert_eq!(engine.physics_state, PhysicsState::Flying);

    // 7. Simulate frames in Fly mode with Jump/Up pressed
    engine.camera_controller.process_action(Action::Jump, true);
    engine.update(0.1, None);
    engine.update(0.1, None);

    // Player should move up, overriding the falling velocity
    assert!(
        engine.camera.eye.y > current_y,
        "Player should fly upward when Levitate and Jump are active"
    );
}
