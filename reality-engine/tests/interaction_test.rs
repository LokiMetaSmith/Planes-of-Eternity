use cgmath::{Point3, Vector3};
use reality_engine::engine::Engine;
use reality_engine::visual_lambda::{NodeType, VisualNode};
use reality_engine::lambda::Term;
use std::rc::Rc;

#[test]
fn test_visual_lambda_interactions() {
    let mut engine = Engine::new(800, 600, None);

    // Give it a dummy node we can interact with
    let node = VisualNode {
        id: 0,
        term: Rc::new(Term::Var("dummy".to_string())),
        position: Point3::new(0.0, 0.0, 0.0),
        target_position: Point3::new(0.0, 0.0, 0.0),
        velocity: Vector3::new(0.0, 0.0, 0.0),
        color: [1.0; 4],
        scale: 1.0,
        node_type: NodeType::Var("dummy".to_string()),
        collapsed: false,
        is_editing: false,
        is_pinned: false,
    };
    engine.lambda_system.nodes.clear();
    engine.lambda_system.nodes.push(node);

    // We mock ray intersection to hit node 0 for our tests
    engine.camera.eye = Point3::new(0.0, 0.0, 5.0);
    engine.camera.target = Point3::new(0.0, 0.0, 0.0);
    engine.camera.up = Vector3::new(0.0, 1.0, 0.0);

    // The node is at 0,0,0 and has radius 1.0
    // Click at center (0,0) in NDC should send ray through origin

    // 1. Test Pinning (Left Click quickly)
    assert_eq!(engine.lambda_system.nodes[0].is_pinned, false);
    engine.process_click(0.0, 0.0); // should toggle pin
    assert_eq!(engine.lambda_system.nodes[0].is_pinned, true);
    engine.process_click(0.0, 0.0);
    assert_eq!(engine.lambda_system.nodes[0].is_pinned, false);

    // 2. Test Editing (Right hold)
    assert_eq!(engine.lambda_system.nodes[0].is_editing, false);
    engine.process_right_hold(0.0, 0.0);
    assert_eq!(engine.lambda_system.nodes[0].is_editing, true);
    engine.process_right_hold(0.0, 0.0);
    assert_eq!(engine.lambda_system.nodes[0].is_editing, false);

    // 3. Test Collapse (Right click)
    assert_eq!(engine.lambda_system.nodes[0].collapsed, false);
    engine.process_right_click(0.0, 0.0);
    assert_eq!(engine.lambda_system.nodes[0].collapsed, true);

    // 4. Test Scroll Wheel (Anchor Distance)
    assert_eq!(engine.anchor_distance, 8.0);
    engine.process_mouse_wheel(100.0);
    assert_eq!(engine.anchor_distance, 9.0);
    engine.process_mouse_wheel(-200.0);
    assert_eq!(engine.anchor_distance, 7.0);
}
