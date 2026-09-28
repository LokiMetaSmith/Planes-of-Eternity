use cgmath::Point3;

/// Simulates the exact WebXR math logic from lib.rs
fn calc_dist_sq(finger: &Point3<f32>, wrist: &Point3<f32>) -> f32 {
    let dwx = finger.x - wrist.x;
    let dwy = finger.y - wrist.y;
    let dwz = finger.z - wrist.z;
    dwx * dwx + dwy * dwy + dwz * dwz
}

fn simulate_xr_gesture(
    thumb: Point3<f32>, index: Point3<f32>, middle: Point3<f32>, ring: Point3<f32>, pinky: Point3<f32>, wrist: Point3<f32>
) -> (bool, bool) {
    let d_thumb_sq = calc_dist_sq(&thumb, &wrist);
    let d_index_sq = calc_dist_sq(&index, &wrist);
    let d_middle_sq = calc_dist_sq(&middle, &wrist);
    let d_ring_sq = calc_dist_sq(&ring, &wrist);
    let d_pinky_sq = calc_dist_sq(&pinky, &wrist);

    let mut palm_pressed = false;
    let mut fist_pressed = false;

    if d_thumb_sq > 0.08 * 0.08 && d_index_sq > 0.11 * 0.11 &&
       d_middle_sq > 0.11 * 0.11 && d_ring_sq > 0.10 * 0.10 &&
       d_pinky_sq > 0.09 * 0.09 {
        palm_pressed = true;
    }

    if d_thumb_sq < 0.08 * 0.08 && d_index_sq < 0.08 * 0.08 &&
       d_middle_sq < 0.08 * 0.08 && d_ring_sq < 0.08 * 0.08 &&
       d_pinky_sq < 0.08 * 0.08 {
        fist_pressed = true;
    }

    (palm_pressed, fist_pressed)
}

#[test]
fn test_open_palm_gesture() {
    let wrist = Point3::new(0.0, 0.0, 0.0);
    // Simulate fingers fully extended away from wrist
    let thumb = Point3::new(0.09, 0.0, 0.0);   // > 0.08
    let index = Point3::new(0.0, 0.12, 0.0);   // > 0.11
    let middle = Point3::new(0.0, 0.12, 0.0);  // > 0.11
    let ring = Point3::new(0.0, 0.11, 0.0);    // > 0.10
    let pinky = Point3::new(0.0, 0.10, 0.0);   // > 0.09

    let (is_palm, is_fist) = simulate_xr_gesture(thumb, index, middle, ring, pinky, wrist);
    assert!(is_palm, "Expected Open Palm gesture");
    assert!(!is_fist, "Did not expect Fist gesture");
}

#[test]
fn test_fist_gesture() {
    let wrist = Point3::new(0.0, 0.0, 0.0);
    // Simulate fingers curled closely to the wrist
    let thumb = Point3::new(0.05, 0.0, 0.0);   // < 0.08
    let index = Point3::new(0.0, 0.05, 0.0);   // < 0.08
    let middle = Point3::new(0.0, 0.05, 0.0);  // < 0.08
    let ring = Point3::new(0.0, 0.05, 0.0);    // < 0.08
    let pinky = Point3::new(0.0, 0.05, 0.0);   // < 0.08

    let (is_palm, is_fist) = simulate_xr_gesture(thumb, index, middle, ring, pinky, wrist);
    assert!(!is_palm, "Did not expect Open Palm gesture");
    assert!(is_fist, "Expected Fist gesture");
}

#[test]
fn test_neutral_gesture() {
    let wrist = Point3::new(0.0, 0.0, 0.0);
    // Simulate half-curled fingers (like resting on a controller)
    let thumb = Point3::new(0.08, 0.0, 0.0);
    let index = Point3::new(0.0, 0.09, 0.0);
    let middle = Point3::new(0.0, 0.09, 0.0);
    let ring = Point3::new(0.0, 0.09, 0.0);
    let pinky = Point3::new(0.0, 0.09, 0.0);

    let (is_palm, is_fist) = simulate_xr_gesture(thumb, index, middle, ring, pinky, wrist);
    assert!(!is_palm, "Did not expect Open Palm gesture");
    assert!(!is_fist, "Did not expect Fist gesture");
}

#[test]
fn test_gesture_debouncing_logic() {
    let mut last_palm_pressed = vec![false; 4];
    let mut action_inscribe_count = 0;

    // Frame 1: Open Palm (first time)
    let is_palm_frame1 = true;
    let idx = 0;

    if is_palm_frame1 && !last_palm_pressed[idx] {
        action_inscribe_count += 1;
    }
    last_palm_pressed[idx] = is_palm_frame1;

    // Ensure it fired
    assert_eq!(action_inscribe_count, 1, "Action should fire on initial press");

    // Frame 2: Still Open Palm (held down)
    let is_palm_frame2 = true;
    if is_palm_frame2 && !last_palm_pressed[idx] {
        action_inscribe_count += 1;
    }
    last_palm_pressed[idx] = is_palm_frame2;

    // Ensure it was debounced
    assert_eq!(action_inscribe_count, 1, "Action should be debounced on hold");

    // Frame 3: Relax hand
    let is_palm_frame3 = false;
    if is_palm_frame3 && !last_palm_pressed[idx] {
        action_inscribe_count += 1;
    }
    last_palm_pressed[idx] = is_palm_frame3;

    // Frame 4: Open Palm again
    let is_palm_frame4 = true;
    if is_palm_frame4 && !last_palm_pressed[idx] {
        action_inscribe_count += 1;
    }
    last_palm_pressed[idx] = is_palm_frame4;

    // Ensure it fired again after resetting
    assert_eq!(action_inscribe_count, 2, "Action should fire again after reset");
}
