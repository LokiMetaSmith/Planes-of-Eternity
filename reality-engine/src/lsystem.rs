use std::collections::HashMap;
use cgmath::{Matrix3, InnerSpace, Vector3, Point3, Deg};

pub struct LSystem {
    axiom: String,
    rules: HashMap<char, String>,
}

impl LSystem {
    pub fn new(axiom: &str) -> Self {
        Self {
            axiom: axiom.to_string(),
            rules: HashMap::new(),
        }
    }

    pub fn add_rule(&mut self, predecessor: char, successor: &str) {
        self.rules.insert(predecessor, successor.to_string());
    }

    pub fn evaluate(&self, iterations: usize) -> String {
        let mut current = self.axiom.clone();
        for _ in 0..iterations {
            let mut next = String::new();
            for c in current.chars() {
                if let Some(successor) = self.rules.get(&c) {
                    next.push_str(successor);
                } else {
                    next.push(c);
                }
            }
            current = next;
        }
        current
    }
}

pub struct TurtleState {
    pub position: Vector3<f32>,
    pub direction: Vector3<f32>,
    pub up: Vector3<f32>,
    pub right: Vector3<f32>,
}

impl TurtleState {
    pub fn new(position: Vector3<f32>) -> Self {
        Self {
            position,
            direction: Vector3::new(0.0, 1.0, 0.0), // Pointing Up
            up: Vector3::new(0.0, 0.0, -1.0),       // "Up" for the turtle is backwards initially
            right: Vector3::new(1.0, 0.0, 0.0),
        }
    }

    fn orthonormalize(&mut self) {
        self.direction = self.direction.normalize();
        // Re-derive right to be perpendicular to direction and current up
        self.right = self.up.cross(self.direction).normalize();
        // Re-derive up to be perpendicular to direction and right
        self.up = self.direction.cross(self.right).normalize();
    }

    // Rotate pitch (around right vector)
    pub fn pitch(&mut self, angle_deg: f32) {
        let rot = Matrix3::from_axis_angle(self.right, Deg(angle_deg));
        self.direction = rot * self.direction;
        self.up = rot * self.up;
        self.orthonormalize();
    }

    // Rotate yaw (around up vector)
    pub fn yaw(&mut self, angle_deg: f32) {
        let rot = Matrix3::from_axis_angle(self.up, Deg(angle_deg));
        self.direction = rot * self.direction;
        self.right = rot * self.right;
        self.orthonormalize();
    }

    // Rotate roll (around direction vector)
    pub fn roll(&mut self, angle_deg: f32) {
        let rot = Matrix3::from_axis_angle(self.direction, Deg(angle_deg));
        self.up = rot * self.up;
        self.right = rot * self.right;
        self.orthonormalize();
    }

    pub fn forward(&mut self, distance: f32) {
        self.position += self.direction * distance;
    }
}

pub struct LSystemTurtle {
    state: TurtleState,
    stack: Vec<TurtleState>,
    angle: f32,
    distance: f32,
    pub voxels: Vec<Point3<i32>>,
}

impl LSystemTurtle {
    pub fn new(start_pos: Point3<i32>, angle: f32, distance: f32) -> Self {
        Self {
            state: TurtleState::new(Vector3::new(start_pos.x as f32, start_pos.y as f32, start_pos.z as f32)),
            stack: Vec::new(),
            angle,
            distance,
            voxels: Vec::new(),
        }
    }

    pub fn generate(&mut self, sentence: &str) {
        // Record starting position
        self.voxels.push(Point3::new(
            self.state.position.x.round() as i32,
            self.state.position.y.round() as i32,
            self.state.position.z.round() as i32,
        ));

        for c in sentence.chars() {
            match c {
                'F' | 'A' | 'B' | '1' | '0' => {
                    self.state.forward(self.distance);
                    let p = Point3::new(
                        self.state.position.x.round() as i32,
                        self.state.position.y.round() as i32,
                        self.state.position.z.round() as i32,
                    );
                    self.voxels.push(p);
                }
                'f' => {
                    self.state.forward(self.distance);
                }
                '+' => self.state.yaw(self.angle),
                '-' => self.state.yaw(-self.angle),
                '&' => self.state.pitch(self.angle),
                '^' => self.state.pitch(-self.angle),
                '<' => self.state.roll(self.angle),
                '>' => self.state.roll(-self.angle),
                '|' => self.state.yaw(180.0), // Turn around
                '[' => {
                    self.stack.push(TurtleState {
                        position: self.state.position,
                        direction: self.state.direction,
                        up: self.state.up,
                        right: self.state.right,
                    });
                }
                ']' => {
                    if let Some(state) = self.stack.pop() {
                        self.state = state;
                    }
                }
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_turtle_orthonormalization() {
        let mut state = TurtleState::new(Vector3::new(0.0, 0.0, 0.0));

        // Calculate initial determinant to ensure handedness is preserved
        let initial_det = state.right.x * (state.direction.y * state.up.z - state.up.y * state.direction.z)
                - state.right.y * (state.direction.x * state.up.z - state.up.x * state.direction.z)
                + state.right.z * (state.direction.x * state.up.y - state.up.x * state.direction.y);

        // Perform many rotations to accumulate potential drift
        for _ in 0..1000 {
            state.pitch(33.3);
            state.yaw(45.0);
            state.roll(12.7);
        }

        // Check unit lengths
        assert_relative_eq!(state.direction.magnitude(), 1.0, epsilon = 1e-6);
        assert_relative_eq!(state.up.magnitude(), 1.0, epsilon = 1e-6);
        assert_relative_eq!(state.right.magnitude(), 1.0, epsilon = 1e-6);

        // Check perpendicularity (dot products should be 0)
        assert_relative_eq!(state.direction.dot(state.up), 0.0, epsilon = 1e-6);
        assert_relative_eq!(state.direction.dot(state.right), 0.0, epsilon = 1e-6);
        assert_relative_eq!(state.up.dot(state.right), 0.0, epsilon = 1e-6);

        // Check handedness preservation
        let det = state.right.x * (state.direction.y * state.up.z - state.up.y * state.direction.z)
                - state.right.y * (state.direction.x * state.up.z - state.up.x * state.direction.z)
                + state.right.z * (state.direction.x * state.up.y - state.up.x * state.direction.y);
        assert_relative_eq!(det, initial_det, epsilon = 1e-6);
    }
}
