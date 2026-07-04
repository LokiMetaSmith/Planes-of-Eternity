use std::collections::HashMap;

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
    pub position: cgmath::Vector3<f32>,
    pub direction: cgmath::Vector3<f32>,
    pub up: cgmath::Vector3<f32>,
    pub right: cgmath::Vector3<f32>,
}

impl TurtleState {
    pub fn new(position: cgmath::Vector3<f32>) -> Self {
        Self {
            position,
            direction: cgmath::Vector3::new(0.0, 1.0, 0.0), // Pointing Up
            up: cgmath::Vector3::new(0.0, 0.0, -1.0),       // "Up" for the turtle is backwards initially
            right: cgmath::Vector3::new(1.0, 0.0, 0.0),
        }
    }

    // Rotate pitch (around right vector)
    pub fn pitch(&mut self, angle_deg: f32) {
        use cgmath::{Matrix3, Rotation3, InnerSpace};
        let rot = Matrix3::from_axis_angle(self.right, cgmath::Deg(angle_deg));
        self.direction = rot * self.direction;
        self.up = rot * self.up;
    }

    // Rotate yaw (around up vector)
    pub fn yaw(&mut self, angle_deg: f32) {
        use cgmath::{Matrix3, Rotation3, InnerSpace};
        let rot = Matrix3::from_axis_angle(self.up, cgmath::Deg(angle_deg));
        self.direction = rot * self.direction;
        self.right = rot * self.right;
    }

    // Rotate roll (around direction vector)
    pub fn roll(&mut self, angle_deg: f32) {
        use cgmath::{Matrix3, Rotation3, InnerSpace};
        let rot = Matrix3::from_axis_angle(self.direction, cgmath::Deg(angle_deg));
        self.up = rot * self.up;
        self.right = rot * self.right;
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
    pub voxels: Vec<cgmath::Point3<i32>>,
}

impl LSystemTurtle {
    pub fn new(start_pos: cgmath::Point3<i32>, angle: f32, distance: f32) -> Self {
        Self {
            state: TurtleState::new(cgmath::Vector3::new(start_pos.x as f32, start_pos.y as f32, start_pos.z as f32)),
            stack: Vec::new(),
            angle,
            distance,
            voxels: Vec::new(),
        }
    }

    pub fn generate(&mut self, sentence: &str) {
        // Record starting position
        self.voxels.push(cgmath::Point3::new(
            self.state.position.x.round() as i32,
            self.state.position.y.round() as i32,
            self.state.position.z.round() as i32,
        ));

        for c in sentence.chars() {
            match c {
                'F' | 'A' | 'B' | '1' | '0' => {
                    self.state.forward(self.distance);
                    let p = cgmath::Point3::new(
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
