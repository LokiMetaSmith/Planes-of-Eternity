use cgmath::*;
use crate::input::Action;

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.5,
    0.0, 0.0, 0.0, 1.0,
);

pub struct Camera {
    pub eye: Point3<f32>,
    pub target: Point3<f32>,
    pub up: Vector3<f32>,
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub projection_override: Option<Matrix4<f32>>,
}

impl Camera {
    pub fn build_view_projection_matrix(&self) -> Matrix4<f32> {
        let view = Matrix4::look_at_rh(self.eye, self.target, self.up);
        let proj = if let Some(p) = self.projection_override {
            p
        } else {
            cgmath::perspective(Deg(self.fovy), self.aspect, self.znear, self.zfar)
        };
        return OPENGL_TO_WGPU_MATRIX * proj * view;
    }

    pub fn rotate(&mut self, dx: f32, dy: f32) {
        self.yaw += dx;
        self.pitch += dy;

        // Clamp pitch to avoid flipping
        self.pitch = self.pitch.clamp(-std::f32::consts::FRAC_PI_2 + 0.1, std::f32::consts::FRAC_PI_2 - 0.1);

        self.update_target();
    }

    pub fn set_rotation(&mut self, yaw: f32, pitch: f32) {
        self.yaw = yaw;
        self.pitch = pitch;
        self.update_target();
    }

    fn update_target(&mut self) {
        let (sin_y, cos_y) = self.yaw.sin_cos();
        let (sin_p, cos_p) = self.pitch.sin_cos();

        // Direction vector from yaw/pitch
        // Y-up system
        let front = Vector3::new(
            cos_p * sin_y,
            sin_p,
            cos_p * cos_y
        ).normalize();

        // Keep distance to target constant (arbitrary, just needs to be non-zero for look_at)
        // We use a fixed distance or keep previous magnitude?
        // For FPS, target is just a point in front.
        let dist = 10.0;
        self.target = self.eye + front * dist;
    }
}

pub struct CameraController {
    speed: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_up_pressed: bool,
    is_down_pressed: bool,
}

impl CameraController {
    pub fn new(speed: f32) -> Self {
        Self {
            speed,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            is_up_pressed: false,
            is_down_pressed: false,
        }
    }

    pub fn process_action(&mut self, action: Action, pressed: bool) {
        match action {
            Action::MoveForward => self.is_forward_pressed = pressed,
            Action::MoveBackward => self.is_backward_pressed = pressed,
            Action::MoveLeft => self.is_left_pressed = pressed,
            Action::MoveRight => self.is_right_pressed = pressed,
            Action::Jump => self.is_up_pressed = pressed,
            Action::Descend => self.is_down_pressed = pressed,
            _ => (),
        }
    }

    // Kept for legacy tests that might still pass strings, though ideally they use Actions now
    pub fn process_events(&mut self, key_code: &str, pressed: bool) -> bool {
        match key_code {
            "KeyW" | "ArrowUp" => {
                self.is_forward_pressed = pressed;
                true
            }
            "KeyA" | "ArrowLeft" => {
                self.is_left_pressed = pressed;
                true
            }
            "KeyS" | "ArrowDown" => {
                self.is_backward_pressed = pressed;
                true
            }
            "KeyD" | "ArrowRight" => {
                self.is_right_pressed = pressed;
                true
            }
            _ => false,
        }
    }

    pub fn update_camera(&self, camera: &mut Camera) {
        use cgmath::InnerSpace;

        // Calculate forward direction on XZ plane for movement
        let (sin_y, cos_y) = camera.yaw.sin_cos();
        let forward_xz = Vector3::new(sin_y, 0.0, cos_y).normalize();
        let right_xz = forward_xz.cross(Vector3::unit_y()).normalize();

        if self.is_forward_pressed {
            camera.eye += forward_xz * self.speed;
        }
        if self.is_backward_pressed {
            camera.eye -= forward_xz * self.speed;
        }
        if self.is_right_pressed {
            camera.eye += right_xz * self.speed;
        }
        if self.is_left_pressed {
            camera.eye -= right_xz * self.speed;
        }

        if self.is_up_pressed {
            camera.eye.y += self.speed;
        }
        if self.is_down_pressed {
            camera.eye.y -= self.speed;
        }

        // Floor collision
        if camera.eye.y < 1.0 {
            camera.eye.y = 1.0;
        }

        // Update target based on new eye position
        camera.update_target();
    }
}
