use cgmath::{EuclideanSpace, InnerSpace, Point3, Vector3};
use crate::camera::{Camera, CameraController};
use crate::lambda;
use crate::persistence::GameState;
use crate::projector::RealityProjector;
use crate::reality_types::{RealityArchetype, RealitySignature};
use crate::visual_lambda::LambdaSystem;
use crate::world::WorldState;

pub struct Engine {
    pub world_state: WorldState,
    pub player_projector: RealityProjector,
    pub active_anomaly: Option<RealityProjector>, // The one we are currently placing/editing
    pub lambda_system: LambdaSystem,
    pub camera: Camera,
    pub camera_controller: CameraController,
    pub global_offset: [f32; 4],
    pub time: f32,
    pub pending_full_sync: bool,
    pub width: u32,
    pub height: u32,
}

impl Engine {
    pub fn new(width: u32, height: u32, initial_state: Option<GameState>) -> Self {
        let mut camera = Camera {
            eye: (0.0, 1.0, 2.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: Vector3::unit_y(),
            aspect: width as f32 / height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        };

        let camera_controller = CameraController::new(0.2);

        let (player_projector, world_state, active_anomaly, mut lambda_system) = if let Some(state) = initial_state {
            log::info!("Restoring game state...");
            // Restore camera position
            camera.eye = state.player.projector.location;

            // Create a default active anomaly
            let mut anomaly_sig = RealitySignature::default();
            anomaly_sig.active_style.archetype = RealityArchetype::SciFi;
            anomaly_sig.active_style.roughness = 0.8;
            anomaly_sig.active_style.scale = 5.0;
            anomaly_sig.active_style.distortion = 0.8;
            anomaly_sig.fidelity = 100.0;
            let active_anomaly = RealityProjector::new(
                Point3::new(0.0, 0.0, 0.0),
                anomaly_sig
            );

            let mut ls = LambdaSystem::new();
            let term = lambda::parse("(\\x.x) y").unwrap();
            ls.set_term(term);

            (state.player.projector, state.world, Some(active_anomaly), ls)
        } else {
            let mut player_sig = RealitySignature::default();
            player_sig.active_style.archetype = RealityArchetype::Fantasy;
            player_sig.active_style.roughness = 0.3;
            player_sig.active_style.scale = 2.0;
            player_sig.active_style.distortion = 0.1;
            player_sig.fidelity = 100.0;
            let player_projector = RealityProjector::new(
                Point3::new(0.0, 1.0, 2.0),
                player_sig
            );

            let mut anomaly_sig = RealitySignature::default();
            anomaly_sig.active_style.archetype = RealityArchetype::SciFi;
            anomaly_sig.active_style.roughness = 0.8;
            anomaly_sig.active_style.scale = 5.0;
            anomaly_sig.active_style.distortion = 0.8;
            anomaly_sig.fidelity = 100.0;
            let active_anomaly = RealityProjector::new(
                Point3::new(0.0, 0.0, 0.0),
                anomaly_sig
            );

            let world_state = WorldState::default();

            let mut ls = LambdaSystem::new();
            let term = lambda::parse("(\\x.x) y").unwrap();
            ls.set_term(term);

            (player_projector, world_state, Some(active_anomaly), ls)
        };

        // Initial Lambda Setup
        // Check if lambda system needs init from state? For now, always reset as it's not persisted.
        // It was initialized above.

        Self {
            world_state,
            player_projector,
            active_anomaly,
            lambda_system,
            camera,
            camera_controller,
            global_offset: [0.0; 4],
            time: 0.0,
            pending_full_sync: false,
            width,
            height,
        }
    }

    pub fn set_global_offset(&mut self, x: f32, z: f32) {
        self.global_offset = [x, z, 0.0, 0.0];
    }

    pub fn resize(&mut self, new_width: u32, new_height: u32) {
        if new_width > 0 && new_height > 0 {
            self.width = new_width;
            self.height = new_height;
            self.camera.aspect = self.width as f32 / self.height as f32;
        }
    }

    pub fn update(&mut self, dt: f32) {
        self.time += dt;
        self.camera_controller.update_camera(&mut self.camera);

        // Update Reality Projector Position (Player follows camera)
        self.player_projector.location = self.camera.eye;

        // Update Lambda System
        let forward = (self.camera.target - self.camera.eye).normalize();
        let anchor = self.camera.eye + forward * 8.0; // Place it 8 units away
        self.lambda_system.set_anchor(anchor);

        // Fixed timestep for lambda physics for now, could use dt
        self.lambda_system.update(0.016);
    }

    pub fn process_keyboard(&mut self, key_code: &str, pressed: bool) {
        if pressed && key_code == "KeyF" {
             // Cast Spell
             log::info!("Casting Lambda Spell!");
             let archetype_id = self.lambda_system.get_archetype_from_term();
             let archetype = match archetype_id {
                 0 => RealityArchetype::Fantasy,
                 1 => RealityArchetype::SciFi,
                 2 => RealityArchetype::Horror,
                 _ => RealityArchetype::Toon,
             };

             // Forward vector: Target - Eye
             let forward = (self.camera.target - self.camera.eye).normalize();
             let spawn_pos = self.camera.eye + forward * 10.0; // 10 units away

             let mut sig = RealitySignature::default();
             sig.active_style.archetype = archetype;
             sig.fidelity = 100.0;
             sig.active_style.roughness = 0.5;
             sig.active_style.scale = 2.0;
             sig.active_style.distortion = 0.5;

             self.world_state.add_anomaly(RealityProjector {
                 location: spawn_pos,
                 reality_signature: sig,
             });
             log::info!("Spell Cast: {:?} at {:?}", archetype, spawn_pos);
        }

        self.camera_controller.process_events(key_code, pressed);
    }

    pub fn process_mouse_down(&mut self, x: f32, y: f32, button: i16) {
        let (ray_origin, ray_dir) = self.get_ray(x, y);

        if button == 0 { // Left
            if let Some(idx) = self.lambda_system.intersect(ray_origin, ray_dir) {
                self.lambda_system.start_drag(idx, ray_origin, ray_dir);
            }
        } else if button == 2 { // Right
             if let Some(idx) = self.lambda_system.intersect(ray_origin, ray_dir) {
                self.lambda_system.toggle_collapse(idx);
            }
        }
    }

    pub fn process_mouse_move(&mut self, x: f32, y: f32) {
        let (ray_origin, ray_dir) = self.get_ray(x, y);
        self.lambda_system.update_drag(ray_origin, ray_dir);
    }

    pub fn process_mouse_up(&mut self) {
        self.lambda_system.end_drag();
    }

    // Returns true if state changed and requires save
    pub fn process_click(&mut self, x: f32, y: f32) -> bool {
        let (ray_origin, ray_dir) = self.get_ray(x, y);

        // 1. Check Lambda Intersection (Reduce)
        if let Some(_idx) = self.lambda_system.intersect(ray_origin, ray_dir) {
             log::info!("Clicked Lambda Node! Reducing...");
             self.lambda_system.reduce_root();
             return false; // Lambda state is not currently persisted in save game, so return false?
             // Actually if we want to save lambda state eventually, we should return true.
             // But the current save format doesn't include lambda.
        }

        // 2. Plane Intersection (Terrain)
        // P = O + tD, P.y = 0
        if ray_dir.y.abs() > 1e-6 {
             let t = -self.camera.eye.y / ray_dir.y;
             if t > 0.0 {
                 let hit_point = self.camera.eye + ray_dir * t;
                 log::warn!("Injection at: {:?}", hit_point);

                 // Move Active Anomaly to click location
                 if let Some(ref mut anomaly) = self.active_anomaly {
                     anomaly.location = hit_point;

                     // "Commit" the anomaly to the world state (Append it)
                     self.world_state.add_anomaly(RealityProjector {
                         location: anomaly.location,
                         reality_signature: anomaly.reality_signature.clone(),
                     });

                     log::info!("World Root Hash Updated: {}", self.world_state.root_hash);
                     return true; // State changed, request save
                 }
             }
        }
        false
    }

    pub fn get_ray(&self, x: f32, y: f32) -> (Point3<f32>, Vector3<f32>) {
        use cgmath::SquareMatrix;
        let view_proj = self.camera.build_view_projection_matrix();
        let inv_view_proj = view_proj.invert().unwrap_or(cgmath::Matrix4::identity());
        let ray_clip = cgmath::Vector4::new(x, y, -1.0, 1.0);
        let ray_world_hom = inv_view_proj * ray_clip;
        let ray_world_vec = ray_world_hom.truncate() / ray_world_hom.w;
        let ray_world_point = Point3::from_vec(ray_world_vec);
        let ray_origin = self.camera.eye;
        let ray_dir = (ray_world_point - ray_origin).normalize();
        (ray_origin, ray_dir)
    }

    pub fn reset(&mut self) {
        // Reset World
        self.world_state = WorldState::default();

        // Reset Player
        let mut player_sig = RealitySignature::default();
        player_sig.active_style.archetype = RealityArchetype::Fantasy;
        player_sig.active_style.roughness = 0.3;
        player_sig.active_style.scale = 2.0;
        player_sig.active_style.distortion = 0.1;
        player_sig.fidelity = 100.0;
        self.player_projector = RealityProjector::new(
            Point3::new(0.0, 1.0, 2.0),
            player_sig
        );
        self.camera.eye = self.player_projector.location;

        // Reset Active Anomaly
        let mut anomaly_sig = RealitySignature::default();
        anomaly_sig.active_style.archetype = RealityArchetype::SciFi;
        anomaly_sig.active_style.roughness = 0.8;
        anomaly_sig.active_style.scale = 5.0;
        anomaly_sig.active_style.distortion = 0.8;
        anomaly_sig.fidelity = 100.0;
        self.active_anomaly = Some(RealityProjector::new(
            Point3::new(0.0, 0.0, 0.0),
            anomaly_sig
        ));

        // Reset Lambda System
        self.lambda_system = LambdaSystem::new();
        let term = lambda::parse("(\\x.x) y").unwrap();
        self.lambda_system.set_term(term);

        log::info!("World State Reset.");
    }
}
