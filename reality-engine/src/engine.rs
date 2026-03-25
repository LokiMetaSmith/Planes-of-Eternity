use crate::audio::AudioManager;
use crate::camera::{Camera, CameraController};
use crate::input::{Action, InputConfig};
use crate::lambda::{self, Primitive, Term};
use crate::persistence::GameState;
use crate::projector::RealityProjector;
use crate::reality_types::{RealityArchetype, RealitySignature};
use crate::visual_lambda::{self, LambdaSystem};
use crate::world::WorldState;
use cgmath::{EuclideanSpace, InnerSpace, Point3, SquareMatrix, Vector3};
use serde::Serialize;
use std::rc::Rc;

#[derive(Serialize, Debug, PartialEq)]
pub struct LabelInfo {
    pub text: String,
    pub x: f32,
    pub y: f32,
    pub color: String,
}

pub struct Engine {
    pub world_state: WorldState,
    pub player_projector: RealityProjector,
    pub active_anomaly: Option<RealityProjector>, // The one we are currently placing/editing
    pub lambda_system: LambdaSystem,
    pub camera: Camera,
    pub camera_controller: CameraController,
    pub input_config: InputConfig,
    pub audio: AudioManager,
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
            yaw: std::f32::consts::PI,
            pitch: -0.4636, // ~ -26.5 degrees
            projection_override: None,
        };

        let camera_controller = CameraController::new(0.2);

        let (player_projector, world_state, active_anomaly, lambda_system, input_config) =
            if let Some(state) = initial_state {
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
                let active_anomaly = RealityProjector::new(Point3::new(0.0, 0.0, 0.0), anomaly_sig);

                let mut ls = LambdaSystem::new();

                // Restore Lambda State
                let source = if state.lambda_source.is_empty() {
                    "FIRE".to_string()
                } else {
                    state.lambda_source
                };
                if let Some(term) = lambda::parse(&source) {
                    ls.set_term(term);
                    if !state.lambda_layout.is_empty() {
                        ls.apply_layout(state.lambda_layout);
                    }
                } else {
                    // Fallback
                    let term = lambda::parse("FIRE").unwrap();
                    ls.set_term(term);
                }

                (
                    state.player.projector,
                    state.world,
                    Some(active_anomaly),
                    ls,
                    state.input_config,
                )
            } else {
                let mut player_sig = RealitySignature::default();
                player_sig.active_style.archetype = RealityArchetype::Fantasy;
                player_sig.active_style.roughness = 0.3;
                player_sig.active_style.scale = 2.0;
                player_sig.active_style.distortion = 0.1;
                player_sig.fidelity = 100.0;
                let player_projector =
                    RealityProjector::new(Point3::new(0.0, 1.0, 2.0), player_sig);

                let mut anomaly_sig = RealitySignature::default();
                anomaly_sig.active_style.archetype = RealityArchetype::SciFi;
                anomaly_sig.active_style.roughness = 0.8;
                anomaly_sig.active_style.scale = 5.0;
                anomaly_sig.active_style.distortion = 0.8;
                anomaly_sig.fidelity = 100.0;
                let active_anomaly = RealityProjector::new(Point3::new(0.0, 0.0, 0.0), anomaly_sig);

                let mut world_state = WorldState::default();

                // Spawn initial NPCs
                let archetypes = vec![
                    RealityArchetype::SciFi,
                    RealityArchetype::Horror,
                    RealityArchetype::HyperNature,
                    RealityArchetype::CyberSpace,
                    RealityArchetype::Dream,
                    RealityArchetype::ObraDinn,
                ];
                for (i, arch) in archetypes.into_iter().enumerate() {
                    let mut sig = RealitySignature::default();
                    sig.active_style.archetype = arch;
                    sig.active_style.roughness = 0.5;
                    sig.active_style.scale = 3.0;
                    sig.active_style.distortion = 0.2;
                    sig.fidelity = 150.0;

                    let angle = (i as f32) * std::f32::consts::PI / 2.0;
                    let radius = 15.0;
                    let loc = Point3::new(angle.cos() * radius, 1.0, angle.sin() * radius);

                    world_state.npcs.push(RealityProjector::new(loc, sig));
                }

                let mut ls = LambdaSystem::new();
                let term = lambda::parse("FIRE").unwrap();
                ls.set_term(term);

                (
                    player_projector,
                    world_state,
                    Some(active_anomaly),
                    ls,
                    InputConfig::default(),
                )
            };

        // Initial Lambda Setup
        // Check if lambda system needs init from state? For now, always reset as it's not persisted.
        // It was initialized above.

        let audio = AudioManager::new();

        Self {
            world_state,
            player_projector,
            active_anomaly,
            lambda_system,
            camera,
            camera_controller,
            input_config,
            audio,
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

        // --- NPC AI Logic ---
        // Simple wandering and reality projection
        let mut npcs_to_update = Vec::new();
        for npc in &mut self.world_state.npcs {
            let speed = 2.0;

            if let Some(target) = npc.target_location {
                // Move towards target
                use cgmath::InnerSpace;
                use cgmath::MetricSpace;
                let dir = target - npc.location;
                // Optimization: Avoid duplicate magnitude calculation by computing squared distance,
                // checking against a squared threshold, and then extracting the square root just once
                // to multiply it back as a scalar, instead of calling `.magnitude()` and then `.normalize()`.
                let dist_sq = dir.magnitude2();
                if dist_sq > 0.01 {
                    // 0.1 squared
                    let dist = dist_sq.sqrt();
                    let move_vec = dir * (speed * dt / dist);
                    npc.location += move_vec;
                } else {
                    npc.target_location = None; // Reached target
                }
            } else {
                // Fallback deterministic wandering
                let mut seed = 0.0;
                for b in npc.uuid.bytes() {
                    seed += b as f32;
                }
                seed = seed % 100.0;

                let move_x = (self.time + seed).cos() * speed * dt;
                let move_z = (self.time + seed * 1.5).sin() * speed * dt;

                npc.location.x += move_x;
                npc.location.z += move_z;
            }

            // Keep them somewhat grounded
            npc.location.y = 1.0;

            npcs_to_update.push(npc.clone());
        }

        // Apply NPC influence
        for npc in npcs_to_update {
            self.world_state.apply_player_influence(&npc, dt);
        }
        // ---------------------

        // --- Dropped Item Physics ---
        for item in &mut self.world_state.dropped_items {
            item.velocity.y -= 9.8 * dt; // Gravity
            item.position += item.velocity * dt;

            // Simple ground collision at y = 0.5 (radius of item approx)
            if item.position.y <= 0.5 {
                item.position.y = 0.5;
                item.velocity.y = -item.velocity.y * 0.5; // Bounce with restitution

                // Apply friction
                item.velocity.x *= 0.9;
                item.velocity.z *= 0.9;
            }
        }

        // Apply Player Archetype Gameplay Effects
        let player_archetype = self
            .player_projector
            .reality_signature
            .active_style
            .archetype;
        match player_archetype {
            RealityArchetype::SciFi => {
                // SciFi is fast and agile
                self.camera_controller.speed = 0.4;
            }
            RealityArchetype::Steampunk => {
                // Steampunk is heavy and slow
                self.camera_controller.speed = 0.15;
            }
            RealityArchetype::Fantasy => {
                // Default / balanced
                self.camera_controller.speed = 0.2;
            }
            _ => {
                self.camera_controller.speed = 0.2; // Default
            }
        }

        self.camera_controller.update_camera(&mut self.camera);

        // Update Reality Projector Position (Player follows camera)
        self.player_projector.location = self.camera.eye;

        // Apply Player Influence to World State
        let player_proj = self.player_projector.clone();
        self.world_state.apply_player_influence(&player_proj, dt);

        // Update Lambda System
        let forward = (self.camera.target - self.camera.eye).normalize();
        let anchor = self.camera.eye + forward * 8.0; // Place it 8 units away
        self.lambda_system.set_anchor(anchor);

        // Fixed timestep for lambda physics for now, could use dt
        let events = self.lambda_system.update(0.016);
        for event in events {
            match event {
                visual_lambda::LambdaEvent::ReductionStarted => self.audio.play_reduce(),
            }
        }
    }

    pub fn process_keyboard(&mut self, key_code: &str, pressed: bool) {
        // Map raw key to Action
        if let Some(action) = self.input_config.map_key(key_code) {
            match action {
                Action::CastSpell => {
                    if pressed {
                        // Cast Spell
                        log::info!("Casting Lambda Spell!");
                        self.audio.play_cast();
                        if let Some(term) = &self.lambda_system.root_term {
                            if let Some(anomaly) = self.compile_spell(term.clone()) {
                                self.world_state.add_anomaly(anomaly.clone());
                                log::info!(
                                    "Spell Cast Successfully: {:?}",
                                    anomaly.reality_signature.active_style.archetype
                                );
                            } else {
                                log::warn!(
                                    "Spell Failed: Term did not compile to a valid anomaly."
                                );
                            }
                        }
                    }
                }
                Action::MoveForward
                | Action::MoveBackward
                | Action::MoveLeft
                | Action::MoveRight
                | Action::Jump
                | Action::Descend => {
                    // Camera controller handles WASD implicitly, but we need to map our Config to it.
                    // The CameraController currently hardcodes KeyW, etc.
                    // We need to update CameraController to accept generic Actions or boolean flags.

                    // For now, let's keep CameraController "dumb" about bindings and just set its flags.
                    // But CameraController takes key_code string.
                    // We should modify CameraController to take Action enum? Or just bool flags.
                }
                Action::Inscribe => {
                    // Handled by GameClient (lib.rs) triggering window.prompt
                }
                Action::ToggleAutoReduce => {
                    if pressed {
                        self.lambda_system.auto_reduce = !self.lambda_system.auto_reduce;
                    }
                }
                Action::Step => {
                    if pressed {
                        self.lambda_system.reduce_root();
                    }
                }
                Action::TogglePause => {
                    if pressed {
                        self.lambda_system.paused = !self.lambda_system.paused;
                    }
                }
                Action::DropItem => {
                    if pressed {
                        use cgmath::InnerSpace;
                        let (sin_y, cos_y) = self.camera.yaw.sin_cos();
                        let forward_xz = cgmath::Vector3::new(sin_y, 0.0, cos_y);

                        let new_item = crate::reality_types::DroppedItem {
                            id: uuid::Uuid::new_v4().to_string(),
                            position: self.camera.eye,
                            velocity: forward_xz * 5.0 + cgmath::Vector3::unit_y() * 2.0,
                            scale: 0.2,
                            color: [1.0, 0.8, 0.2, 1.0], // Goldish color
                        };
                        self.world_state.dropped_items.push(new_item);
                        log::info!("Dropped item at {:?}", self.camera.eye);
                    }
                }
                // Ignore Voxel Actions (Handled by lib.rs / wrapper)
                _ => {}
            }
        }

        // TEMPORARY: CameraController still hardcodes keys.
        // We will pass the key_code to it, but really we should refactor it to use Actions.
        // To support rebinding for movement, we must map the *bound key* to the *hardcoded internal logic* OR update CameraController.
        // Let's do the clean way: Update CameraController to take Action.

        // Wait, Engine owns CameraController.
        // Let's act as the bridge.
        if let Some(action) = self.input_config.map_key(key_code) {
            self.camera_controller.process_action(action, pressed);
        }
    }

    pub fn process_mouse_down(&mut self, x: f32, y: f32, button: i16) {
        self.audio.resume_context();
        let (ray_origin, ray_dir) = self.get_ray(x, y);

        if button == 0 {
            // Left
            if let Some(idx) = self.lambda_system.intersect(ray_origin, ray_dir) {
                self.lambda_system.start_drag(idx, ray_origin, ray_dir);
            }
        } else if button == 2 {
            // Right
            if let Some(idx) = self.lambda_system.intersect(ray_origin, ray_dir) {
                self.lambda_system.toggle_collapse(idx);
            }
        }
    }

    pub fn process_mouse_move(&mut self, x: f32, y: f32) {
        let last_hover = self.lambda_system.hovered_node;
        let (ray_origin, ray_dir) = self.get_ray(x, y);
        self.lambda_system.update_drag(ray_origin, ray_dir);
        self.lambda_system.update_hover(ray_origin, ray_dir);

        if self.lambda_system.hovered_node.is_some()
            && self.lambda_system.hovered_node != last_hover
        {
            self.audio.play_hover();
        }
    }

    pub fn process_mouse_look(&mut self, dx: f32, dy: f32) {
        let sensitivity = 0.002;
        self.camera.rotate(-dx * sensitivity, -dy * sensitivity);
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
                log::info!("Injection at: {:?}", hit_point);

                // Move Active Anomaly to click location
                if let Some(ref mut anomaly) = self.active_anomaly {
                    anomaly.location = hit_point;

                    // "Commit" the anomaly to the world state (Append it)
                    self.world_state.add_anomaly(RealityProjector::new(
                        anomaly.location,
                        anomaly.reality_signature.clone(),
                    ));

                    log::info!("World Root Hash Updated: {}", self.world_state.root_hash);
                    return true; // State changed, request save
                }
            }
        }
        false
    }

    // Legacy function preserved for benchmarking against get_node_labels_flat
    pub fn get_node_labels_json(&self) -> String {
        let view_proj = self.camera.build_view_projection_matrix();
        let mut labels = Vec::new();

        let status_text = if self.lambda_system.paused {
            "PAUSED".to_string()
        } else if self.lambda_system.auto_reduce {
            "AUTO-RUN".to_string()
        } else {
            "STEP MODE".to_string()
        };

        let status_color = if self.lambda_system.paused {
            "#FFFF00".to_string() // Yellow
        } else if self.lambda_system.auto_reduce {
            "#00FF00".to_string() // Green
        } else {
            "#00F0FF".to_string() // Cyan
        };

        labels.push(LabelInfo {
            text: status_text,
            x: 0.05,
            y: 0.05,
            color: status_color,
        });

        for node in &self.lambda_system.nodes {
            if node.scale < 0.01 {
                continue;
            }

            let p = Point3::new(node.position.x, node.position.y, node.position.z);
            let clip = view_proj * p.to_homogeneous();

            if clip.w > 0.0 {
                let ndc_x = clip.x / clip.w;
                let ndc_y = clip.y / clip.w;

                if (-1.2..=1.2).contains(&ndc_x) && (-1.2..=1.2).contains(&ndc_y) {
                    let screen_x = (ndc_x + 1.0) * 0.5;
                    let screen_y = (1.0 - ndc_y) * 0.5;

                    let text = match &node.node_type {
                        visual_lambda::NodeType::Var(s) => s.clone(),
                        visual_lambda::NodeType::Abs(s) => format!("λ{}", s),
                        visual_lambda::NodeType::Prim(p) => format!("{:?}", p).to_uppercase(),
                        visual_lambda::NodeType::Port => {
                            if let crate::lambda::Term::Var(name) = &*node.term {
                                name.clone()
                            } else {
                                continue;
                            }
                        }
                        _ => continue,
                    };

                    let r = (node.color[0] * 255.0) as u8;
                    let g = (node.color[1] * 255.0) as u8;
                    let b = (node.color[2] * 255.0) as u8;
                    let color = format!("#{:02x}{:02x}{:02x}", r, g, b);

                    labels.push(LabelInfo {
                        text,
                        x: screen_x,
                        y: screen_y,
                        color,
                    });
                }
            }
        }
        serde_json::to_string(&labels).unwrap_or_else(|_| "[]".to_string())
    }

    pub fn get_node_labels_flat(&self) -> Vec<u8> {
        let view_proj = self.camera.build_view_projection_matrix();
        let mut bytes = Vec::new();

        // We will write the count later, so we reserve 4 bytes
        bytes.extend_from_slice(&0u32.to_le_bytes());
        let mut count: u32 = 0;

        // Status Label
        let status_text = if self.lambda_system.paused {
            "PAUSED"
        } else if self.lambda_system.auto_reduce {
            "AUTO-RUN"
        } else {
            "STEP MODE"
        };

        let (sr, sg, sb) = if self.lambda_system.paused {
            (255u8, 255u8, 0u8) // Yellow
        } else if self.lambda_system.auto_reduce {
            (0u8, 255u8, 0u8) // Green
        } else {
            (0u8, 240u8, 255u8) // Cyan
        };

        let x: f32 = 0.05;
        let y: f32 = 0.05;

        bytes.extend_from_slice(&x.to_le_bytes());
        bytes.extend_from_slice(&y.to_le_bytes());
        bytes.push(sr);
        bytes.push(sg);
        bytes.push(sb);
        bytes.push(255); // a
        let text_bytes = status_text.as_bytes();
        bytes.extend_from_slice(&(text_bytes.len() as u16).to_le_bytes());
        bytes.extend_from_slice(text_bytes);
        count += 1;

        for node in &self.lambda_system.nodes {
            if node.scale < 0.01 {
                continue;
            }

            let p = Point3::new(node.position.x, node.position.y, node.position.z);
            let clip = view_proj * p.to_homogeneous();

            if clip.w > 0.0 {
                let ndc_x = clip.x / clip.w;
                let ndc_y = clip.y / clip.w;

                if (-1.2..=1.2).contains(&ndc_x) && (-1.2..=1.2).contains(&ndc_y) {
                    let screen_x = (ndc_x + 1.0) * 0.5;
                    let screen_y = (1.0 - ndc_y) * 0.5;

                    let text = match &node.node_type {
                        visual_lambda::NodeType::Var(s) => s.clone(),
                        visual_lambda::NodeType::Abs(s) => format!("λ{}", s),
                        visual_lambda::NodeType::Prim(p) => format!("{:?}", p).to_uppercase(),
                        visual_lambda::NodeType::Port => {
                            if let crate::lambda::Term::Var(name) = &*node.term {
                                name.clone()
                            } else {
                                continue;
                            }
                        }
                        _ => continue,
                    };

                    let r = (node.color[0] * 255.0) as u8;
                    let g = (node.color[1] * 255.0) as u8;
                    let b = (node.color[2] * 255.0) as u8;
                    let a = (node.color[3] * 255.0) as u8;

                    bytes.extend_from_slice(&screen_x.to_le_bytes());
                    bytes.extend_from_slice(&screen_y.to_le_bytes());
                    bytes.push(r);
                    bytes.push(g);
                    bytes.push(b);
                    bytes.push(a);

                    let t_bytes = text.as_bytes();
                    bytes.extend_from_slice(&(t_bytes.len() as u16).to_le_bytes());
                    bytes.extend_from_slice(t_bytes);
                    count += 1;
                }
            }
        }

        // Write actual count
        let count_bytes = count.to_le_bytes();
        bytes[0..4].copy_from_slice(&count_bytes);

        bytes
    }

    pub fn get_ray(&self, x: f32, y: f32) -> (Point3<f32>, Vector3<f32>) {
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

    fn compile_spell(&self, term: Rc<Term>) -> Option<RealityProjector> {
        // 1. Fully reduce the term
        let mut current = term;
        for _ in 0..100 {
            // Max reduction steps to prevent infinite loops
            let (next, changed) = current.reduce();
            if !changed {
                current = next;
                break;
            }
            current = next;
        }

        // 2. Interpret the result
        // Default spawn position
        let forward = (self.camera.target - self.camera.eye).normalize();
        let spawn_pos = self.camera.eye + forward * 10.0;

        match &*current {
            Term::Prim(p) => self.primitive_to_anomaly(*p, spawn_pos),
            Term::App(func, arg) => {
                // Check if it's Prim App Prim
                if let Term::Prim(op) = &**func {
                    if let Term::Prim(target) = &**arg {
                        return self.combine_primitives(*op, *target, spawn_pos);
                    }
                }
                // Also check if it's Prim App (App...) - recursive evaluation is hard without specific logic.
                // For now, only support 1 level of application (Modifier Target).
                None
            }
            _ => None,
        }
    }

    fn primitive_to_anomaly(&self, p: Primitive, pos: Point3<f32>) -> Option<RealityProjector> {
        let mut sig = RealitySignature {
            fidelity: 100.0,
            ..Default::default()
        };
        sig.active_style.scale = 5.0;

        match p {
            Primitive::Fire => {
                sig.active_style.archetype = RealityArchetype::Horror; // Use Horror for Fire visual
                sig.active_style.roughness = 1.0;
                sig.active_style.distortion = 0.8;
            }
            Primitive::Water => {
                sig.active_style.archetype = RealityArchetype::HyperNature; // Watery?
                sig.active_style.roughness = 0.2;
                sig.active_style.distortion = 0.5;
            }
            Primitive::Growth => {
                sig.active_style.archetype = RealityArchetype::Fantasy;
                sig.active_style.roughness = 0.5;
            }
            Primitive::Energy => {
                sig.active_style.archetype = RealityArchetype::SciFi;
                sig.active_style.roughness = 0.0;
                sig.active_style.distortion = 0.2;
            }
            Primitive::Void => {
                sig.active_style.archetype = RealityArchetype::Void;
            }
            _ => return None,
        }

        Some(RealityProjector::new(pos, sig))
    }

    fn combine_primitives(
        &self,
        op: Primitive,
        target: Primitive,
        pos: Point3<f32>,
    ) -> Option<RealityProjector> {
        let mut base = self.primitive_to_anomaly(target, pos)?;

        match op {
            Primitive::Growth => {
                base.reality_signature.active_style.scale *= 2.0;
            }
            Primitive::Decay => {
                base.reality_signature.active_style.scale *= 0.5;
                base.reality_signature.active_style.distortion = 1.0;
            }
            Primitive::Energy => {
                base.reality_signature.fidelity = 200.0; // Boost fidelity?
                base.reality_signature.active_style.roughness = 0.0;
            }
            Primitive::Fire => {
                // Fire + Something = Burnt Something?
                base.reality_signature.active_style.archetype = RealityArchetype::Horror;
            }
            _ => {}
        }

        Some(base)
    }

    pub fn process_inscription(&mut self, text: &str) {
        log::info!("Inscribing: {}", text);

        // Security Enhancement: Prevent DoS by limiting input length
        // Parsing extremely long lambda terms could cause excessive memory allocation
        // and CPU usage during parsing or reduction.
        const MAX_INSCRIPTION_LEN: usize = 256;
        if text.len() > MAX_INSCRIPTION_LEN {
            log::warn!(
                "Security Warning: Inscription exceeded maximum length limit ({} bytes).",
                MAX_INSCRIPTION_LEN
            );
            return;
        }

        // Clean up input (trim)
        let clean_text = text.trim();
        if let Some(term) = lambda::parse(clean_text) {
            self.lambda_system.set_term(term);
            log::info!("Inscription successful. Term updated.");
        } else {
            log::warn!("Invalid inscription syntax.");
        }
    }

    pub fn reset(&mut self) {
        // Reset World
        self.world_state = WorldState::default();

        // Spawn initial NPCs
        let archetypes = vec![
            RealityArchetype::SciFi,
            RealityArchetype::Horror,
            RealityArchetype::HyperNature,
            RealityArchetype::CyberSpace,
            RealityArchetype::Dream,
            RealityArchetype::ObraDinn,
        ];
        for (i, arch) in archetypes.into_iter().enumerate() {
            let mut sig = RealitySignature::default();
            sig.active_style.archetype = arch;
            sig.active_style.roughness = 0.5;
            sig.active_style.scale = 3.0;
            sig.active_style.distortion = 0.2;
            sig.fidelity = 150.0;

            let angle = (i as f32) * std::f32::consts::PI / 2.0;
            let radius = 15.0;
            let loc = Point3::new(angle.cos() * radius, 1.0, angle.sin() * radius);

            self.world_state.npcs.push(RealityProjector::new(loc, sig));
        }

        // Reset Player
        let mut player_sig = RealitySignature::default();
        player_sig.active_style.archetype = RealityArchetype::Fantasy;
        player_sig.active_style.roughness = 0.3;
        player_sig.active_style.scale = 2.0;
        player_sig.active_style.distortion = 0.1;
        player_sig.fidelity = 100.0;
        self.player_projector = RealityProjector::new(Point3::new(0.0, 1.0, 2.0), player_sig);
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
            anomaly_sig,
        ));

        // Reset Lambda System
        self.lambda_system = LambdaSystem::new();
        let term = lambda::parse("FIRE").unwrap();
        self.lambda_system.set_term(term);

        log::info!("World State Reset.");
    }
}
