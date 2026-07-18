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
use noise::{Fbm, Simplex};
use serde::{Deserialize, Serialize};
use std::rc::Rc;

#[derive(Serialize, Debug, PartialEq)]
pub struct LabelInfo {
    pub text: String,
    pub x: f32,
    pub y: f32,
    pub color: String,
}

pub struct SpellEffect {
    pub position: cgmath::Point3<f32>,
    pub color: [f32; 4],
    pub scale: f32,
    pub timer: f32,
    pub max_time: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicsState {
    Flying,
    Gravity,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnimationState {
    Idle,
    Walk,
    Attack,
    Interact,
}

pub struct Splat4DPlayer {
    pub animations: std::collections::HashMap<AnimationState, Vec<Vec<crate::splat::SplatVertex>>>,
    pub current_state: AnimationState,
    pub next_state: Option<AnimationState>,
    pub current_frame: usize,
    pub timer: f32,
    pub blend_timer: f32,
    pub blend_duration: f32,
    pub frame_rate: f32,
    pub loop_playback: bool,
    pub position: cgmath::Point3<f32>,
    pub archetype_override: Option<u32>,
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
    pub spell_effects: Vec<SpellEffect>,
    pub pending_dreams: Vec<String>,
    pub pending_models: Vec<(String, [f32; 3])>,
    pub pending_extensions: Vec<[f32; 3]>,

    pub is_recording_splats: bool,
    pub splat_recording_buffer: Vec<Vec<crate::splat::SplatVertex>>,

    pub active_4d_splats: Vec<Splat4DPlayer>,

    pub show_3d_inventory: bool,
    pub show_minimap: bool,
    pub show_ui: bool,

    pub anchor_distance: f32,
    pub fbm_noise: Fbm<Simplex>,
    pub physics_state: PhysicsState,
}

pub struct CollisionResult {
    pub position: cgmath::Point3<f32>,
    pub velocity: cgmath::Vector3<f32>,
    pub hit: bool,
    pub smashed: bool,
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
            velocity: Vector3::new(0.0, 0.0, 0.0),
            is_grounded: true,
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
                    RealityArchetype::Tron,
                    RealityArchetype::Clockwork,
                    RealityArchetype::Cottagecore,
                    RealityArchetype::WildWest,
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

                    let mut npc = RealityProjector::new(loc, sig);
                    npc.behavior = Some(crate::projector::NpcBehavior {
                        preferred_archetype: arch,
                        energy: 100.0,
                        mutation_progress: 0.0,
                        hostile: arch == RealityArchetype::Horror,
                        goal_stack: Vec::new(),
                        animation_playback: Some(crate::projector::AnimationPlayback::default()),
                    });
                    world_state.npcs.push(npc);
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
            spell_effects: Vec::new(),
            pending_dreams: Vec::new(),
            pending_models: Vec::new(),
            pending_extensions: Vec::new(),
            is_recording_splats: false,
            splat_recording_buffer: Vec::new(),
            active_4d_splats: Vec::new(),
            show_3d_inventory: false,
            show_minimap: false,
            show_ui: true,
            anchor_distance: 8.0,
            fbm_noise: noise::Fbm::<noise::Simplex>::new(1337),
            physics_state: PhysicsState::Flying,
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

    fn is_occupied(
        &mut self,
        pos: Point3<f32>,
        radius: f32,
        voxel_world: &mut crate::voxel::VoxelWorld,
        can_smash: bool,
        smashed: &mut bool,
    ) -> bool {
        let min_x = (pos.x - radius).floor() as i32;
        let max_x = (pos.x + radius).floor() as i32;
        let min_y = (pos.y - radius).floor() as i32;
        let max_y = (pos.y + radius).floor() as i32;
        let min_z = (pos.z - radius).floor() as i32;
        let max_z = (pos.z + radius).floor() as i32;

        let mut occupied = false;
        for vx in min_x..=max_x {
            for vy in min_y..=max_y {
                for vz in min_z..=max_z {
                    if let Some(voxel) = voxel_world.get_voxel_at(vx, vy, vz) {
                        if voxel.id != 0 {
                            if can_smash {
                                voxel_world.set_voxel_at(vx, vy, vz, crate::voxel::Voxel { id: 0 });
                                *smashed = true;
                                self.audio.play_impact();
                                let color = crate::voxel::Chunk::get_color_for_id(voxel.id);
                                self.spell_effects.push(SpellEffect {
                                    position: Point3::new(vx as f32 + 0.5, vy as f32 + 0.5, vz as f32 + 0.5),
                                    color: [color[0], color[1], color[2], 1.0],
                                    scale: 1.0,
                                    timer: 0.0,
                                    max_time: 0.5,
                                });
                            } else {
                                occupied = true;
                            }
                        }
                    }
                }
            }
        }
        occupied
    }

    pub fn check_collision(
        &mut self,
        current_pos: Point3<f32>,
        velocity: Vector3<f32>,
        dt: f32,
        radius: f32,
        voxel_world: &mut crate::voxel::VoxelWorld,
    ) -> CollisionResult {
        let mut pos = current_pos;
        let mut vel = velocity;
        let mut smashed_any = false;
        let mut hit_any = false;

        let smash_threshold = 8.0; // Speed required to smash blocks

        // Sub-stepping for anti-tunneling
        let speed = vel.magnitude();
        let num_steps = ((speed * dt * 2.0).ceil() as usize).max(1);
        let step_dt = dt / num_steps as f32;

        for _ in 0..num_steps {
            // Try X movement
            let mut next_pos_x = pos;
            next_pos_x.x += vel.x * step_dt;
            if !self.is_occupied(next_pos_x, radius, voxel_world, speed > smash_threshold, &mut smashed_any) {
                pos.x = next_pos_x.x;
            } else {
                vel.x = 0.0;
                hit_any = true;
            }

            // Try Y movement
            let mut next_pos_y = pos;
            next_pos_y.y += vel.y * step_dt;
            if !self.is_occupied(next_pos_y, radius, voxel_world, speed > smash_threshold, &mut smashed_any) {
                pos.y = next_pos_y.y;
            } else {
                vel.y = 0.0;
                hit_any = true;
            }

            // Try Z movement
            let mut next_pos_z = pos;
            next_pos_z.z += vel.z * step_dt;
            if !self.is_occupied(next_pos_z, radius, voxel_world, speed > smash_threshold, &mut smashed_any) {
                pos.z = next_pos_z.z;
            } else {
                vel.z = 0.0;
                hit_any = true;
            }
        }

        CollisionResult {
            position: pos,
            velocity: if smashed_any { vel * 0.9 } else { vel },
            hit: hit_any,
            smashed: smashed_any,
        }
    }

    pub fn update(
        &mut self,
        dt: f32,
        mut voxel_world: Option<&mut crate::voxel::VoxelWorld>,
    ) -> bool {
        self.time += dt;

        // Process any queued EXTEND spells
        let mut extensions = Vec::new();
        std::mem::swap(&mut self.pending_extensions, &mut extensions);
        for pos in extensions {
            if let Some(ref mut vw) = voxel_world {
                let key_x = (pos[0] / crate::voxel::CHUNK_SIZE as f32).floor() as i32;
                let key_y = (pos[1] / crate::voxel::CHUNK_SIZE as f32).floor() as i32;
                let key_z = (pos[2] / crate::voxel::CHUNK_SIZE as f32).floor() as i32;

                let trajectory = vec![pos];

                // Gather neighbors
                let mut neighbors = Vec::new();
                let offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)];
                for (dx, dz) in offsets {
                    let nx = key_x + dx;
                    let nz = key_z + dz;
                    let nkey = crate::voxel::ChunkKey { x: nx, y: key_y, z: nz };
                    if let Some(n_chunk) = vw.chunks.get(&nkey) {
                        let mut n_heights = vec![0.0; crate::voxel::CHUNK_SIZE * crate::voxel::CHUNK_SIZE];
                        for lz in 0..crate::voxel::CHUNK_SIZE {
                            for lx in 0..crate::voxel::CHUNK_SIZE {
                                let mut h = -1.0;
                                for ly in (0..crate::voxel::CHUNK_SIZE).rev() {
                                    let voxel = n_chunk.get(lx, ly, lz);
                                    if voxel.id != 0 {
                                        h = ly as f32;
                                        break;
                                    }
                                }
                                n_heights[lx + lz * crate::voxel::CHUNK_SIZE] = h;
                            }
                        }
                        neighbors.push((nx, nz, n_heights));
                    }
                }

                vw.genie.request_iterative_extension(key_x, key_y, key_z, crate::voxel::CHUNK_SIZE, trajectory, neighbors);
            }
        }

        let mut _chunks_generated = false;
        if let Some(vw) = &mut voxel_world {
            if vw.ensure_chunks_around(self.camera.eye, 3) {
                _chunks_generated = true;
            }

            while let Some(res) = vw.genie.poll_terrain() {
                if let Some(chunk) = vw.chunks.get_mut(&res.chunk_key) {
                    let _wx_base = chunk.key.x * crate::voxel::CHUNK_SIZE as i32;
                    let wy_base = chunk.key.y * crate::voxel::CHUNK_SIZE as i32;
                    let _wz_base = chunk.key.z * crate::voxel::CHUNK_SIZE as i32;

                    for z in 0..crate::voxel::CHUNK_SIZE {
                        for y in 0..crate::voxel::CHUNK_SIZE {
                            for x in 0..crate::voxel::CHUNK_SIZE {
                                let wy = wy_base + y as i32;
                                let terrain_height = res.heightmap[x + z * crate::voxel::CHUNK_SIZE] as i32;

                                let mut voxel = crate::voxel::Voxel::default();
                                if wy <= terrain_height {
                                    if wy == terrain_height {
                                        voxel.id = 5; // Grass
                                    } else if wy > terrain_height - 3 {
                                        voxel.id = 8; // Dirt
                                    } else {
                                        voxel.id = 1; // Stone
                                    }
                                }

                                // Water
                                if wy <= -2 && voxel.id == 0 {
                                    voxel.id = 4;
                                }

                                let idx = x + y * crate::voxel::CHUNK_SIZE + z * crate::voxel::CHUNK_SIZE * crate::voxel::CHUNK_SIZE;
                                chunk.data[idx] = voxel;
                            }
                        }
                    }
                    _chunks_generated = true;
                }
            }
        }

        // --- NPC AI Logic ---
        let current_time = crate::projector::get_current_timestamp();
        for npc in &mut self.world_state.npcs {
            if let Some((_, timestamp)) = npc.chat_message {
                if current_time > timestamp + 5000 {
                    npc.chat_message = None;
                }
            }
        }

        // Simple wandering and reality projection

        let mut npcs_to_update = Vec::new();
        // First we extract the npcs into a temporary vector so we don't have multiple mutable borrows to self.world_state
        let mut npcs = std::mem::take(&mut self.world_state.npcs);

        // We need to re-borrow voxel_world for each NPC, so we use a shadow binding
        let mut vw_opt = voxel_world;

        for npc in &mut npcs {
            use crate::projector::{Goal, GoalStatus};
            use cgmath::InnerSpace;

            // 1. Evaluate Interrupts (Reactive events)
            if let Some(behavior) = &mut npc.behavior {
                let dir_to_player = self.camera.eye - npc.location;
                let dist_to_player_sq = dir_to_player.magnitude2();

                let mut interrupt_triggered = false;

                if behavior.hostile && dist_to_player_sq < 400.0 {
                    // Aggro radius (20 units squared)
                    let current_is_attack = match behavior.goal_stack.last() {
                        Some(Goal::Attack(_)) => true,
                        _ => false,
                    };

                    if !current_is_attack {
                        behavior.goal_stack.clear();
                        behavior.goal_stack.push(Goal::Attack("player".to_string()));
                        interrupt_triggered = true;
                        let _ = interrupt_triggered;
                    }
                } else if dist_to_player_sq < 25.0 {
                    // Startled by player getting too close (5 units)
                    let current_is_escape = match behavior.goal_stack.last() {
                        Some(Goal::Escape) => true,
                        Some(Goal::Attack(_)) => true, // hostile overrides
                        _ => false,
                    };

                    if !current_is_escape && !behavior.hostile {
                        behavior.goal_stack.clear();
                        behavior.goal_stack.push(Goal::Escape);
                        interrupt_triggered = true;
                        let _ = interrupt_triggered;
                    }
                }

                // If remote script sets target location, interrupt and wander there
                if !interrupt_triggered && npc.target_location.is_some() {
                    let current_is_wander = match behavior.goal_stack.last() {
                        Some(Goal::Wander) => true,
                        _ => false,
                    };

                    if !current_is_wander {
                        behavior.goal_stack.clear();
                        behavior.goal_stack.push(Goal::Wander);
                        interrupt_triggered = true;
                        let _ = interrupt_triggered;
                    }
                }

                // 2. High-level brain (Assign top-level goals if stack is empty)
                if behavior.goal_stack.is_empty() {
                    let current_archetype = self.world_state.get_dominant_archetype_at(npc.location);

                    if let Some(arch) = current_archetype {
                        if arch == behavior.preferred_archetype {
                            // Thriving
                            if rand::random::<f32>() < 0.2 {
                                behavior.goal_stack.push(Goal::ExpandInfluence);
                            } else if rand::random::<f32>() < 0.5 {
                                behavior.goal_stack.push(Goal::GatherResource("".to_string()));
                            } else {
                                behavior.goal_stack.push(Goal::Wander);
                                behavior.goal_stack.push(Goal::Idle(2.0));
                            }
                        } else {
                            // Agitated
                            if behavior.mutation_progress >= 100.0 {
                                behavior.goal_stack.push(Goal::Evolve);
                            } else if rand::random::<f32>() < 0.6 {
                                behavior.goal_stack.push(Goal::Escape);
                            } else {
                                behavior.goal_stack.push(Goal::Wander);
                            }
                        }
                    } else {
                        // Void / Neutral
                        behavior.goal_stack.push(Goal::Wander);
                    }
                }
            }

            // 3. Update top goal
            let mut goal_status = GoalStatus::Continue;
            let mut top_goal = None;
            if let Some(behavior) = &mut npc.behavior {
                if let Some(goal) = behavior.goal_stack.last().cloned() {
                    top_goal = Some(goal.clone());

                    // Map Goal to AnimationState
                    if let Some(anim_playback) = &mut behavior.animation_playback {
                        let desired_state = match goal {
                            Goal::Idle(_) => AnimationState::Idle,
                            Goal::Wander | Goal::Escape | Goal::GatherResource(_) | Goal::ExpandInfluence => AnimationState::Walk,
                            Goal::Attack(_) => AnimationState::Attack,
                            Goal::Evolve => AnimationState::Interact,
                        };

                        if anim_playback.current_state != desired_state {
                            if anim_playback.next_state != Some(desired_state) {
                                anim_playback.next_state = Some(desired_state);
                                anim_playback.blend_timer = 0.0;
                            }
                        } else {
                            anim_playback.next_state = None;
                        }
                    }
                }
            }

            if let Some(goal) = top_goal {
                // Execute goal
                goal_status = match goal {
                    Goal::Idle(time_left) => {
                        if let Some(behavior) = &mut npc.behavior {
                            let new_time = time_left - dt;
                            if new_time <= 0.0 {
                                GoalStatus::Success
                            } else {
                                // Replace with updated time
                                behavior.goal_stack.pop();
                                behavior.goal_stack.push(Goal::Idle(new_time));
                                GoalStatus::Continue
                            }
                        } else {
                            GoalStatus::Failure
                        }
                    }
                    Goal::Wander => {
                        let mut status = GoalStatus::Continue;
                        if npc.target_location.is_none() {
                            let angle = rand::random::<f32>() * std::f32::consts::PI * 2.0;
                            let radius = 10.0 + rand::random::<f32>() * 15.0;
                            npc.target_location = Some(cgmath::Point3::new(
                                npc.location.x + angle.cos() * radius,
                                1.0,
                                npc.location.z + angle.sin() * radius,
                            ));
                        }

                        if let Some(target) = npc.target_location {
                            let dir = target - npc.location;
                            let dist_sq = dir.magnitude2();
                            if dist_sq > 0.1 && dist_sq.is_finite() {
                                let move_vel = dir * (2.0 / dist_sq.sqrt());
                                if let Some(vw) = vw_opt.as_mut() {
                                    let res = self.check_collision(npc.location, move_vel, dt, 0.5, vw);
                                    npc.location = res.position;
                                } else {
                                    npc.location += move_vel * dt;
                                }
                            } else {
                                npc.target_location = None;
                                status = GoalStatus::Success;
                            }
                        } else {
                            status = GoalStatus::Failure;
                        }

                        if let Some(behavior) = &mut npc.behavior {
                            let current_arch = self.world_state.get_dominant_archetype_at(npc.location);
                            if current_arch == Some(behavior.preferred_archetype) {
                                behavior.energy = (behavior.energy + 5.0 * dt).min(100.0);
                                behavior.mutation_progress = (behavior.mutation_progress - 10.0 * dt).max(0.0);
                            } else {
                                behavior.energy = (behavior.energy - 2.0 * dt).max(0.0);
                                behavior.mutation_progress += 2.0 * dt;
                            }
                        }

                        status
                    }
                    Goal::Escape => {
                        let mut status = GoalStatus::Continue;
                        if npc.target_location.is_none() {
                            let angle = rand::random::<f32>() * std::f32::consts::PI * 2.0;
                            let radius = 20.0 + rand::random::<f32>() * 20.0;
                            npc.target_location = Some(cgmath::Point3::new(
                                npc.location.x + angle.cos() * radius,
                                1.0,
                                npc.location.z + angle.sin() * radius,
                            ));
                        }

                        if let Some(target) = npc.target_location {
                            let dir = target - npc.location;
                            let dist_sq = dir.magnitude2();
                            if dist_sq > 0.1 && dist_sq.is_finite() {
                                let move_vel = dir * (5.0 / dist_sq.sqrt());
                                if let Some(vw) = vw_opt.as_mut() {
                                    let res = self.check_collision(npc.location, move_vel, dt, 0.5, vw);
                                    npc.location = res.position;
                                } else {
                                    npc.location += move_vel * dt;
                                }
                            } else {
                                npc.target_location = None;
                                status = GoalStatus::Success;
                            }
                        }

                        if let Some(behavior) = &mut npc.behavior {
                            behavior.energy = (behavior.energy - 5.0 * dt).max(0.0);
                            behavior.mutation_progress += 15.0 * dt;
                        }

                        status
                    }
                    Goal::Evolve => {
                        if let Some(behavior) = &mut npc.behavior {
                            let current_arch = self.world_state.get_dominant_archetype_at(npc.location);
                            if let Some(arch) = current_arch {
                                behavior.preferred_archetype = arch;
                                npc.reality_signature.active_style.archetype = arch;
                                behavior.mutation_progress = 0.0;
                                behavior.energy = 100.0;
                                GoalStatus::Success
                            } else {
                                GoalStatus::Failure
                            }
                        } else {
                            GoalStatus::Failure
                        }
                    }
                    Goal::Attack(_) => {
                        npc.target_location = Some(self.camera.eye);
                        let dir = self.camera.eye - npc.location;
                        let dist_sq = dir.magnitude2();

                        if dist_sq < 4.0 {
                            // Player hit
                            let push_dir = dir * (10.0 / dist_sq.sqrt().max(0.1));
                            self.camera.eye += push_dir;
                            // Attack successful, end goal
                            npc.target_location = None;
                            GoalStatus::Success
                        } else if dist_sq > 625.0 {
                            // Player escaped (25 units away)
                            npc.target_location = None;
                            GoalStatus::Failure
                        } else {
                            // Chase
                            if dist_sq.is_finite() && dist_sq > 0.01 {
                                let move_vel = dir * (6.0 / dist_sq.sqrt());
                                if let Some(vw) = vw_opt.as_mut() {
                                    let res = self.check_collision(npc.location, move_vel, dt, 0.5, *vw);
                                    npc.location = res.position;
                                } else {
                                    npc.location += move_vel * dt;
                                }
                            }
                            GoalStatus::Continue
                        }
                    }
                    Goal::GatherResource(_) => {
                        let mut status = GoalStatus::Continue;
                        // Find closest item
                        let mut closest_item_idx = None;
                        let mut min_dist_sq = 900.0; // max search radius 30 units

                        for (i, item) in self.world_state.dropped_items.iter().enumerate() {
                            let d_sq = (item.position - npc.location).magnitude2();
                            if d_sq < min_dist_sq {
                                min_dist_sq = d_sq;
                                closest_item_idx = Some(i);
                            }
                        }

                        if let Some(idx) = closest_item_idx {
                            let item_pos = self.world_state.dropped_items[idx].position;
                            if min_dist_sq < 2.0 {
                                // Pick it up
                                self.world_state.dropped_items.remove(idx);
                                if let Some(behavior) = &mut npc.behavior {
                                    behavior.energy = (behavior.energy + 20.0).min(100.0);
                                }
                                status = GoalStatus::Success;
                            } else {
                                // Move towards it
                                let dir = item_pos - npc.location;
                                if min_dist_sq.is_finite() && min_dist_sq > 0.01 {
                                    let move_vel = dir * (3.0 / min_dist_sq.sqrt());
                                    if let Some(vw) = vw_opt.as_mut() {
                                        let res = self.check_collision(npc.location, move_vel, dt, 0.5, *vw);
                                        npc.location = res.position;
                                    } else {
                                        npc.location += move_vel * dt;
                                    }
                                }
                            }
                        } else {
                            status = GoalStatus::Failure; // No resources found
                        }

                        status
                    }
                    Goal::ExpandInfluence => {
                        // Drop a potential node to expand reality
                        if let Some(behavior) = &mut npc.behavior {
                            if behavior.energy > 50.0 {
                                behavior.energy -= 30.0;
                                let drop_pos = npc.location + cgmath::Vector3::new(0.0, 1.0, 0.0);
                                let new_item = crate::reality_types::DroppedItem::new_cube(
                                    format!("potential_node_npc_{}", uuid::Uuid::new_v4()),
                                    drop_pos,
                                    cgmath::Vector3::new(0.0, 2.0, 0.0),
                                    0.3,
                                    [1.0, 0.5, 0.0, 1.0], // Orange node
                                    3,
                                );
                                self.world_state.dropped_items.push(new_item);
                                GoalStatus::Success
                            } else {
                                GoalStatus::Failure
                            }
                        } else {
                            GoalStatus::Failure
                        }
                    }
                };
            }

            // 4. Pop stack on Success/Failure
            if goal_status != GoalStatus::Continue {
                if let Some(behavior) = &mut npc.behavior {
                    behavior.goal_stack.pop();

                    if goal_status == GoalStatus::Failure {
                        // Clear stack on failure to cause re-evaluation next frame
                        behavior.goal_stack.clear();
                    }
                }
            }

            // Keep somewhat grounded if no voxel world for height check
            if vw_opt.is_none() {
                npc.location.y = 1.0;
            } else if let Some(vw) = vw_opt.as_mut() {
                // Apply a bit of gravity to NPCs too
                let res = self.check_collision(npc.location, Vector3::new(0.0, -9.8, 0.0), dt, 0.5, vw);
                npc.location = res.position;
            }

            npcs_to_update.push(npc.clone());
        }
        // Put the modified npcs back
        self.world_state.npcs = npcs;
        voxel_world = vw_opt;

        // Apply NPC influence
        for npc in npcs_to_update {
            self.world_state.apply_player_influence(&npc, dt);
        }
        // ---------------------

        // --- Dropped Item Physics ---
        let mut voxel_destroyed = false;
        for item in &mut self.world_state.dropped_items {
            item.velocity.y -= 9.8 * dt; // Gravity
            let next_pos = item.position + item.velocity * dt;

            let mut hit_ground = false;

            // Voxel Collision (Starbase-inspired physics)
            if let Some(ref mut vw) = voxel_world {
                let vx = next_pos.x.floor() as i32;
                // Check foot/bottom of the item (approx radius 0.5)
                let vy = (next_pos.y - 0.5).floor() as i32;
                let vz = next_pos.z.floor() as i32;

                if let Some(voxel) = vw.get_voxel_at(vx, vy, vz) {
                    if voxel.id != 0 {
                        // Collision!
                        hit_ground = true;

                        // Check for high-velocity impact (Destruction)
                        let speed_sq = item.velocity.magnitude2();
                        if speed_sq > 100.0 {
                            // Speed > 10.0
                            vw.set_voxel_at(vx, vy, vz, crate::voxel::Voxel { id: 0 });
                            log::info!("Voxel at {}, {}, {} destroyed by impact!", vx, vy, vz);
                            voxel_destroyed = true;

                            // Self-damage to the local item voxel grid (Fracturing)
                            if item.size[0] > 0 && !item.voxels.is_empty() {
                                // Pseudo-random to avoid trait bound issues with different rand versions
                                let local_idx = ((self.time * 1000.0) as usize
                                    + item.voxels.len() / 2)
                                    % item.voxels.len();
                                if item.voxels[local_idx] != 0 {
                                    item.voxels[local_idx] = 0; // Destroy a local piece
                                    log::info!("Dropped item {} fractured!", item.id);
                                }
                            }

                            // Reduce item velocity heavily after smashing through
                            item.velocity *= 0.5;
                            // Continue falling instead of bouncing
                            hit_ground = false;
                        }
                    }
                }
            }

            // Fallback ground collision
            if !hit_ground && next_pos.y <= 0.5 {
                hit_ground = true;
                item.position.y = 0.5;
            }

            if hit_ground {
                item.position.x = next_pos.x;
                item.position.z = next_pos.z;
                item.position.y = next_pos.y.max(0.5); // Ensure it doesn't sink below global floor
                item.velocity.y = -item.velocity.y * 0.5; // Bounce with restitution
                // Apply friction
                item.velocity.x *= 0.9;
                item.velocity.z *= 0.9;

                if item.velocity.magnitude2() > 4.0 {
                    self.audio.play_impact();
                }
            } else {
                item.position = next_pos;
            }
        }

        if voxel_destroyed {
            // Signal lib.rs to mark voxel dirty by returning true, or we can just mark it in lib.rs if we know
            // Actually, we pass a mutable reference to voxel_world, so it gets updated.
            // But we need a way to tell the caller that voxels were dirtied so it can trigger mesh updates.
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

        let has_gravity = self.physics_state == PhysicsState::Gravity;

        // --- Player Movement & Collision ---
        let mut move_vel = self.camera_controller.get_movement_velocity(&self.camera);
        if has_gravity {
            self.camera.velocity.y -= 9.8 * dt; // Gravity
        } else {
            self.camera.velocity.y = 0.0;
            // Flying vertical movement
            if self.camera_controller.is_up_pressed() {
                move_vel.y += self.camera_controller.speed;
            }
            if self.camera_controller.is_down_pressed() {
                move_vel.y -= self.camera_controller.speed;
            }
        }

        // Handle Jump
        if has_gravity && self.camera_controller.is_up_pressed() && self.camera.is_grounded {
            self.camera.velocity.y = 5.0;
            self.camera.is_grounded = false;
        }

        let combined_vel = move_vel / dt.max(0.001) + self.camera.velocity;

        if let Some(vw) = voxel_world.as_mut() {
            let player_radius = 0.4;
            let res = self.check_collision(self.camera.eye, combined_vel, dt, player_radius, vw);

            self.camera.eye = res.position;
            self.camera.velocity = res.velocity;

            // Simple grounding check
            if has_gravity {
                let foot_pos = self.camera.eye - Vector3::unit_y() * 0.1;
                let mut smashed_dummy = false;
                if self.is_occupied(foot_pos, player_radius, vw, false, &mut smashed_dummy) {
                    if self.camera.velocity.y <= 0.0 {
                        self.camera.velocity.y = 0.0;
                        self.camera.is_grounded = true;
                    }
                } else {
                    self.camera.is_grounded = false;
                }
            }
        } else {
            self.camera.eye += combined_vel * dt;
        }

        // Global Floor Fallback
        let target_floor = if self.camera_controller.is_crouching { 0.5 } else { 1.0 };
        if self.camera.eye.y < target_floor {
            self.camera.eye.y = target_floor;
            if self.camera.velocity.y < 0.0 {
                self.camera.velocity.y = 0.0;
                self.camera.is_grounded = true;
            }
        }

        self.camera_controller.update_camera_target(&mut self.camera);

        // Update Reality Projector Position (Player follows camera)
        self.player_projector.location = self.camera.eye;

        // Apply Player Influence to World State
        let player_proj = self.player_projector.clone();
        self.world_state.apply_player_influence(&player_proj, dt);

        // Update Lambda System
        let forward = (self.camera.target - self.camera.eye).normalize();
        let anchor = self.camera.eye + forward * self.anchor_distance;
        self.lambda_system.set_anchor(anchor);

        // Fixed timestep for lambda physics for now, could use dt
        let events = self.lambda_system.update(0.016);
        for event in events {
            match event {
                visual_lambda::LambdaEvent::ReductionStarted => self.audio.play_reduce(),
            }
        }

        // Update Spell Effects
        for effect in &mut self.spell_effects {
            effect.timer += dt;
        }
        self.spell_effects.retain(|e| e.timer < e.max_time);

        // --- Background Potential Node Spawning ---
        // We use 3D Simplex noise with fBm to determine if a node of raw potential should spawn.
        use noise::NoiseFn;

        let grid_size = 50.0;
        let px = self.camera.eye.x;
        let py = self.camera.eye.y;
        let pz = self.camera.eye.z;

        let grid_x = (px / grid_size).floor() as i32;
        let grid_y = (py / grid_size).floor() as i32;
        let grid_z = (pz / grid_size).floor() as i32;

        let grid_coord = [grid_x, grid_y, grid_z];

        if !self.world_state.spawned_nodes_grid.contains(&grid_coord) {
            // Evaluate noise at the center of the grid cell
            let nx = (grid_x as f64) * 0.1;
            let ny = (grid_y as f64) * 0.1;
            let nz = (grid_z as f64) * 0.1;

            let noise_val = self.fbm_noise.get([nx, ny, nz]);

            if noise_val > 0.7 {
                // High potential threshold
                let spawn_x = (grid_x as f32) * grid_size + (grid_size / 2.0);
                let spawn_y = py + 5.0; // Spawn slightly above player height
                let spawn_z = (grid_z as f32) * grid_size + (grid_size / 2.0);

                let spawn_pos = cgmath::Point3::new(spawn_x, spawn_y, spawn_z);

                let new_node = crate::reality_types::DroppedItem::new_cube(
                    format!("potential_node_{}_{}_{}", grid_x, grid_y, grid_z),
                    spawn_pos,
                    cgmath::Vector3::new(0.0, 0.0, 0.0), // Floating or drops down
                    0.5,                                 // Larger scale
                    [0.0, 1.0, 1.0, 1.0],                // Cyan glowing color
                    5,                                   // 5x5x5 grid so it looks substantial
                );

                self.world_state.dropped_items.push(new_node);
                log::info!("Spawned raw potential node at {:?}", spawn_pos);
            }

            self.world_state.spawned_nodes_grid.insert(grid_coord);
        }

        voxel_destroyed
    }

    pub fn process_keyboard(&mut self, key_code: &str, pressed: bool) -> Option<Action> {
        // Check for custom spell bindings first
        if let Some(spell_str) = self.input_config.custom_spell_bindings.get(key_code) {
            if pressed {
                log::info!("Casting Custom Spell from Binding: {}", spell_str);
                if let Some(term) = crate::lambda::parse(spell_str) {
                    self.audio.play_cast();
                    if let Some(anomaly) = self.compile_spell(term) {
                        self.world_state.add_anomaly(anomaly.clone());
                        log::info!(
                            "Custom Spell Cast Successfully: {:?}",
                            anomaly.reality_signature.active_style.archetype
                        );

                        let color = match anomaly.reality_signature.active_style.archetype {
                            crate::reality_types::RealityArchetype::Fractal => [1.0, 0.5, 0.0, 1.0],
                            crate::reality_types::RealityArchetype::Prehistoric => [0.3, 0.4, 0.1, 1.0],
                            crate::reality_types::RealityArchetype::SciFi => [0.0, 1.0, 1.0, 1.0],
                            crate::reality_types::RealityArchetype::Horror => [1.0, 0.0, 0.0, 1.0],
                            crate::reality_types::RealityArchetype::Fantasy => [0.0, 1.0, 0.0, 1.0],
                            crate::reality_types::RealityArchetype::Toon => [1.0, 1.0, 0.0, 1.0],
                            crate::reality_types::RealityArchetype::HyperNature => {
                                [0.0, 0.8, 0.2, 1.0]
                            }
                            crate::reality_types::RealityArchetype::Genie => [1.0, 0.0, 1.0, 1.0],
                            crate::reality_types::RealityArchetype::Void => [0.1, 0.1, 0.1, 1.0],
                            crate::reality_types::RealityArchetype::Glitch => [0.0, 1.0, 0.0, 1.0],
                            crate::reality_types::RealityArchetype::Steampunk => {
                                [0.8, 0.5, 0.2, 1.0]
                            }
                            crate::reality_types::RealityArchetype::Vaporwave => {
                                [1.0, 0.0, 1.0, 1.0]
                            }
                            crate::reality_types::RealityArchetype::Noir => [0.5, 0.5, 0.5, 1.0],
                            crate::reality_types::RealityArchetype::CyberSpace => {
                                [0.0, 1.0, 1.0, 1.0]
                            }
                            crate::reality_types::RealityArchetype::Dream => [0.8, 0.6, 1.0, 1.0],
                            crate::reality_types::RealityArchetype::ObraDinn => {
                                [0.9, 0.9, 0.8, 1.0]
                            }
                            crate::reality_types::RealityArchetype::SolarPunk => {
                                [0.2, 0.9, 0.4, 1.0]
                            }
                            crate::reality_types::RealityArchetype::Biopunk => [0.8, 0.2, 0.4, 1.0],
                            crate::reality_types::RealityArchetype::Tron => [0.0, 1.0, 1.0, 1.0],
                            crate::reality_types::RealityArchetype::ColdStorage => {
                                [0.6, 0.9, 1.0, 1.0]
                            }
                            crate::reality_types::RealityArchetype::LiminalSpace => {
                                [0.95, 0.95, 0.8, 1.0]
                            }
                            crate::reality_types::RealityArchetype::Clockwork => {
                                [0.8, 0.6, 0.2, 1.0]
                            }
                            crate::reality_types::RealityArchetype::Cottagecore => {
                                [0.4, 0.7, 0.3, 1.0]
                            }
                            crate::reality_types::RealityArchetype::WildWest => {
                                [0.8, 0.5, 0.2, 1.0]
                            }
                        };

                        self.spell_effects.push(SpellEffect {
                            position: anomaly.location,
                            color,
                            scale: anomaly.reality_signature.active_style.scale,
                            timer: 0.0,
                            max_time: 1.5,
                        });
                    }
                }
            }
            return None;
        }

        // Map raw key to Action
        if let Some(action) = self.input_config.map_key(key_code) {
            // Send engine actions via lambda evaluation for scriptability
            let term = match action {
                Action::Jump => Some(Term::prim(Primitive::Jump)),
                Action::Descend => Some(Term::prim(Primitive::Descend)),
                Action::MoveForward => Some(Term::app(
                    Term::prim(Primitive::Move),
                    Term::prim(Primitive::Forward),
                )),
                Action::MoveBackward => Some(Term::app(
                    Term::prim(Primitive::Move),
                    Term::prim(Primitive::Backward),
                )),
                Action::MoveLeft => Some(Term::app(
                    Term::prim(Primitive::Move),
                    Term::prim(Primitive::Left),
                )),
                Action::MoveRight => Some(Term::app(
                    Term::prim(Primitive::Move),
                    Term::prim(Primitive::Right),
                )),
                Action::DropItem => Some(Term::prim(Primitive::Drop)),
                Action::PickupItem => Some(Term::prim(Primitive::Pickup)),
                _ => None,
            };

            if let Some(t) = term {
                let final_term = if pressed {
                    t
                } else {
                    Term::app(Term::prim(Primitive::Stop), t)
                };
                self.compile_spell(final_term);
            }

            match action {
                Action::CastSpell => {
                    if pressed {
                        // Cast Spell
                        log::info!("Casting Lambda Spell!");
                        self.audio.play_cast();
                        if let Some(term) = self.lambda_system.root_term.clone() {
                            if let Some(anomaly) = self.compile_spell(term) {
                                self.world_state.add_anomaly(anomaly.clone());
                                log::info!(
                                    "Spell Cast Successfully: {:?}",
                                    anomaly.reality_signature.active_style.archetype
                                );

                                // Spawn visual effect
                                let color = match anomaly.reality_signature.active_style.archetype {
                                    crate::reality_types::RealityArchetype::SciFi => {
                                        [0.0, 1.0, 1.0, 1.0]
                                    }
                                    crate::reality_types::RealityArchetype::Horror => {
                                        [1.0, 0.0, 0.0, 1.0]
                                    }
                                    crate::reality_types::RealityArchetype::Fantasy => {
                                        [0.0, 1.0, 0.0, 1.0]
                                    }
                                    crate::reality_types::RealityArchetype::Toon => {
                                        [1.0, 1.0, 0.0, 1.0]
                                    }
                                    crate::reality_types::RealityArchetype::HyperNature => {
                                        [0.0, 0.8, 0.2, 1.0]
                                    }
                                    crate::reality_types::RealityArchetype::Genie => {
                                        [1.0, 0.0, 1.0, 1.0]
                                    }
                                    crate::reality_types::RealityArchetype::Void => {
                                        [0.1, 0.1, 0.1, 1.0]
                                    }
                                    crate::reality_types::RealityArchetype::Glitch => {
                                        [0.0, 1.0, 0.0, 1.0]
                                    }
                                    crate::reality_types::RealityArchetype::Steampunk => {
                                        [0.8, 0.5, 0.2, 1.0]
                                    }
                                    crate::reality_types::RealityArchetype::Vaporwave => {
                                        [1.0, 0.0, 1.0, 1.0]
                                    }
                                    crate::reality_types::RealityArchetype::Noir => {
                                        [0.5, 0.5, 0.5, 1.0]
                                    }
                                    crate::reality_types::RealityArchetype::CyberSpace => {
                                        [0.0, 1.0, 1.0, 1.0]
                                    }
                                    crate::reality_types::RealityArchetype::Dream => {
                                        [0.8, 0.6, 1.0, 1.0]
                                    }
                                    crate::reality_types::RealityArchetype::ObraDinn => {
                                        [0.9, 0.9, 0.8, 1.0]
                                    }
                                    crate::reality_types::RealityArchetype::SolarPunk => {
                                        [0.2, 0.9, 0.4, 1.0]
                                    }
                                    crate::reality_types::RealityArchetype::Biopunk => {
                                        [0.8, 0.2, 0.4, 1.0]
                                    }
                                    crate::reality_types::RealityArchetype::Tron => {
                                        [0.0, 1.0, 1.0, 1.0]
                                    }
                                    crate::reality_types::RealityArchetype::ColdStorage => {
                                        [0.6, 0.9, 1.0, 1.0]
                                    }
                                    crate::reality_types::RealityArchetype::LiminalSpace => {
                                        [0.95, 0.95, 0.8, 1.0]
                                    }
                                    crate::reality_types::RealityArchetype::Clockwork => {
                                        [0.8, 0.6, 0.2, 1.0]
                                    }
                                    crate::reality_types::RealityArchetype::Cottagecore => {
                                        [0.4, 0.7, 0.3, 1.0]
                                    }
                                    crate::reality_types::RealityArchetype::WildWest => {
                                        [0.8, 0.5, 0.2, 1.0]
                                    }
                                    crate::reality_types::RealityArchetype::Fractal => {
                                        [1.0, 0.5, 0.0, 1.0]
                                    }
                                    crate::reality_types::RealityArchetype::Prehistoric => {
                                        [0.3, 0.4, 0.1, 1.0]
                                    }
                                };

                                self.spell_effects.push(SpellEffect {
                                    position: anomaly.location,
                                    color,
                                    scale: anomaly.reality_signature.active_style.scale,
                                    timer: 0.0,
                                    max_time: 1.5,
                                });
                            } else {
                                log::warn!(
                                    "Spell Failed: Term did not compile to a valid anomaly."
                                );
                            }
                        }
                    }
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
                Action::StoreSpell => {
                    if pressed {
                        if let Some(term) = &self.lambda_system.root_term {
                            let term_string = term.to_string();
                            let new_item = crate::reality_types::DroppedItem::new_cube(
                                format!("spell_{}", term_string),
                                self.camera.eye,
                                cgmath::Vector3::new(0.0, 0.0, 0.0),
                                0.2,
                                [0.0, 0.5, 1.0, 1.0], // Blueish color for spells
                                3,
                            );
                            self.world_state.player_inventory.push(new_item);
                            log::info!("Stored spell into inventory: {}", term_string);

                            // Clear visual plane
                            self.lambda_system.root_term = None;
                            self.lambda_system.nodes.clear();
                            self.lambda_system.edges.clear();
                        }
                    }
                }
                Action::DropItem => {
                    if pressed {
                        let (sin_y, cos_y) = self.camera.yaw.sin_cos();
                        let forward_xz = cgmath::Vector3::new(sin_y, 0.0, cos_y);

                        // If inventory has items, drop the last one
                        if let Some(mut item) = self.world_state.player_inventory.pop() {
                            item.position = self.camera.eye;
                            item.velocity = forward_xz * 5.0 + cgmath::Vector3::unit_y() * 2.0;
                            self.world_state.dropped_items.push(item);
                            log::info!("Dropped item from inventory at {:?}", self.camera.eye);
                        } else {
                            // Security Enhancement: Prevent DoS by limiting maximum spawned items
                            // An attacker could spam the DropItem action to spawn infinite items, exhausting memory and network bandwidth during sync.
                            const MAX_SPAWNED_ITEMS: usize = 100;
                            if self.world_state.dropped_items.len() < MAX_SPAWNED_ITEMS {
                                // Otherwise, create a default gold cube to drop
                                let new_item = crate::reality_types::DroppedItem::new_cube(
                                    uuid::Uuid::new_v4().to_string(),
                                    self.camera.eye,
                                    forward_xz * 5.0 + cgmath::Vector3::unit_y() * 2.0,
                                    0.2,
                                    [1.0, 0.8, 0.2, 1.0], // Goldish color
                                    3,                    // 3x3x3 grid
                                );
                                self.world_state.dropped_items.push(new_item);
                                log::info!("Dropped new spawned item at {:?}", self.camera.eye);
                            } else {
                                log::warn!("Security Warning: Maximum spawned items limit reached ({}). Cannot drop new item.", MAX_SPAWNED_ITEMS);
                            }
                        }
                    }
                }
                Action::ToggleInventory => {
                    if pressed {
                        self.show_3d_inventory = !self.show_3d_inventory;
                    }
                }
                Action::ToggleMinimap => {
                    if pressed {
                        self.show_minimap = !self.show_minimap;
                    }
                }
                Action::ToggleUI => {
                    if pressed {
                        self.show_ui = !self.show_ui;
                    }
                }
                Action::CloseInventory => {
                    if pressed {
                        self.show_3d_inventory = false;
                    }
                }
                // Ignore Voxel Actions (Handled by lib.rs / wrapper)
                _ => {}
            }
            return Some(action);
        }
        None
    }

    pub fn process_mouse_down(&mut self, x: f32, y: f32, button: i16) {
        self.audio.resume_context();
        let (ray_origin, ray_dir) = self.get_ray(x, y);

        if button == 0 {
            // Left
            if let Some(idx) = self.lambda_system.intersect(ray_origin, ray_dir) {
                self.lambda_system.start_drag(idx, ray_origin, ray_dir);
            }
        }
    }

    pub fn process_right_click(&mut self, x: f32, y: f32) {
        let (ray_origin, ray_dir) = self.get_ray(x, y);
        if let Some(idx) = self.lambda_system.intersect(ray_origin, ray_dir) {
            self.lambda_system.toggle_collapse(idx);
        }
    }

    pub fn process_right_hold(&mut self, x: f32, y: f32) {
        let (ray_origin, ray_dir) = self.get_ray(x, y);
        if let Some(idx) = self.lambda_system.intersect(ray_origin, ray_dir) {
            self.lambda_system.toggle_edit(idx);
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
        if let Some(idx) = self.lambda_system.dragged_node {
            self.lambda_system.set_pin(idx, true);
        }
        self.lambda_system.end_drag();
    }

    pub fn process_double_click(&mut self, x: f32, y: f32) {
        let (ray_origin, ray_dir) = self.get_ray(x, y);
        if let Some(_idx) = self.lambda_system.intersect(ray_origin, ray_dir) {
            log::info!("Double Clicked Lambda Node! Reducing...");
            self.lambda_system.reduce_root();
        }
    }

    pub fn process_mouse_wheel(&mut self, delta_y: f32) {
        // Adjust anchor distance based on wheel scroll
        let scroll_speed = 0.01;
        self.anchor_distance += delta_y * scroll_speed;

        // Clamp distance between reasonable min and max limits
        self.anchor_distance = self.anchor_distance.clamp(2.0, 25.0);
    }

    // Returns true if state changed and requires save
    pub fn process_click(&mut self, x: f32, y: f32) -> bool {
        let (ray_origin, ray_dir) = self.get_ray(x, y);

        // 1. Check Lambda Intersection (Pin)
        if let Some(idx) = self.lambda_system.intersect(ray_origin, ray_dir) {
            log::info!("Clicked Lambda Node! Toggling pin...");
            self.lambda_system.toggle_pin(idx);
            return false;
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

    pub fn get_node_labels_flat(&self, bytes: &mut Vec<u8>) {
        let view_proj = self.camera.build_view_projection_matrix();
        bytes.clear();

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

    fn compile_spell(&mut self, term: Rc<Term>) -> Option<RealityProjector> {
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
        self.execute_side_effect(&current, true)
    }

    fn execute_side_effect(&mut self, term: &Term, pressed: bool) -> Option<RealityProjector> {
        // Default spawn position
        let forward = (self.camera.target - self.camera.eye).normalize();
        let spawn_pos = self.camera.eye + forward * 10.0;

        match term {
            Term::Prim(p) => {
                match p {
                    Primitive::Jump => {
                        self.camera_controller.process_action(Action::Jump, pressed);
                        None
                    }
                    Primitive::Descend => {
                        self.camera_controller
                            .process_action(Action::Descend, pressed);
                        None
                    }
                    Primitive::Drop => {
                        if pressed {
                            let (sin_y, cos_y) = self.camera.yaw.sin_cos();
                            let forward_xz = cgmath::Vector3::new(sin_y, 0.0, cos_y);

                            if let Some(mut item) = self.world_state.player_inventory.pop() {
                                item.position = self.camera.eye;
                                item.velocity = forward_xz * 5.0 + cgmath::Vector3::unit_y() * 2.0;
                                self.world_state.dropped_items.push(item);
                                log::info!(
                                    "Lambda dropped item from inventory at {:?}",
                                    self.camera.eye
                                );
                            } else {
                                const MAX_SPAWNED_ITEMS: usize = 100;
                                if self.world_state.dropped_items.len() < MAX_SPAWNED_ITEMS {
                                    let new_item = crate::reality_types::DroppedItem::new_cube(
                                        uuid::Uuid::new_v4().to_string(),
                                        self.camera.eye,
                                        forward_xz * 5.0 + cgmath::Vector3::unit_y() * 2.0,
                                        0.2,
                                        [1.0, 0.8, 0.2, 1.0], // Goldish color
                                        3,                    // 3x3x3 grid
                                    );
                                    self.world_state.dropped_items.push(new_item);
                                    log::info!(
                                        "Lambda dropped new spawned item at {:?}",
                                        self.camera.eye
                                    );
                                } else {
                                    log::warn!("Security Warning: Maximum spawned items limit reached ({}). Cannot drop new item.", MAX_SPAWNED_ITEMS);
                                }
                            }
                        }
                        None
                    }
                    Primitive::Pickup => {
                        if pressed {
                            let mut closest_idx = None;
                            let mut min_dist_sq = 25.0; // 5.0 units radius squared
                            let player_pos = self.camera.eye;

                            for (i, item) in self.world_state.dropped_items.iter().enumerate() {
                                let dist_sq = (item.position - player_pos).magnitude2();
                                if dist_sq < min_dist_sq {
                                    min_dist_sq = dist_sq;
                                    closest_idx = Some(i);
                                }
                            }

                            if let Some(idx) = closest_idx {
                                let item = self.world_state.dropped_items.remove(idx);
                                log::info!(
                                    "Lambda picked up item {} from {:?}",
                                    item.id,
                                    item.position
                                );

                                if item.id.starts_with("potential_node_") {
                                    // Absorb the raw potential to expand reality
                                    self.player_projector.reality_signature.influence_radius +=
                                        50.0;
                                    log::info!(
                                        "Absorbed raw potential! New influence radius: {}",
                                        self.player_projector.reality_signature.influence_radius
                                    );

                                    // Trigger Gaussian Splat generation to anchor the landscape
                                    let prompt = format!(
                                        "{:?} landscape",
                                        self.player_projector
                                            .reality_signature
                                            .active_style
                                            .archetype
                                    );
                                    // Push a dream so GenieBridge will generate splats around this new anchor
                                    self.pending_dreams.push(prompt.clone());
                                    // Also generate procedural voxel models
                                    self.pending_models.push((
                                        prompt,
                                        [self.camera.eye.x, self.camera.eye.y, self.camera.eye.z],
                                    ));
                                } else {
                                    // Normal item pickup
                                    self.world_state.player_inventory.push(item);
                                }
                            }
                        }
                        None
                    }
                    Primitive::Heal => {
                        if pressed {
                            self.player_projector.reality_signature.fidelity =
                                (self.player_projector.reality_signature.fidelity + 20.0)
                                    .min(100.0);
                            log::info!(
                                "Player healed. Current fidelity: {}",
                                self.player_projector.reality_signature.fidelity
                            );
                        }
                        None
                    }
                    Primitive::Gravity => {
                        if pressed {
                            self.physics_state = PhysicsState::Gravity;
                            log::info!("Gravity enabled.");
                        }
                        None
                    }
                    Primitive::Levitate => {
                        if pressed {
                            self.physics_state = PhysicsState::Flying;
                            log::info!("Levitate enabled (flying).");
                        }
                        None
                    }
                    Primitive::Extend => {
                        if pressed {
                            log::info!("Extend primitive invoked! Queueing GCT extension.");
                            self.pending_extensions.push([self.camera.eye.x, self.camera.eye.y, self.camera.eye.z]);
                        }
                        None
                    }
                    _ => self.primitive_to_anomaly(*p, spawn_pos),
                }
            }
            Term::App(func, arg) => {
                if let Term::Prim(op) = &**func {
                    if *op == Primitive::Stop {
                        return self.execute_side_effect(arg, false);
                    }

                    if *op == Primitive::Move {
                        if let Term::Prim(target) = &**arg {
                            match target {
                                Primitive::Forward => self
                                    .camera_controller
                                    .process_action(Action::MoveForward, pressed),
                                Primitive::Backward => self
                                    .camera_controller
                                    .process_action(Action::MoveBackward, pressed),
                                Primitive::Left => self
                                    .camera_controller
                                    .process_action(Action::MoveLeft, pressed),
                                Primitive::Right => self
                                    .camera_controller
                                    .process_action(Action::MoveRight, pressed),
                                _ => (),
                            }
                        }
                        return None;
                    }

                    if *op == Primitive::Dream {
                        if pressed {
                            if let Term::Var(prompt) = &**arg {
                                log::info!("Triggering DREAM with prompt: {}", prompt);
                                self.pending_dreams.push(prompt.clone());
                            } else if let Term::Prim(prim_prompt) = &**arg {
                                log::info!(
                                    "Triggering DREAM with primitive prompt: {}",
                                    prim_prompt
                                );
                                self.pending_dreams.push(prim_prompt.to_string());
                            }
                        }
                        return None;
                    }

                    if let Term::Prim(target) = &**arg {
                        return self.combine_primitives(*op, *target, spawn_pos);
                    } else if let Term::Var(target) = &**arg {
                        if *op == Primitive::SetArchetype {
                            let arch = match target.to_uppercase().as_str() {
                                "FANTASY" => Some(crate::reality_types::RealityArchetype::Fantasy),
                                "SCIFI" => Some(crate::reality_types::RealityArchetype::SciFi),
                                "HORROR" => Some(crate::reality_types::RealityArchetype::Horror),
                                "TOON" => Some(crate::reality_types::RealityArchetype::Toon),
                                "HYPERNATURE" => {
                                    Some(crate::reality_types::RealityArchetype::HyperNature)
                                }
                                "GENIE" => Some(crate::reality_types::RealityArchetype::Genie),
                                "GLITCH" => Some(crate::reality_types::RealityArchetype::Glitch),
                                "STEAMPUNK" => {
                                    Some(crate::reality_types::RealityArchetype::Steampunk)
                                }
                                "VAPORWAVE" => {
                                    Some(crate::reality_types::RealityArchetype::Vaporwave)
                                }
                                "NOIR" => Some(crate::reality_types::RealityArchetype::Noir),
                                "CYBERSPACE" => {
                                    Some(crate::reality_types::RealityArchetype::CyberSpace)
                                }
                                "DREAM" => Some(crate::reality_types::RealityArchetype::Dream),
                                "OBRADINN" => {
                                    Some(crate::reality_types::RealityArchetype::ObraDinn)
                                }
                                "SOLARPUNK" => {
                                    Some(crate::reality_types::RealityArchetype::SolarPunk)
                                }
                                "BIOPUNK" => Some(crate::reality_types::RealityArchetype::Biopunk),
                                "CLOCKWORK" => {
                                    Some(crate::reality_types::RealityArchetype::Clockwork)
                                }
                                "COTTAGECORE" => {
                                    Some(crate::reality_types::RealityArchetype::Cottagecore)
                                }
                                "WILDWEST" => {
                                    Some(crate::reality_types::RealityArchetype::WildWest)
                                }
                                "FRACTAL" => Some(crate::reality_types::RealityArchetype::Fractal),
                                "PREHISTORIC" => Some(crate::reality_types::RealityArchetype::Prehistoric),
                                "VOID" => Some(crate::reality_types::RealityArchetype::Void),
                                _ => None,
                            };

                            if let Some(a) = arch {
                                self.player_projector
                                    .reality_signature
                                    .active_style
                                    .archetype = a;
                                log::info!("Player archetype set to {:?}", a);
                            } else {
                                log::warn!("Unknown archetype: {}", target);
                            }
                            return None;
                        }
                    }
                }

                // Fallback mechanism to ensure the grammar works like a context-free grammar,
                // where every spell, even invalid ones, resolves into a functioning spell/anomaly.
                log::info!("Term resulted in an invalid application: {:?}", term);

                let mut base = self.primitive_to_anomaly(Primitive::Void, spawn_pos)?;
                base.reality_signature.active_style.archetype =
                    crate::reality_types::RealityArchetype::Glitch;
                Some(base)
            }

            // Fallback for unresolved terms like plain variables
            _ => {
                log::info!("Term unresolved: {:?}", term);
                let mut base = self.primitive_to_anomaly(Primitive::Void, spawn_pos)?;
                base.reality_signature.active_style.archetype =
                    crate::reality_types::RealityArchetype::Glitch;
                Some(base)
            }
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
            Primitive::Acid => {
                sig.active_style.archetype = RealityArchetype::Biopunk;
                sig.active_style.roughness = 0.8;
                sig.active_style.distortion = 0.9;
            }
            Primitive::Fog => {
                sig.active_style.archetype = RealityArchetype::Noir;
                sig.active_style.roughness = 0.1;
                sig.active_style.distortion = 0.2;
            }
            Primitive::Cloud => {
                sig.active_style.archetype = RealityArchetype::Dream;
                sig.active_style.roughness = 0.4;
                sig.active_style.scale = 2.0;
            }
            Primitive::Rain => {
                sig.active_style.archetype = RealityArchetype::Noir;
                sig.active_style.roughness = 0.5;
                sig.active_style.distortion = 0.6;
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
            RealityArchetype::Tron,
            RealityArchetype::Clockwork,
            RealityArchetype::Cottagecore,
            RealityArchetype::WildWest,
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

            let mut npc = RealityProjector::new(loc, sig);
            npc.behavior = Some(crate::projector::NpcBehavior {
                preferred_archetype: arch,
                energy: 100.0,
                mutation_progress: 0.0,
                hostile: arch == RealityArchetype::Horror,
                goal_stack: Vec::new(),
                animation_playback: Some(crate::projector::AnimationPlayback::default()),
            });
            self.world_state.npcs.push(npc);
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

        self.physics_state = PhysicsState::Flying;
        self.pending_dreams.clear();
        self.pending_models.clear();
        self.pending_extensions.clear();

        log::info!("World State Reset.");
    }
}
