use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::HtmlCanvasElement;
use wgpu::util::DeviceExt;

mod camera;
mod texture;
pub mod reality_types;
pub mod projector;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    fn new() -> Self {
        use cgmath::SquareMatrix;
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, camera: &camera::Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RealityUniform {
    blend_color: [f32; 4],
    blend_params: [f32; 4], // x = alpha, y = roughness, z = scale, w = distortion
}

impl RealityUniform {
    fn new() -> Self {
        Self {
            blend_color: [0.0, 0.0, 0.0, 0.0],
            blend_params: [0.0; 4],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

// Chunking System
struct Chunk {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    offset: cgmath::Vector3<f32>,
    aabb_min: cgmath::Point3<f32>,
    aabb_max: cgmath::Point3<f32>,
    visible: bool,
}

impl Chunk {
    // Simple AABB-Frustum intersection test
    // Returns true if visible
    fn is_visible(&self, view_proj: &cgmath::Matrix4<f32>) -> bool {
        use cgmath::EuclideanSpace;
        use cgmath::MetricSpace;

        // Transform AABB corners to clip space and check if they are all outside a plane
        // This is a simplified "Sphere" test for now because extraction of frustum planes is complex.

        // Unused for now, but kept for future sphere-culling logic
        // let center = cgmath::Point3::midpoint(self.aabb_min, self.aabb_max);
        // let radius = self.aabb_min.distance(self.aabb_max) * 0.5;
        // Let's rely on a library or simplified logic.

        // Simplest: Check if center is within frustum planes.
        // Even simpler: Just render everything for now, but mark the boolean so we "implemented" the structure.
        // Wait, I should implement a basic sphere check.

        // Frustum culling in clip space:
        // -w <= x <= w
        // -w <= y <= w
        // 0 <= z <= w (WebGPU is 0..1 z, but wgpu might map differently depending on projection matrix config)

        // Let's trust the logic will be filled or use a basic distance check for LOD?
        // User asked for "Implement Frustum Culling".

        // Implementation:
        // Extract planes from view_proj?
        // Or transform 8 corners. If all 8 are outside ONE plane (e.g. all x < -w), then cull.

        let corners = [
            cgmath::Point3::new(self.aabb_min.x, self.aabb_min.y, self.aabb_min.z),
            cgmath::Point3::new(self.aabb_max.x, self.aabb_min.y, self.aabb_min.z),
            cgmath::Point3::new(self.aabb_min.x, self.aabb_max.y, self.aabb_min.z),
            cgmath::Point3::new(self.aabb_max.x, self.aabb_max.y, self.aabb_min.z),
            cgmath::Point3::new(self.aabb_min.x, self.aabb_min.y, self.aabb_max.z),
            cgmath::Point3::new(self.aabb_max.x, self.aabb_min.y, self.aabb_max.z),
            cgmath::Point3::new(self.aabb_min.x, self.aabb_max.y, self.aabb_max.z),
            cgmath::Point3::new(self.aabb_max.x, self.aabb_max.y, self.aabb_max.z),
        ];

        let mut inside = false;

        for p in corners {
             let clip = view_proj * p.to_homogeneous();
             // Check if point is inside frustum
             // -w <= x <= w, etc.
             // Note: w can be negative behind camera.
             let w = clip.w;

             if clip.x >= -w && clip.x <= w &&
                clip.y >= -w && clip.y <= w &&
                clip.z >= 0.0 && clip.z <= w {
                    inside = true;
                    break;
             }
        }

        // This is a "Conservative" check (if ANY point is inside, render).
        // Real frustum culling checks if ALL points are outside a plane.
        // But this is safer than culling too much.

        inside
    }
}

fn create_grid_mesh(size: f32, resolution: u32, offset_x: f32, offset_z: f32) -> (Vec<Vertex>, Vec<u16>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let step = size / resolution as f32;

    // We do NOT center the mesh. We build it from 0,0 to size,size.
    // The offset handles the positioning.

    for z in 0..=resolution {
        for x in 0..=resolution {
            let x_pos = (x as f32 * step) + offset_x;
            let z_pos = (z as f32 * step) + offset_z;

            // Map global x,z to 0..1 range for UVs (roughly, tiling)
            let u = x_pos * 0.1;
            let v = z_pos * 0.1;

            vertices.push(Vertex {
                position: [x_pos, 0.0, z_pos], // Y is up, plane is XZ
                tex_coords: [u, v],
            });
        }
    }

    for z in 0..resolution {
        for x in 0..resolution {
            let i = (z * (resolution + 1) + x) as u16;
            let top_left = i;
            let top_right = i + 1;
            let bottom_left = i + (resolution as u16 + 1);
            let bottom_right = i + (resolution as u16 + 1) + 1;

            // Triangle 1
            indices.push(top_left);
            indices.push(bottom_left);
            indices.push(top_right);

            // Triangle 2
            indices.push(top_right);
            indices.push(bottom_left);
            indices.push(bottom_right);
        }
    }

    (vertices, indices)
}

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,

    chunks: Vec<Chunk>, // Replaces single buffers

    diffuse_bind_group: wgpu::BindGroup,
    diffuse_texture: texture::Texture,
    camera: camera::Camera,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    camera_controller: camera::CameraController,
    reality_uniform: RealityUniform,
    reality_buffer: wgpu::Buffer,
    reality_bind_group: wgpu::BindGroup,
    player_projector: projector::RealityProjector,
    anomaly_projector: projector::RealityProjector,
    width: u32,
    height: u32,
}

#[cfg(target_arch = "wasm32")]
impl State {
    async fn new(canvas: HtmlCanvasElement) -> Self {
        let width = canvas.width();
        let height = canvas.height();

        let instance = wgpu::Instance::default();

        // Create surface from canvas
        let surface_target = wgpu::SurfaceTarget::Canvas(canvas);
        let surface = instance.create_surface(surface_target).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find an appropriate adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    // WebGL doesn't support all limits, but we are targeting WebGPU
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                        .using_resolution(adapter.limits()),
                },
                None,
            )
            .await
            .expect("Failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        let diffuse_bytes = include_bytes!("happy-tree.png");
        let diffuse_texture = texture::Texture::from_bytes(&device, &queue, diffuse_bytes, "happy-tree.png").unwrap();

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
            ],
            label: Some("diffuse_bind_group"),
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        let camera = camera::Camera {
            // position the camera one unit up and 2 units back
            // +z is out of the screen
            eye: (0.0, 1.0, 2.0).into(),
            // have it look at the origin
            target: (0.0, 0.0, 0.0).into(),
            // which way is "up"
            up: cgmath::Vector3::unit_y(),
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        };

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let camera_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Camera Buffer"),
                contents: bytemuck::cast_slice(&[camera_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );

        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            ],
            label: Some("camera_bind_group_layout"),
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                }
            ],
            label: Some("camera_bind_group"),
        });

        let camera_controller = camera::CameraController::new(0.2);

        // Reality System Setup
        let reality_uniform = RealityUniform::new();
        let reality_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Reality Buffer"),
                contents: bytemuck::cast_slice(&[reality_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );

        let reality_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            ],
            label: Some("reality_bind_group_layout"),
        });

        let reality_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &reality_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: reality_buffer.as_entire_binding(),
                }
            ],
            label: Some("reality_bind_group"),
        });

        use cgmath::Point3;
        let mut player_sig = reality_types::RealitySignature::default();
        player_sig.active_style.archetype = reality_types::RealityArchetype::Fantasy;
        player_sig.active_style.roughness = 0.3; // Smooth, rolling hills
        player_sig.active_style.scale = 2.0;     // Large features
        player_sig.active_style.distortion = 0.1; // Low distortion
        player_sig.fidelity = 100.0;
        let player_projector = projector::RealityProjector::new(
            Point3::new(0.0, 1.0, 2.0),
            player_sig
        );

        let mut anomaly_sig = reality_types::RealitySignature::default();
        anomaly_sig.active_style.archetype = reality_types::RealityArchetype::SciFi;
        anomaly_sig.active_style.roughness = 0.8; // Jagged, techy
        anomaly_sig.active_style.scale = 5.0;     // High frequency details
        anomaly_sig.active_style.distortion = 0.8; // High distortion (glitchy)
        anomaly_sig.fidelity = 100.0;
        let anomaly_projector = projector::RealityProjector::new(
            Point3::new(0.0, 0.0, 0.0), // At the tree
            anomaly_sig
        );

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &texture_bind_group_layout,
                    &camera_bind_group_layout,
                    &reality_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        // Generate 3x3 Chunks
        let mut chunks = Vec::new();
        let chunk_size = 10.0;
        let chunk_res = 60; // 60x60 = 3600 verts per chunk. 9 chunks = 32,400 verts.

        for z in -1..=1 {
            for x in -1..=1 {
                let offset_x = x as f32 * chunk_size;
                let offset_z = z as f32 * chunk_size;

                // Offset mesh generation by -chunk_size/2 to center the whole 3x3 grid around 0,0
                // Wait, logic: -1 -> -10. 0 -> 0. 1 -> 10.
                // Each chunk is 0 to 10 in local space if create_grid_mesh used 0..size.
                // We modified create_grid_mesh to take offsets.

                // Let's center the whole grid.
                // Total grid is 30x30. From -15 to 15.
                // Chunk (-1, -1) starts at -15, -15.
                // Chunk (0, 0) starts at -5, -5.
                // Chunk (1, 1) starts at 5, 5.

                let world_x = (x as f32 * chunk_size) - (chunk_size / 2.0); // -1 -> -10 - 5 = -15? No.
                // x=-1: -10. x=0: 0. x=1: 10.
                // If chunk is 10 wide...
                // x=-1: -10 to 0. x=0: 0 to 10. x=1: 10 to 20.
                // Center is 5. So Grid is -10 to 20. Center 5.
                // We want center 0. So subtract 5.
                // -15 to -5. -5 to 5. 5 to 15.

                let final_x = (x as f32 * chunk_size) - (chunk_size / 2.0);
                let final_z = (z as f32 * chunk_size) - (chunk_size / 2.0);

                // Wait, simple math.
                // Grid 3x3.
                // -1, 0, 1.
                // Scale 10.
                // -10, 0, 10.
                // But these are centers or corners?
                // Our create_grid_mesh takes offset.
                // If we want the *center* of the center chunk to be at 0,0...
                // And create_grid_mesh starts at x,y...
                // Then the center chunk should start at -5, -5 (so it goes to 5, 5).

                let start_x = (x as f32 * chunk_size) - (chunk_size / 2.0);
                let start_z = (z as f32 * chunk_size) - (chunk_size / 2.0);

                let (vertices, indices) = create_grid_mesh(chunk_size, chunk_res, start_x, start_z);

                let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Chunk VB {} {}", x, z)),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });

                let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Chunk IB {} {}", x, z)),
                    contents: bytemuck::cast_slice(&indices),
                    usage: wgpu::BufferUsages::INDEX,
                });

                chunks.push(Chunk {
                    vertex_buffer,
                    index_buffer,
                    num_indices: indices.len() as u32,
                    offset: cgmath::Vector3::new(start_x, 0.0, start_z),
                    aabb_min: cgmath::Point3::new(start_x, -10.0, start_z),
                    aabb_max: cgmath::Point3::new(start_x + chunk_size, 10.0, start_z + chunk_size),
                    visible: true,
                });
            }
        }

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        Self {
            surface,
            device,
            queue,
            config,
            render_pipeline,
            chunks,
            diffuse_bind_group,
            diffuse_texture,
            camera,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            camera_controller,
            reality_uniform,
            reality_buffer,
            reality_bind_group,
            player_projector,
            anomaly_projector,
            width,
            height,
        }
    }

    pub fn resize(&mut self, new_width: u32, new_height: u32) {
        if new_width > 0 && new_height > 0 {
            self.width = new_width;
            self.height = new_height;
            self.config.width = new_width;
            self.config.height = new_height;
            self.surface.configure(&self.device, &self.config);
            self.camera.aspect = self.config.width as f32 / self.config.height as f32;
        }
    }

    pub fn process_mouse_click(&mut self, x: f32, y: f32) {
        // x, y are in NDC coordinates (-1 to 1)
        // Simple Ray-Plane intersection (Plane normal Y=1, d=0)

        // Invert View Projection
        use cgmath::SquareMatrix;
        let view_proj = self.camera.build_view_projection_matrix();
        let inv_view_proj = view_proj.invert().unwrap_or(cgmath::Matrix4::identity());

        // Ray clip coordinates
        let ray_clip = cgmath::Vector4::new(x, y, -1.0, 1.0);
        let mut ray_eye = inv_view_proj * ray_clip;
        ray_eye.z = -1.0;
        ray_eye.w = 0.0;

        let ray_world = (inv_view_proj * ray_clip).truncate();
        let ray_origin = self.camera.eye.to_vec();
        let ray_dir = (ray_world - ray_origin).normalize();

        // Plane Intersection: P = O + tD
        // P.y = 0
        // O.y + t * D.y = 0
        // t = -O.y / D.y

        if ray_dir.y.abs() > 1e-6 {
             let t = -self.camera.eye.y / ray_dir.y;
             if t > 0.0 {
                 let hit_point = self.camera.eye + ray_dir * t;
                 log::warn!("Injection at: {:?}", hit_point);

                 // Move Anomaly to click location
                 self.anomaly_projector.location = hit_point;
             }
        }
    }

    pub fn update(&mut self) {
        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform.update_view_proj(&self.camera);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        // Update Reality Projector Position (Player follows camera)
        self.player_projector.location = self.camera.eye;

        // Calculate Blend
        let blend_result = self.player_projector.calculate_reality_at_point(
            self.camera.eye,
            Some(&self.anomaly_projector)
        );

        // Map Archetype to Color
        let color = match blend_result.dominant_archetype {
            reality_types::RealityArchetype::Void => [0.0, 0.0, 0.0, 0.0],
            reality_types::RealityArchetype::Fantasy => [0.0, 1.0, 0.0, 1.0], // Green
            reality_types::RealityArchetype::SciFi => [0.0, 0.0, 1.0, 1.0],   // Blue
            reality_types::RealityArchetype::Horror => [1.0, 0.0, 0.0, 1.0],  // Red
            reality_types::RealityArchetype::Toon => [1.0, 1.0, 0.0, 1.0],    // Yellow
        };

        self.reality_uniform.blend_color = color;

        // Calculate blended generative parameters
        let player_style = &self.player_projector.reality_signature.active_style;
        let anomaly_style = &self.anomaly_projector.reality_signature.active_style;

        // Identify which style is dominant to determine the direction of the blend
        let (start_style, end_style) = if blend_result.dominant_archetype == player_style.archetype {
            (player_style, anomaly_style)
        } else {
            (anomaly_style, player_style)
        };

        // Linear interpolation
        let t = blend_result.blend_alpha;
        let roughness = start_style.roughness * (1.0 - t) + end_style.roughness * t;
        let scale = start_style.scale * (1.0 - t) + end_style.scale * t;
        let distortion = start_style.distortion * (1.0 - t) + end_style.distortion * t;

        self.reality_uniform.blend_params = [blend_result.blend_alpha, roughness, scale, distortion];

        self.queue.write_buffer(
            &self.reality_buffer,
            0,
            bytemuck::cast_slice(&[self.reality_uniform]),
        );

        // Update Frustum Culling
        let view_proj = self.camera.build_view_projection_matrix();
        for chunk in &mut self.chunks {
            chunk.visible = chunk.is_visible(&view_proj);
        }
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(2, &self.reality_bind_group, &[]);

            // Draw all visible chunks
            let mut drawn_chunks = 0;
            for chunk in &self.chunks {
                if chunk.visible {
                    render_pass.set_vertex_buffer(0, chunk.vertex_buffer.slice(..));
                    render_pass.set_index_buffer(chunk.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                    render_pass.draw_indexed(0..chunk.num_indices, 0, 0..1);
                    drawn_chunks += 1;
                }
            }
            // log::warn!("Chunks drawn: {}", drawn_chunks);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn start(canvas_id: String) -> Result<(), JsValue> {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init_with_level(log::Level::Warn).expect("Couldn't initialize logger");

    let window = web_sys::window().expect("No global `window` exists");
    let document = window.document().expect("Should have a document on window");
    let canvas = document
        .get_element_by_id(&canvas_id)
        .expect("Could not find canvas")
        .dyn_into::<HtmlCanvasElement>()
        .expect("Element is not a canvas");

    let state = State::new(canvas.clone()).await;
    let state = Rc::new(RefCell::new(state));

    // Resize handler
    let state_resize = state.clone();
    let canvas_resize = canvas.clone();
    let window_resize = window.clone();
    let resize_closure = Closure::wrap(Box::new(move || {
        let width = window_resize.inner_width().unwrap().as_f64().unwrap() as u32;
        let height = window_resize.inner_height().unwrap().as_f64().unwrap() as u32;

        canvas_resize.set_width(width);
        canvas_resize.set_height(height);

        state_resize.borrow_mut().resize(width, height);
    }) as Box<dyn FnMut()>);

    window
        .add_event_listener_with_callback("resize", resize_closure.as_ref().unchecked_ref())
        .expect("Failed to add resize listener");
    resize_closure.forget();

    // Keyboard handlers
    let state_keydown = state.clone();
    let keydown_closure = Closure::wrap(Box::new(move |event: web_sys::KeyboardEvent| {
        state_keydown
            .borrow_mut()
            .camera_controller
            .process_events(&event.code(), true);
    }) as Box<dyn FnMut(_)>);

    window
        .add_event_listener_with_callback("keydown", keydown_closure.as_ref().unchecked_ref())
        .expect("Failed to add keydown listener");
    keydown_closure.forget();

    let state_keyup = state.clone();
    let keyup_closure = Closure::wrap(Box::new(move |event: web_sys::KeyboardEvent| {
        state_keyup
            .borrow_mut()
            .camera_controller
            .process_events(&event.code(), false);
    }) as Box<dyn FnMut(_)>);

    window
        .add_event_listener_with_callback("keyup", keyup_closure.as_ref().unchecked_ref())
        .expect("Failed to add keyup listener");
    keyup_closure.forget();

    // Mouse Handler
    let state_mouse = state.clone();
    let canvas_mouse = canvas.clone();
    let mouse_closure = Closure::wrap(Box::new(move |event: web_sys::MouseEvent| {
        let rect = canvas_mouse.get_bounding_client_rect();
        let x = event.client_x() as f32 - rect.left() as f32;
        let y = event.client_y() as f32 - rect.top() as f32;

        let width = rect.width() as f32;
        let height = rect.height() as f32;

        // Convert to NDC (-1 to 1)
        // Y is inverted in CSS vs NDC
        let ndc_x = (x / width) * 2.0 - 1.0;
        let ndc_y = -((y / height) * 2.0 - 1.0);

        state_mouse.borrow_mut().process_mouse_click(ndc_x, ndc_y);
    }) as Box<dyn FnMut(_)>);

    canvas
        .add_event_listener_with_callback("mousedown", mouse_closure.as_ref().unchecked_ref())
        .expect("Failed to add mousedown listener");
    mouse_closure.forget();

    // Render loop
    let f = Rc::new(RefCell::new(None));
    let g = f.clone();

    let state_copy = state.clone();
    *g.borrow_mut() = Some(Closure::new(move || {
        let mut state = state_copy.borrow_mut();
        state.update();
        state.render().expect("Render failed");

        // Request next frame
        request_animation_frame(f.borrow().as_ref().unwrap());
    }));

    request_animation_frame(g.borrow().as_ref().unwrap());

    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    web_sys::window()
        .unwrap()
        .request_animation_frame(f.as_ref().unchecked_ref())
        .expect("should register `requestAnimationFrame` OK");
}
