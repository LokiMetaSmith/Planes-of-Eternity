#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;
#[cfg(target_arch = "wasm32")]
use std::rc::Rc;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use web_sys::{HtmlCanvasElement, XrSession, XrWebGlLayer, XrRenderStateInit, XrReferenceSpace, XrReferenceSpaceType, XrFrame, XrViewerPose, XrView, XrEye};
use wgpu::util::DeviceExt;
use cgmath::{InnerSpace, SquareMatrix, Rotation};
use serde::Serialize;
use std::collections::HashMap;

pub mod camera;
mod texture;
pub mod reality_types;
pub mod projector;
pub mod persistence;
pub mod world;
pub mod voxel;
pub mod genie_bridge;
pub mod network;
pub mod lambda;
pub mod visual_lambda;
pub mod input;
pub mod engine;
pub mod audio;
#[cfg(target_arch = "wasm32")]
pub mod xr;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4], // xyz, w=padding
}

impl CameraUniform {
    fn new() -> Self {
        use cgmath::SquareMatrix;
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
            camera_pos: [0.0; 4],
        }
    }

    fn update_view_proj(&mut self, camera: &camera::Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
        self.camera_pos = [camera.eye.x, camera.eye.y, camera.eye.z, 1.0];
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RealityUniform {
    proj1_pos_fid: [f32; 4],
    proj1_params: [f32; 4], // x=roughness, y=scale, z=distortion, w=archetype_id
    proj1_color: [f32; 4],
    proj2_pos_fid: [f32; 4],
    proj2_params: [f32; 4],
    proj2_color: [f32; 4],
    global_offset: [f32; 4],
}

impl RealityUniform {
    fn new() -> Self {
        Self {
            proj1_pos_fid: [0.0; 4],
            proj1_params: [0.0; 4],
            proj1_color: [0.0; 4],
            proj2_pos_fid: [0.0; 4],
            proj2_params: [0.0; 4],
            proj2_color: [0.0; 4],
            global_offset: [0.0; 4],
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

// Helper for Frustum Culling
fn is_aabb_visible(min: cgmath::Point3<f32>, max: cgmath::Point3<f32>, view_proj: &cgmath::Matrix4<f32>) -> bool {
    let corners = [
        cgmath::Point3::new(min.x, min.y, min.z),
        cgmath::Point3::new(max.x, min.y, min.z),
        cgmath::Point3::new(min.x, max.y, min.z),
        cgmath::Point3::new(max.x, max.y, min.z),
        cgmath::Point3::new(min.x, min.y, max.z),
        cgmath::Point3::new(max.x, min.y, max.z),
        cgmath::Point3::new(min.x, max.y, max.z),
        cgmath::Point3::new(max.x, max.y, max.z),
    ];

    for p in corners {
         let clip = view_proj * p.to_homogeneous();
         let w = clip.w;
         // Check if point is inside frustum (conservative check)
         if clip.x >= -w && clip.x <= w &&
            clip.y >= -w && clip.y <= w &&
            clip.z >= 0.0 && clip.z <= w {
                return true;
         }
    }
    false
}

fn create_grid_mesh(size: f32, resolution: u32) -> (Vec<Vertex>, Vec<u16>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let step = size / resolution as f32;
    // Center the mesh around 0,0 locally
    let offset = size / 2.0;

    for z in 0..=resolution {
        for x in 0..=resolution {
            let x_pos = (x as f32 * step) - offset;
            let z_pos = (z as f32 * step) - offset;

            // UVs 0..1
            let u = x as f32 / resolution as f32;
            let v = z as f32 / resolution as f32;

            vertices.push(Vertex {
                position: [x_pos, 0.0, z_pos],
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

            indices.push(top_left);
            indices.push(bottom_left);
            indices.push(top_right);

            indices.push(top_right);
            indices.push(bottom_left);
            indices.push(bottom_right);
        }
    }

    (vertices, indices)
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Instance {
    model: [[f32; 4]; 4],
    stability: f32,
    padding: [f32; 3],
}

impl Instance {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Instance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

pub struct ChunkMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    pub lod_level: usize,
}

struct ChunkData {
    position: cgmath::Vector3<f32>,
    aabb_min: cgmath::Point3<f32>,
    aabb_max: cgmath::Point3<f32>,
}

#[cfg(target_arch = "wasm32")]
pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,

    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,

    instance_buffer: wgpu::Buffer,
    num_instances: u32,
    chunk_data: Vec<ChunkData>, // CPU side data for culling

    diffuse_bind_group: wgpu::BindGroup,
    diffuse_texture: texture::Texture,
    // camera moved to engine
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    // camera_controller moved to engine
    reality_uniform: RealityUniform,
    reality_buffer: wgpu::Buffer,
    reality_bind_group: wgpu::BindGroup,
    // player_projector moved to engine
    // world_state moved to engine
    // active_anomaly moved to engine
    // width/height logic in engine, but config needs it
    width: u32,
    height: u32,
    // global_offset moved to engine
    lambda_renderer: visual_lambda::LambdaRenderer,
    // lambda_system moved to engine
    // pending_full_sync moved to engine
    // time moved to engine

    pub engine: engine::Engine,

    pub depth_texture: texture::Texture,
    pub voxel_world: voxel::VoxelWorld,
    pub voxel_pipeline: wgpu::RenderPipeline,
    pub voxel_meshes: HashMap<voxel::ChunkKey, ChunkMesh>,
    pub voxel_dirty: bool,
    pub last_lod_update_pos: cgmath::Point3<f32>,
    pub voxel_atlas: texture::Texture,
    pub voxel_bind_group: wgpu::BindGroup,
    pub voxel_density_texture: wgpu::Texture,
    pub voxel_density_view: wgpu::TextureView,
    pub in_xr: bool,
    pub canvas: HtmlCanvasElement,
}

#[cfg(target_arch = "wasm32")]
impl State {
    async fn new(canvas: HtmlCanvasElement) -> Self {
        let width = canvas.width();
        let height = canvas.height();

        let instance = wgpu::Instance::default();

        // Create surface from canvas
        let surface_target = wgpu::SurfaceTarget::Canvas(canvas.clone());
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
                    memory_hints: Default::default(),
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

        // Enable transparent background for AR overlay
        let alpha_mode = surface_caps.alpha_modes.iter()
            .find(|&&m| m == wgpu::CompositeAlphaMode::PreMultiplied)
            .cloned()
            .unwrap_or(surface_caps.alpha_modes[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        let diffuse_bytes = include_bytes!("happy-tree.png");
        let diffuse_texture = match texture::Texture::from_bytes(&device, &queue, diffuse_bytes, "happy-tree.png") {
            Ok(tex) => tex,
            Err(e) => {
                log::warn!("Failed to load happy-tree.png: {:?}. Using fallback.", e);
                texture::Texture::create_fallback(&device, &queue, "happy-tree-fallback")
            }
        };

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

        let depth_texture = texture::Texture::create_depth_texture(&device, &config, "depth_texture");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        // Init Engine Logic
        let mut loaded_state = persistence::load_from_local_storage("reality_engine_save");

        // Extract voxel_world if present
        let loaded_voxel_world = if let Some(ref mut state) = loaded_state {
             state.voxel_world.take()
        } else {
             None
        };

        let engine = engine::Engine::new(width, height, loaded_state);

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&engine.camera);

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
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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

        // Generate single chunk mesh (template)
        let chunk_size = 10.0;
        let chunk_res = 60;
        let (vertices, indices) = create_grid_mesh(chunk_size, chunk_res);

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Chunk Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Chunk Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let num_indices = indices.len() as u32;

        // Define logical chunks (Positions)
        let mut chunk_data = Vec::new();
        let grid_count = 3; // 3x3
        let half_grid = (grid_count as f32 * chunk_size) / 2.0;
        let offset = chunk_size / 2.0; // Mesh is centered, so we move centers.

        for z in 0..grid_count {
            for x in 0..grid_count {
                let x_pos = (x as f32 * chunk_size) - chunk_size; // 0->-10, 1->0, 2->10
                let z_pos = (z as f32 * chunk_size) - chunk_size;

                chunk_data.push(ChunkData {
                    position: cgmath::Vector3::new(x_pos, 0.0, z_pos),
                    aabb_min: cgmath::Point3::new(x_pos - 5.0, -10.0, z_pos - 5.0),
                    aabb_max: cgmath::Point3::new(x_pos + 5.0, 10.0, z_pos + 5.0),
                });
            }
        }

        // Initial Instance Buffer
        let instances = vec![Instance { model: cgmath::Matrix4::identity().into(), stability: 1.0, padding: [0.0; 3] }; chunk_data.len()];
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc(), Instance::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
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
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let lambda_renderer = visual_lambda::LambdaRenderer::new(
            &device,
            config.format,
            &camera_bind_group_layout,
        );

        // --- Voxel Initialization ---
        let voxel_world = if let Some(vw) = loaded_voxel_world {
            log::info!("Restored Voxel World from save.");
            vw
        } else {
            let mut vw = voxel::VoxelWorld::new();
            vw.generate_default_world();
            vw
        };

        // Voxel Texture Atlas
        let voxel_atlas = texture::Texture::create_procedural_atlas(&device, &queue);

        // Voxel Density Texture (3D)
        let density_size = wgpu::Extent3d {
            width: 128,
            height: 128,
            depth_or_array_layers: 128,
        };
        let voxel_density_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Voxel Density Texture"),
            size: density_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R8Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let voxel_density_view = voxel_density_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let voxel_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D3,
                        sample_type: wgpu::TextureSampleType::Uint,
                    },
                    count: None,
                },
            ],
            label: Some("voxel_bind_group_layout"),
        });

        let voxel_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &voxel_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&voxel_atlas.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&voxel_atlas.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&voxel_density_view),
                },
            ],
            label: Some("voxel_bind_group"),
        });

        let voxel_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Voxel Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shader_voxel.wgsl"))),
        });

        let voxel_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Voxel Pipeline Layout"),
            bind_group_layouts: &[
                &camera_bind_group_layout, // Group 0
                &voxel_bind_group_layout,  // Group 1
                &reality_bind_group_layout, // Group 2
            ],
            push_constant_ranges: &[],
        });

        let voxel_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Voxel Render Pipeline"),
            layout: Some(&voxel_pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &voxel_shader,
                entry_point: Some("vs_main"),
                buffers: &[voxel::VoxelVertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &voxel_shader,
                entry_point: Some("fs_main"),
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
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let mut state = Self {
            surface,
            device,
            queue,
            config,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            instance_buffer,
            num_instances: chunk_data.len() as u32,
            chunk_data,
            diffuse_bind_group,
            diffuse_texture,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            reality_uniform,
            reality_buffer,
            reality_bind_group,
            width,
            height,
            lambda_renderer,
            engine,
            depth_texture,
            voxel_world,
            voxel_pipeline,
            voxel_meshes: HashMap::new(),
            voxel_dirty: false,
            last_lod_update_pos: cgmath::Point3::new(0.0, 0.0, 0.0),
            voxel_atlas,
            voxel_bind_group,
            voxel_density_texture,
            voxel_density_view,
            in_xr: false,
            canvas: canvas.clone(),
        };

        state.last_lod_update_pos = state.engine.camera.eye;
        state.update_lods(true);
        state.update_voxel_texture(); // Initial upload

        state
    }

    pub fn update_voxel_texture(&self) {
        let size = 128;
        let mut data = vec![0u8; size * size * size];

        // World Bounds: X: -64..64, Y: -32..96, Z: -64..64
        // Texture: 0..128
        // Mapping: tx = wx + 64, ty = wy + 32, tz = wz + 64

        // Iterate over active chunks and stamp them into the buffer
        for chunk in self.voxel_world.chunks.values() {
             let cx = chunk.key.x; // e.g. -2, -1, 0, 1
             let cy = chunk.key.y; // e.g. -1, 0, 1
             let cz = chunk.key.z; // e.g. -2, -1, 0, 1

             let base_x = cx * 32 + 64;
             let base_y = cy * 32 + 32;
             let base_z = cz * 32 + 64;

             if base_x < 0 || base_x >= 128 || base_y < 0 || base_y >= 128 || base_z < 0 || base_z >= 128 {
                 continue; // Out of texture bounds
             }

             for z in 0..32 {
                 for y in 0..32 {
                     for x in 0..32 {
                         let v = chunk.get(x, y, z);
                         if v.id != 0 {
                             let tx = base_x + x as i32;
                             let ty = base_y + y as i32;
                             let tz = base_z + z as i32;

                             if tx >= 0 && tx < 128 && ty >= 0 && ty < 128 && tz >= 0 && tz < 128 {
                                 let idx = tx as usize + (ty as usize) * 128 + (tz as usize) * 128 * 128;
                                 data[idx] = 1; // Solid
                             }
                         }
                     }
                 }
             }
        }

        let density_size = wgpu::Extent3d {
            width: 128,
            height: 128,
            depth_or_array_layers: 128,
        };

        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &self.voxel_density_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            &data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(128),
                rows_per_image: Some(128),
            },
            density_size,
        );
    }

    pub fn resize(&mut self, new_width: u32, new_height: u32) {
        if new_width > 0 && new_height > 0 {
            self.width = new_width;
            self.height = new_height;
            self.config.width = new_width;
            self.config.height = new_height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture = texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
            self.engine.resize(new_width, new_height);
        }
    }

    pub fn process_keyboard(&mut self, key_code: &str) {
        // Voxel Inputs via Config
        if let Some(action) = self.engine.input_config.map_key(key_code) {
             match action {
                 input::Action::VoxelDiffusion => {
                     self.voxel_world.update_dynamics();
                     self.voxel_world.save_all_states();
                     self.voxel_dirty = true;
                 },
                 input::Action::VoxelTimeReverse => {
                     self.voxel_world.revert_all_states();
                     self.voxel_dirty = true;
                 },
                 input::Action::VoxelDream => {
                     self.voxel_world.dream();
                     self.voxel_world.save_all_states();
                     self.voxel_dirty = true;
                 },
                 _ => {}
             }
        }

        self.engine.process_keyboard(key_code, true);
    }

    // Helper for keyup
    pub fn process_keyup(&mut self, key_code: &str) {
        self.engine.process_keyboard(key_code, false);
    }

    pub fn process_mouse_down(&mut self, x: f32, y: f32, button: i16) {
        self.engine.process_mouse_down(x, y, button);
    }

    pub fn process_mouse_move(&mut self, x: f32, y: f32) {
        self.engine.process_mouse_move(x, y);
    }

    pub fn process_mouse_up(&mut self) {
        self.engine.process_mouse_up();
    }

    pub fn process_click(&mut self, x: f32, y: f32) {
        if self.engine.process_click(x, y) {
             // State changed, save
             let lambda_source = if let Some(term) = &self.engine.lambda_system.root_term {
                 term.to_string()
             } else {
                 "".to_string()
             };

             let game_state = persistence::GameState {
                player: persistence::PlayerState {
                    projector: self.engine.player_projector.clone(),
                },
                world: self.engine.world_state.clone(),
                lambda_source,
                lambda_layout: self.engine.lambda_system.get_layout(),
                input_config: self.engine.input_config.clone(),
                voxel_world: Some(self.voxel_world.clone()),
                timestamp: js_sys::Date::now() as u64,
                version: persistence::SAVE_VERSION,
            };
            persistence::save_to_local_storage("reality_engine_save", &game_state);
        }
    }

    pub fn update_lods(&mut self, force: bool) {
        let cam_pos = self.engine.camera.eye;

        let mut to_update = Vec::new();
        let mut valid_keys = Vec::new();

        for (key, chunk) in &self.voxel_world.chunks {
            valid_keys.push(*key);

            let chunk_world_x = chunk.key.x as f32 * voxel::CHUNK_SIZE as f32 + (voxel::CHUNK_SIZE as f32 / 2.0);
            let chunk_world_y = chunk.key.y as f32 * voxel::CHUNK_SIZE as f32 + (voxel::CHUNK_SIZE as f32 / 2.0);
            let chunk_world_z = chunk.key.z as f32 * voxel::CHUNK_SIZE as f32 + (voxel::CHUNK_SIZE as f32 / 2.0);

            let dist_sq = (cam_pos.x - chunk_world_x).powi(2) +
                          (cam_pos.y - chunk_world_y).powi(2) +
                          (cam_pos.z - chunk_world_z).powi(2);
            let dist = dist_sq.sqrt();

            let desired_lod = if dist > 128.0 {
                4
            } else if dist > 64.0 {
                2
            } else {
                1
            };

            let needs_update = force || match self.voxel_meshes.get(key) {
                Some(mesh) => mesh.lod_level != desired_lod,
                None => true,
            };

            if needs_update {
                to_update.push((*key, desired_lod));
            }
        }

        // Remove meshes for chunks that no longer exist
        self.voxel_meshes.retain(|k, _| valid_keys.contains(k));

        for (key, lod) in to_update {
            if let Some(chunk) = self.voxel_world.chunks.get(&key) {
                let mesh_chunk = chunk.create_lod(lod);
                let (v, i) = mesh_chunk.generate_mesh();

                if !i.is_empty() {
                    let v_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Voxel Chunk Vertex"),
                        contents: bytemuck::cast_slice(&v),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    let i_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Voxel Chunk Index"),
                        contents: bytemuck::cast_slice(&i),
                        usage: wgpu::BufferUsages::INDEX,
                    });

                    self.voxel_meshes.insert(key, ChunkMesh {
                        vertex_buffer: v_buf,
                        index_buffer: i_buf,
                        index_count: i.len() as u32,
                        lod_level: lod,
                    });
                } else {
                    self.voxel_meshes.remove(&key);
                }
            }
        }

        self.voxel_dirty = false;
        self.last_lod_update_pos = cam_pos;
    }

    pub fn update(&mut self) {
        self.engine.update(0.016); // Assuming ~60fps fixed step in update

        // Rebuild meshes if dirty or if camera moved significantly (LOD update trigger)
        let cam_pos = self.engine.camera.eye;
        let dist_sq = (cam_pos.x - self.last_lod_update_pos.x).powi(2) +
                      (cam_pos.y - self.last_lod_update_pos.y).powi(2) +
                      (cam_pos.z - self.last_lod_update_pos.z).powi(2);

        if self.voxel_dirty || dist_sq > 256.0 {
            self.update_lods(self.voxel_dirty);
            if self.voxel_dirty {
                self.update_voxel_texture();
            }
        }

        // Sync WGPU buffers with Engine State
        self.camera_uniform.update_view_proj(&self.engine.camera);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        // Helper to map archetype to ID and Color
        fn get_archetype_data(archetype: reality_types::RealityArchetype) -> (f32, [f32; 4]) {
             match archetype {
                reality_types::RealityArchetype::Void => ( -1.0, [0.0, 0.0, 0.0, 0.0] ),
                reality_types::RealityArchetype::Fantasy => ( 0.0, [0.0, 1.0, 0.0, 1.0] ),
                reality_types::RealityArchetype::SciFi => ( 1.0, [0.0, 0.0, 1.0, 1.0] ),
                reality_types::RealityArchetype::Horror => ( 2.0, [1.0, 0.0, 0.0, 1.0] ),
                reality_types::RealityArchetype::Toon => ( 3.0, [1.0, 1.0, 0.0, 1.0] ),
                reality_types::RealityArchetype::HyperNature => ( 4.0, [0.2, 0.8, 0.2, 1.0] ),
                reality_types::RealityArchetype::Genie => ( 5.0, [1.0, 0.8, 0.0, 1.0] ), // Gold
            }
        }

        let p1 = &self.engine.player_projector;
        let (id1, color1) = get_archetype_data(p1.reality_signature.active_style.archetype);
        self.reality_uniform.proj1_pos_fid = [p1.location.x, p1.location.y, p1.location.z, p1.reality_signature.fidelity];
        self.reality_uniform.proj1_params = [
            p1.reality_signature.active_style.roughness,
            p1.reality_signature.active_style.scale,
            p1.reality_signature.active_style.distortion,
            id1
        ];
        self.reality_uniform.proj1_color = color1;

        let p2 = if let Some(ref anomaly) = self.engine.active_anomaly {
             anomaly
        } else {
             &self.engine.player_projector
        };

        let (id2, color2) = get_archetype_data(p2.reality_signature.active_style.archetype);
        self.reality_uniform.proj2_pos_fid = [p2.location.x, p2.location.y, p2.location.z, p2.reality_signature.fidelity];
        self.reality_uniform.proj2_params = [
            p2.reality_signature.active_style.roughness,
            p2.reality_signature.active_style.scale,
            p2.reality_signature.active_style.distortion,
            id2
        ];
        self.reality_uniform.proj2_color = color2;
        self.reality_uniform.global_offset = self.engine.global_offset;
        self.reality_uniform.global_offset[2] = self.engine.time;

        self.queue.write_buffer(
            &self.reality_buffer,
            0,
            bytemuck::cast_slice(&[self.reality_uniform]),
        );

        // Update Frustum Culling & Instances
        let view_proj = self.engine.camera.build_view_projection_matrix();

        let mut visible_instances = Vec::new();
        for chunk_data in &self.chunk_data {
            if is_aabb_visible(chunk_data.aabb_min, chunk_data.aabb_max, &view_proj) {
                let model = cgmath::Matrix4::from_translation(chunk_data.position);

                // Get Stability from World State
                let chunk_size = 10.0;
                let id = world::ChunkId::from_world_pos(chunk_data.position.x, chunk_data.position.z, chunk_size);
                let stability = if let Some(chunk) = self.engine.world_state.chunks.get(&id) {
                    chunk.stability
                } else {
                    1.0
                };

                visible_instances.push(Instance {
                    model: model.into(),
                    stability,
                    padding: [0.0; 3],
                });
            }
        }

        self.num_instances = visible_instances.len() as u32;
        if self.num_instances > 0 {
            self.queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&visible_instances),
            );
        }

        // Update Lambda System buffers
        self.lambda_renderer.update_buffers(
            &self.device,
            &self.queue,
            &self.engine.lambda_system.nodes,
            &self.engine.lambda_system.edges,
            self.engine.lambda_system.hovered_node,
            self.engine.lambda_system.hovered_edge
        );
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
            let clear_color = if self.in_xr {
                wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }
            } else {
                wgpu::Color { r: 0.1, g: 0.1, b: 0.2, a: 1.0 }
            };

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            // Draw Terrain
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(2, &self.reality_bind_group, &[]);

            if self.num_instances > 0 {
                render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
                render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..self.num_indices, 0, 0..self.num_instances);
            }

            // Draw Voxels
            render_pass.set_pipeline(&self.voxel_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(1, &self.voxel_bind_group, &[]);
            render_pass.set_bind_group(2, &self.reality_bind_group, &[]);

            for mesh in self.voxel_meshes.values() {
                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }

            if !self.engine.lambda_system.nodes.is_empty() {
                 self.lambda_renderer.render(&mut render_pass, &self.camera_bind_group, self.engine.lambda_system.nodes.len() as u32);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

#[cfg(target_arch = "wasm32")]
fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    web_sys::window()
        .unwrap()
        .request_animation_frame(f.as_ref().unchecked_ref())
        .expect("should register `requestAnimationFrame` OK");
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct GameClient {
    state: Rc<RefCell<State>>,
    network: Option<Rc<RefCell<network::NetworkManager>>>,
    current_save_slot: Rc<RefCell<String>>,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl GameClient {
    pub fn get_node_labels(&self) -> String {
        let state = self.state.borrow();
        let labels = state.engine.get_node_labels();
        serde_json::to_string(&labels).unwrap_or_else(|_| "[]".to_string())
    }

    pub fn set_anomaly_params(&self, roughness: f32, scale: f32, distortion: f32) {
        let mut state = self.state.borrow_mut();
        if let Some(ref mut anomaly) = state.engine.active_anomaly {
            anomaly.reality_signature.active_style.roughness = roughness;
            anomaly.reality_signature.active_style.scale = scale;
            anomaly.reality_signature.active_style.distortion = distortion;
        }
        self.save_state(&state);
    }

    pub fn set_world_origin(&self, lat: f32, lon: f32) {
        let mut state = self.state.borrow_mut();
        let scale = 111000.0;
        let x = lon * scale * 0.1;
        let z = lat * scale * 0.1;

        state.engine.set_global_offset(x, z);
    }

    pub fn set_anomaly_archetype(&self, archetype_id: i32) {
        let mut state = self.state.borrow_mut();
        let archetype = match archetype_id {
            0 => reality_types::RealityArchetype::Fantasy,
            1 => reality_types::RealityArchetype::SciFi,
            2 => reality_types::RealityArchetype::Horror,
            3 => reality_types::RealityArchetype::Toon,
            4 => reality_types::RealityArchetype::HyperNature,
            5 => reality_types::RealityArchetype::Genie,
            _ => reality_types::RealityArchetype::Void,
        };
        if let Some(ref mut anomaly) = state.engine.active_anomaly {
            anomaly.reality_signature.active_style.archetype = archetype;
        }
        self.save_state(&state);
    }

    pub fn get_key_binding(&self, action_name: String) -> String {
        let state = self.state.borrow();
        if let Some(action) = input::Action::from_string(&action_name) {
             if let Some(key) = state.engine.input_config.get_binding(action) {
                 return key.clone();
             }
        }
        "".to_string()
    }

    pub fn set_key_binding(&self, action_name: String, key_code: String) {
        let mut state = self.state.borrow_mut();
        if let Some(action) = input::Action::from_string(&action_name) {
             state.engine.input_config.set_binding(action, key_code);
             self.save_state(&state);
        }
    }

    pub fn get_network_status(&self) -> String {
        if let Some(network_manager) = &self.network {
             let status = network_manager.borrow().get_status();
             serde_json::to_string(&status).unwrap_or_else(|_| "{}".to_string())
        } else {
             "{}".to_string()
        }
    }

    pub fn save_game(&self, slot_name: String) {
        *self.current_save_slot.borrow_mut() = slot_name;
        let state = self.state.borrow();
        self.save_state(&state);
    }

    pub fn load_game(&self, slot_name: String) {
        *self.current_save_slot.borrow_mut() = slot_name.clone();
        let key = persistence::get_save_key(&slot_name);
        if let Some(loaded_state) = persistence::load_from_local_storage(&key) {
            let mut state = self.state.borrow_mut();

            state.engine.world_state = loaded_state.world;
            state.engine.player_projector = loaded_state.player.projector;
            state.engine.camera.eye = state.engine.player_projector.location;

            use cgmath::Point3;
            let mut anomaly_sig = reality_types::RealitySignature::default();
            anomaly_sig.active_style.archetype = reality_types::RealityArchetype::SciFi;
            anomaly_sig.active_style.roughness = 0.8;
            anomaly_sig.active_style.scale = 5.0;
            anomaly_sig.active_style.distortion = 0.8;
            anomaly_sig.fidelity = 100.0;
            state.engine.active_anomaly = Some(projector::RealityProjector::new(
                Point3::new(0.0, 0.0, 0.0),
                anomaly_sig
            ));

            state.engine.lambda_system = visual_lambda::LambdaSystem::new();
            state.engine.input_config = loaded_state.input_config;

            // Restore Voxel World
            if let Some(voxel_world) = loaded_state.voxel_world {
                state.voxel_world = voxel_world;
                state.voxel_dirty = true;
                state.update_voxel_texture();
                state.update_lods(true);
            }

            // Restore Lambda State
            let source = if loaded_state.lambda_source.is_empty() { "FIRE".to_string() } else { loaded_state.lambda_source };
            if let Some(term) = lambda::parse(&source) {
                state.engine.lambda_system.set_term(term);
                if !loaded_state.lambda_layout.is_empty() {
                    state.engine.lambda_system.apply_layout(loaded_state.lambda_layout);
                }
            } else {
                 // Fallback
                 let term = lambda::parse("FIRE").unwrap();
                 state.engine.lambda_system.set_term(term);
            }

            log::info!("Game Loaded from slot: {}", slot_name);
        } else {
            log::warn!("Failed to load save slot: {}", slot_name);
        }
    }

    pub fn list_saves(&self) -> String {
        let saves = persistence::list_saves();
        serde_json::to_string(&saves).unwrap_or_else(|_| "[]".to_string())
    }

    pub fn delete_save(&self, slot_name: String) {
        persistence::delete_save(&slot_name);
    }

    pub fn reset_world(&self) {
        let mut state = self.state.borrow_mut();
        state.engine.reset();
        self.save_state(&state);
    }

    fn save_state(&self, state: &State) {
        let lambda_source = if let Some(term) = &state.engine.lambda_system.root_term {
            term.to_string()
        } else {
            "".to_string()
        };

        let game_state = persistence::GameState {
            player: persistence::PlayerState {
                projector: state.engine.player_projector.clone(),
            },
            world: state.engine.world_state.clone(),
            lambda_source,
            lambda_layout: state.engine.lambda_system.get_layout(),
            input_config: state.engine.input_config.clone(),
            voxel_world: Some(state.voxel_world.clone()),
            timestamp: js_sys::Date::now() as u64,
            version: persistence::SAVE_VERSION,
        };
        let slot = self.current_save_slot.borrow().clone();
        let key = persistence::get_save_key(&slot);
        persistence::save_to_local_storage(&key, &game_state);
    }

    pub async fn enter_ar_session(&self) -> Result<(), JsValue> {
        if !xr::is_ar_supported().await? {
             return Err(JsValue::from_str("AR not supported"));
        }

        let session = xr::request_ar_session().await?;

        // 1. Get State & Canvas (Brief Borrow)
        let canvas = {
            let mut state = self.state.borrow_mut();
            state.in_xr = true;
            state.canvas.clone()
        };

        // 2. Get Context (No Borrow)
        let gl_context = canvas.get_context("webgl2")?
            .ok_or_else(|| JsValue::from_str("No WebGL2 context found"))?
            .dyn_into::<web_sys::WebGl2RenderingContext>()?;

        // 3. Create Layer (No Borrow)
        let layer = XrWebGlLayer::new_with_web_gl2_rendering_context(&session, &gl_context)?;

        let mut render_state_init = XrRenderStateInit::new();
        render_state_init.set_base_layer(Some(&layer));
        session.update_render_state_with_state(&render_state_init);

        // 4. Request Reference Space (Async - Must not hold borrow)
        let reference_space = wasm_bindgen_futures::JsFuture::from(
            session.request_reference_space(XrReferenceSpaceType::Local)
        ).await?.unchecked_into::<XrReferenceSpace>();

        let state_xr = self.state.clone();
        let session_clone = session.clone();
        let reference_space_clone = reference_space.clone();

        let f_xr: Rc<RefCell<Option<Closure<dyn FnMut(f64, XrFrame)>>>> = Rc::new(RefCell::new(None));
        let g_xr = f_xr.clone();

        *g_xr.borrow_mut() = Some(Closure::new(move |_time: f64, frame: XrFrame| {
            let mut state = state_xr.borrow_mut();

            let pose = frame.get_viewer_pose(&reference_space_clone);
            if let Some(pose) = pose {
                 let views = pose.views();
                 if views.length() > 0 {
                      let view = views.get(0).unchecked_into::<XrView>();

                      // Tracking
                      let transform = view.transform();
                      let pos = transform.position();
                      let orient = transform.orientation();

                      let q = cgmath::Quaternion::new(orient.w() as f32, orient.x() as f32, orient.y() as f32, orient.z() as f32);
                      let forward = q.rotate_vector(cgmath::Vector3::new(0.0, 0.0, -1.0));
                      let up = q.rotate_vector(cgmath::Vector3::new(0.0, 1.0, 0.0));

                      state.engine.camera.eye = cgmath::Point3::new(pos.x() as f32, pos.y() as f32, pos.z() as f32);
                      state.engine.camera.target = state.engine.camera.eye + forward;
                      state.engine.camera.up = up;

                      // Projection
                      let proj_data = view.projection_matrix();

                      if proj_data.len() >= 16 {
                          state.engine.camera.projection_override = Some(cgmath::Matrix4::from_cols(
                              cgmath::Vector4::new(proj_data[0], proj_data[1], proj_data[2], proj_data[3]),
                              cgmath::Vector4::new(proj_data[4], proj_data[5], proj_data[6], proj_data[7]),
                              cgmath::Vector4::new(proj_data[8], proj_data[9], proj_data[10], proj_data[11]),
                              cgmath::Vector4::new(proj_data[12], proj_data[13], proj_data[14], proj_data[15]),
                          ));
                      }

                      if let Some(viewport) = layer.get_viewport(&view) {
                           let vp_width = viewport.width() as u32;
                           let vp_height = viewport.height() as u32;
                           if vp_width != state.width || vp_height != state.height {
                                state.resize(vp_width, vp_height);
                                state.canvas.set_width(vp_width);
                                state.canvas.set_height(vp_height);
                           }
                      }
                 }
            }

            state.update();

            // Render to Canvas (wgpu Surface)
            let _ = state.render();

            // Blit Canvas to XR Framebuffer
            let width = state.width as i32;
            let height = state.height as i32;
            let framebuffer = layer.framebuffer();

            gl_context.bind_framebuffer(web_sys::WebGl2RenderingContext::READ_FRAMEBUFFER, None);
            gl_context.bind_framebuffer(web_sys::WebGl2RenderingContext::DRAW_FRAMEBUFFER, framebuffer.as_ref());

            gl_context.blit_framebuffer(
                0, 0, width, height,
                0, 0, width, height,
                web_sys::WebGl2RenderingContext::COLOR_BUFFER_BIT,
                web_sys::WebGl2RenderingContext::NEAREST
            );

            session_clone.request_animation_frame(f_xr.borrow().as_ref().unwrap().as_ref().unchecked_ref());
        }));

        session.request_animation_frame(g_xr.borrow().as_ref().unwrap().as_ref().unchecked_ref());

        let state_end = self.state.clone();
        let onend_closure = Closure::wrap(Box::new(move |_| {
            let mut state = state_end.borrow_mut();
            state.in_xr = false;
            log::info!("XR Session Ended");
        }) as Box<dyn FnMut(JsValue)>);
        session.set_onend(Some(onend_closure.as_ref().unchecked_ref()));
        onend_closure.forget();

        Ok(())
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn start(canvas_id: String) -> Result<GameClient, JsValue> {
    std::panic::set_hook(Box::new(|info| {
        web_sys::console::error_1(&wasm_bindgen::JsValue::from_str(&format!("Panic: {}", info)));
    }));
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

    let current_save_slot = Rc::new(RefCell::new("default".to_string()));

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
        let code = event.code();

        // Check for Inscribe Key
        let is_inscribe = {
             let state = state_keydown.borrow();
             state.engine.input_config.get_binding(input::Action::Inscribe) == Some(&code)
        };

        if is_inscribe {
             // Pause/Unlock pointer if needed?
             // web_sys::window().unwrap().document().unwrap().exit_pointer_lock();

             if let Ok(Some(text)) = web_sys::window().unwrap().prompt_with_message("Inscribe Reality (e.g. 'FIRE', 'GROWTH TREE'):") {
                 if !text.is_empty() {
                    state_keydown.borrow_mut().engine.process_inscription(&text);
                 }
             }
             return;
        }

        state_keydown
            .borrow_mut()
            .process_keyboard(&code);
    }) as Box<dyn FnMut(_)>);

    window
        .add_event_listener_with_callback("keydown", keydown_closure.as_ref().unchecked_ref())
        .expect("Failed to add keydown listener");
    keydown_closure.forget();

    let state_keyup = state.clone();
    let keyup_closure = Closure::wrap(Box::new(move |event: web_sys::KeyboardEvent| {
        state_keyup
            .borrow_mut()
            .process_keyup(&event.code());
    }) as Box<dyn FnMut(_)>);

    window
        .add_event_listener_with_callback("keyup", keyup_closure.as_ref().unchecked_ref())
        .expect("Failed to add keyup listener");
    keyup_closure.forget();

    // Mouse Handler
    let state_mouse = state.clone();
    let canvas_mouse = canvas.clone();

    // MouseState tracking for click detection
    struct MouseState {
        down_pos: Option<(f32, f32)>,
        down_time: f64,
    }
    let mouse_state = Rc::new(RefCell::new(MouseState { down_pos: None, down_time: 0.0 }));

    let state_down = state.clone();
    let canvas_down = canvas.clone();
    let mouse_state_down = mouse_state.clone();
    let mousedown_closure = Closure::wrap(Box::new(move |event: web_sys::MouseEvent| {
        let document = web_sys::window().unwrap().document().unwrap();
        let locked = document.pointer_lock_element().is_some();

        if !locked {
             canvas_down.request_pointer_lock();
        }

        let (ndc_x, ndc_y) = if locked {
            (0.0, 0.0)
        } else {
            let rect = canvas_down.get_bounding_client_rect();
            let x = event.client_x() as f32 - rect.left() as f32;
            let y = event.client_y() as f32 - rect.top() as f32;
            let width = rect.width() as f32;
            let height = rect.height() as f32;
            ((x / width) * 2.0 - 1.0, -((y / height) * 2.0 - 1.0))
        };

        let button = event.button();
        state_down.borrow_mut().process_mouse_down(ndc_x, ndc_y, button);

        mouse_state_down.borrow_mut().down_pos = Some((ndc_x, ndc_y));
        mouse_state_down.borrow_mut().down_time = js_sys::Date::now();
    }) as Box<dyn FnMut(_)>);

    canvas.add_event_listener_with_callback("mousedown", mousedown_closure.as_ref().unchecked_ref()).unwrap();
    mousedown_closure.forget();

    let state_move = state.clone();
    let canvas_move = canvas.clone();
    let mousemove_closure = Closure::wrap(Box::new(move |event: web_sys::MouseEvent| {
        let document = web_sys::window().unwrap().document().unwrap();
        let locked = document.pointer_lock_element().is_some();

        if locked {
            state_move.borrow_mut().engine.process_mouse_look(
                event.movement_x() as f32,
                event.movement_y() as f32
            );
            // Also update drag with center ray to allow carrying items
            state_move.borrow_mut().process_mouse_move(0.0, 0.0);
        } else {
            let rect = canvas_move.get_bounding_client_rect();
            let x = event.client_x() as f32 - rect.left() as f32;
            let y = event.client_y() as f32 - rect.top() as f32;
            let width = rect.width() as f32;
            let height = rect.height() as f32;
            let ndc_x = (x / width) * 2.0 - 1.0;
            let ndc_y = -((y / height) * 2.0 - 1.0);

            state_move.borrow_mut().process_mouse_move(ndc_x, ndc_y);
        }
    }) as Box<dyn FnMut(_)>);

    // Mousemove on window
    window.add_event_listener_with_callback("mousemove", mousemove_closure.as_ref().unchecked_ref()).unwrap();
    mousemove_closure.forget();

    let state_up = state.clone();
    let canvas_up = canvas.clone();
    let mouse_state_up = mouse_state.clone();
    let mouseup_closure = Closure::wrap(Box::new(move |event: web_sys::MouseEvent| {
        let rect = canvas_up.get_bounding_client_rect();
        let x = event.client_x() as f32 - rect.left() as f32;
        let y = event.client_y() as f32 - rect.top() as f32;
        let width = rect.width() as f32;
        let height = rect.height() as f32;
        let ndc_x = (x / width) * 2.0 - 1.0;
        let ndc_y = -((y / height) * 2.0 - 1.0);

        state_up.borrow_mut().process_mouse_up();

        // Check for click
        let mut ms = mouse_state_up.borrow_mut();
        if let Some((sx, sy)) = ms.down_pos {
             let dist = ((ndc_x - sx).powi(2) + (ndc_y - sy).powi(2)).sqrt();
             let elapsed = js_sys::Date::now() - ms.down_time;

             if dist < 0.05 && elapsed < 300.0 {
                 state_up.borrow_mut().process_click(ndc_x, ndc_y);
             }
        }
        ms.down_pos = None;
    }) as Box<dyn FnMut(_)>);

    window.add_event_listener_with_callback("mouseup", mouseup_closure.as_ref().unchecked_ref()).unwrap();
    mouseup_closure.forget();

    // Device Orientation Handler
    let state_orient = state.clone();
    let orient_closure = Closure::wrap(Box::new(move |event: web_sys::DeviceOrientationEvent| {
        if let (Some(alpha), Some(beta), Some(_gamma)) = (event.alpha(), event.beta(), event.gamma()) {
            let yaw = -alpha.to_radians() as f32;
            let pitch = (beta - 90.0).to_radians() as f32;

            let mut state = state_orient.borrow_mut();
            state.engine.camera.set_rotation(yaw, pitch);
        }
    }) as Box<dyn FnMut(_)>);

    window
        .add_event_listener_with_callback("deviceorientation", orient_closure.as_ref().unchecked_ref())
        .expect("Failed to add deviceorientation listener");
    orient_closure.forget();

    // Geolocation Handler
    let state_geo = state.clone();
    if let Ok(navigator) = window.navigator().dyn_into::<web_sys::Navigator>() {
        if let Ok(geolocation) = navigator.geolocation() {
            let success_callback = Closure::wrap(Box::new(move |position: web_sys::Position| {
                 let coords = position.coords();
                 let lat = coords.latitude() as f32;
                 let lon = coords.longitude() as f32;

                 let scale = 111000.0;
                 let x = lon * scale * 0.1;
                 let z = lat * scale * 0.1;

                 state_geo.borrow_mut().engine.set_global_offset(x, z);
                 log::info!("Geolocation Acquired: {}, {}", lat, lon);
            }) as Box<dyn FnMut(_)>);

            let error_callback = Closure::wrap(Box::new(move |_error: web_sys::PositionError| {
                log::warn!("Geolocation failed or denied.");
            }) as Box<dyn FnMut(_)>);

            geolocation.get_current_position_with_error_callback(
                success_callback.as_ref().unchecked_ref(),
                Some(error_callback.as_ref().unchecked_ref())
            ).ok();

            success_callback.forget();
            error_callback.forget();
        }
    }

    // Render loop
    let f = Rc::new(RefCell::new(None));
    let g = f.clone();

    let state_copy = state.clone();
    *g.borrow_mut() = Some(Closure::new(move || {
        let mut state = state_copy.borrow_mut();
        if !state.in_xr {
            state.update();
            state.render().expect("Render failed");
        }
        request_animation_frame(f.borrow().as_ref().unwrap());
    }));

    request_animation_frame(g.borrow().as_ref().unwrap());

    // Initialize Network
    let network_manager = match network::NetworkManager::new("ws://localhost:9000/ws") {
        Ok(m) => {
            // Setup Sync Callback
            let state_sync = state.clone();
            m.borrow_mut().set_sync_callback(move |msg| {
                match msg {
                    network::SyncMessage::WorldUpdate(remote_world) => {
                        let mut state = state_sync.borrow_mut();
                        if state.engine.world_state.merge(remote_world) {
                            log::info!("Anomaly Conflict Resolved! World merged.");
                        }
                    }
                    network::SyncMessage::ChunkUpdate(chunks) => {
                        let mut state = state_sync.borrow_mut();
                        if state.engine.world_state.merge_chunks(chunks) {
                            log::info!("Chunk Update Merged.");
                        }
                    }
                    network::SyncMessage::RequestSync => {
                        let mut state = state_sync.borrow_mut();
                        state.engine.pending_full_sync = true;
                        log::info!("Received Sync Request. Scheduled Full Broadcast.");
                    }
                }
            });
            Some(m)
        },
        Err(e) => {
            log::warn!("Failed to connect to signaling server: {:?}", e);
            None
        }
    };

    // Autosave loop (every 5 seconds)
    let state_autosave = state.clone();
    let network_autosave = network_manager.clone();
    let slot_autosave = current_save_slot.clone();

    let autosave_closure = Closure::wrap(Box::new(move || {
        let mut state = state_autosave.borrow_mut();
        let lambda_source = if let Some(term) = &state.engine.lambda_system.root_term {
             term.to_string()
        } else {
             "".to_string()
        };

        let game_state = persistence::GameState {
            player: persistence::PlayerState {
                projector: state.engine.player_projector.clone(),
            },
            world: state.engine.world_state.clone(),
            lambda_source,
            lambda_layout: state.engine.lambda_system.get_layout(),
            input_config: state.engine.input_config.clone(),
            voxel_world: Some(state.voxel_world.clone()),
            timestamp: js_sys::Date::now() as u64,
            version: persistence::SAVE_VERSION,
        };
        let slot = slot_autosave.borrow().clone();
        let key = persistence::get_save_key(&slot);
        persistence::save_to_local_storage(&key, &game_state);

        // Pollinate (Broadcast Presence)
        if let Some(net) = &network_autosave {
             let loc = state.engine.player_projector.location;
             let net = net.borrow();

             net.pollinate(&state.engine.world_state.root_hash, [loc.x, loc.y, loc.z]);

             // Check pending full sync
             if state.engine.pending_full_sync {
                 net.broadcast_world_state(&state.engine.world_state);
                 state.engine.pending_full_sync = false;
             }
             // Check if world empty (Request Sync)
             else if state.engine.world_state.chunks.is_empty() {
                 net.broadcast_request_sync();
             }
             // Check dirty chunks (Delta Sync)
             else {
                 let dirty = state.engine.world_state.extract_dirty_chunks();
                 if !dirty.is_empty() {
                     net.broadcast_chunk_update(dirty);
                 }
             }
        }
    }) as Box<dyn FnMut()>);

    window.set_interval_with_callback_and_timeout_and_arguments_0(
        autosave_closure.as_ref().unchecked_ref(),
        5000, // 5 seconds
    ).expect("Failed to set autosave interval");
    autosave_closure.forget();

    Ok(GameClient { state, network: network_manager, current_save_slot })
}
