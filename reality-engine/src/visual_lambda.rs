use cgmath::{Point3, Vector3, InnerSpace, MetricSpace, EuclideanSpace};
use std::rc::Rc;
use std::collections::HashMap;
use crate::lambda::{Term, Primitive};
use wgpu::util::DeviceExt;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LambdaEvent {
    ReductionStarted,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LambdaVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

impl LambdaVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<LambdaVertex>() as wgpu::BufferAddress,
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
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LambdaInstance {
    pub position: [f32; 3],
    pub color: [f32; 4],
    pub scale: f32,
}

impl LambdaInstance {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<LambdaInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 7]>() as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LambdaLineVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

impl LambdaLineVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<LambdaLineVertex>() as wgpu::BufferAddress,
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
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

pub struct LambdaRenderer {
    pipeline: wgpu::RenderPipeline,
    transparent_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    instance_buffer: wgpu::Buffer,
    capacity: usize,
    opaque_count: u32,

    // Line Rendering
    line_pipeline: wgpu::RenderPipeline,
    line_vertex_buffer: wgpu::Buffer,
    line_vertex_count: u32,
    line_capacity: usize,
}

impl LambdaRenderer {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, camera_bind_group_layout: &wgpu::BindGroupLayout) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Lambda Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shader_lambda.wgsl"))),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Lambda Pipeline Layout"),
            bind_group_layouts: &[camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        let depth_stencil_state = Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Lambda Opaque Pipeline"),
            layout: Some(&pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[LambdaVertex::desc(), LambdaInstance::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
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
            depth_stencil: depth_stencil_state.clone(),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let transparent_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Lambda Transparent Pipeline"),
            layout: Some(&pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[LambdaVertex::desc(), LambdaInstance::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // Line Pipeline
        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Lambda Line Pipeline"),
            layout: Some(&pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_line"),
                buffers: &[LambdaLineVertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_line"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // Create Sphere Mesh
        let (vertices, indices) = create_sphere_mesh(1.0, 16, 16);

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Lambda Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Lambda Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Initial empty instance buffer (size for 100 nodes for now)
        let capacity = 100;
        let instances = vec![LambdaInstance { position: [0.0; 3], color: [0.0; 4], scale: 0.0 }; capacity];
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Lambda Instance Buffer"),
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        // Initial Line Buffer
        let line_capacity = 200; // 100 edges = 200 vertices
        let line_vertices = vec![LambdaLineVertex { position: [0.0; 3], color: [0.0; 4] }; line_capacity];
        let line_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Lambda Line Vertex Buffer"),
            contents: bytemuck::cast_slice(&line_vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            pipeline,
            transparent_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices: indices.len() as u32,
            instance_buffer,
            capacity,
            opaque_count: 0,
            line_pipeline,
            line_vertex_buffer,
            line_vertex_count: 0,
            line_capacity,
        }
    }

    pub fn update_buffers(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, nodes: &[VisualNode], edges: &[(usize, usize)], hovered_node: Option<usize>) {
        // Update Instances
        if nodes.len() > self.capacity {
            self.capacity = (nodes.len() * 2).max(self.capacity * 2);
            let instances = vec![LambdaInstance { position: [0.0; 3], color: [0.0; 4], scale: 0.0 }; self.capacity];
            self.instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Lambda Instance Buffer Resized"),
                contents: bytemuck::cast_slice(&instances),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
        }

        // Helper: Collect subtree indices if hovered
        let mut highlighted_nodes = std::collections::HashSet::new();
        if let Some(hover_idx) = hovered_node {
            highlighted_nodes.insert(hover_idx);

            // BFS to highlight descendants
            let mut stack = vec![hover_idx];
            while let Some(parent) = stack.pop() {
                for &(p, c) in edges {
                    if p == parent {
                        highlighted_nodes.insert(c);
                        stack.push(c);
                    }
                }
            }

            // Also highlight ports pointing to this binder (if it's an Abs)
            // (Edges for ports are (Port, Binder), but our BFS follows Structure edges (Parent, Child))
            // Wait, edges are mixed.
            // Structure: (Parent, Child)
            // Wire: (Port, Binder)
            // We need to differentiate to correctly highlight subtree vs usage.
            // But `edges` is just a list of (usize, usize).
            // `build_subtree` pushes (Port, Binder) for wires.
            // If we traverse (Port, Binder), then if we hover Port, we highlight Binder?
            // If we hover Binder, we don't necessarily highlight Ports (backward edge).
            // But we want to see USAGE.

            // Let's iterate all edges to find usage
            if matches!(nodes[hover_idx].node_type, NodeType::Abs(_)) {
                for &(src, dst) in edges {
                    if dst == hover_idx && matches!(nodes[src].node_type, NodeType::Port) {
                        highlighted_nodes.insert(src);
                    }
                }
            }
        }

        // Sort Nodes: Opaque first, then Transparent
        let mut instances: Vec<LambdaInstance> = nodes.iter().enumerate().map(|(i, n)| {
            let mut color = n.color;

            // Apply Highlight
            if let Some(hover_idx) = hovered_node {
                if i == hover_idx {
                    // Direct Hover: Brighten significantly
                    color[0] = (color[0] + 0.5).min(1.0);
                    color[1] = (color[1] + 0.5).min(1.0);
                    color[2] = (color[2] + 0.5).min(1.0);
                    color[3] = (color[3] + 0.2).min(1.0);
                } else if highlighted_nodes.contains(&i) {
                    // Subtree/Related: Slight brighten
                    color[0] = (color[0] + 0.2).min(1.0);
                    color[1] = (color[1] + 0.2).min(1.0);
                    color[2] = (color[2] + 0.2).min(1.0);
                } else {
                    // Unrelated: Dim slightly
                    color[0] *= 0.5;
                    color[1] *= 0.5;
                    color[2] *= 0.5;
                    color[3] *= 0.8; // Make bubbles ghostlier
                }
            }

            LambdaInstance {
                position: [n.position.x, n.position.y, n.position.z],
                color,
                scale: n.scale,
            }
        }).collect();

        // Sort: Opaque (Alpha >= 0.95) first
        instances.sort_by(|a, b| {
            let a_opaque = a.color[3] >= 0.95;
            let b_opaque = b.color[3] >= 0.95;
            b_opaque.cmp(&a_opaque) // True > False, so Opaque first
        });

        self.opaque_count = instances.iter().take_while(|i| i.color[3] >= 0.95).count() as u32;

        if !instances.is_empty() {
             queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&instances));
        }

        // Update Lines
        // We use curves now, so we need more vertices per edge (e.g. 10 segments = 20 vertices)
        let segments_per_edge = 10;
        let required_line_vertices = edges.len() * segments_per_edge * 2;

        if required_line_vertices > self.line_capacity {
            self.line_capacity = (required_line_vertices * 2).max(self.line_capacity * 2);
             let line_vertices = vec![LambdaLineVertex { position: [0.0; 3], color: [0.0; 4] }; self.line_capacity];
            self.line_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Lambda Line Vertex Buffer Resized"),
                contents: bytemuck::cast_slice(&line_vertices),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
        }

        let mut line_data = Vec::with_capacity(required_line_vertices);
        for &(start_idx, end_idx) in edges {
            if start_idx < nodes.len() && end_idx < nodes.len() {
                let start_node = &nodes[start_idx];
                let end_node = &nodes[end_idx];

                // Skip edges connected to invisible nodes (collapsed)
                if start_node.scale < 0.001 || end_node.scale < 0.001 {
                    continue;
                }

                let p0 = start_node.position;
                let p2 = end_node.position;

                // Color Logic
                let mut color = [1.0, 1.0, 1.0, 0.5];

                // Determine Control Point and Color based on type
                // Is this a Structure Edge or a Wire?
                // Wire connects Port -> Binder. Port is `start_node`.
                // Actually edges are (start, end). In build_subtree:
                // Structure: Abs -> Body (edges.push((node_index, child_idx)))
                // Structure: App -> Child (edges.push((node_index, left_idx)))
                // Wire: Port -> Binder (edges.push((node_index, binder_idx)))

                let is_wire = matches!(start_node.node_type, NodeType::Port);

                let p1 = if is_wire {
                    // Wires should curve "out".
                    // Since nodes are usually below binder, we curve sideways/up.
                    // Simple heuristic: midpoint + outward offset
                    // Let's use cross product with Up vector?
                    // Or just add Y offset.
                    color = [0.8, 0.8, 1.0, 0.8]; // Blueish wires

                    let mid = p0 + (p2 - p0) * 0.5;
                    // Curve away from center?
                    // Let's just curve "up" relative to the straight line to loop back.
                    // Port is usually below Binder.
                    // So we want curve to go OUT then UP.

                    // Simple hack: Curve perpendicular to the line in XZ plane
                    let dir = p2 - p0;
                    let right = Vector3::new(-dir.z, 0.0, dir.x).normalize();
                    // If dir is vertical, this fails.

                    if dir.x.abs() < 0.1 && dir.z.abs() < 0.1 {
                         mid + Vector3::new(2.0, 0.0, 0.0) // Side loop
                    } else {
                         mid + right * 2.0 // Curve side
                    }
                } else {
                    // Structure Edge
                    // Gentle S-curve or just straight?
                    // Let's keep it mostly straight but slightly curved for aesthetics
                    let mid = p0 + (p2 - p0) * 0.5;
                    mid
                };

                // Generate Curve Segments (Quadratic Bezier)
                // B(t) = (1-t)^2 P0 + 2(1-t)t P1 + t^2 P2
                let mut prev_pos = p0;

                for i in 1..=segments_per_edge {
                    let t = i as f32 / segments_per_edge as f32;
                    let it = 1.0 - t;

                    // Bezier using Vectors for math
                    let v0 = p0.to_vec();
                    let v1 = p1.to_vec();
                    let v2 = p2.to_vec();

                    let pos_vec = (v0 * (it * it)) + (v1 * (2.0 * it * t)) + (v2 * (t * t));
                    let pos = Point3::from_vec(pos_vec);

                    line_data.push(LambdaLineVertex {
                        position: [prev_pos.x, prev_pos.y, prev_pos.z],
                        color,
                    });
                    line_data.push(LambdaLineVertex {
                        position: [pos.x, pos.y, pos.z],
                        color,
                    });

                    prev_pos = pos;
                }
            }
        }

        self.line_vertex_count = line_data.len() as u32;

        if !line_data.is_empty() {
             queue.write_buffer(&self.line_vertex_buffer, 0, bytemuck::cast_slice(&line_data));
        }
    }

    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>, camera_bind_group: &'a wgpu::BindGroup, num_instances: u32) {
        // Draw Lines
        if self.line_vertex_count > 0 {
            render_pass.set_pipeline(&self.line_pipeline);
            render_pass.set_bind_group(0, camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.line_vertex_buffer.slice(..));
            render_pass.draw(0..self.line_vertex_count, 0..1);
        }

        // Draw Opaque Nodes
        if self.opaque_count > 0 {
            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..self.opaque_count);
        }

        // Draw Transparent Nodes
        if num_instances > self.opaque_count {
            render_pass.set_pipeline(&self.transparent_pipeline);
            render_pass.set_bind_group(0, camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..self.num_indices, 0, self.opaque_count..num_instances);
        }
    }
}

fn create_sphere_mesh(radius: f32, sectors: u32, stacks: u32) -> (Vec<LambdaVertex>, Vec<u16>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let sector_step = std::f32::consts::PI * 2.0 / sectors as f32;
    let stack_step = std::f32::consts::PI / stacks as f32;

    for i in 0..=stacks {
        let stack_angle = std::f32::consts::PI / 2.0 - i as f32 * stack_step;
        let xy = radius * stack_angle.cos();
        let z = radius * stack_angle.sin();

        for j in 0..=sectors {
            let sector_angle = j as f32 * sector_step;
            let x = xy * sector_angle.cos();
            let y = xy * sector_angle.sin();

            // For a sphere, normal is normalized position
            let nx = x / radius;
            let ny = y / radius;
            let nz = z / radius;

            vertices.push(LambdaVertex {
                position: [x, y, z],
                normal: [nx, ny, nz],
            });
        }
    }

    for i in 0..stacks {
        let k1 = i * (sectors + 1);
        let k2 = k1 + sectors + 1;

        for j in 0..sectors {
            if i != 0 {
                indices.push((k1 + j) as u16);
                indices.push((k2 + j) as u16);
                indices.push((k1 + j + 1) as u16);
            }

            if i != (stacks - 1) {
                indices.push((k1 + j + 1) as u16);
                indices.push((k2 + j) as u16);
                indices.push((k2 + j + 1) as u16);
            }
        }
    }

    (vertices, indices)
}

#[derive(Clone, Debug)]
pub struct VisualNode {
    pub id: u64,
    pub term: Rc<Term>, // Back reference to logic
    pub position: Point3<f32>,
    pub target_position: Point3<f32>,
    pub velocity: Vector3<f32>,
    pub color: [f32; 4],
    pub scale: f32,
    pub node_type: NodeType,
    pub collapsed: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum NodeType {
    Var(String),
    Abs(String),
    App,
    Prim(Primitive),
    Port, // New: Represents a variable usage port (bound variable)
}

#[derive(Clone, Debug)]
pub enum AnimationState {
    Idle,
    Reducing {
        // We are reducing: (\x.Body) Arg
        // Abs index is the bubble. Arg index is the root of the argument.
        // We want to visually move Arg to Ports.
        abs_idx: usize,
        arg_idx: usize,
        ports: Vec<usize>,
        progress: f32,
        new_term: Rc<Term>, // The result to switch to after animation
    },
}

pub struct LambdaSystem {
    pub nodes: Vec<VisualNode>,
    pub edges: Vec<(usize, usize)>, // Indices into nodes
    pub root_term: Option<Rc<Term>>,
    next_id: u64,
    pub dragged_node: Option<usize>,
    pub drag_distance: f32,
    pub hovered_node: Option<usize>,
    pub anchor_pos: Point3<f32>,
    pub animation_state: AnimationState,
    pub paused: bool,
    pub auto_reduce: bool,
    pub auto_reduce_timer: f32,
}

impl LambdaSystem {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            root_term: None,
            next_id: 0,
            dragged_node: None,
            drag_distance: 0.0,
            hovered_node: None,
            anchor_pos: Point3::new(0.0, 5.0, 0.0),
            animation_state: AnimationState::Idle,
            paused: false,
            auto_reduce: false,
            auto_reduce_timer: 0.0,
        }
    }

    pub fn set_anchor(&mut self, new_anchor: Point3<f32>) {
        let delta = new_anchor - self.anchor_pos;
        self.anchor_pos = new_anchor;
        for node in &mut self.nodes {
            node.position += delta;
            node.target_position += delta;
        }
    }

    pub fn set_term(&mut self, term: Rc<Term>) {
        self.root_term = Some(term.clone());
        self.rebuild_graph();
    }

    pub fn intersect(&self, ray_origin: Point3<f32>, ray_dir: Vector3<f32>) -> Option<usize> {
        let mut closest_dist = std::f32::MAX;
        let mut closest_idx = None;

        for (i, node) in self.nodes.iter().enumerate() {
            // Ray-Sphere Intersection
            let radius = node.scale; // Use node scale as radius!
            let l = node.position - ray_origin;
            let tca = l.dot(ray_dir);
            if tca < 0.0 { continue; }

            let d2 = l.dot(l) - tca * tca;
            if d2 > radius * radius { continue; }

            let thc = (radius * radius - d2).sqrt();
            let t0 = tca - thc;
            let t1 = tca + thc;

            let t = if t0 < 0.0 { t1 } else { t0 };

            if t < closest_dist {
                closest_dist = t;
                closest_idx = Some(i);
            }
        }

        closest_idx
    }

    pub fn update_hover(&mut self, ray_origin: Point3<f32>, ray_dir: Vector3<f32>) {
        self.hovered_node = self.intersect(ray_origin, ray_dir);
    }

    pub fn start_drag(&mut self, idx: usize, ray_origin: Point3<f32>, ray_dir: Vector3<f32>) {
        if idx < self.nodes.len() {
            self.dragged_node = Some(idx);
            // Calculate distance to object
            let diff = self.nodes[idx].position - ray_origin;
            self.drag_distance = diff.dot(ray_dir);
        }
    }

    pub fn update_drag(&mut self, ray_origin: Point3<f32>, ray_dir: Vector3<f32>) {
        if let Some(idx) = self.dragged_node {
            if idx < self.nodes.len() {
                // Project ray to distance
                let target = ray_origin + ray_dir * self.drag_distance;
                self.nodes[idx].target_position = target;
                self.nodes[idx].position = target; // Snap or smooth? Snap for responsiveness
                self.nodes[idx].velocity = Vector3::new(0.0, 0.0, 0.0); // Reset physics
            }
        }
    }

    pub fn end_drag(&mut self) {
        self.dragged_node = None;
    }

    pub fn toggle_collapse(&mut self, idx: usize) {
        if idx < self.nodes.len() {
            self.nodes[idx].collapsed = !self.nodes[idx].collapsed;
            let is_collapsed = self.nodes[idx].collapsed;
            self.set_subtree_visibility(idx, !is_collapsed);
            // Ensure the node itself remains visible
            self.nodes[idx].scale = if matches!(self.nodes[idx].node_type, NodeType::Abs(_)) { 3.0 } else { 1.0 };
        }
    }

    fn set_subtree_visibility(&mut self, root_idx: usize, visible: bool) {
        // Simple visibility toggle
        let mut stack = vec![root_idx];
        while let Some(parent) = stack.pop() {
            for &(p, c) in &self.edges {
                if p == parent {
                    // Set scale based on visibility and type
                    if !visible {
                        self.nodes[c].scale = 0.0;
                    } else {
                        // Restore default scale
                         self.nodes[c].scale = match self.nodes[c].node_type {
                            NodeType::Abs(_) => 3.0,
                            NodeType::Port => 0.2,
                            NodeType::App => 0.5,
                            _ => 1.0,
                         };
                    }
                    stack.push(c);
                }
            }
        }
    }

    pub fn reduce_root(&mut self) -> bool {
        // Don't interrupt animation
        if let AnimationState::Reducing { .. } = self.animation_state {
            return false;
        }

        if let Some(root) = &self.root_term {
             let (new_root, reduced) = root.reduce();
             if reduced {
                 // Detect Reduction Type for Animation
                 // We want to animate Beta Reduction: (\x.M) N
                 // Check if root is App(Abs(x, M), N)
                 if let Term::App(func, _arg) = &**root {
                     if let Term::Abs(_param, _body) = &**func {
                         // Found a Beta Reduction!
                         // Need to find visual nodes corresponding to Abs and Arg
                         // This is tricky because `nodes` is flat.
                         // But we built the graph. The root is node 0.
                         // App is node 0.
                         // Func (Abs) is the left child of 0.
                         // Arg is the right child of 0.

                         // Find children of 0
                         let mut left_child = None;
                         let mut right_child = None;
                         for &(p, c) in &self.edges {
                             if p == 0 {
                                 // How to distinguish left/right?
                                 // In build_subtree, we pushed left then right.
                                 // So left_child < right_child usually?
                                 // Or check node type/term?
                                 // We know Func is Abs.
                                 // Edges were pushed in order: Left then Right in build_subtree.
                                 // But we can't rely on edge order in self.edges (it's a vec but iterate might be unstable if we used map? No, it's vec).
                                 // But build_subtree pushes (node_index, left_idx) then (node_index, right_idx).
                                 // So left child has smaller index usually? No, recursive.
                                 // Left child is built first. So left_idx < right_idx (usually).

                                 // Better heuristic: Check if we already found left_child.
                                 if matches!(self.nodes[c].node_type, NodeType::Abs(_)) && left_child.is_none() {
                                     left_child = Some(c);
                                 } else {
                                     right_child = Some(c);
                                 }
                             }
                         }

                         if let (Some(abs_idx), Some(arg_idx)) = (left_child, right_child) {
                             // Find Ports for 'param' inside Abs
                             // We need to scan nodes for NodeType::Port that link to abs_idx
                             // In build_subtree, Port -> Binder edge is created.
                             // Wait, Edge direction is (Port, Binder)?
                             // Yes: `self.edges.push((node_index, binder_idx));`

                             let mut ports = Vec::new();
                             for i in 0..self.nodes.len() {
                                 if matches!(self.nodes[i].node_type, NodeType::Port) {
                                     // Check if this port connects to abs_idx
                                     for &(src, dst) in &self.edges {
                                         if src == i && dst == abs_idx {
                                             ports.push(i);
                                         }
                                     }
                                 }
                             }

                             self.animation_state = AnimationState::Reducing {
                                 abs_idx,
                                 arg_idx,
                                 ports,
                                 progress: 0.0,
                                 new_term: new_root,
                             };
                             return true;
                         }
                     }
                 }

                 // Fallback for other reductions (e.g. inner)
                 self.set_term(new_root);
                 return true;
             }
        }
        false
    }

    pub fn update(&mut self, dt: f32) -> Vec<LambdaEvent> {
        let mut events = Vec::new();

        // Auto-Reduce Logic
        if self.auto_reduce && !self.paused && matches!(self.animation_state, AnimationState::Idle) {
            self.auto_reduce_timer += dt;
            if self.auto_reduce_timer > 0.5 {
                self.auto_reduce_timer = 0.0;
                if self.reduce_root() {
                    events.push(LambdaEvent::ReductionStarted);
                }
            }
        }

        // Animation Logic
        if let AnimationState::Reducing { ref abs_idx, ref arg_idx, ref ports, ref mut progress, ref new_term } = self.animation_state {
            if !self.paused {
                *progress += dt * 2.0; // 0.5s duration
            }

            if *progress >= 1.0 {
                // Animation Complete
                let term = new_term.clone();
                self.animation_state = AnimationState::Idle;
                self.set_term(term);
                return events;
            }

            // Animate!
            // 1. Move Arg towards Abs (Consumption)
            // 2. Move Arg towards Ports (Substitution) - if we could duplicate it visualy...
            // For now, let's just move Arg towards Abs center.

            // We can't easily move the whole subtree of Arg without updating all children.
            // But `set_anchor` does that via delta.
            // Let's calculate delta for arg_idx to move towards abs_idx.

            let target_pos = self.nodes[*abs_idx].position;
            let current_pos = self.nodes[*arg_idx].position;
            let dir = target_pos - current_pos;

            // We need frame delta.
            // Actually, let's just modify the `target_position` of the arg root?
            // The physics loop below will pull it.
            // But physics is springy. We want deterministic animation.

            // Let's force position.
            let lerp_pos = current_pos + dir * (dt * 5.0); // Move fast

            // Move subtree
            let delta = lerp_pos - current_pos;

            // BFS to move arg subtree
            let mut stack = vec![*arg_idx];
            let mut visited = std::collections::HashSet::new();
            while let Some(idx) = stack.pop() {
                if visited.contains(&idx) { continue; }
                visited.insert(idx);

                self.nodes[idx].position += delta;
                self.nodes[idx].target_position += delta;

                // 1. Shrink Effect: As progress -> 1.0, scale -> 0.1
                // But only if we are being consumed.
                // The argument is being consumed by the Abs bubble.
                // Let's shrink it as it gets closer.
                let shrink_factor = 1.0 - (*progress * 0.9);
                if matches!(self.nodes[idx].node_type, NodeType::Abs(_)) {
                     // Keep bubbles somewhat visible
                     self.nodes[idx].scale = 3.0 * shrink_factor;
                } else {
                     self.nodes[idx].scale = shrink_factor;
                }

                // Find children
                for &(p, c) in &self.edges {
                    if p == idx {
                        stack.push(c);
                    }
                }
            }

            // 2. Bubble Pulse: Pulse the Abs bubble
            // Base scale 3.0. Pulse up to 3.5.
            let pulse = (*progress * std::f32::consts::PI).sin(); // 0 -> 1 -> 0
            self.nodes[*abs_idx].scale = 3.0 + (pulse * 0.5);

            // 3. Color Shift: Shift Abs color to Reddish to show activity
            // Base is [1.0, 1.0, 1.0, 0.3]
            // Shift to [1.0, 0.5, 0.5, 0.5]
            self.nodes[*abs_idx].color = [
                1.0,
                1.0 - (pulse * 0.5),
                1.0 - (pulse * 0.5),
                0.3 + (pulse * 0.2)
            ];

            // Glow effect on Ports?
            // We can pulse their scale
            for &port_idx in ports {
                self.nodes[port_idx].scale = 0.2 + (pulse * 0.3);
                 // Also highlight color
                 self.nodes[port_idx].color = [
                     0.2 + (pulse * 0.8),
                     0.2,
                     0.2,
                     1.0
                 ];
            }
        }

        let node_count = self.nodes.len();
        if node_count == 0 { return events; }

        let repulsion_strength = 50.0; // Increased for bubbles
        let centering_strength = 0.5;
        let damping = 0.9;
        let base_rest_length = 3.0;

        // 1. Repulsion
        for i in 0..node_count {
            if self.nodes[i].scale < 0.001 { continue; } // Skip invisible
            for j in 0..node_count {
                if i == j { continue; }
                if self.nodes[j].scale < 0.001 { continue; } // Skip invisible

                let dir = self.nodes[i].position - self.nodes[j].position;
                let dist_sq = dir.magnitude2();
                if dist_sq < 0.0001 {
                     self.nodes[i].velocity += Vector3::new(0.1, 0.0, 0.0);
                } else {
                     // Check radii collision
                     let r1 = self.nodes[i].scale;
                     let r2 = self.nodes[j].scale;
                     let min_dist = (r1 + r2) * 1.1; // Add padding

                     if dist_sq < (min_dist * min_dist) {
                         // Hard push (collision)
                         let force = dir.normalize() * (repulsion_strength * 5.0);
                         self.nodes[i].velocity += force * dt;
                     } else if dist_sq < 150.0 {
                         // Soft push (repulsion)
                         let force = dir.normalize() * (repulsion_strength / dist_sq);
                         self.nodes[i].velocity += force * dt;
                     }
                }
            }
        }

        // 2. Spring Forces (Edges)
        for &(start, end) in &self.edges {
             if start >= node_count || end >= node_count { continue; }
             if self.nodes[start].scale < 0.001 || self.nodes[end].scale < 0.001 { continue; }

             let dir = self.nodes[end].position - self.nodes[start].position;
             let dist = dir.magnitude();

             // Dynamic Spring Params based on Edge Type
             let is_wire = matches!(self.nodes[start].node_type, NodeType::Port);

             let (stiffness, rest_len) = if is_wire {
                 (0.5, base_rest_length * 2.0) // Loose, long wires
             } else {
                 (5.0, base_rest_length) // Stiff structure
             };

             if dist > 0.0001 {
                 let force = (dist - rest_len) * stiffness;
                 let force_vec = dir.normalize() * force;
                 self.nodes[start].velocity += force_vec * dt;
                 self.nodes[end].velocity -= force_vec * dt;
             }
        }

        // 3. Centering / Target Force & Integration
        for node in &mut self.nodes {
            if node.scale < 0.001 {
                node.position = node.target_position; // Snap hidden nodes
                node.velocity = Vector3::new(0.0, 0.0, 0.0);
                continue;
            }

            // Pull towards "ideal" tree layout target (weakly)
            let to_target = node.target_position - node.position;
            node.velocity += to_target * centering_strength * dt;

            // Update Position
            node.position += node.velocity * dt;

            // Damping
            node.velocity *= damping;
        }

        events
    }

    fn rebuild_graph(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.next_id = 0;

        let scope = HashMap::new();

        if let Some(term) = &self.root_term {
            // Calculate layout positions
            // Start at anchor_pos relative to World Origin
            self.build_subtree(term.clone(), self.anchor_pos, 0, &scope);
        }
    }

    // Returns the index of the node representing this term
    fn build_subtree(&mut self, term: Rc<Term>, pos: Point3<f32>, depth: i32, scope: &HashMap<String, usize>) -> usize {
        let id = self.next_id;
        self.next_id += 1;

        match &*term {
            Term::Abs(param, body) => {
                // Create Bubble Node
                let color = [1.0, 1.0, 1.0, 0.3]; // Translucent White
                let scale = 3.0; // Large Bubble
                let node_index = self.nodes.len();

                self.nodes.push(VisualNode {
                    id,
                    term: term.clone(),
                    position: pos,
                    target_position: pos,
                    velocity: Vector3::new(0.0, 0.0, 0.0),
                    color,
                    scale,
                    node_type: NodeType::Abs(param.clone()),
                    collapsed: false,
                });

                // Update Scope
                let mut new_scope = scope.clone();
                new_scope.insert(param.clone(), node_index);

                // Build Body
                let child_pos = pos + Vector3::new(0.0, -2.0, 0.0);
                let child_idx = self.build_subtree(body.clone(), child_pos, depth + 1, &new_scope);

                // Edge from Abs to Body (Structure)
                self.edges.push((node_index, child_idx));

                node_index
            },
            Term::Var(name) => {
                // Check if Bound
                if let Some(&binder_idx) = scope.get(name) {
                    // It is bound!
                    // Create a "Port" node at the usage site
                    let color = [0.2, 0.2, 0.2, 1.0]; // Dark Grey Port
                    let scale = 0.2;
                    let node_index = self.nodes.len();

                    self.nodes.push(VisualNode {
                        id,
                        term: term.clone(),
                        position: pos,
                        target_position: pos,
                        velocity: Vector3::new(0.0, 0.0, 0.0),
                        color,
                        scale,
                        node_type: NodeType::Port,
                        collapsed: false,
                    });

                    // Create WIRE Edge (Port -> Binder)
                    self.edges.push((node_index, binder_idx));

                    node_index
                } else {
                    // Free Variable
                    let color = self.hash_color(name);
                    let scale = 1.0;
                    let node_index = self.nodes.len();

                    self.nodes.push(VisualNode {
                        id,
                        term: term.clone(),
                        position: pos,
                        target_position: pos,
                        velocity: Vector3::new(0.0, 0.0, 0.0),
                        color,
                        scale,
                        node_type: NodeType::Var(name.clone()),
                        collapsed: false,
                    });

                    node_index
                }
            },
            Term::App(func, arg) => {
                // Application Node
                let color = [0.5, 0.5, 0.5, 0.8];
                let scale = 0.5; // Connector
                let node_index = self.nodes.len();

                self.nodes.push(VisualNode {
                    id,
                    term: term.clone(),
                    position: pos,
                    target_position: pos,
                    velocity: Vector3::new(0.0, 0.0, 0.0),
                    color,
                    scale,
                    node_type: NodeType::App,
                    collapsed: false,
                });

                // Build Children
                let horizontal_spacing = 3.0 * (0.8f32).powi(depth);
                let left_pos = pos + Vector3::new(-horizontal_spacing, -2.0, 0.0);
                let right_pos = pos + Vector3::new(horizontal_spacing, -2.0, 0.0);

                let left_idx = self.build_subtree(func.clone(), left_pos, depth + 1, scope);
                let right_idx = self.build_subtree(arg.clone(), right_pos, depth + 1, scope);

                self.edges.push((node_index, left_idx));
                self.edges.push((node_index, right_idx));

                node_index
            },
            Term::Prim(p) => {
                 let color = self.get_primitive_color(*p);
                 let scale = 1.0;
                 let node_index = self.nodes.len();

                 self.nodes.push(VisualNode {
                    id,
                    term: term.clone(),
                    position: pos,
                    target_position: pos,
                    velocity: Vector3::new(0.0, 0.0, 0.0),
                    color,
                    scale,
                    node_type: NodeType::Prim(*p),
                    collapsed: false,
                });

                node_index
            }
        }
    }

    fn hash_color(&self, s: &str) -> [f32; 4] {
        let mut sum = 0u32;
        for b in s.bytes() {
            sum = sum.wrapping_add(b as u32);
        }

        // Simple procedural color generation
        let r = ((sum * 123) % 255) as f32 / 255.0;
        let g = ((sum * 456) % 255) as f32 / 255.0;
        let b = ((sum * 789) % 255) as f32 / 255.0;

        [r, g, b, 1.0]
    }

    fn get_primitive_color(&self, p: Primitive) -> [f32; 4] {
        match p {
            Primitive::Fire => [1.0, 0.0, 0.0, 1.0],
            Primitive::Water => [0.0, 0.0, 1.0, 1.0],
            Primitive::Earth => [0.6, 0.4, 0.2, 1.0],
            Primitive::Air => [0.0, 1.0, 1.0, 0.5],
            Primitive::Growth => [0.0, 1.0, 0.0, 1.0],
            Primitive::Decay => [0.5, 0.0, 0.5, 1.0],
            Primitive::Energy => [1.0, 1.0, 0.0, 1.0],
            Primitive::Stable => [1.0, 1.0, 1.0, 1.0],
            Primitive::Void => [0.1, 0.1, 0.1, 1.0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lambda;

    #[test]
    fn test_auto_reduce() {
        let mut sys = LambdaSystem::new();
        // (\x.x) y -> y
        let term = lambda::parse("(\\x.x) y").unwrap();
        sys.set_term(term);

        sys.auto_reduce = true;
        sys.paused = false;

        // Update 1: Accumulate some time
        sys.update(0.4);
        assert!(matches!(sys.animation_state, AnimationState::Idle));

        // Update 2: Cross the 0.5s threshold
        // Timer was 0.4. Add 0.2 -> 0.6 (>0.5). Triggers Reduce.
        // Animation Logic runs with dt=0.2. Progress += 0.4. Total Progress 0.4.
        // Should remain in Reducing state.
        sys.update(0.2);

        // Should have triggered reduction.
        // In (\x.x) y, it IS a beta reduction, so it should enter AnimationState::Reducing.
        match sys.animation_state {
            AnimationState::Reducing { .. } => assert!(true),
            _ => assert!(false, "Should be reducing"),
        }
    }

    #[test]
    fn test_paused_no_reduce() {
        let mut sys = LambdaSystem::new();
        let term = lambda::parse("(\\x.x) y").unwrap();
        sys.set_term(term);

        sys.auto_reduce = true;
        sys.paused = true;

        sys.update(0.6);

        // Should NOT be reducing
        match sys.animation_state {
            AnimationState::Idle => assert!(true),
            _ => assert!(false, "Should stay idle when paused"),
        }
    }
}
