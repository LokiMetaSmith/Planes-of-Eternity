use cgmath::{Point3, Vector3, InnerSpace, MetricSpace};
use std::rc::Rc;
use crate::lambda::Term;
use wgpu::util::DeviceExt;

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
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    instance_buffer: wgpu::Buffer,
    capacity: usize,

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

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Lambda Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[LambdaVertex::desc(), LambdaInstance::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
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
            depth_stencil: None, // Ensure depth testing matches main pass if used
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // Line Pipeline
        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Lambda Line Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_line",
                buffers: &[LambdaLineVertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_line",
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
            vertex_buffer,
            index_buffer,
            num_indices: indices.len() as u32,
            instance_buffer,
            capacity,
            line_pipeline,
            line_vertex_buffer,
            line_vertex_count: 0,
            line_capacity,
        }
    }

    pub fn update_buffers(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, nodes: &[VisualNode], edges: &[(usize, usize)]) {
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

        let instances: Vec<LambdaInstance> = nodes.iter().map(|n| LambdaInstance {
            position: [n.position.x, n.position.y, n.position.z],
            color: n.color,
            scale: n.scale,
        }).collect();

        if !instances.is_empty() {
             queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&instances));
        }

        // Update Lines
        let required_line_vertices = edges.len() * 2;
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

                // Let's use white for edges, 0.5 alpha
                let color = [1.0, 1.0, 1.0, 0.5];

                line_data.push(LambdaLineVertex {
                    position: [start_node.position.x, start_node.position.y, start_node.position.z],
                    color,
                });
                 line_data.push(LambdaLineVertex {
                    position: [end_node.position.x, end_node.position.y, end_node.position.z],
                    color,
                });
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

        // Draw Nodes
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..self.num_indices, 0, 0..num_instances);
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
}

pub struct LambdaSystem {
    pub nodes: Vec<VisualNode>,
    pub edges: Vec<(usize, usize)>, // Indices into nodes
    pub root_term: Option<Rc<Term>>,
    next_id: u64,
    pub dragged_node: Option<usize>,
    pub drag_distance: f32,
    pub anchor_pos: Point3<f32>,
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
            anchor_pos: Point3::new(0.0, 5.0, 0.0),
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
            // P = O + tD
            // |P - C|^2 = R^2
            // |O + tD - C|^2 = R^2
            // Let L = C - O
            // t^2 - 2t(L.D) + |L|^2 - R^2 = 0

            let radius = 1.0; // Sphere radius (mesh is size 1.0)
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
            // TODO: In a real implementation, this would hide children.
            // Since we use 'rebuild_graph', we might lose this on next update.
            // For MVP visual toggle, we can just set scale of children to 0?
            // Implementing full collapse is complex with current architecture.
            // Let's change scale of children to 0 if collapsed.
            // But we need to recurse down.

            let is_collapsed = self.nodes[idx].collapsed;
            self.set_subtree_visibility(idx, !is_collapsed);
            // Ensure the node itself remains visible
            self.nodes[idx].scale = 1.0;
        }
    }

    fn set_subtree_visibility(&mut self, root_idx: usize, visible: bool) {
        // BFS or DFS to set visibility
        // We have edges: (parent, child). We need to find children of root_idx.
        // This is slow if edges are just a list.
        // But edges are built in order... no they aren't guaranteed.
        // Let's scan edges.

        let mut stack = vec![root_idx];
        while let Some(parent) = stack.pop() {
            for &(p, c) in &self.edges {
                if p == parent {
                    self.nodes[c].scale = if visible { 1.0 } else { 0.0 };
                    // If we are making visible, and the child itself is collapsed, don't recurse?
                    // Complexity... let's just show/hide all.
                    stack.push(c);
                }
            }
        }
    }

    pub fn reduce_root(&mut self) -> bool {
        if let Some(root) = &self.root_term {
             let (new_root, reduced) = root.reduce();
             if reduced {
                 self.set_term(new_root);
                 return true;
             }
        }
        false
    }

    pub fn get_archetype_from_term(&self) -> i32 {
        // Simple heuristic to map term to archetype
        // Count nodes?
        let count = self.nodes.len();
        match count {
            0..=3 => 0, // Fantasy (Simple)
            4..=7 => 1, // SciFi
            8..=15 => 2, // Horror
            _ => 3,     // Toon (Complex)
        }
    }

    pub fn update(&mut self, dt: f32) {
        // Spring physics / Interpolation to target
        for node in &mut self.nodes {
            let diff = node.target_position - node.position;
            let dist = diff.magnitude();

            if dist > 0.001 {
                let speed = 5.0; // Units per second
                let step = diff.normalize() * speed * dt;

                if step.magnitude() >= dist {
                    node.position = node.target_position;
                } else {
                    node.position += step;
                }
            }
        }
    }

    fn rebuild_graph(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.next_id = 0;

        if let Some(term) = &self.root_term {
            // Calculate layout positions
            // Start at anchor_pos relative to World Origin
            self.build_subtree(term.clone(), self.anchor_pos, 0);
        }
    }

    // Returns the index of the root node of this subtree
    fn build_subtree(&mut self, term: Rc<Term>, pos: Point3<f32>, depth: i32) -> usize {
        let id = self.next_id;
        self.next_id += 1;

        let (node_type, color) = match &*term {
            Term::Var(name) => (NodeType::Var(name.clone()), self.hash_color(name)),
            Term::Abs(param, _) => (NodeType::Abs(param.clone()), [1.0, 1.0, 1.0, 1.0]), // White for Lambda
            Term::App(_, _) => (NodeType::App, [0.5, 0.5, 0.5, 1.0]), // Grey for App
        };

        let node_index = self.nodes.len();
        self.nodes.push(VisualNode {
            id,
            term: term.clone(),
            position: pos, // Spawn at target for now (snap)
            target_position: pos,
            velocity: Vector3::new(0.0, 0.0, 0.0),
            color,
            scale: 1.0,
            node_type,
            collapsed: false,
        });

        // Recursively build children
        let vertical_spacing = 2.0;
        let horizontal_spacing = 2.0 * (0.8f32).powi(depth); // Tighten spacing as we go down

        // Note: We don't respect 'collapsed' here because rebuild_graph is called on Logic change.
        // If we want to persist collapsed state, we need a map.
        // For MVP, we'll implement collapse as just "hide children".
        // But if we hide children, we shouldn't build them?
        // Let's implement toggle logic later in 'toggle_collapse'.
        // For now, just build everything.

        match &*term {
            Term::Var(_) => {
                // Leaf
            }
            Term::Abs(_, body) => {
                let child_pos = pos + Vector3::new(0.0, -vertical_spacing, 0.0);
                let child_idx = self.build_subtree(body.clone(), child_pos, depth + 1);
                self.edges.push((node_index, child_idx));
            }
            Term::App(func, arg) => {
                let left_pos = pos + Vector3::new(-horizontal_spacing, -vertical_spacing, 0.0);
                let right_pos = pos + Vector3::new(horizontal_spacing, -vertical_spacing, 0.0);

                let left_idx = self.build_subtree(func.clone(), left_pos, depth + 1);
                let right_idx = self.build_subtree(arg.clone(), right_pos, depth + 1);

                self.edges.push((node_index, left_idx));
                self.edges.push((node_index, right_idx));
            }
        }

        node_index
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
}
