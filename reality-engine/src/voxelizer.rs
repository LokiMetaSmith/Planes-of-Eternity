use crate::voxel::{Chunk, ChunkKey, Voxel, CHUNK_SIZE};
use cgmath::{InnerSpace, Point3, Vector3};

pub struct Voxelizer;

impl Voxelizer {
    /// Voxelizes a 3D mesh (given by vertices and indices) into a Chunk.
    ///
    /// The mesh should be scaled and translated such that it fits within the
    /// bounding box `(0.0, 0.0, 0.0)` to `(CHUNK_SIZE as f32, CHUNK_SIZE as f32, CHUNK_SIZE as f32)`.
    pub fn voxelize_mesh(vertices: &[Point3<f32>], indices: &[u32], voxel_id: u8) -> Chunk {
        // Since we are returning a Chunk without knowing its world key, we can use a dummy key
        // like x: 0, y: 0, z: 0. The caller can always reassign or place its data appropriately.
        let mut chunk = Chunk::new(ChunkKey { x: 0, y: 0, z: 0 });

        // Simple and robust approach: evaluate every voxel against every triangle
        // For larger meshes or higher resolution, a more optimized approach (like AABB
        // tree or 3D scanline conversion) should be used.
        let num_triangles = indices.len() / 3;

        for z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    let mut is_inside = false;

                    // The center of the current voxel
                    let voxel_center = Point3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);

                    // A simple bounding box check per voxel to see if it intersects any triangle
                    // Or cast a ray to determine inside/outside status (solid voxelization)
                    // For now, we implement a surface voxelization: if the voxel AABB intersects
                    // a triangle, it's solid.

                    let voxel_half_size = Vector3::new(0.5, 0.5, 0.5);

                    for t in 0..num_triangles {
                        let i0 = indices[t * 3] as usize;
                        let i1 = indices[t * 3 + 1] as usize;
                        let i2 = indices[t * 3 + 2] as usize;

                        let v0 = vertices[i0];
                        let v1 = vertices[i1];
                        let v2 = vertices[i2];

                        if tri_box_overlap(voxel_center, voxel_half_size, v0, v1, v2) {
                            is_inside = true;
                            break;
                        }
                    }

                    if is_inside {
                        let idx = chunk.index(x, y, z);
                        chunk.data[idx] = Voxel { id: voxel_id };
                        // Request detail splat generation for the deformed voxel
                        // (Requires integration with GenieBridge, but since this is isolated we just log/mark)
                    }
                }
            }
        }

        chunk
    }
}

// AABB-Triangle intersection (Akenine-Möller algorithm)
fn tri_box_overlap(
    box_center: Point3<f32>,
    box_half_size: Vector3<f32>,
    tv0: Point3<f32>,
    tv1: Point3<f32>,
    tv2: Point3<f32>,
) -> bool {
    let v0 = tv0 - box_center;
    let v1 = tv1 - box_center;
    let v2 = tv2 - box_center;

    let f0 = v1 - v0;
    let f1 = v2 - v1;
    let f2 = v0 - v2;

    // Test axes a00..a22 (cross products of edge vectors and AABB normals)
    let a00 = Vector3::new(0.0, -f0.z, f0.y);
    let a01 = Vector3::new(0.0, -f1.z, f1.y);
    let a02 = Vector3::new(0.0, -f2.z, f2.y);
    let a10 = Vector3::new(f0.z, 0.0, -f0.x);
    let a11 = Vector3::new(f1.z, 0.0, -f1.x);
    let a12 = Vector3::new(f2.z, 0.0, -f2.x);
    let a20 = Vector3::new(-f0.y, f0.x, 0.0);
    let a21 = Vector3::new(-f1.y, f1.x, 0.0);
    let a22 = Vector3::new(-f2.y, f2.x, 0.0);

    if !axis_test(a00, v0, v1, v2, box_half_size) {
        return false;
    }
    if !axis_test(a01, v0, v1, v2, box_half_size) {
        return false;
    }
    if !axis_test(a02, v0, v1, v2, box_half_size) {
        return false;
    }
    if !axis_test(a10, v0, v1, v2, box_half_size) {
        return false;
    }
    if !axis_test(a11, v0, v1, v2, box_half_size) {
        return false;
    }
    if !axis_test(a12, v0, v1, v2, box_half_size) {
        return false;
    }
    if !axis_test(a20, v0, v1, v2, box_half_size) {
        return false;
    }
    if !axis_test(a21, v0, v1, v2, box_half_size) {
        return false;
    }
    if !axis_test(a22, v0, v1, v2, box_half_size) {
        return false;
    }

    // Test the three axes corresponding to the face normals of AABB
    if find_min(v0.x, v1.x, v2.x) > box_half_size.x || find_max(v0.x, v1.x, v2.x) < -box_half_size.x
    {
        return false;
    }
    if find_min(v0.y, v1.y, v2.y) > box_half_size.y || find_max(v0.y, v1.y, v2.y) < -box_half_size.y
    {
        return false;
    }
    if find_min(v0.z, v1.z, v2.z) > box_half_size.z || find_max(v0.z, v1.z, v2.z) < -box_half_size.z
    {
        return false;
    }

    // Test separating axis corresponding to triangle face normal
    let normal = f0.cross(f1);
    if !plane_box_overlap(normal, v0, box_half_size) {
        return false;
    }

    true
}

fn axis_test(
    a: Vector3<f32>,
    v0: Vector3<f32>,
    v1: Vector3<f32>,
    v2: Vector3<f32>,
    box_half_size: Vector3<f32>,
) -> bool {
    let p0 = a.dot(v0);
    let p1 = a.dot(v1);
    let p2 = a.dot(v2);

    let min = find_min(p0, p1, p2);
    let max = find_max(p0, p1, p2);

    let r = box_half_size.x * a.x.abs() + box_half_size.y * a.y.abs() + box_half_size.z * a.z.abs();

    if min > r || max < -r {
        return false;
    }
    true
}

fn plane_box_overlap(normal: Vector3<f32>, vert: Vector3<f32>, maxbox: Vector3<f32>) -> bool {
    let mut vmin = Vector3::new(0.0, 0.0, 0.0);
    let mut vmax = Vector3::new(0.0, 0.0, 0.0);

    if normal.x > 0.0 {
        vmin.x = -maxbox.x;
        vmax.x = maxbox.x;
    } else {
        vmin.x = maxbox.x;
        vmax.x = -maxbox.x;
    }
    if normal.y > 0.0 {
        vmin.y = -maxbox.y;
        vmax.y = maxbox.y;
    } else {
        vmin.y = maxbox.y;
        vmax.y = -maxbox.y;
    }
    if normal.z > 0.0 {
        vmin.z = -maxbox.z;
        vmax.z = maxbox.z;
    } else {
        vmin.z = maxbox.z;
        vmax.z = -maxbox.z;
    }

    let min_dot = normal.dot(vmin);
    let max_dot = normal.dot(vmax);
    let d = -normal.dot(vert);

    if min_dot + d > 0.0 {
        return false;
    }
    if max_dot + d >= 0.0 {
        return true;
    }

    false
}

fn find_min(a: f32, b: f32, c: f32) -> f32 {
    a.min(b).min(c)
}

fn find_max(a: f32, b: f32, c: f32) -> f32 {
    a.max(b).max(c)
}
