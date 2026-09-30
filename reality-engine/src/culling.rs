use cgmath::{Matrix3, Point3, Vector2, Vector3};

const BUFFER_WIDTH: usize = 256;
const BUFFER_HEIGHT: usize = 128;

pub struct SoftwareOcclusionCuller {
    pub buffer: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

impl SoftwareOcclusionCuller {
    pub fn new() -> Self {
        Self {
            buffer: vec![f32::INFINITY; BUFFER_WIDTH * BUFFER_HEIGHT],
            width: BUFFER_WIDTH,
            height: BUFFER_HEIGHT,
        }
    }

    pub fn clear(&mut self) {
        self.buffer.fill(f32::INFINITY);
    }

    /// Transforms a point into view space
    fn transform(
        p: Point3<f32>,
        view_rot: &Matrix3<f32>,
        view_trans: &Vector3<f32>,
        focal_len_px: &Vector2<f32>,
        near_plane: f32,
    ) -> Option<(f32, f32, f32)> {
        // Apply view matrix: R * p + T
        let view_pos = view_rot * Vector3::new(p.x, p.y, p.z) + view_trans;

        // Note: standard view matrices look down -Z. Depth in front of camera is negative Z.
        // We will work with linear depth as `dist = -view_pos.z`.
        let dist = -view_pos.z;

        if dist < near_plane {
            return None;
        }

        // Perspective divide
        let inv_z = 1.0 / dist;
        let x_screen = view_pos.x * inv_z * focal_len_px.x + (BUFFER_WIDTH as f32 * 0.5);
        let y_screen = -view_pos.y * inv_z * focal_len_px.y + (BUFFER_HEIGHT as f32 * 0.5); // Usually -y is up in view, screen Y is down

        Some((x_screen, y_screen, dist))
    }

    fn project_aabb(
        &self,
        min: Point3<f32>,
        max: Point3<f32>,
        view_rot: &Matrix3<f32>,
        view_trans: &Vector3<f32>,
        focal_len_px: &Vector2<f32>,
        near_plane: f32,
    ) -> Option<Vec<(f32, f32, f32)>> {
        let corners = [
            Point3::new(min.x, min.y, min.z),
            Point3::new(max.x, min.y, min.z),
            Point3::new(min.x, max.y, min.z),
            Point3::new(max.x, max.y, min.z),
            Point3::new(min.x, min.y, max.z),
            Point3::new(max.x, min.y, max.z),
            Point3::new(min.x, max.y, max.z),
            Point3::new(max.x, max.y, max.z),
        ];

        let mut projected = Vec::with_capacity(8);
        for &corner in &corners {
            if let Some(p) = Self::transform(corner, view_rot, view_trans, focal_len_px, near_plane) {
                projected.push(p);
            } else {
                // If any corner is behind the camera, we conservatively assume we can't reliably occlusion test it
                // and skip drawing it to the depth buffer / consider it visible.
                return None;
            }
        }
        Some(projected)
    }

    /// Rasterizes a fully solid occluder (Write)
    pub fn rasterize_occluder(
        &mut self,
        min: Point3<f32>,
        max: Point3<f32>,
        view_rot: &Matrix3<f32>,
        view_trans: &Vector3<f32>,
        focal_len_px: &Vector2<f32>,
        near_plane: f32,
    ) {
        let projected_opt = self.project_aabb(min, max, view_rot, view_trans, focal_len_px, near_plane);
        if projected_opt.is_none() {
            return;
        }
        let projected = projected_opt.unwrap();

        // Find min/max Y for the scanline loop and furthest distance
        let mut min_y_f = f32::INFINITY;
        let mut max_y_f = f32::NEG_INFINITY;
        let mut max_depth = f32::NEG_INFINITY;

        for &(_px, py, pz) in &projected {
            if py < min_y_f { min_y_f = py; }
            if py > max_y_f { max_y_f = py; }
            if pz > max_depth { max_depth = pz; }
        }

        // Add conservative 1 pixel inset on the Y bounds
        let min_y = (min_y_f.ceil() as i32) + 1;
        let max_y = (max_y_f.floor() as i32) - 1;

        if min_y > max_y {
            return;
        }

        let min_y_clamped = min_y.max(0).min(self.height as i32 - 1);
        let max_y_clamped = max_y.max(0).min(self.height as i32 - 1);

        // We'll calculate min_x and max_x per scanline.
        // The projected cube forms a convex polygon, so we can trace its edges.
        let mut row_min_x = vec![f32::INFINITY; self.height];
        let mut row_max_x = vec![f32::NEG_INFINITY; self.height];

        // 12 edges of the cube
        let edges = [
            (0, 1), (1, 3), (3, 2), (2, 0), // front face
            (4, 5), (5, 7), (7, 6), (6, 4), // back face
            (0, 4), (1, 5), (2, 6), (3, 7), // connecting edges
        ];

        for &(i, j) in &edges {
            let (mut x0, mut y0, _) = projected[i];
            let (mut x1, mut y1, _) = projected[j];

            if y0 > y1 {
                std::mem::swap(&mut x0, &mut x1);
                std::mem::swap(&mut y0, &mut y1);
            }

            // Skip horizontal edges
            if (y1 - y0).abs() < 1e-4 {
                continue;
            }

            let inv_slope = (x1 - x0) / (y1 - y0);

            // Rasterize edge
            let start_y = (y0.ceil() as i32).max(0);
            let end_y = (y1.floor() as i32).min(self.height as i32 - 1);

            for y in start_y..=end_y {
                let x = x0 + (y as f32 - y0) * inv_slope;
                if x < row_min_x[y as usize] { row_min_x[y as usize] = x; }
                if x > row_max_x[y as usize] { row_max_x[y as usize] = x; }
            }
        }

        // Fill buffer
        for y in min_y_clamped..=max_y_clamped {
            let row = y as usize;
            if row_min_x[row] > row_max_x[row] {
                continue;
            }

            // Apply 1-pixel conservative inset on X axis
            let start_x = (row_min_x[row].ceil() as i32) + 1;
            let end_x = (row_max_x[row].floor() as i32) - 1;

            let start_x_clamped = start_x.max(0).min(self.width as i32 - 1);
            let end_x_clamped = end_x.max(0).min(self.width as i32 - 1);

            for x in start_x_clamped..=end_x_clamped {
                let idx = row * self.width + x as usize;
                // We overwrite with max_depth. The depth buffer tracks the CLOSEST occluder.
                // Wait, if we use f32::INFINITY and test if candidate < buffer, we want to store the CLOSEST of all written occluders,
                // but for a single occluder we write its MAX depth so we don't accidentally over-occlude.
                if max_depth < self.buffer[idx] {
                    self.buffer[idx] = max_depth;
                }
            }
        }
    }

    /// Queries if a chunk is occluded (Read)
    pub fn is_occluded(
        &self,
        min: Point3<f32>,
        max: Point3<f32>,
        view_rot: &Matrix3<f32>,
        view_trans: &Vector3<f32>,
        focal_len_px: &Vector2<f32>,
        near_plane: f32,
    ) -> bool {
        let projected_opt = self.project_aabb(min, max, view_rot, view_trans, focal_len_px, near_plane);
        if projected_opt.is_none() {
            // Intersects near plane, assume visible
            return false;
        }
        let projected = projected_opt.unwrap();

        let mut min_x_f = f32::INFINITY;
        let mut max_x_f = f32::NEG_INFINITY;
        let mut min_y_f = f32::INFINITY;
        let mut max_y_f = f32::NEG_INFINITY;
        let mut min_depth = f32::INFINITY;

        for &(px, py, pz) in &projected {
            if px < min_x_f { min_x_f = px; }
            if px > max_x_f { max_x_f = px; }
            if py < min_y_f { min_y_f = py; }
            if py > max_y_f { max_y_f = py; }
            if pz < min_depth { min_depth = pz; } // Closest part of the chunk
        }

        // Apply 1-pixel conservative OUTSET for queries
        let min_x = (min_x_f.floor() as i32) - 1;
        let max_x = (max_x_f.ceil() as i32) + 1;
        let min_y = (min_y_f.floor() as i32) - 1;
        let max_y = (max_y_f.ceil() as i32) + 1;

        // If the expanded box is off-screen entirely, it's NOT occluded by our depth buffer
        // (but might be frustum culled elsewhere, so returning false/visible is safe)
        if max_x < 0 || min_x >= self.width as i32 || max_y < 0 || min_y >= self.height as i32 {
            return false;
        }

        let start_x_clamped = min_x.max(0).min(self.width as i32 - 1);
        let end_x_clamped = max_x.max(0).min(self.width as i32 - 1);
        let start_y_clamped = min_y.max(0).min(self.height as i32 - 1);
        let end_y_clamped = max_y.max(0).min(self.height as i32 - 1);

        // Check if ANY pixel within the footprint has a depth > min_depth
        // If min_depth (closest point of chunk) > buffer value, then the buffer occludes that pixel.
        // For the chunk to be FULLY occluded, it must be occluded at ALL pixels.
        for y in start_y_clamped..=end_y_clamped {
            for x in start_x_clamped..=end_x_clamped {
                let idx = (y as usize) * self.width + (x as usize);

                // If our closest part is in front of the occluder (min_depth < buffer), it's visible.
                if min_depth <= self.buffer[idx] {
                    return false;
                }
            }
        }

        // Fully occluded!
        true
    }
}
