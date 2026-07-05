use crate::splat::{
    Splat4DContainer, Splat4DDelta, Splat4DGop, Splat4DHeader, Splat4DKeyframe,
    Splat4DStaticSection, SplatVertex,
};

pub struct Splat4DEncoder {
    pub pos_bound: f32,
    pub scale_bound: f32,
    pub color_bound: f32,
    pub gop_size: usize,
}

impl Default for Splat4DEncoder {
    fn default() -> Self {
        Self {
            pos_bound: 0.002,
            scale_bound: 0.02,
            color_bound: 4.0 / 255.0,
            gop_size: 60,
        }
    }
}

impl Splat4DEncoder {
    pub fn encode_container(&self, sequences: &[Vec<SplatVertex>]) -> Option<Splat4DContainer> {
        if sequences.is_empty() {
            return None;
        }

        // 1. Identify Static vs. Dynamic Splats
        // For prototype, we assume all splats are dynamic if they appear in any frame.
        // A more advanced implementation would compare first and last frames.
        let dynamic_splats_raw = &sequences[0]; // Assume consistent splat count across frames
        let static_splats = Vec::new(); // Empty static section for now

        // 2. Encode GOPs
        let mut gops = Vec::new();
        for chunk in sequences.chunks(self.gop_size) {
            if let Some(gop) = self.encode_gop(chunk, gops.len() as u32 * self.gop_size as u32) {
                gops.push(gop);
            }
        }

        let header = Splat4DHeader {
            magic: "SP4D".to_string(),
            version: 1,
            pos_bound: self.pos_bound,
            scale_bound: self.scale_bound,
            color_bound: self.color_bound,
            static_count: static_splats.len() as u32,
            dynamic_count: dynamic_splats_raw.len() as u32,
            gop_size: self.gop_size as u32,
        };

        Some(Splat4DContainer {
            header,
            static_section: Splat4DStaticSection {
                splats: static_splats,
            },
            gops,
        })
    }

    pub fn encode_gop(&self, frames: &[Vec<SplatVertex>], start_frame: u32) -> Option<Splat4DGop> {
        if frames.is_empty() {
            return None;
        }

        let keyframe_raw = &frames[0];
        let keyframe = Splat4DKeyframe {
            positions: keyframe_raw
                .iter()
                .map(|s| self.quantize_pos(s.position))
                .collect(),
            rotations: keyframe_raw
                .iter()
                .map(|s| self.quantize_rot(s.rotation))
                .collect(),
            scales: keyframe_raw
                .iter()
                .map(|s| self.quantize_scale(s.scale))
                .collect(),
            colors: keyframe_raw
                .iter()
                .map(|s| self.quantize_color(s.color))
                .collect(),
        };

        let mut delta_frames = Vec::new();
        let mut prev_raw = keyframe_raw;

        for frame_raw in frames.iter().skip(1) {
            let mut motion_deltas = Vec::new();
            let mut rotation_deltas = Vec::new();
            let mut scale_deltas = Vec::new();
            let mut color_deltas = Vec::new();
            let mut active_mask = vec![0u8; (frame_raw.len() + 7) / 8];

            for (i, current) in frame_raw.iter().enumerate() {
                if i >= prev_raw.len() {
                    break;
                }
                let prev = &prev_raw[i];

                // Check if attributes changed significantly (Deadband "hold")
                let pos_diff = [
                    current.position[0] - prev.position[0],
                    current.position[1] - prev.position[1],
                    current.position[2] - prev.position[2],
                ];

                let is_active = pos_diff[0].abs() > self.pos_bound
                    || pos_diff[1].abs() > self.pos_bound
                    || pos_diff[2].abs() > self.pos_bound;

                if is_active {
                    active_mask[i / 8] |= 1 << (i % 8);
                    motion_deltas.push(self.diff_pos(current.position, prev.position));
                    rotation_deltas.push(self.diff_rot(current.rotation, prev.rotation));
                    scale_deltas.push(self.diff_scale(current.scale, prev.scale));
                    color_deltas.push(self.diff_color(current.color, prev.color));
                }
            }

            delta_frames.push(Splat4DDelta {
                active_mask,
                motion_deltas,
                rotation_deltas,
                scale_deltas,
                color_deltas,
            });
            prev_raw = frame_raw;
        }

        Some(Splat4DGop {
            start_frame,
            keyframe,
            delta_frames,
        })
    }

    fn quantize_pos(&self, pos: [f32; 3]) -> [i16; 3] {
        [
            (pos[0] / self.pos_bound) as i16,
            (pos[1] / self.pos_bound) as i16,
            (pos[2] / self.pos_bound) as i16,
        ]
    }

    fn diff_pos(&self, cur: [f32; 3], prev: [f32; 3]) -> [i8; 3] {
        [
            ((cur[0] - prev[0]) / self.pos_bound).clamp(-127.0, 127.0) as i8,
            ((cur[1] - prev[1]) / self.pos_bound).clamp(-127.0, 127.0) as i8,
            ((cur[2] - prev[2]) / self.pos_bound).clamp(-127.0, 127.0) as i8,
        ]
    }

    fn quantize_rot(&self, rot: [f32; 4]) -> [i8; 4] {
        [
            (rot[0] * 127.0) as i8,
            (rot[1] * 127.0) as i8,
            (rot[2] * 127.0) as i8,
            (rot[3] * 127.0) as i8,
        ]
    }

    fn diff_rot(&self, cur: [f32; 4], prev: [f32; 4]) -> [i8; 4] {
        [
            ((cur[0] - prev[0]) * 127.0).clamp(-127.0, 127.0) as i8,
            ((cur[1] - prev[1]) * 127.0).clamp(-127.0, 127.0) as i8,
            ((cur[2] - prev[2]) * 127.0).clamp(-127.0, 127.0) as i8,
            ((cur[3] - prev[3]) * 127.0).clamp(-127.0, 127.0) as i8,
        ]
    }

    fn quantize_scale(&self, scale: [f32; 3]) -> [i8; 3] {
        [
            (scale[0].max(1e-6).ln() / self.scale_bound).clamp(-127.0, 127.0) as i8,
            (scale[1].max(1e-6).ln() / self.scale_bound).clamp(-127.0, 127.0) as i8,
            (scale[2].max(1e-6).ln() / self.scale_bound).clamp(-127.0, 127.0) as i8,
        ]
    }

    fn diff_scale(&self, cur: [f32; 3], prev: [f32; 3]) -> [i8; 3] {
        [
            ((cur[0].max(1e-6).ln() - prev[0].max(1e-6).ln()) / self.scale_bound)
                .clamp(-127.0, 127.0) as i8,
            ((cur[1].max(1e-6).ln() - prev[1].max(1e-6).ln()) / self.scale_bound)
                .clamp(-127.0, 127.0) as i8,
            ((cur[2].max(1e-6).ln() - prev[2].max(1e-6).ln()) / self.scale_bound)
                .clamp(-127.0, 127.0) as i8,
        ]
    }

    fn quantize_color(&self, color: [f32; 4]) -> [u8; 4] {
        [
            (color[0].clamp(0.0, 1.0) * 255.0) as u8,
            (color[1].clamp(0.0, 1.0) * 255.0) as u8,
            (color[2].clamp(0.0, 1.0) * 255.0) as u8,
            (color[3].clamp(0.0, 1.0) * 255.0) as u8,
        ]
    }

    fn diff_color(&self, cur: [f32; 4], prev: [f32; 4]) -> [i8; 4] {
        [
            ((cur[0] - prev[0]) * 255.0).clamp(-127.0, 127.0) as i8,
            ((cur[1] - prev[1]) * 255.0).clamp(-127.0, 127.0) as i8,
            ((cur[2] - prev[2]) * 255.0).clamp(-127.0, 127.0) as i8,
            ((cur[3] - prev[3]) * 255.0).clamp(-127.0, 127.0) as i8,
        ]
    }
}
