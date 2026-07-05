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
            gop_size: 30,
        }
    }
}

impl Splat4DEncoder {
    pub fn encode_sequence(
        &self,
        sequence: &[Vec<SplatVertex>],
        static_count: usize,
    ) -> Splat4DContainer {
        let total_frames = sequence.len();
        let dynamic_count = if total_frames > 0 {
            sequence[0].len().saturating_sub(static_count)
        } else {
            0
        };

        let mut gops = Vec::new();
        for i in (0..total_frames).step_by(self.gop_size) {
            let end = (i + self.gop_size).min(total_frames);
            let gop = self.encode_gop(&sequence[i..end], i as u32, dynamic_count);
            gops.push(gop);
        }

        let static_splats = if total_frames > 0 {
            sequence[0][dynamic_count..].to_vec()
        } else {
            Vec::new()
        };

        Splat4DContainer {
            header: Splat4DHeader {
                magic: "SP4D".to_string(),
                version: 1,
                pos_bound: self.pos_bound,
                scale_bound: self.scale_bound,
                color_bound: self.color_bound,
                static_count: static_count as u32,
                dynamic_count: dynamic_count as u32,
                gop_size: self.gop_size as u32,
            },
            static_section: Splat4DStaticSection {
                splats: static_splats,
            },
            gops,
        }
    }

    fn encode_gop(
        &self,
        frames: &[Vec<SplatVertex>],
        start_frame: u32,
        dynamic_count: usize,
    ) -> Splat4DGop {
        let keyframe_data = &frames[0];
        let keyframe = Splat4DKeyframe {
            positions: keyframe_data.iter().take(dynamic_count).map(|s| self.quantize_pos(s.position)).collect(),
            rotations: keyframe_data.iter().take(dynamic_count).map(|s| self.quantize_rot(s.rotation)).collect(),
            scales: keyframe_data.iter().take(dynamic_count).map(|s| self.quantize_scale(s.scale)).collect(),
            colors: keyframe_data.iter().take(dynamic_count).map(|s| self.quantize_color(s.color)).collect(),
            archetype_ids: keyframe_data.iter().take(dynamic_count).map(|s| s.archetype_id as u16).collect(),
            morph_weights: keyframe_data.iter().take(dynamic_count).map(|s| (s.morph_weight * 255.0) as u8).collect(),
        };

        let mut delta_frames = Vec::new();
        let mut prev_frame = keyframe_data;

        for frame in frames.iter().skip(1) {
            let mut active_mask = vec![0u8; (dynamic_count + 7) / 8];
            let mut motion_deltas = Vec::new();
            let mut rotation_deltas = Vec::new();
            let mut scale_deltas = Vec::new();
            let mut color_deltas = Vec::new();
            let mut morph_deltas = Vec::new();

            for i in 0..dynamic_count {
                let curr = &frame[i];
                let prev = &prev_frame[i];

                let d_pos = [
                    ((curr.position[0] - prev.position[0]) / self.pos_bound).round() as i8,
                    ((curr.position[1] - prev.position[1]) / self.pos_bound).round() as i8,
                    ((curr.position[2] - prev.position[2]) / self.pos_bound).round() as i8,
                ];

                let d_rot = [
                    ((curr.rotation[0] - prev.rotation[0]) * 127.0).round() as i8,
                    ((curr.rotation[1] - prev.rotation[1]) * 127.0).round() as i8,
                    ((curr.rotation[2] - prev.rotation[2]) * 127.0).round() as i8,
                    ((curr.rotation[3] - prev.rotation[3]) * 127.0).round() as i8,
                ];

                let d_scale = [
                    ((curr.scale[0].ln() - prev.scale[0].ln()) / self.scale_bound).round() as i8,
                    ((curr.scale[1].ln() - prev.scale[1].ln()) / self.scale_bound).round() as i8,
                    ((curr.scale[2].ln() - prev.scale[2].ln()) / self.scale_bound).round() as i8,
                ];

                let d_color = [
                    ((curr.color[0] - prev.color[0]) * 255.0).round() as i8,
                    ((curr.color[1] - prev.color[1]) * 255.0).round() as i8,
                    ((curr.color[2] - prev.color[2]) * 255.0).round() as i8,
                    ((curr.color[3] - prev.color[3]) * 255.0).round() as i8,
                ];

                let d_morph = ((curr.morph_weight - prev.morph_weight) * 255.0).round() as i8;

                let is_active = d_pos.iter().any(|&d| d != 0)
                    || d_rot.iter().any(|&d| d != 0)
                    || d_scale.iter().any(|&d| d != 0)
                    || d_color.iter().any(|&d| d != 0)
                    || d_morph != 0;

                if is_active {
                    active_mask[i / 8] |= 1 << (i % 8);
                    motion_deltas.push(d_pos);
                    rotation_deltas.push(d_rot);
                    scale_deltas.push(d_scale);
                    color_deltas.push(d_color);
                    morph_deltas.push(d_morph);
                }
            }

            delta_frames.push(Splat4DDelta {
                active_mask,
                motion_deltas,
                rotation_deltas,
                scale_deltas,
                color_deltas,
                morph_deltas,
            });
            prev_frame = frame;
        }

        Splat4DGop {
            start_frame,
            keyframe,
            delta_frames,
        }
    }

    fn quantize_pos(&self, pos: [f32; 3]) -> [i16; 3] {
        [
            (pos[0] / self.pos_bound).round() as i16,
            (pos[1] / self.pos_bound).round() as i16,
            (pos[2] / self.pos_bound).round() as i16,
        ]
    }

    fn quantize_rot(&self, rot: [f32; 4]) -> [i8; 4] {
        [
            (rot[0] * 127.0).round() as i8,
            (rot[1] * 127.0).round() as i8,
            (rot[2] * 127.0).round() as i8,
            (rot[3] * 127.0).round() as i8,
        ]
    }

    fn quantize_scale(&self, scale: [f32; 3]) -> [i8; 3] {
        [
            (scale[0].ln() / self.scale_bound).round() as i8,
            (scale[1].ln() / self.scale_bound).round() as i8,
            (scale[2].ln() / self.scale_bound).round() as i8,
        ]
    }

    fn quantize_color(&self, color: [f32; 4]) -> [u8; 4] {
        [
            (color[0] * 255.0).round() as u8,
            (color[1] * 255.0).round() as u8,
            (color[2] * 255.0).round() as u8,
            (color[3] * 255.0).round() as u8,
        ]
    }
}
