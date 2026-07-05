use crate::splat::{Splat4DContainer, Splat4DGop, SplatVertex};

pub struct Splat4DDecoder {
    pub pos_bound: f32,
    pub scale_bound: f32,
    pub color_bound: f32,
}

impl Default for Splat4DDecoder {
    fn default() -> Self {
        Self {
            pos_bound: 0.002,
            scale_bound: 0.02,
            color_bound: 4.0 / 255.0,
        }
    }
}

impl Splat4DDecoder {
    pub fn decode_container(&self, container: &Splat4DContainer) -> Vec<Vec<SplatVertex>> {
        let mut all_frames = Vec::new();

        for gop in &container.gops {
            let mut gop_frames = self.decode_gop(gop, container.header.dynamic_count as usize);

            for frame in &mut gop_frames {
                frame.extend(container.static_section.splats.iter().cloned());
            }

            all_frames.extend(gop_frames);
        }

        all_frames
    }

    pub fn decode_gop(&self, gop: &Splat4DGop, dynamic_count: usize) -> Vec<Vec<SplatVertex>> {
        let mut frames = Vec::new();

        let mut current_frame = Vec::with_capacity(dynamic_count);
        for i in 0..dynamic_count {
            let pos = self.dequantize_pos(gop.keyframe.positions[i]);
            let rot = self.dequantize_rot(gop.keyframe.rotations[i]);
            let scale = self.dequantize_scale(gop.keyframe.scales[i]);
            let color = self.dequantize_color(gop.keyframe.colors[i]);
            let archetype_id = gop.keyframe.archetype_ids[i] as u32;
            let morph_weight = gop.keyframe.morph_weights[i] as f32 / 255.0;

            current_frame.push(SplatVertex {
                position: pos,
                rotation: rot,
                scale,
                color,
                previous_position: pos,
                archetype_id,
                target_archetype_id: archetype_id,
                morph_weight,
            });
        }
        frames.push(current_frame.clone());

        for delta in &gop.delta_frames {
            let mut next_frame = current_frame.clone();
            let mut delta_idx = 0;

            for i in 0..dynamic_count {
                let is_active = (delta.active_mask[i / 8] & (1 << (i % 8))) != 0;

                if is_active {
                    let prev = &current_frame[i];

                    let pos = self.apply_delta_pos(prev.position, delta.motion_deltas[delta_idx]);
                    let rot = self.apply_delta_rot(prev.rotation, delta.rotation_deltas[delta_idx]);
                    let scale = self.apply_delta_scale(prev.scale, delta.scale_deltas[delta_idx]);
                    let color = self.apply_delta_color(prev.color, delta.color_deltas[delta_idx]);
                    let morph_weight = (prev.morph_weight + delta.morph_deltas[delta_idx] as f32 / 255.0).clamp(0.0, 1.0);

                    next_frame[i] = SplatVertex {
                        position: pos,
                        rotation: rot,
                        scale,
                        color,
                        previous_position: prev.position,
                        archetype_id: prev.archetype_id,
                        target_archetype_id: prev.target_archetype_id,
                        morph_weight,
                    };
                    delta_idx += 1;
                } else {
                    next_frame[i].previous_position = current_frame[i].position;
                }
            }
            frames.push(next_frame.clone());
            current_frame = next_frame;
        }

        frames
    }

    fn dequantize_pos(&self, qpos: [i16; 3]) -> [f32; 3] {
        [
            qpos[0] as f32 * self.pos_bound,
            qpos[1] as f32 * self.pos_bound,
            qpos[2] as f32 * self.pos_bound,
        ]
    }

    fn apply_delta_pos(&self, prev: [f32; 3], delta: [i8; 3]) -> [f32; 3] {
        [
            prev[0] + delta[0] as f32 * self.pos_bound,
            prev[1] + delta[1] as f32 * self.pos_bound,
            prev[2] + delta[2] as f32 * self.pos_bound,
        ]
    }

    fn dequantize_rot(&self, qrot: [i8; 4]) -> [f32; 4] {
        [
            qrot[0] as f32 / 127.0,
            qrot[1] as f32 / 127.0,
            qrot[2] as f32 / 127.0,
            qrot[3] as f32 / 127.0,
        ]
    }

    fn apply_delta_rot(&self, prev: [f32; 4], delta: [i8; 4]) -> [f32; 4] {
        [
            (prev[0] + delta[0] as f32 / 127.0).clamp(-1.0, 1.0),
            (prev[1] + delta[1] as f32 / 127.0).clamp(-1.0, 1.0),
            (prev[2] + delta[2] as f32 / 127.0).clamp(-1.0, 1.0),
            (prev[3] + delta[3] as f32 / 127.0).clamp(-1.0, 1.0),
        ]
    }

    fn dequantize_scale(&self, qscale: [i8; 3]) -> [f32; 3] {
        [
            (qscale[0] as f32 * self.scale_bound).exp(),
            (qscale[1] as f32 * self.scale_bound).exp(),
            (qscale[2] as f32 * self.scale_bound).exp(),
        ]
    }

    fn apply_delta_scale(&self, prev: [f32; 3], delta: [i8; 3]) -> [f32; 3] {
        [
            prev[0] * (delta[0] as f32 * self.scale_bound).exp(),
            prev[1] * (delta[1] as f32 * self.scale_bound).exp(),
            prev[2] * (delta[2] as f32 * self.scale_bound).exp(),
        ]
    }

    fn dequantize_color(&self, qcolor: [u8; 4]) -> [f32; 4] {
        [
            qcolor[0] as f32 / 255.0,
            qcolor[1] as f32 / 255.0,
            qcolor[2] as f32 / 255.0,
            qcolor[3] as f32 / 255.0,
        ]
    }

    fn apply_delta_color(&self, prev: [f32; 4], delta: [i8; 4]) -> [f32; 4] {
        [
            (prev[0] + delta[0] as f32 / 255.0).clamp(0.0, 1.0),
            (prev[1] + delta[1] as f32 / 255.0).clamp(0.0, 1.0),
            (prev[2] + delta[2] as f32 / 255.0).clamp(0.0, 1.0),
            (prev[3] + delta[3] as f32 / 255.0).clamp(0.0, 1.0),
        ]
    }
}
