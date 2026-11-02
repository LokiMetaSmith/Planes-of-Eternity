#[cfg(target_arch = "wasm32")]
use web_sys::{AudioContext, AudioNode, OscillatorType};

pub struct AudioManager {
    #[cfg(target_arch = "wasm32")]
    ctx: Option<AudioContext>,
}

impl Default for AudioManager {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioManager {
    pub fn new() -> Self {
        #[cfg(target_arch = "wasm32")]
        {
            let ctx = AudioContext::new().ok();

            // Expose to window for UI resume hook
            if let Some(ref c) = ctx {
                if let Some(window) = web_sys::window() {
                    let _ = js_sys::Reflect::set(&window, &wasm_bindgen::JsValue::from_str("audioContext"), c);
                }
            }

            Self { ctx }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self {}
        }
    }

    pub fn resume_context(&self) {
        #[cfg(target_arch = "wasm32")]
        if let Some(ctx) = &self.ctx {
            if ctx.state() == web_sys::AudioContextState::Suspended {
                let _ = ctx.resume();
            }
        }
    }

    pub fn play_hover(&self) {
        #[cfg(target_arch = "wasm32")]
        self.play_tone(880.0, 0.05, "sine", 0.05);
    }

    pub fn play_cast(&self) {
        #[cfg(target_arch = "wasm32")]
        {
            // Simple major triad arpeggio
            let now = self.current_time();
            self.play_tone_at(440.0, 0.3, "sine", 0.1, now);
            self.play_tone_at(554.37, 0.3, "sine", 0.1, now + 0.05); // C#
            self.play_tone_at(659.25, 0.3, "sine", 0.1, now + 0.1); // E
            self.play_tone_at(880.0, 0.5, "sine", 0.1, now + 0.15); // A
        }
    }

    pub fn play_reduce(&self) {
        #[cfg(target_arch = "wasm32")]
        self.play_tone(220.0, 0.1, "triangle", 0.1);
    }

    pub fn play_impact(&self) {
        #[cfg(target_arch = "wasm32")]
        self.play_tone(100.0, 0.2, "sawtooth", 0.2);
    }

    #[cfg(target_arch = "wasm32")]
    fn current_time(&self) -> f64 {
        if let Some(ctx) = &self.ctx {
            ctx.current_time()
        } else {
            0.0
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn play_tone(&self, freq: f32, duration: f64, type_: &str, volume: f32) {
        self.play_tone_at(freq, duration, type_, volume, self.current_time());
    }

    #[cfg(target_arch = "wasm32")]
    fn play_tone_at(&self, freq: f32, duration: f64, type_: &str, volume: f32, start_time: f64) {
        if let Some(ctx) = &self.ctx {
            if let Ok(osc) = ctx.create_oscillator() {
                if let Ok(gain) = ctx.create_gain() {
                    osc.set_type(match type_ {
                        "sine" => OscillatorType::Sine,
                        "square" => OscillatorType::Square,
                        "sawtooth" => OscillatorType::Sawtooth,
                        "triangle" => OscillatorType::Triangle,
                        _ => OscillatorType::Sine,
                    });
                    osc.frequency().set_value(freq);

                    let _ = osc.connect_with_audio_node(gain.as_ref() as &AudioNode);
                    let _ = gain.connect_with_audio_node(ctx.destination().as_ref() as &AudioNode);

                    // Envelope
                    // Attack
                    let _ = gain.gain().set_value_at_time(0.0, start_time);
                    let _ = gain
                        .gain()
                        .linear_ramp_to_value_at_time(volume, start_time + 0.01);
                    // Decay/Release
                    let _ = gain
                        .gain()
                        .exponential_ramp_to_value_at_time(0.001, start_time + duration);

                    let _ = osc.start_with_when(start_time);
                    let _ = osc.stop_with_when(start_time + duration);
                }
            }
        }
    }

    /// Plays synthesized parametric waveform buffer from SfxGenie
    pub fn play_sfx_buffer(&self, buffer: &[f32], sample_rate: f32) {
        #[cfg(target_arch = "wasm32")]
        if let Some(ctx) = &self.ctx {
            if let Ok(audio_buf) = ctx.create_buffer(1, buffer.len() as u32, sample_rate) {
                let mut buf_copy = buffer.to_vec();
                let _ = audio_buf.copy_to_channel(&mut buf_copy, 0);
                if let Ok(source) = ctx.create_buffer_source() {
                    source.set_buffer(Some(&audio_buf));
                    let _ = source.connect_with_audio_node(ctx.destination().as_ref() as &AudioNode);
                    let _ = source.start();
                }
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = (buffer, sample_rate);
        }
    }

    /// Updates the 3D HRTF AudioListener orientation & position for WebXR headsets
    pub fn update_listener_pose(&self, pos: [f32; 3], forward: [f32; 3], up: [f32; 3]) {
        #[cfg(target_arch = "wasm32")]
        if let Some(ctx) = &self.ctx {
            let listener = ctx.listener();
            listener.set_position(pos[0] as f64, pos[1] as f64, pos[2] as f64);
            listener.set_orientation(
                forward[0] as f64, forward[1] as f64, forward[2] as f64,
                up[0] as f64, up[1] as f64, up[2] as f64,
            );
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = (pos, forward, up);
        }
    }

    /// Plays a 3D HRTF-spatialized audio buffer emitted from an anomaly or splat coordinate
    pub fn play_spatial_sfx(&self, buffer: &[f32], sample_rate: f32, source_pos: [f32; 3]) {
        #[cfg(target_arch = "wasm32")]
        if let Some(ctx) = &self.ctx {
            if let Ok(audio_buf) = ctx.create_buffer(1, buffer.len() as u32, sample_rate) {
                let mut buf_copy = buffer.to_vec();
                let _ = audio_buf.copy_to_channel(&mut buf_copy, 0);

                if let (Ok(source), Ok(panner)) = (ctx.create_buffer_source(), ctx.create_panner()) {
                    source.set_buffer(Some(&audio_buf));
                    panner.set_panning_model(web_sys::PanningModelType::Hrtf);
                    panner.set_distance_model(web_sys::DistanceModelType::Inverse);
                    panner.set_ref_distance(1.0);
                    panner.set_max_distance(50.0);
                    panner.set_rolloff_factor(1.5);
                    panner.set_position(source_pos[0] as f64, source_pos[1] as f64, source_pos[2] as f64);

                    let _ = source.connect_with_audio_node(panner.as_ref() as &AudioNode);
                    let _ = panner.connect_with_audio_node(ctx.destination().as_ref() as &AudioNode);
                    let _ = source.start();
                }
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = (buffer, sample_rate, source_pos);
        }
    }

    /// Plays stereo multi-channel PCM audio packet from Audio-DiT co-diffuser
    pub fn play_audio_dit_packet(&self, left_channel: &[f32], right_channel: &[f32], sample_rate: u32) {
        #[cfg(target_arch = "wasm32")]
        if let Some(ctx) = &self.ctx {
            let num_frames = left_channel.len().min(right_channel.len()) as u32;
            if num_frames == 0 {
                return;
            }
            if let Ok(audio_buf) = ctx.create_buffer(2, num_frames, sample_rate as f32) {
                let mut left_copy = left_channel[..num_frames as usize].to_vec();
                let mut right_copy = right_channel[..num_frames as usize].to_vec();
                let _ = audio_buf.copy_to_channel(&mut left_copy, 0);
                let _ = audio_buf.copy_to_channel(&mut right_copy, 1);

                if let Ok(source) = ctx.create_buffer_source() {
                    source.set_buffer(Some(&audio_buf));
                    let _ = source.connect_with_audio_node(ctx.destination().as_ref() as &AudioNode);
                    let _ = source.start();
                }
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = (left_channel, right_channel, sample_rate);
        }
    }
}
