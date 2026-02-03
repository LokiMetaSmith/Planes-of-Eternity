#[cfg(target_arch = "wasm32")]
use web_sys::{AudioContext, AudioNode, OscillatorType};

pub struct AudioManager {
    #[cfg(target_arch = "wasm32")]
    ctx: Option<AudioContext>,
}

impl AudioManager {
    pub fn new() -> Self {
        #[cfg(target_arch = "wasm32")]
        {
            let ctx = AudioContext::new().ok();
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
                        "sine" => web_sys::OscillatorType::Sine,
                        "square" => web_sys::OscillatorType::Square,
                        "sawtooth" => web_sys::OscillatorType::Sawtooth,
                        "triangle" => web_sys::OscillatorType::Triangle,
                        _ => web_sys::OscillatorType::Sine,
                    });
                    osc.frequency().set_value(freq);

                    let _ = osc.connect_with_audio_node(&gain);
                    let _ = gain.connect_with_audio_node(&ctx.destination());

                    // Envelope
                    // Attack
                    let _ = gain.gain().set_value_at_time(0.0, start_time);
                    let _ = gain.gain().linear_ramp_to_value_at_time(volume, start_time + 0.01);
                    // Decay/Release
                    let _ = gain.gain().exponential_ramp_to_value_at_time(0.001, start_time + duration);

                    let _ = osc.start_with_when(start_time);
                    let _ = osc.stop_with_when(start_time + duration);
                }
            }
        }
    }
}
