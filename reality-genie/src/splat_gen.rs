pub trait SplatGenerator {
    fn generate_splats_from_prompt(&self, prompt: &str) -> Vec<[f32; 14]>;
}

pub struct DummySplatGenerator;

impl DummySplatGenerator {
    pub fn new() -> Self {
        Self
    }
}

impl SplatGenerator for DummySplatGenerator {
    fn generate_splats_from_prompt(&self, prompt: &str) -> Vec<[f32; 14]> {
        let mut splats = Vec::new();
        let color = if prompt.to_lowercase().contains("fire") {
            [1.0, 0.2, 0.0, 0.8]
        } else if prompt.to_lowercase().contains("water") {
            [0.1, 0.3, 1.0, 0.6]
        } else {
            [0.8, 0.8, 0.8, 0.9]
        };

        for i in 0..10 {
            splats.push([
                (i as f32) * 0.1, 0.5, (i as f32) * 0.1,
                0.0, 0.0, 0.0, 1.0,
                0.1, 0.1, 0.1,
                color[0], color[1], color[2], color[3]
            ]);
        }
        splats
    }
}
