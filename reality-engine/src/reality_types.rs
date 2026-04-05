use serde::{Deserialize, Serialize};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default, Hash, Serialize, Deserialize)]
pub enum RealityArchetype {
    #[default]
    Void, // Default/Empty
    Fantasy,     // High Fantasy
    SciFi,       // Cyber Punk
    Horror,      // Eldritch Horror
    Toon,        // Toon Logic
    HyperNature, // Procedural Nature
    Genie,       // Generative Dream
    Glitch,      // Digital Distortion
    Steampunk,   // Brass and Steam
    Vaporwave,   // Synthwave / Outrun
    Noir,        // Monochrome, High Contrast, Rain
    CyberSpace,  // Matrix / Digital Grid
    Dream,       // Soft Pastel Clouds
    ObraDinn,    // 1-bit Dithered Sphere Mapping
    SolarPunk,   // High-tech harmony with nature
    Biopunk,     // Organic, fleshy, mutated structures
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RealityStyle {
    pub archetype: RealityArchetype,
    pub sub_theme: String, // e.g., "Necromancer_Castle"
    pub seed: i32,         // The deterministic random number for WFC generation

    // Generative Parameters (Stable Diffusion-like control)
    pub roughness: f32,  // High frequency noise intensity
    pub scale: f32,      // Frequency of the noise (Feature size)
    pub distortion: f32, // Domain warping intensity
}

impl Default for RealityStyle {
    fn default() -> Self {
        Self {
            archetype: RealityArchetype::default(),
            sub_theme: String::new(),
            seed: 0,
            roughness: 0.5,
            scale: 1.0,
            distortion: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct RealityInjection {
    pub injection_id: String,
    // Add other properties as needed
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RealitySignature {
    // The visual look the player is projecting
    pub active_style: RealityStyle,

    // The "Power" of their computer/magic.
    // Higher number = Higher Poly Count, Better Physics, Harder to overwite.
    pub fidelity: f32,

    // The radius of their influence bubble in world units (cm)
    pub influence_radius: f32,

    // A list of "Injections" (e.g., The Sci-Fi Turret inside a Fantasy World)
    pub active_injections: Vec<RealityInjection>,
}

impl Default for RealitySignature {
    fn default() -> Self {
        Self {
            active_style: RealityStyle::default(),
            fidelity: 100.0,
            influence_radius: 1000.0,
            active_injections: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlendResult {
    pub dominant_archetype: RealityArchetype, // Who won?
    pub blend_alpha: f32,                     // 0.0 to 1.0 (How much "Bleed" is happening?)
    pub is_conflict: bool, // True if genres are opposites (e.g., SciFi vs Fantasy)
    pub total_strength: f32, // Total signal strength at this point
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DroppedItem {
    pub id: String,
    pub position: cgmath::Point3<f32>,
    pub velocity: cgmath::Vector3<f32>,
    pub scale: f32,
    pub color: [f32; 4],
    #[serde(default)]
    pub voxels: Vec<u8>, // Local voxel grid representing the item
    #[serde(default)]
    pub size: [usize; 3], // Dimensions of the local voxel grid
}

impl DroppedItem {
    pub fn new_cube(
        id: String,
        position: cgmath::Point3<f32>,
        velocity: cgmath::Vector3<f32>,
        scale: f32,
        color: [f32; 4],
        size: usize,
    ) -> Self {
        let volume = size * size * size;
        let mut voxels = vec![0; volume];

        // Fill as a solid cube initially (ID 1 = solid)
        for i in 0..volume {
            voxels[i] = 1;
        }

        Self {
            id,
            position,
            velocity,
            scale,
            color,
            voxels,
            size: [size, size, size],
        }
    }
}

impl Default for BlendResult {
    fn default() -> Self {
        Self {
            dominant_archetype: RealityArchetype::Void,
            blend_alpha: 0.0,
            is_conflict: false,
            total_strength: 0.0,
        }
    }
}
