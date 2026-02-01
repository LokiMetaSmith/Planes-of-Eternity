use cgmath::{MetricSpace, Point3};
use serde::{Serialize, Deserialize};
use crate::reality_types::{BlendResult, RealitySignature};
use uuid::Uuid;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RealityProjector {
    pub location: Point3<f32>,
    pub reality_signature: RealitySignature,
    #[serde(default = "default_uuid")]
    pub uuid: String,
    #[serde(default = "default_timestamp")]
    pub last_updated: u64,
}

fn default_uuid() -> String {
    Uuid::new_v4().to_string()
}

fn default_timestamp() -> u64 {
    0
}

pub fn get_current_timestamp() -> u64 {
    #[cfg(target_arch = "wasm32")]
    {
        js_sys::Date::now() as u64
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::time::SystemTime;
        SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default().as_millis() as u64
    }
}

impl RealityProjector {
    pub fn new(location: Point3<f32>, signature: RealitySignature) -> Self {
        Self {
            location,
            reality_signature: signature,
            uuid: Uuid::new_v4().to_string(),
            last_updated: get_current_timestamp(),
        }
    }

    pub fn get_blend_weight_at_location(&self, location: Point3<f32>, rival: Option<&RealityProjector>) -> f32 {
        let rival_ref = match rival {
            Some(r) => r,
            None => return 1.0,
        };

        let dist_a = self.location.distance(location).max(1.0);
        let dist_b = rival_ref.location.distance(location).max(1.0);

        // Note: Legacy C++ implementation uses Fidelity / Distance and ignores InfluenceRadius for strength calculation.
        // We maintain this behavior for parity, though visual debugs in C++ used the radius.
        let strength_a = self.reality_signature.fidelity / dist_a;
        let strength_b = rival_ref.reality_signature.fidelity / dist_b;

        const KINDA_SMALL_NUMBER: f32 = 1e-4;

        if strength_a + strength_b <= KINDA_SMALL_NUMBER {
            return 0.5;
        }

        // Use standard blending formula to avoid singularities at equal strength
        strength_a / (strength_a + strength_b)
    }

    pub fn calculate_reality_at_point(&self, point: Point3<f32>, rival: Option<&RealityProjector>) -> BlendResult {
        let mut result = BlendResult::default();

        let rival_ref = match rival {
            Some(r) => r,
            None => {
                result.dominant_archetype = self.reality_signature.active_style.archetype;
                result.blend_alpha = 0.0;
                return result;
            }
        };

        let dist_a = self.location.distance(point).max(1.0);
        let dist_b = rival_ref.location.distance(point).max(1.0);

        let strength_a = self.reality_signature.fidelity / dist_a;
        let strength_b = rival_ref.reality_signature.fidelity / dist_b;

        result.total_strength = strength_a + strength_b;

        const KINDA_SMALL_NUMBER: f32 = 1e-4;

        if strength_a >= strength_b {
            result.dominant_archetype = self.reality_signature.active_style.archetype;
            if strength_a + strength_b > KINDA_SMALL_NUMBER {
                // Use smooth blending: weak / (strong + weak)
                // At equal strength, this becomes 0.5
                result.blend_alpha = strength_b / (strength_a + strength_b);
            } else {
                result.blend_alpha = 0.0;
            }
        } else {
            result.dominant_archetype = rival_ref.reality_signature.active_style.archetype;
            if strength_a + strength_b > KINDA_SMALL_NUMBER {
                result.blend_alpha = strength_a / (strength_a + strength_b);
            } else {
                result.blend_alpha = 0.0;
            }
        }

        if self.reality_signature.active_style.archetype != rival_ref.reality_signature.active_style.archetype {
            result.is_conflict = true;
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reality_types::RealityArchetype;

    #[test]
    fn test_calculate_reality_at_point() {
        let mut sig_a = RealitySignature::default();
        sig_a.active_style.archetype = RealityArchetype::Fantasy;
        sig_a.fidelity = 100.0;

        let mut sig_b = RealitySignature::default();
        sig_b.active_style.archetype = RealityArchetype::SciFi;
        sig_b.fidelity = 100.0;

        let proj_a = RealityProjector::new(Point3::new(0.0, 0.0, 0.0), sig_a);
        let proj_b = RealityProjector::new(Point3::new(10.0, 0.0, 0.0), sig_b);

        // Point closer to A
        let result = proj_a.calculate_reality_at_point(Point3::new(2.0, 0.0, 0.0), Some(&proj_b));
        assert_eq!(result.dominant_archetype, RealityArchetype::Fantasy);
        assert!(result.is_conflict);

        // Point closer to B
        let result = proj_a.calculate_reality_at_point(Point3::new(8.0, 0.0, 0.0), Some(&proj_b));
        assert_eq!(result.dominant_archetype, RealityArchetype::SciFi);
        assert!(result.is_conflict);

        // Midpoint
        let result = proj_a.calculate_reality_at_point(Point3::new(5.0, 0.0, 0.0), Some(&proj_b));
        // Equal strength, code says StrengthA >= StrengthB -> A wins
        assert_eq!(result.dominant_archetype, RealityArchetype::Fantasy);
        assert!( (result.blend_alpha - 0.5).abs() < 1e-4 ); // Blend should be 0.5 at equal strength

        // Strength A = 100/5 = 20. Strength B = 100/5 = 20. Total = 40.
        assert!( (result.total_strength - 40.0).abs() < 1e-4 );
    }

    #[test]
    fn test_get_blend_weight() {
        let mut sig_a = RealitySignature::default();
        sig_a.fidelity = 100.0;
        let mut sig_b = RealitySignature::default();
        sig_b.fidelity = 100.0;

        let proj_a = RealityProjector::new(Point3::new(0.0, 0.0, 0.0), sig_a);
        let proj_b = RealityProjector::new(Point3::new(10.0, 0.0, 0.0), sig_b);

        // At A's location, weight should be 1.0 (actually slightly less because B has some influence)
        // Distance A = 1.0 (clamped), Strength A = 100.
        // Distance B = 10.0, Strength B = 10.
        // A wins.
        // Strength A = 100. Strength B = 10.
        // Weight = 100 / 110 = 0.90909...
        let weight = proj_a.get_blend_weight_at_location(Point3::new(0.0, 0.0, 0.0), Some(&proj_b));
        assert!((weight - (100.0 / 110.0)).abs() < 1e-4);
    }

    #[test]
    fn test_partial_eq() {
        let sig = RealitySignature::default();
        let proj1 = RealityProjector::new(Point3::new(0.0, 0.0, 0.0), sig.clone());
        let mut proj2 = RealityProjector::new(Point3::new(0.0, 0.0, 0.0), sig.clone());
        let proj3 = RealityProjector::new(Point3::new(1.0, 0.0, 0.0), sig.clone());

        // Because of unique UUIDs, these should be NE
        assert_ne!(proj1, proj2);

        // proj3 is definitely different location
        assert_ne!(proj1, proj3);

        // Force equality
        proj2.uuid = proj1.uuid.clone();
        proj2.last_updated = proj1.last_updated;
        assert_eq!(proj1, proj2);
    }
}
