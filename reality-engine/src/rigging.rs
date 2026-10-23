use cgmath::{Matrix4, Quaternion, Vector3, SquareMatrix, One, Zero};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bone {
    pub name: String,
    pub parent_index: Option<usize>,

    // Local transform
    pub local_translation: Vector3<f32>,
    pub local_rotation: Quaternion<f32>,
    pub local_scale: Vector3<f32>,

    // The inverse bind matrix (to transform vertices from model space to bone space)
    // For programmatic rigging from scratch, this can often be identity if we define vertices in bone space.
    pub inverse_bind_matrix: Matrix4<f32>,
}

impl Bone {
    pub fn new(name: &str, parent_index: Option<usize>) -> Self {
        Self {
            name: name.to_string(),
            parent_index,
            local_translation: Vector3::zero(),
            local_rotation: Quaternion::one(),
            local_scale: Vector3::new(1.0, 1.0, 1.0),
            inverse_bind_matrix: Matrix4::identity(),
        }
    }

    pub fn local_matrix(&self) -> Matrix4<f32> {
        let t = Matrix4::from_translation(self.local_translation);
        let r = Matrix4::from(self.local_rotation);
        let s = Matrix4::from_nonuniform_scale(self.local_scale.x, self.local_scale.y, self.local_scale.z);
        t * r * s
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skeleton {
    pub bones: Vec<Bone>,
}

impl Skeleton {
    pub fn new() -> Self {
        Self { bones: Vec::new() }
    }

    pub fn add_bone(&mut self, bone: Bone) -> usize {
        let idx = self.bones.len();
        self.bones.push(bone);
        idx
    }

    /// Evaluates the global transform matrix for each bone.
    /// The returned vector has the same length as the number of bones.
    pub fn evaluate_global_transforms(&self) -> Vec<Matrix4<f32>> {
        let mut globals = Vec::with_capacity(self.bones.len());

        for (i, bone) in self.bones.iter().enumerate() {
            let local = bone.local_matrix();
            if let Some(parent_idx) = bone.parent_index {
                // Ensure parent transforms are calculated (requires bones to be topologically sorted,
                // i.e. parent_idx < i)
                if parent_idx < i {
                    globals.push(globals[parent_idx] * local);
                } else {
                    // Fallback if not sorted correctly
                    globals.push(local);
                }
            } else {
                globals.push(local);
            }
        }

        globals
    }
}
