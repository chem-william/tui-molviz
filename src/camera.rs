/// View orientation (radians) and zoom factor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    pub yaw: f64,
    pub pitch: f64,
    pub zoom: f64,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            yaw: 0.6,
            pitch: 0.4,
            zoom: 1.3,
        }
    }
}

impl Camera {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn rotate(&mut self, yaw_delta: f64, pitch_delta: f64) {
        self.yaw += yaw_delta;
        self.pitch += pitch_delta;
    }

    pub fn zoom_by(&mut self, factor: f64) {
        self.zoom = (self.zoom * factor).clamp(0.1, 100.0);
    }

    /// Orthographic projection of a world point under the camera. Rotates by yaw
    /// (about vertical Y) then pitch (about horizontal X); the projected (x, y) are
    /// the screen plane and the surviving z is the depth used for shading/occlusion.
    #[must_use]
    pub fn project_point(self, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();

        let x1 = x * cy + z * sy;
        let z1 = -x * sy + z * cy;
        let y2 = y * cp - z1 * sp;
        let z2 = y * sp + z1 * cp;
        (x1, y2, z2)
    }
}
