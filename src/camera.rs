use std::f64::consts::{PI, TAU};

/// View orientation (radians) and zoom factor.
///
/// [`Camera::rotate`] keeps both angles wrapped to `[-PI, PI)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    yaw: f64,
    pitch: f64,
    zoom: f64,
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
    fn clamp_zoom(zoom: f64) -> f64 {
        zoom.clamp(0.1, 100.0)
    }

    /// Creates a new [`Self`] where `zoom` is clamped to [0.1, 100]
    /// and `yaw` and `pitch` has been folded in `[-PI, PI)`.
    #[must_use]
    pub fn new(yaw: f64, pitch: f64, zoom: f64) -> Self {
        Self {
            yaw: wrap_angle(yaw),
            pitch: wrap_angle(pitch),
            zoom: Self::clamp_zoom(zoom),
        }
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    #[must_use]
    pub fn yaw(&self) -> f64 {
        self.yaw
    }

    #[must_use]
    pub fn pitch(&self) -> f64 {
        self.pitch
    }

    #[must_use]
    pub fn zoom(&self) -> f64 {
        self.zoom
    }

    /// Turn the camera. Both angles are periodic: they wrap into `[-PI, PI)`, so
    /// pitching all the way over the top brings the view back around to where it
    /// started.
    ///
    /// Yaw is applied in screen terms rather than about the fixed world axis.
    /// Past the poles the camera hangs upside-down, and a world-axis yaw then
    /// sweeps the atoms now facing the viewer the other way — pressing "left"
    /// would visibly turn the molecule right. Negating the delta while inverted
    /// keeps "left" meaning left all the way around.
    pub fn rotate(&mut self, yaw_delta: f64, pitch_delta: f64) {
        let yaw_delta = if self.is_inverted() {
            -yaw_delta
        } else {
            yaw_delta
        };
        self.yaw = wrap_angle(self.yaw + yaw_delta);
        self.pitch = wrap_angle(self.pitch + pitch_delta);
    }

    /// Whether the camera currently hangs upside-down, i.e. pitched more than a
    /// quarter turn away from upright, so world "up" projects to screen "down".
    #[must_use]
    pub fn is_inverted(self) -> bool {
        self.pitch.cos() < 0.0
    }

    pub fn zoom_by(&mut self, factor: f64) {
        self.zoom = Camera::clamp_zoom(self.zoom * factor);
    }

    /// Orthographic projection of a world point under the camera. Rotates by yaw
    /// (about vertical Y) then pitch (about horizontal X); the projected (x, y) are
    /// the screen plane and the surviving z is the depth used for shading/occlusion.
    #[must_use]
    pub fn project_point(self, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();

        let x1 = x * cy - z * sy;
        let z1 = x * sy + z * cy;
        let y2 = y * cp - z1 * sp;
        let z2 = y * sp + z1 * cp;
        (x1, y2, z2)
    }
}

/// Fold an angle into `[-PI, PI)`.
fn wrap_angle(angle: f64) -> f64 {
    (angle + PI).rem_euclid(TAU) - PI
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera(yaw: f64, pitch: f64) -> Camera {
        Camera {
            yaw,
            pitch,
            zoom: 1.0,
        }
    }

    #[test]
    fn yaw_wraps_to_an_equivalent_angle() {
        let mut cam = camera(0.0, 0.0);
        cam.rotate(TAU + 0.5, 0.0);

        assert!((cam.yaw() - 0.5).abs() < 1e-12);
        assert!(cam.yaw() >= -PI && cam.yaw() < PI);
    }

    #[test]
    fn yaw_wraps_downwards_too() {
        let mut cam = camera(0.0, 0.0);
        cam.rotate(-TAU - 0.5, 0.0);

        assert!((cam.yaw + 0.5).abs() < 1e-12);
    }

    #[test]
    fn full_yaw_turn_projects_identically() {
        let point = (1.0, 2.0, 3.0);
        let mut cam = camera(0.6, 0.4);
        let before = cam.project_point(point.0, point.1, point.2);

        cam.rotate(TAU, 0.0);
        let after = cam.project_point(point.0, point.1, point.2);

        assert!((before.0 - after.0).abs() < 1e-12);
        assert!((before.1 - after.1).abs() < 1e-12);
        assert!((before.2 - after.2).abs() < 1e-12);
    }

    #[test]
    fn pitch_wraps_and_keeps_tumbling() {
        let mut cam = camera(0.0, 0.0);
        cam.rotate(0.0, TAU + 0.5);

        assert!((cam.pitch - 0.5).abs() < 1e-12);
        assert!(cam.pitch >= -PI && cam.pitch < PI);
    }

    #[test]
    fn a_half_turn_of_pitch_shows_the_backside() {
        let cam = camera(0.0, PI);
        // The atom behind the origin now faces the viewer, and the one in front
        // is occluded — the backside, not a broken depth sort.
        assert!(cam.project_point(0.0, 0.0, -1.0).2 > cam.project_point(0.0, 0.0, 1.0).2);
    }

    #[test]
    fn a_full_pitch_tumble_returns_to_the_start() {
        let point = (1.0, 2.0, 3.0);
        let start = camera(0.6, 0.4);
        let mut cam = start;

        // Step over both poles rather than jumping, so pitch has to wrap all the
        // way round and land back on its start.
        for _ in 0..100 {
            cam.rotate(0.0, TAU / 100.0);
        }

        assert!((cam.pitch - start.pitch).abs() < 1e-12);
        assert!((cam.yaw - start.yaw).abs() < 1e-12);
        let (sx, sy, sz) = start.project_point(point.0, point.1, point.2);
        let (cx, cy, cz) = cam.project_point(point.0, point.1, point.2);
        assert!((sx - cx).abs() < 1e-12 && (sy - cy).abs() < 1e-12 && (sz - cz).abs() < 1e-12);
    }

    #[test]
    fn yaw_turns_the_same_way_at_every_pitch() {
        // Whichever face is towards the viewer, a positive yaw must sweep it in
        // one consistent screen direction; upside-down used to reverse it.
        let mut cam = camera(0.0, 0.0);
        for _ in 0..200 {
            cam.rotate(0.0, 0.1);

            // Track the point currently nearest the viewer along the z axis.
            let front = if cam.project_point(0.0, 0.0, 1.0).2 >= cam.project_point(0.0, 0.0, -1.0).2
            {
                1.0
            } else {
                -1.0
            };
            let before = cam.project_point(0.0, 0.0, front).0;
            let mut turned = cam;
            turned.rotate(0.01, 0.0);
            let after = turned.project_point(0.0, 0.0, front).0;

            assert!(
                after <= before,
                "yaw reversed at pitch {}: {before} -> {after}",
                cam.pitch
            );
        }
    }
}
