use std::f64::consts::{PI, TAU};

/// View orientation (radians), zoom factor, and screen offset.
///
/// [`Camera::rotate`] keeps both angles wrapped to `[-PI, PI)`, and
/// [`Camera::translate`] shifts the view in the screen plane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    yaw: f64,
    pitch: f64,
    zoom: f64,
    tx: f64,
    ty: f64,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            yaw: 0.6,
            pitch: 0.4,
            zoom: 1.3,
            tx: 0.0,
            ty: 0.0,
        }
    }
}

impl Camera {
    fn clamp_zoom(zoom: f64) -> f64 {
        zoom.clamp(0.1, 100.0)
    }

    /// Creates a new [`Self`] where `zoom` is clamped to [0.1, 100]
    /// and `yaw` and `pitch` has been folded in `[-PI, PI)`.
    ///
    /// The screen offset starts at the origin: the molecule is centered in the
    /// view. Use [`Camera::translate`] to pan it.
    #[must_use]
    pub fn new(yaw: f64, pitch: f64, zoom: f64) -> Self {
        Self {
            yaw: wrap_angle(yaw),
            pitch: wrap_angle(pitch),
            zoom: Self::clamp_zoom(zoom),
            tx: 0.0,
            ty: 0.0,
        }
    }

    /// Restores the default orientation, zoom, and offset.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Clears only the screen offset, keeping orientation and zoom. The
    /// molecule's centroid moves back to the center of the view.
    pub fn recenter(&mut self) {
        self.tx = 0.0;
        self.ty = 0.0;
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

    /// Shifts the view by `dx` right and `dy` up, in the same units as atom
    /// coordinates (Å). The offset lives in the screen plane, applied after
    /// the rotation, so panning and then rotating orbits around the panned
    /// center.
    ///
    /// To pan by terminal cells, convert them with
    /// [`MoleculeCanvas::cell_delta_to_world`](crate::MoleculeCanvas::cell_delta_to_world),
    /// which also flips `dy` for screen rows.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tui_molviz::camera::Camera;
    ///
    /// let mut camera = Camera::default();
    /// camera.translate(0.5, -0.25);
    /// assert_eq!(camera.offset(), (0.5, -0.25));
    ///
    /// // The projection follows the offset in x and y, not depth.
    /// let (x, y, z) = camera.project_point(0.0, 0.0, 0.0);
    /// assert!((x - 0.5).abs() < 1e-12 && (y + 0.25).abs() < 1e-12);
    /// let (_, _, z0) = Camera::default().project_point(0.0, 0.0, 0.0);
    /// assert!((z - z0).abs() < 1e-12);
    /// ```
    pub fn translate(&mut self, dx: f64, dy: f64) {
        self.tx += dx;
        self.ty += dy;
    }

    /// The current screen offset (right, up), in the same units as atom
    /// coordinates (Å). `offset()` of a fresh camera is `(0.0, 0.0)`.
    #[must_use]
    pub fn offset(self) -> (f64, f64) {
        (self.tx, self.ty)
    }

    pub fn zoom_by(&mut self, factor: f64) {
        self.zoom = Camera::clamp_zoom(self.zoom * factor);
    }

    /// Zooms by `factor` while keeping the point under the cursor stationary
    /// on screen. `cursor` is in canvas data coordinates, i.e.
    /// [`project_point`](Self::project_point) output including the offset.
    ///
    /// Zoom rescales the canvas rather than the data coordinates, so the
    /// point's *screen* position is what stays fixed; in data coordinates it
    /// moves by the same factor the zoom applies. Zooming at the canvas
    /// center `(0, 0)` is equivalent to [`zoom_by`](Self::zoom_by).
    ///
    /// When the cursor comes from a terminal cell, convert it with
    /// [`MoleculeCanvas::cell_to_data`](crate::MoleculeCanvas::cell_to_data).
    ///
    /// # Example
    ///
    /// ```rust
    /// use tui_molviz::camera::Camera;
    ///
    /// let mut camera = Camera::new(0.0, 0.0, 1.0);
    /// let point = camera.project_point(1.0, 0.0, 0.0);
    ///
    /// camera.zoom_around(2.0, (point.0, point.1));
    /// // In data coordinates the point halves, keeping its screen position
    /// // under a canvas whose bounds halve with the 2x zoom.
    /// assert_eq!(camera.project_point(1.0, 0.0, 0.0).0, 0.5);
    /// ```
    pub fn zoom_around(&mut self, factor: f64, cursor: (f64, f64)) {
        let factor = factor.max(1e-9);
        let zoom = Self::clamp_zoom(self.zoom * factor);
        // The zoom the view actually applied; the offset math must use this,
        // not the raw factor, or a clamped zoom leaves the cursor drifting.
        let applied = zoom / self.zoom;
        // The world point under the cursor, in the un-offset rotated frame.
        let (px, py) = (cursor.0 - self.tx, cursor.1 - self.ty);
        // A point's screen position is its offset-frame position (point +
        // offset) relative to a canvas whose bounds scale with the zoom, so
        // keeping it fixed divides the offset-frame position by the applied
        // factor.
        self.tx = (px + self.tx) / applied - px;
        self.ty = (py + self.ty) / applied - py;
        self.zoom = zoom;
    }

    /// Orthographic projection of a world point under the camera. Rotates by yaw
    /// (about vertical Y) then pitch (about horizontal X), then shifts by the
    /// screen offset from [`translate`](Self::translate); the projected (x, y) are
    /// the screen plane and the surviving z is the depth used for shading/occlusion.
    ///
    /// The offset is applied after the rotation, so it does not depend on
    /// orientation and never affects depth.
    #[must_use]
    pub fn project_point(self, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();

        let x1 = x * cy - z * sy;
        let z1 = x * sy + z * cy;
        let y2 = y * cp - z1 * sp;
        let z2 = y * sp + z1 * cp;
        (x1 + self.tx, y2 + self.ty, z2)
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
            tx: 0.0,
            ty: 0.0,
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
    fn translate_shifts_projection_in_xy_not_z() {
        let mut cam = camera(0.6, 0.4);
        cam.translate(0.5, -0.25);

        let (x, y, z) = cam.project_point(1.0, 2.0, 3.0);
        let (x0, y0, z0) = camera(0.6, 0.4).project_point(1.0, 2.0, 3.0);

        assert!((x - x0 - 0.5).abs() < 1e-12);
        assert!((y - y0 + 0.25).abs() < 1e-12);
        assert!((z - z0).abs() < 1e-12);
    }

    #[test]
    fn translate_accumulates_and_is_reported() {
        let mut cam = camera(0.0, 0.0);

        assert_eq!(cam.offset(), (0.0, 0.0));

        cam.translate(0.5, 0.25);
        cam.translate(-0.25, 0.75);

        assert_eq!(cam.offset(), (0.25, 1.0));
    }

    #[test]
    fn reset_clears_the_offset() {
        let mut cam = camera(0.0, 0.0);
        cam.translate(1.0, 2.0);
        cam.rotate(0.5, -0.5);
        cam.zoom_by(3.0);

        cam.reset();

        assert_eq!(cam.offset(), (0.0, 0.0));
        assert_eq!(cam, Camera::default());
    }

    #[test]
    fn recenter_clears_only_the_offset() {
        let mut cam = camera(0.3, 0.7);
        cam.translate(1.0, -2.0);
        cam.zoom_by(2.0);

        cam.recenter();

        assert_eq!(cam.offset(), (0.0, 0.0));
        assert!((cam.yaw() - 0.3).abs() < 1e-12);
        assert!((cam.pitch() - 0.7).abs() < 1e-12);
        assert!((cam.zoom() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn pan_then_rotate_keeps_the_panned_center() {
        // Panning moves the molecule, and a later rotation orbits around the
        // panned center: the centroid stays at the offset, wherever it points.
        let mut cam = camera(0.0, 0.0);
        cam.translate(0.8, -0.4);

        for _ in 0..40 {
            cam.rotate(0.1, 0.03);
        }

        // The centroid (origin in world coords) still projects to the offset.
        let (x, y, _) = cam.project_point(0.0, 0.0, 0.0);
        assert!((x - 0.8).abs() < 1e-12);
        assert!((y + 0.4).abs() < 1e-12);
    }

    /// A point's screen position is its data coordinate relative to the
    /// canvas half-width, which shrinks in proportion to the zoom.
    fn screen_fraction(x: f64, half_width: f64) -> f64 {
        (x + half_width) / (2.0 * half_width)
    }

    #[test]
    fn zoom_around_keeps_the_cursor_cell_fixed() {
        let mut cam = camera(0.6, 0.4);
        cam.translate(0.3, -0.2);
        let cursor = cam.project_point(1.0, 2.0, 3.0);
        let (half_width, zoom) = (2.0, cam.zoom());

        cam.zoom_around(2.5, (cursor.0, cursor.1));

        let after = cam.project_point(1.0, 2.0, 3.0);
        let half_width_after = half_width * zoom / cam.zoom();
        assert!(
            (screen_fraction(cursor.0, half_width) - screen_fraction(after.0, half_width_after))
                .abs()
                < 1e-9
        );
        assert!(
            (screen_fraction(cursor.1, half_width) - screen_fraction(after.1, half_width_after))
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn zoom_around_at_origin_is_zoom_by() {
        let mut cam = camera(0.6, 0.4);
        let mut reference = camera(0.6, 0.4);

        cam.zoom_around(2.5, (0.0, 0.0));
        reference.zoom_by(2.5);

        assert_eq!(cam, reference);
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
