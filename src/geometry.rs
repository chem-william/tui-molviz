//! Scalar and vector maths on raw `[f64; 3]` atom coordinates.
//!
//! The crate deliberately carries no linear-algebra dependency — these few
//! helpers are everything the visualizer and the measurements need. They live
//! together so that interatomic distance is computed in exactly one place.

use std::f64::consts::{PI, TAU};

/// Fold an angle into `[-PI, PI)`.
pub(crate) fn wrap_angle(angle: f64) -> f64 {
    (angle + PI).rem_euclid(TAU) - PI
}

/// Below this length (Å) a vector is treated as having no direction rather than
/// a very short one. Normalizing it would divide by ~zero and leak `NaN` into
/// the canvas, so callers return `None` instead.
pub(crate) const DEGENERATE_LENGTH: f64 = 1e-9;

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub(crate) fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

pub(crate) fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm(sub(a, b))
}

/// The component of `v` perpendicular to `axis`, which must be a unit vector.
fn reject(v: [f64; 3], axis: [f64; 3]) -> [f64; 3] {
    let k = dot(v, axis);
    [v[0] - k * axis[0], v[1] - k * axis[1], v[2] - k * axis[2]]
}

/// The angle at `vertex` between the arms reaching `a` and `c`, in radians over
/// `[0, PI]`.
///
/// `None` when either arm is shorter than [`DEGENERATE_LENGTH`] — an atom sits
/// on the vertex, so its arm points nowhere. Collinear atoms are *not*
/// degenerate: they are a perfectly good straight angle.
pub(crate) fn angle(a: [f64; 3], vertex: [f64; 3], c: [f64; 3]) -> Option<f64> {
    let (u, v) = (sub(a, vertex), sub(c, vertex));
    let (lu, lv) = (norm(u), norm(v));
    if lu < DEGENERATE_LENGTH || lv < DEGENERATE_LENGTH {
        return None;
    }

    // Rounding can push the cosine a hair outside [-1, 1] for (anti)parallel
    // arms, where `acos` returns NaN. Clamping keeps an exactly straight angle
    // reading as PI instead of poisoning every value derived from it.
    Some((dot(u, v) / (lu * lv)).clamp(-1.0, 1.0).acos())
}

/// The signed dihedral of `a`-`b`-`c`-`d` about the `b`–`c` axis, in radians
/// over `(-PI, PI]`, following the IUPAC sign convention.
///
/// `None` when the axis is degenerate, or when either terminal atom lies on it
/// (`a`-`b`-`c` or `b`-`c`-`d` collinear) — there is then no plane to measure
/// the torsion between.
pub(crate) fn dihedral(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> Option<f64> {
    let axis = sub(c, b);
    let len = norm(axis);
    if len < DEGENERATE_LENGTH {
        return None;
    }
    let axis = [axis[0] / len, axis[1] / len, axis[2] / len];

    // Strip each terminal bond of its component along the axis; what is left is
    // its projection into the plane the torsion is read in.
    let v = reject(sub(a, b), axis);
    let w = reject(sub(d, c), axis);
    if norm(v) < DEGENERATE_LENGTH || norm(w) < DEGENERATE_LENGTH {
        return None;
    }

    // `atan2` of the two projections is signed and needs no clamping: `axis × v`
    // completes a right-handed frame with `v`, so the arguments are the sine and
    // cosine of the torsion scaled by the same positive factor.
    Some(dot(cross(axis, v), w).atan2(dot(v, w)))
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{FRAC_PI_2, PI};

    use super::*;

    /// A tolerance for values that pass through a trig round trip.
    const EPS: f64 = 1e-12;

    fn assert_close(actual: f64, expected: f64, what: &str) {
        assert!(
            (actual - expected).abs() < 1e-9,
            "{what}: expected {expected}, got {actual}"
        );
    }

    #[test]
    fn distance_is_the_euclidean_norm_of_the_separation() {
        assert_close(
            distance([0.0, 0.0, 0.0], [3.0, 4.0, 0.0]),
            5.0,
            "3-4-5 triangle",
        );
        assert_eq!(distance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 0.0);
    }

    #[test]
    fn angle_measures_the_arms_at_the_vertex() {
        let right = angle([1.0, 0.0, 0.0], [0.0; 3], [0.0, 1.0, 0.0]).unwrap();
        assert_close(right, FRAC_PI_2, "right angle");

        // Water's H–O–H, with the oxygen as the vertex.
        let water = angle(
            [0.9572, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000],
            [-0.2390, 0.9270, 0.0000],
        )
        .unwrap();
        assert!(
            (water.to_degrees() - 104.5).abs() < 0.1,
            "water should be about 104.5 degrees, got {}",
            water.to_degrees()
        );
    }

    #[test]
    fn collinear_atoms_are_a_straight_angle_not_a_nan() {
        // The cosine here rounds to just past -1, where `acos` returns NaN
        // without the clamp. A straight angle is a legitimate measurement.
        let straight = angle([2.0, 0.0, 0.0], [0.0; 3], [-3.0, 0.0, 0.0]).unwrap();

        assert!(straight.is_finite(), "a straight angle must not be NaN");
        assert_close(straight, PI, "straight angle");
    }

    #[test]
    fn angle_is_none_when_an_arm_has_no_direction() {
        assert_eq!(angle([0.0; 3], [0.0; 3], [0.0, 1.0, 0.0]), None);
        assert_eq!(angle([1.0, 0.0, 0.0], [0.0; 3], [0.0; 3]), None);
    }

    /// The canonical torsion fixture: `a` and the axis fixed, `d` swept around
    /// it by `phi`, which is then the dihedral.
    fn swept_dihedral(phi: f64) -> Option<f64> {
        let (sin, cos) = phi.sin_cos();
        dihedral(
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, cos, sin],
        )
    }

    #[test]
    fn dihedral_follows_the_swept_torsion() {
        assert_close(swept_dihedral(0.0).unwrap(), 0.0, "syn");
        assert_close(
            swept_dihedral(PI / 3.0).unwrap().to_degrees(),
            60.0,
            "gauche",
        );
        assert_close(swept_dihedral(PI).unwrap().abs(), PI, "anti");
    }

    #[test]
    fn dihedral_sign_flips_with_the_mirror_image() {
        let gauche = swept_dihedral(PI / 3.0).unwrap();
        let mirrored = swept_dihedral(-PI / 3.0).unwrap();

        assert!(gauche > 0.0, "a positive sweep should read positive");
        assert_close(mirrored, -gauche, "mirrored torsion");
    }

    #[test]
    fn dihedral_is_none_when_there_is_no_plane_to_measure() {
        // A degenerate b–c axis.
        assert_eq!(
            dihedral([0.0, 1.0, 0.0], [0.0; 3], [0.0; 3], [1.0, 1.0, 0.0]),
            None
        );
        // `a` lies on the axis, so a-b-c is collinear.
        assert_eq!(
            dihedral([-1.0, 0.0, 0.0], [0.0; 3], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]),
            None
        );
        // `d` lies on the axis, so b-c-d is collinear.
        assert_eq!(
            dihedral([0.0, 1.0, 0.0], [0.0; 3], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]),
            None
        );
    }

    #[test]
    fn cross_is_perpendicular_to_both_inputs() {
        let (a, b) = ([1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]);
        let n = cross(a, b);

        assert!(dot(n, a).abs() < EPS);
        assert!(dot(n, b).abs() < EPS);
    }
}
