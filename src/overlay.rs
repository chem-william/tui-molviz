//! Turning a [`Measurement`] into something drawable on the braille canvas.
//!
//! This module deals only in projected canvas-data coordinates: it takes the
//! screen positions of the measured atoms and hands back braille points and
//! text labels. It knows nothing about the widget, the molecule, or the camera,
//! which is what lets it be tested without rendering anything.
//!
//! One honesty note runs through the whole module. The dashes and the arc are
//! drawn in the *projection*, because they have to touch the atoms they
//! annotate; the number beside them is measured in the molecule's true 3-D
//! coordinates. Rotating the camera therefore opens and closes the arc while
//! the printed angle holds still. The drawing is a locator — it says *which*
//! quantity the number describes — not a second rendering of the value.

use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};

use crate::Measurement;
use crate::geometry::wrap_angle;

/// Samples per braille dot along a dashed segment. Two keeps a dash solid at
/// any angle without lighting the same braille bit twice.
const SAMPLES_PER_DOT: u32 = 2;
/// Samples in one dash — two braille dots lit.
const DASH_SAMPLES: u32 = 4;
/// Samples in one dash-and-gap period — two dots on, two off.
const PERIOD_SAMPLES: u32 = 8;
/// Hard cap on the points one segment may contribute, so no canvas can make the
/// sampler allocate without bound.
const MAX_SAMPLES_PER_SEGMENT: u32 = 4096;
/// Clearance, in braille dots, between an atom's highlight ring and the first
/// dash, so the ring still reads as a closed circle.
pub(crate) const RING_CLEARANCE_DOTS: f64 = 1.0;
/// Trimmed segments shorter than this (braille dots) are dropped: the atoms sit
/// on top of each other on screen, so the direction means nothing.
const MIN_SEGMENT_DOTS: f64 = 1.0;
/// Nominal arc radius, in braille dots.
const ARC_RADIUS_DOTS: f64 = 4.0;
/// Greatest fraction of the shorter projected arm the arc may take, so it stays
/// inside the angle it annotates instead of running past an atom.
const ARC_MAX_ARM_FRACTION: f64 = 0.45;
/// Points sampled along the arc.
const ARC_STEPS: u32 = 24;
/// Projected sweeps below this (radians) get no arc: the arms overlap on
/// screen, so it would just retrace the dashes.
const ARC_MIN_SWEEP: f64 = 0.05;
/// Distance, in braille dots, a label is nudged off what it annotates. One
/// terminal row is four braille dots tall, so this clears the line beneath.
const LABEL_OFFSET_DOTS: f64 = 4.0;
/// Braille dots per terminal cell across, for sizing a label in canvas units.
const DOTS_PER_CELL_X: f64 = 2.0;

/// The canvas measurements the overlay needs, in canvas-data units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Metrics {
    /// One braille dot.
    pub(crate) dot: f64,
    /// Canvas half-width and half-height, for keeping labels on screen.
    pub(crate) bx: f64,
    pub(crate) by: f64,
}

impl Metrics {
    fn is_usable(self) -> bool {
        self.dot > 0.0 && self.dot.is_finite() && self.bx.is_finite() && self.by.is_finite()
    }
}

/// A projected atom, with the clearance the overlay keeps from it: its drawn
/// disk, the gap its highlight ring sits in, and a dash clearance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Anchor {
    pub(crate) x: f64,
    pub(crate) y: f64,
    pub(crate) clearance: f64,
}

impl Anchor {
    fn is_usable(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.clearance.is_finite()
    }
}

/// A measurement rendered into canvas-data coordinates.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Overlay {
    /// Dash and arc points, ready for a single `Points` draw.
    pub(crate) points: Vec<(f64, f64)>,
    /// The `(x, y, line)` value label. `'static` because the canvas paint
    /// closure is `for<'a> Fn(&mut Context<'a>)`: nothing borrowed from the
    /// widget can reach `Context::print`.
    pub(crate) label: (f64, f64, Line<'static>),
    pub(crate) color: Color,
}

/// The drawing for `value` over the projected `anchors`, or `None` when there
/// is nothing worth drawing.
///
/// `anchors` must be as long as `value` has atoms — 2, 3, or 4 — and in the same
/// order, so the middle anchor of three is the angle's vertex.
pub(crate) fn measurement(
    m: Metrics,
    anchors: &[Anchor],
    value: &Measurement,
    color: Color,
) -> Option<Overlay> {
    // Everything downstream normalizes vectors and divides by lengths. Refusing
    // non-finite input here is what keeps NaN off the canvas: ratatui's `Points`
    // does not reject a NaN coordinate, it paints it in the top-left cell.
    if !m.is_usable() || !anchors.iter().all(|a| a.is_usable()) {
        return None;
    }

    let mut points = Vec::new();
    let line = Line::from(Span::styled(
        value.value_label(),
        Style::default().fg(color),
    ));
    let cells = line.width();

    let (at, nudge, offset) = match (anchors, value) {
        ([a, b], Measurement::Distance { .. }) => {
            let (at, offset) = anchored_between(&mut points, m, *a, *b);
            (at, upward_normal(b.x - a.x, b.y - a.y), offset)
        }
        ([a, vertex, c], Measurement::Angle { .. }) => {
            push_dashes(&mut points, m, *vertex, *a);
            push_dashes(&mut points, m, *vertex, *c);
            // The arc spans the angle *as projected*, so it meets both arms; the
            // label beside it is the true 3-D angle. See the module docs.
            let radius = push_arc(&mut points, m, *a, *vertex, *c).unwrap_or(vertex.clearance);
            let bisector = bisector(*a, *vertex, *c);
            (
                (
                    vertex.x + bisector.0 * radius,
                    vertex.y + bisector.1 * radius,
                ),
                bisector,
                LABEL_OFFSET_DOTS * m.dot,
            )
        }
        ([a, b, c, d], Measurement::Dihedral { .. }) => {
            push_dashes(&mut points, m, *a, *b);
            let (at, offset) = anchored_between(&mut points, m, *b, *c);
            push_dashes(&mut points, m, *c, *d);
            (at, upward_normal(c.x - b.x, c.y - b.y), offset)
        }
        _ => return None,
    };

    let (lx, ly) = label_anchor(m, at, nudge, offset, cells);
    Some(Overlay {
        points,
        label: (lx, ly, line),
        color,
    })
}

/// Whether a computed length can be divided by: finite, and strictly positive.
///
/// Every normalization below goes through this first. `is_finite` is what
/// rejects NaN — a NaN length would sail through a bare `> 0.0` test and end up
/// as a NaN coordinate, which ratatui paints in the canvas corner rather than
/// rejecting.
fn usable_length(len: f64) -> bool {
    len.is_finite() && len > 0.0
}

/// Draws the connector between `a` and `b` and reports where its label hangs
/// off it: the midpoint of what was drawn, and how far to nudge the text clear.
///
/// When the connector is dropped because the atoms overlap on screen, the label
/// falls back to their shared centre — and has to clear their highlight rings by
/// itself, since there is no connector holding it away from them.
fn anchored_between(
    points: &mut Vec<(f64, f64)>,
    m: Metrics,
    a: Anchor,
    b: Anchor,
) -> ((f64, f64), f64) {
    let nudge = LABEL_OFFSET_DOTS * m.dot;
    match push_dashes(points, m, a, b) {
        Some(mid) => (mid, nudge),
        None => (
            (f64::midpoint(a.x, b.x), f64::midpoint(a.y, b.y)),
            nudge + a.clearance.max(b.clearance),
        ),
    }
}

/// The drawable span of the segment from `a` to `b`: its unit direction, and the
/// distances along it where drawing starts and stops once both clearances are
/// taken off.
///
/// `None` when the two anchors project to (nearly) the same point, or the
/// clearances swallow the segment. That guard is what keeps a zero-length
/// normalization — and the NaNs it would produce — off the canvas.
fn span(m: Metrics, a: Anchor, b: Anchor) -> Option<((f64, f64), f64, f64)> {
    let (dx, dy) = (b.x - a.x, b.y - a.y);
    let len = dx.hypot(dy);
    if !usable_length(len) {
        return None;
    }
    let (start, end) = (a.clearance, len - b.clearance);
    if end - start < MIN_SEGMENT_DOTS * m.dot {
        return None;
    }
    Some(((dx / len, dy / len), start, end))
}

/// Appends the braille points of a dashed segment from `a` to `b`, each end
/// pulled back by that anchor's clearance, and returns the midpoint of what was
/// actually drawn.
fn push_dashes(
    points: &mut Vec<(f64, f64)>,
    m: Metrics,
    a: Anchor,
    b: Anchor,
) -> Option<(f64, f64)> {
    let ((ux, uy), start, end) = span(m, a, b)?;

    // Stepping by whole samples rather than accumulating a float phase means
    // there is no drifting loop counter and no float modulo to mis-handle.
    let step = m.dot / f64::from(SAMPLES_PER_DOT);
    let samples = ((end - start) / step)
        .ceil()
        .clamp(0.0, f64::from(MAX_SAMPLES_PER_SEGMENT)) as u32;
    for k in 0..samples {
        if k % PERIOD_SAMPLES >= DASH_SAMPLES {
            continue;
        }
        let along = start + f64::from(k) * step;
        points.push((a.x + ux * along, a.y + uy * along));
    }

    let mid = f64::midpoint(start, end);
    Some((a.x + ux * mid, a.y + uy * mid))
}

/// Appends an arc at `vertex` sweeping the projected angle from the `a` arm to
/// the `c` arm, the short way round, and returns the radius it used.
fn push_arc(
    points: &mut Vec<(f64, f64)>,
    m: Metrics,
    a: Anchor,
    vertex: Anchor,
    c: Anchor,
) -> Option<f64> {
    let arm_a = (a.x - vertex.x).hypot(a.y - vertex.y);
    let arm_c = (c.x - vertex.x).hypot(c.y - vertex.y);
    if !usable_length(arm_a) || !usable_length(arm_c) {
        return None;
    }

    let lower = vertex.clearance;
    let upper = ARC_MAX_ARM_FRACTION * arm_a.min(arm_c);
    // Bail rather than clamp: `f64::clamp` panics when the range is inverted,
    // which is exactly what a large atom on a short arm produces.
    if lower > upper {
        return None;
    }
    let radius = (ARC_RADIUS_DOTS * m.dot).clamp(lower, upper);

    let theta_a = (a.y - vertex.y).atan2(a.x - vertex.x);
    let theta_c = (c.y - vertex.y).atan2(c.x - vertex.x);
    let sweep = wrap_angle(theta_c - theta_a);
    if sweep.abs() < ARC_MIN_SWEEP {
        return None;
    }

    for k in 0..=ARC_STEPS {
        let t = f64::from(k) / f64::from(ARC_STEPS);
        let (sin, cos) = (theta_a + t * sweep).sin_cos();
        points.push((vertex.x + radius * cos, vertex.y + radius * sin));
    }
    Some(radius)
}

/// The unit vector from the origin towards `(dx, dy)`, or `(1.0, 0.0)` when it
/// has no direction.
fn unit(dx: f64, dy: f64) -> (f64, f64) {
    let len = dx.hypot(dy);
    if !usable_length(len) {
        return (1.0, 0.0);
    }
    (dx / len, dy / len)
}

/// The unit normal of a segment, oriented so a label nudged along it sits above
/// the line rather than under it; ties break to the right.
fn upward_normal(dx: f64, dy: f64) -> (f64, f64) {
    let (ux, uy) = unit(dx, dy);
    let (nx, ny) = (-uy, ux);
    if ny < 0.0 || (ny == 0.0 && nx < 0.0) {
        (-nx, -ny)
    } else {
        (nx, ny)
    }
}

/// The unit vector bisecting the projected angle at `vertex`, pointing out into
/// the opening. Falls back to a perpendicular when the arms are antiparallel on
/// screen and so have no bisector.
fn bisector(a: Anchor, vertex: Anchor, c: Anchor) -> (f64, f64) {
    let ua = unit(a.x - vertex.x, a.y - vertex.y);
    let uc = unit(c.x - vertex.x, c.y - vertex.y);
    let (sx, sy) = (ua.0 + uc.0, ua.1 + uc.1);
    let len = sx.hypot(sy);
    if !usable_length(len) {
        return (-ua.1, ua.0);
    }
    (sx / len, sy / len)
}

/// Where a label of `cells` columns must *start* so its text reads as centered
/// on `at` and `offset` clear of the drawing in the (unit) direction `nudge`.
fn label_anchor(
    m: Metrics,
    at: (f64, f64),
    nudge: (f64, f64),
    offset: f64,
    cells: usize,
) -> (f64, f64) {
    // `Buffer::set_line` ignores a `Line`'s alignment and draws left-to-right
    // from the print position, so centering has to happen here.
    let width = cells as f64 * DOTS_PER_CELL_X * m.dot;
    // Slide the text along its own length so it leans the way the nudge points:
    // centered under a vertical nudge, and pushed fully clear of the drawing
    // when the nudge is sideways, which is what keeps an angle's label off the
    // arm it sits beside.
    let lean = -width * (1.0 - nudge.0) / 2.0;
    let x = at.0 + nudge.0 * offset + lean;
    let y = at.1 + nudge.1 * offset;

    // A label anchored outside the bounds is dropped whole, and one starting
    // near the right edge is truncated mid-number; a shifted number beats
    // either. `.max(-m.bx)` keeps the range from inverting when the label is
    // wider than the canvas, which would panic in `clamp`.
    (
        x.clamp(-m.bx, (m.bx - width).max(-m.bx)),
        y.clamp(-m.by, m.by),
    )
}

#[cfg(test)]
mod tests {
    use mendeleev::Element;

    use crate::AtomIndex;

    use super::*;

    const COLOR: Color = Color::White;

    fn metrics() -> Metrics {
        Metrics {
            dot: 0.1,
            bx: 10.0,
            by: 5.0,
        }
    }

    fn anchor(x: f64, y: f64) -> Anchor {
        Anchor {
            x,
            y,
            clearance: 0.2,
        }
    }

    fn distance() -> Measurement {
        Measurement::Distance {
            atoms: [AtomIndex::new(0), AtomIndex::new(1)],
            elements: [Element::C, Element::O],
            angstroms: 1.21,
        }
    }

    fn angle() -> Measurement {
        Measurement::Angle {
            atoms: [0, 1, 2].map(AtomIndex::new),
            elements: [Element::C; 3],
            radians: std::f64::consts::FRAC_PI_2,
        }
    }

    fn dihedral() -> Measurement {
        Measurement::Dihedral {
            atoms: [0, 1, 2, 3].map(AtomIndex::new),
            elements: [Element::C; 4],
            radians: std::f64::consts::PI,
        }
    }

    /// Every degenerate shape the overlay has to survive, paired with a
    /// measurement of the matching arity.
    fn degenerate_cases() -> Vec<(&'static str, Metrics, Vec<Anchor>, Measurement)> {
        let m = metrics();
        let tiny = Metrics {
            dot: 0.1,
            bx: 0.2,
            by: 0.2,
        };
        let fat = |x: f64, y: f64| Anchor {
            x,
            y,
            clearance: 50.0,
        };
        vec![
            (
                "coincident pair",
                m,
                vec![anchor(1.0, 1.0), anchor(1.0, 1.0)],
                distance(),
            ),
            (
                "clearances swallow the segment",
                m,
                vec![fat(0.0, 0.0), fat(1.0, 0.0)],
                distance(),
            ),
            (
                "canvas narrower than the label",
                tiny,
                vec![anchor(0.0, 0.0), anchor(1.0, 0.0)],
                distance(),
            ),
            (
                "collinear arms",
                m,
                vec![anchor(-1.0, 0.0), anchor(0.0, 0.0), anchor(1.0, 0.0)],
                angle(),
            ),
            (
                "arms overlapping on screen",
                m,
                vec![anchor(1.0, 0.0), anchor(0.0, 0.0), anchor(2.0, 0.0)],
                angle(),
            ),
            (
                "every atom on one spot",
                m,
                vec![anchor(0.0, 0.0); 3],
                angle(),
            ),
            (
                "vertex clearance exceeds both arms",
                m,
                vec![anchor(0.1, 0.0), fat(0.0, 0.0), anchor(0.0, 0.1)],
                angle(),
            ),
            (
                "four coincident atoms",
                m,
                vec![anchor(0.0, 0.0); 4],
                dihedral(),
            ),
        ]
    }

    #[test]
    fn every_overlay_point_and_label_anchor_is_finite() {
        // The firewall, tested at the arithmetic rather than at a pixel: a NaN
        // reaching `Points` is not a panic, it is a stray dot in the canvas
        // corner, which is far harder to notice.
        for (name, m, anchors, value) in degenerate_cases() {
            let Some(overlay) = measurement(m, &anchors, &value, COLOR) else {
                continue;
            };
            for (x, y) in &overlay.points {
                assert!(x.is_finite() && y.is_finite(), "{name}: point ({x}, {y})");
            }
            let (x, y, _) = &overlay.label;
            assert!(
                x.is_finite() && y.is_finite(),
                "{name}: label at ({x}, {y})"
            );
        }
    }

    #[test]
    fn a_degenerate_shape_never_panics_and_always_labels() {
        // Whatever the geometry does, the number itself must still be printed:
        // it is the part the user actually reads.
        for (name, m, anchors, value) in degenerate_cases() {
            let overlay = measurement(m, &anchors, &value, COLOR)
                .unwrap_or_else(|| panic!("{name}: expected an overlay"));
            assert!(overlay.label.2.width() > 0, "{name}");
        }
    }

    #[test]
    fn dashes_stay_clear_of_both_atoms() {
        let (a, b) = (anchor(0.0, 0.0), anchor(4.0, 0.0));
        let overlay = measurement(metrics(), &[a, b], &distance(), COLOR).unwrap();

        assert!(!overlay.points.is_empty());
        for &(x, y) in &overlay.points {
            assert!(
                (x - a.x).hypot(y - a.y) >= a.clearance - 1e-12,
                "point ({x}, {y}) intrudes on the first atom"
            );
            assert!(
                (x - b.x).hypot(y - b.y) >= b.clearance - 1e-12,
                "point ({x}, {y}) intrudes on the second atom"
            );
        }
    }

    #[test]
    fn a_connector_is_dashed_rather_than_solid() {
        let m = metrics();
        let overlay =
            measurement(m, &[anchor(0.0, 0.0), anchor(4.0, 0.0)], &distance(), COLOR).unwrap();
        let step = m.dot / f64::from(SAMPLES_PER_DOT);

        let gaps: Vec<f64> = overlay
            .points
            .windows(2)
            .map(|pair| pair[1].0 - pair[0].0)
            .collect();

        assert!(
            gaps.iter().any(|g| (g - step).abs() < 1e-9),
            "a dash should be a run of adjacent samples"
        );
        assert!(
            gaps.iter().any(|g| *g > step * 1.5),
            "there should be gaps between the dashes"
        );
    }

    #[test]
    fn coincident_atoms_draw_no_connector() {
        let overlay = measurement(
            metrics(),
            &[anchor(1.0, 1.0), anchor(1.0, 1.0)],
            &distance(),
            COLOR,
        )
        .unwrap();

        assert!(
            overlay.points.is_empty(),
            "atoms on one spot have no direction to draw along"
        );
        assert!(overlay.label.2.width() > 0, "the value is still printed");
    }

    #[test]
    fn the_arc_shrinks_to_fit_the_shorter_arm() {
        let m = metrics();
        let arc_of = |arm: f64| {
            let mut points = Vec::new();
            push_arc(
                &mut points,
                m,
                anchor(arm, 0.0),
                anchor(0.0, 0.0),
                anchor(0.0, 10.0),
            )
        };

        let roomy = arc_of(10.0).unwrap();
        let cramped = arc_of(0.5).unwrap();

        assert!(
            cramped < roomy,
            "a short arm ({cramped}) should pull the arc in from {roomy}"
        );
    }

    #[test]
    fn the_arc_is_skipped_when_it_would_not_read() {
        let m = metrics();
        let mut points = Vec::new();

        // Arms pointing the same way on screen: the arc would retrace the dashes.
        assert_eq!(
            push_arc(
                &mut points,
                m,
                anchor(1.0, 0.0),
                anchor(0.0, 0.0),
                anchor(2.0, 0.0)
            ),
            None
        );
        // A vertex whose clearance is larger than the arms allow. An inverted
        // range here would panic inside `clamp` rather than return.
        assert_eq!(
            push_arc(
                &mut points,
                m,
                anchor(0.1, 0.0),
                Anchor {
                    x: 0.0,
                    y: 0.0,
                    clearance: 50.0
                },
                anchor(0.0, 0.1)
            ),
            None
        );
        assert!(points.is_empty());
    }

    #[test]
    fn label_anchor_centres_the_text_on_its_target() {
        let m = metrics();
        let cells = 5;
        let width = cells as f64 * DOTS_PER_CELL_X * m.dot;

        // Nudged straight up, the text centres on its target.
        let (x, y) = label_anchor(m, (0.0, 0.0), (0.0, 1.0), LABEL_OFFSET_DOTS * m.dot, cells);
        assert!(
            (x + width / 2.0).abs() < 1e-12,
            "the text's middle should land on the target"
        );
        assert!(
            (y - LABEL_OFFSET_DOTS * m.dot).abs() < 1e-12,
            "nudged clear"
        );

        // Nudged sideways, it leans fully clear instead, so it never lies over
        // the arm it annotates.
        let (right, _) = label_anchor(m, (0.0, 0.0), (1.0, 0.0), LABEL_OFFSET_DOTS * m.dot, cells);
        assert!(
            right > 0.0,
            "a rightward nudge should start right of the target"
        );
        let (left, _) = label_anchor(m, (0.0, 0.0), (-1.0, 0.0), LABEL_OFFSET_DOTS * m.dot, cells);
        assert!(
            left + width < 0.0,
            "a leftward nudge should end left of the target"
        );
    }

    #[test]
    fn label_anchor_keeps_a_wide_label_inside_the_canvas() {
        // A label wider than the canvas inverts the clamp range, which panics
        // unless the upper bound is floored.
        let narrow = Metrics {
            dot: 0.1,
            bx: 0.2,
            by: 0.2,
        };
        let (x, y) = label_anchor(
            narrow,
            (0.0, 0.0),
            (0.0, 1.0),
            LABEL_OFFSET_DOTS * narrow.dot,
            20,
        );

        assert!(x >= -narrow.bx && x <= narrow.bx);
        assert!(y >= -narrow.by && y <= narrow.by);
    }

    #[test]
    fn upward_normal_never_points_down() {
        for (dx, dy) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (0.0, 0.0)] {
            let (nx, ny) = upward_normal(dx, dy);
            assert!(ny > 0.0 || (ny == 0.0 && nx > 0.0), "({dx}, {dy})");
            assert!(
                (nx.hypot(ny) - 1.0).abs() < 1e-12,
                "should be a unit vector"
            );
        }
    }

    #[test]
    fn anchors_that_do_not_match_the_measurement_draw_nothing() {
        let m = metrics();
        let two = [anchor(0.0, 0.0), anchor(1.0, 0.0)];

        assert_eq!(measurement(m, &two, &angle(), COLOR), None);
        assert_eq!(measurement(m, &[], &distance(), COLOR), None);
    }

    #[test]
    fn non_finite_input_is_refused_outright() {
        let m = metrics();
        let bad = [anchor(f64::NAN, 0.0), anchor(1.0, 0.0)];

        assert_eq!(measurement(m, &bad, &distance(), COLOR), None);
    }
}
