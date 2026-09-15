//! # A molecular visualizer for Ratatui
//!
//! [Ratatui](https://ratatui.rs/) is an immediate-mode terminal user interface (TUI) library.
//! `tui-molviz` allows you to show molecules in a Ratatui app.
//!
//! # Quick start
//! ```rust,no_run
//! use ratatui::Frame;
//! use ratatui::crossterm::event;
//! use ratatui::widgets::Block;
//! use tui_molviz::molecule::{Atom, Molecule};
//! use tui_molviz::{Element, MoleculeVisualizer};
//!
//! fn main() -> color_eyre::Result<()> {
//!     color_eyre::install()?;
//!
//!     let water = Molecule::from_atoms([
//!         Atom::new(Element::O, [0.0000, 0.0000, 0.0000]),
//!         Atom::new(Element::H, [0.9572, 0.0000, 0.0000]),
//!         Atom::new(Element::H, [-0.2390, 0.9270, 0.0000]),
//!     ]);
//!
//!     ratatui::run(|terminal| loop {
//!         terminal.draw(|frame| render(frame, &water))?;
//!
//!         if event::read()?.is_key_press() {
//!             break Ok(());
//!         }
//!     })
//! }
//!
//! fn render(frame: &mut Frame<'_>, water: &Molecule) {
//!     let widget = MoleculeVisualizer::new(water).block(Block::bordered().title("Water"));
//!     frame.render_widget(widget, frame.area());
//! }
//! ```
//!
//! # Examples
//!
//! * `examples/quickstart.rs` is a simple example plotting a water molecule.
//! * `examples/showcase.rs` is a more complex example that showcases zoom, rotation, panning, and selection of atoms

pub mod camera;
mod geometry;
pub mod measurement;
pub mod molecule;
mod overlay;
pub mod selection;
use std::collections::HashSet;

use crate::camera::Camera;
use crate::molecule::{BondOrder, Molecule};

pub use measurement::{Measurement, MeasurementError};
pub use mendeleev::Color as CpkColor;
pub use mendeleev::Element;
pub use molecule::AtomIndex;
use ratatui::{
    buffer::Buffer,
    layout::{Position, Rect},
    style::{Style, Styled},
    text::{Line, Span},
    widgets::{
        Block, StatefulWidget, Widget,
        canvas::{Canvas, Line as CanvasLine, Points},
    },
};
pub use selection::Selection;

/// The braille canvas a molecule is drawn on.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MoleculeCanvas {
    inner: Rect,
    bx: f64,
    by: f64,
    dpu: f64,
}

impl MoleculeCanvas {
    /// Headroom so the molecule's bounding sphere doesn't touch the canvas edge.
    const EDGE_PADDING: f64 = 1.15;
    /// Atom disk radius, in braille dots, as a fraction of covalent radius.
    const ATOM_RADIUS_SCALE: f64 = 0.55;
    const MIN_ATOM_RADIUS_DOTS: f64 = 1.5;
    const MAX_ATOM_RADIUS_DOTS: f64 = 5.0;

    #[must_use]
    pub fn contains_cell(&self, position: impl Into<Position>) -> bool {
        self.inner.contains(position.into())
    }

    /// Fits a molecule of the given radius into `inner` at the camera's zoom.
    /// Braille packs 2 dots per cell across and 4 down.
    ///
    /// Normally obtained from a stateful render via
    /// [`MoleculeVisualizerState::canvas`], but construct it directly when
    /// hit-testing without rendering — `radius` and `zoom` must match what
    /// the drawn frame used.
    ///
    /// # Panic
    ///
    /// Panics if either `radius` is not [finite](f64::is_finite) or if `zoom <= 0.0`.
    #[must_use]
    pub fn new(inner: Rect, radius: f64, zoom: f64) -> Self {
        assert!(radius.is_finite(), "radius must be finite");
        assert!(zoom > 0.0, "zoom must be positive");
        let w = f64::from(inner.width.max(1));
        let h = f64::from(inner.height.max(1));
        let (rx, ry) = (2.0 * w, 4.0 * h);
        let by = (radius * Self::EDGE_PADDING) / zoom;
        let bx = by * (rx / ry);
        let dpu = ry / (2.0 * by);
        Self { inner, bx, by, dpu }
    }

    /// One braille dot, in canvas-data units.
    pub(crate) fn dot(&self) -> f64 {
        1.0 / self.dpu
    }

    /// The canvas half-extents, in canvas-data units.
    pub(crate) fn half_bounds(&self) -> (f64, f64) {
        (self.bx, self.by)
    }

    /// Drawn radius of an atom, in braille dots.
    #[allow(clippy::manual_clamp)]
    fn atom_radius_dots(&self, cov: f64) -> f64 {
        (cov * Self::ATOM_RADIUS_SCALE * self.dpu)
            .max(Self::MIN_ATOM_RADIUS_DOTS)
            .min(Self::MAX_ATOM_RADIUS_DOTS)
    }

    /// Inverse of the canvas mapping: the [`AtomIndex`] of the atom whose drawn
    /// disk a clicked terminal cell lands in (front-most on overlap), or `None`
    /// for empty space. The canvas maps data x in `[-bx, bx]` left→right and data y in
    /// `[-by, by]` bottom→top, so the row axis is flipped relative to screen rows.
    ///
    /// `camera` and `molecule` must match what the last render drew so the
    /// projection lines up with the pixels on screen. A camera that has been
    /// panned with [`Camera::translate`](crate::camera::Camera::translate)
    /// keeps the hit-test aligned with the molecule on screen, since the
    /// projection includes the offset.
    ///
    /// # Example
    ///
    /// The typical flow is a stateful render, which hands back the canvas
    /// mapping, then a hit-test whenever a terminal cell comes in from any
    /// event source:
    ///
    /// ```rust
    /// use ratatui::{buffer::Buffer, layout::Rect, widgets::StatefulWidget};
    /// use tui_molviz::camera::Camera;
    /// use tui_molviz::molecule::{Atom, Molecule, AtomIndex};
    /// use tui_molviz::{Element, Measurement, MoleculeVisualizer, MoleculeVisualizerState, Selection};
    ///
    /// let molecule = Molecule::from_atoms([
    ///     Atom::new(Element::O, [0.0000, 0.0000, 0.0000]),
    ///     Atom::new(Element::H, [0.9572, 0.0000, 0.0000]),
    ///     Atom::new(Element::H, [-0.2390, 0.9270, 0.0000]),
    /// ]);
    /// let camera = Camera::default();
    ///
    /// // Render statefully (`frame.render_stateful_widget` in a real app).
    /// let mut state = MoleculeVisualizerState::default();
    /// let area = Rect::new(0, 0, 30, 10);
    /// let mut buffer = Buffer::empty(area);
    /// StatefulWidget::render(
    ///     &MoleculeVisualizer::new(&molecule).camera(camera),
    ///     area,
    ///     &mut buffer,
    ///     &mut state,
    /// );
    ///
    /// // A mouse click reports the cell it landed in.
    /// let (col, row) = (13u16, 6u16);
    ///
    /// // Hit-test with the same camera and molecule the frame was drawn with.
    /// let hit = state.canvas().unwrap().pick_atom(camera, &molecule, (col, row));
    /// assert_eq!(hit, Some(AtomIndex::new(0)), "the click landed on the oxygen");
    ///
    /// // Collect hits into a `Selection` to build up a measurement, then feed
    /// // it to `.highlight` for the next frame.
    /// let mut selection = Selection::with_limit(4);
    /// if let Some(hit) = hit {
    ///     selection.toggle(hit);
    /// }
    /// selection.toggle(AtomIndex::new(1));
    ///
    /// let measured = Measurement::of(&molecule, &selection)?.expect("two atoms");
    /// assert_eq!(measured.to_string(), "O0–H1  0.957 Å");
    ///
    /// let _next = MoleculeVisualizer::new(&molecule).camera(camera).highlight(&selection);
    /// # Ok::<(), tui_molviz::MeasurementError>(())
    /// ```
    #[must_use]
    pub fn pick_atom(
        &self,
        camera: Camera,
        molecule: &Molecule,
        position: impl Into<Position>,
    ) -> Option<AtomIndex> {
        let position = position.into();
        let (px, py) = self.cell_to_data(position)?;

        molecule
            .atoms()
            .iter()
            .enumerate()
            .filter_map(|(i, atom)| {
                let [x, y, z] = atom.position();
                let p = camera.project_point(x, y, z);
                let d2 = (p.0 - px).powi(2) + (p.1 - py).powi(2);
                let r_world = self.atom_radius_dots(atom.covalent_radius()) / self.dpu;
                // On overlap, prefer the front-most atom (largest projected z).
                (d2 <= r_world * r_world).then_some((i, p.2))
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| AtomIndex::new(i))
    }

    /// Convert a terminal cell to canvas data coordinates — the same space
    /// [`Camera::project_point`](crate::camera::Camera::project_point) and
    /// [`pick_atom`](Self::pick_atom) use — sampling the cell's center, or
    /// `None` if the cell is outside the canvas area.
    ///
    /// Data x runs left→right over `[-bx, bx]` and data y runs bottom→top over
    /// `[-by, by]`, so the row axis is flipped relative to screen rows.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ratatui::layout::Rect;
    /// use tui_molviz::MoleculeCanvas;
    ///
    /// let canvas = MoleculeCanvas::new(Rect::new(0, 0, 20, 10), 1.0, 1.0);
    /// let (x, y) = canvas.cell_to_data((10, 5)).unwrap();
    /// // The center cell of a 20x10 canvas samples half a cell off the exact
    /// // origin (no cell is centered on it): half a cell is 0.025 of the
    /// // width and 0.05 of the height.
    /// assert!((x - 0.05 * 1.15).abs() < 1e-9);
    /// assert!((y + 0.10 * 1.15).abs() < 1e-9);
    /// assert_eq!(canvas.cell_to_data((20, 0)), None);
    /// ```
    #[must_use]
    pub fn cell_to_data(&self, position: impl Into<Position>) -> Option<(f64, f64)> {
        let position = position.into();
        if !self.inner.contains(position) {
            return None;
        }
        let fx = (f64::from(position.x - self.inner.x) + 0.5) / f64::from(self.inner.width.max(1));
        let fy = (f64::from(position.y - self.inner.y) + 0.5) / f64::from(self.inner.height.max(1));
        Some((-self.bx + fx * 2.0 * self.bx, self.by - fy * 2.0 * self.by))
    }

    /// Convert a terminal-cell delta into
    /// [`Camera::translate`](crate::camera::Camera::translate) units (Å in the
    /// screen plane), for panning. This is how a mouse-drag delta in cells
    /// becomes a camera pan: translate by this on each move event.
    ///
    /// `drow` follows terminal rows (down positive); the returned `dy` follows
    /// data coordinates (up positive), so a drag down the screen yields a
    /// negative `dy`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ratatui::layout::Rect;
    /// use tui_molviz::{MoleculeCanvas, camera::Camera};
    ///
    /// let canvas = MoleculeCanvas::new(Rect::new(0, 0, 20, 10), 1.0, 1.0);
    /// // Two cells right, one cell down.
    /// let (dx, dy) = canvas.cell_delta_to_world(2, 1);
    /// assert!((dx - 2.0 * 2.0 * 1.15 / 20.0).abs() < 1e-9);
    /// assert!((dy + 1.0 * 2.0 * 1.15 / 10.0).abs() < 1e-9);
    ///
    /// let mut camera = Camera::new(0.0, 0.0, 1.0);
    /// camera.translate(dx, dy);
    /// assert_eq!(camera.offset(), (dx, dy));
    /// ```
    #[must_use]
    pub fn cell_delta_to_world(&self, dcol: i32, drow: i32) -> (f64, f64) {
        let (w, h) = (
            f64::from(self.inner.width.max(1)),
            f64::from(self.inner.height.max(1)),
        );
        (
            f64::from(dcol) * 2.0 * self.bx / w,
            -f64::from(drow) * 2.0 * self.by / h,
        )
    }
}

/// State handed back by rendering so a later mouse event can hit-test.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MoleculeVisualizerState {
    canvas: Option<MoleculeCanvas>,
}

impl MoleculeVisualizerState {
    /// The canvas mapping from the most recent render, or `None` if the widget
    /// has not been rendered with this state yet. Pass it to
    /// [`MoleculeCanvas::pick_atom`] to hit-test a terminal cell.
    #[must_use]
    pub fn canvas(&self) -> Option<MoleculeCanvas> {
        self.canvas
    }
}

/// A compact widget for displaying atoms and molecules.
#[derive(Debug, Clone, PartialEq)]
pub struct MoleculeVisualizer<'a> {
    /// The molecule to visualize
    molecule: &'a Molecule,
    /// Optional block to wrap the molecular visualizer
    block: Option<Block<'a>>,
    /// Base style for the entire widget
    style: Style,
    /// Whether to show molecule legend or not. Default is `true`
    show_molecule_legend: bool,
    /// Whether to show bonds between atoms. Default is `true`
    show_bonds: bool,
    /// The camera used to display the molecule. Used to control rotation, zooming, and panning
    camera: Camera,
    /// Atoms to draw highlight markers on, in the order they were selected.
    /// Out-of-range indices are ignored at render time. Default is empty.
    highlight: Vec<AtomIndex>,
    /// Whether to draw the measurement overlay for 2, 3, or 4 highlighted
    /// atoms. Default is `true`.
    show_measurement: bool,
    /// Style of the highlight marker (its `fg` color is used). `None` disables
    /// the highlight even when [`highlight`](Self::highlight) is set.
    highlight_style: Option<Style>,
}

impl<'a> MoleculeVisualizer<'a> {
    /// Creates a new `MoleculeVisualizer` with the given molecule
    ///
    /// # Example
    ///
    /// This visualizes a simple [`Molecule`]
    ///
    /// ```rust
    /// use tui_molviz::molecule::{Atom, Molecule};
    /// use tui_molviz::{Element, MoleculeVisualizer};
    ///
    /// let molecule = Molecule::from_atoms([
    ///     Atom::new(Element::O, [0.0000, 0.0000, 0.0000]),
    ///     Atom::new(Element::H, [0.9572, 0.0000, 0.0000]),
    ///     Atom::new(Element::H, [-0.2390, 0.9270, 0.0000]),
    /// ]);
    /// let visualizer = MoleculeVisualizer::new(&molecule);
    /// ```
    #[must_use]
    pub fn new(molecule: &'a Molecule) -> Self {
        Self {
            molecule,
            block: None,
            style: Style::default(),
            show_molecule_legend: true,
            show_bonds: true,
            camera: Camera::default(),
            highlight: Vec::new(),
            show_measurement: true,
            highlight_style: Some(Style::default().fg(Self::DEFAULT_HIGHLIGHT_COLOR)),
        }
    }

    /// Wraps the visualizer with the given block.
    ///
    /// This is a fluent setter method which must be chained or used as it consumes self
    #[must_use = "method moves the value of self and returns the modified value"]
    pub fn block(mut self, block: Block<'a>) -> Self {
        self.block = Some(block);
        self
    }

    /// Sets the camera the molecule is drawn from. Hit-testing with
    /// [`MoleculeCanvas::pick_atom`] must use this same camera.
    ///
    /// This is a fluent setter method which must be chained or used as it consumes self
    #[must_use = "method moves the value of self and returns the modified value"]
    pub fn camera(mut self, camera: Camera) -> Self {
        self.camera = camera;
        self
    }

    /// Sets whether to show a legend with a color key for each atom in the visualized
    /// molecule. Empty when the molecule has no atoms.
    ///
    /// This is a fluent setter method which must be chained or used as it consumes self
    #[must_use = "method moves the value of self and returns the modified value"]
    pub const fn show_molecule_legend(mut self, molecule_legend: bool) -> Self {
        self.show_molecule_legend = molecule_legend;
        self
    }

    /// Sets whether to show bonds when drawing the molecule.
    ///
    /// This is a fluent setter method which must be chained or used as it consumes self
    #[must_use = "method moves the value of self and returns the modified value"]
    pub const fn show_bonds(mut self, show_bonds: bool) -> Self {
        self.show_bonds = show_bonds;
        self
    }

    /// Sets the base style of the widget.
    ///
    /// `style` accepts any type that is convertible to [`Style`] (e.g. [`Style`], or
    /// your own type that implements [`Into<Style>`]).
    ///
    /// This is a fluent setter method which must be chained or used as it consumes self
    #[must_use]
    pub fn style<S: Into<Style>>(mut self, style: S) -> Self {
        self.style = style.into();
        self
    }

    /// Highlights each of `atoms` by drawing a marker ring around it, replacing
    /// any previous highlight. Pass `[]` to clear it.
    ///
    /// The indices are typically ones returned by
    /// [`MoleculeCanvas::pick_atom`], collected into a [`Selection`]. Indices
    /// that name no atom of the molecule are ignored at render time, and repeats
    /// collapse onto their first occurrence.
    ///
    /// Order matters: with [`show_measurement`](Self::show_measurement) on, two
    /// atoms are labelled with their distance, three with the angle at the
    /// *middle* one, and four with the dihedral about the middle pair.
    ///
    /// ```rust
    /// use tui_molviz::molecule::{Atom, Molecule};
    /// use tui_molviz::{AtomIndex, Element, MoleculeVisualizer, Selection};
    ///
    /// let water = Molecule::from_atoms([
    ///     Atom::new(Element::O, [0.0000, 0.0000, 0.0000]),
    ///     Atom::new(Element::H, [0.9572, 0.0000, 0.0000]),
    ///     Atom::new(Element::H, [-0.2390, 0.9270, 0.0000]),
    /// ]);
    ///
    /// // A selection, a bare array, and a single optional index all work.
    /// let selection: Selection = [AtomIndex::new(1), AtomIndex::new(0)].into_iter().collect();
    /// let _ = MoleculeVisualizer::new(&water).highlight(&selection);
    /// let _ = MoleculeVisualizer::new(&water).highlight([AtomIndex::new(0)]);
    /// let _ = MoleculeVisualizer::new(&water).highlight(Some(AtomIndex::new(0)));
    /// let _ = MoleculeVisualizer::new(&water).highlight([]); // cleared
    /// ```
    ///
    /// This is a fluent setter method which must be chained or used as it consumes self
    #[must_use = "method moves the value of self and returns the modified value"]
    pub fn highlight(mut self, atoms: impl IntoIterator<Item = AtomIndex>) -> Self {
        self.highlight = atoms.into_iter().collect();
        self
    }

    /// Sets whether to annotate the highlighted atoms with the quantity they
    /// measure — a distance for two, an angle for three, a dihedral for four —
    /// drawing dashed connectors and printing the value on the canvas. Default
    /// is `true`; it draws nothing until at least two atoms are highlighted.
    ///
    /// The connectors and the arc are drawn in the projection, so they meet the
    /// atoms on screen, but the printed number is measured in the molecule's
    /// true 3-D coordinates. Rotating the camera therefore opens and closes the
    /// arc while the number holds still: it marks *which* angle is meant rather
    /// than redrawing its value.
    ///
    /// The value is the one [`Measurement::of`] returns for the same atoms, so a
    /// status line built from that agrees with the canvas.
    ///
    /// This is a fluent setter method which must be chained or used as it consumes self
    #[must_use = "method moves the value of self and returns the modified value"]
    pub const fn show_measurement(mut self, show_measurement: bool) -> Self {
        self.show_measurement = show_measurement;
        self
    }

    /// Sets the style of the highlight marker; the marker ring is drawn in the
    /// style's foreground color. Passing `None` disables the highlight entirely,
    /// even when an atom is selected via [`highlight`](Self::highlight).
    ///
    /// This is a fluent setter method which must be chained or used as it consumes self
    #[must_use = "method moves the value of self and returns the modified value"]
    pub const fn highlight_style(mut self, style: Option<Style>) -> Self {
        self.highlight_style = style;
        self
    }
}

impl Styled for MoleculeVisualizer<'_> {
    type Item = Self;

    fn style(&self) -> Style {
        self.style
    }

    fn set_style<S: Into<Style>>(self, style: S) -> Self::Item {
        self.style(style)
    }
}

impl Widget for MoleculeVisualizer<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        Widget::render(&self, area, buf);
    }
}

impl Widget for &MoleculeVisualizer<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let _ = self.render_inner(area, buf);
    }
}

impl StatefulWidget for MoleculeVisualizer<'_> {
    type State = MoleculeVisualizerState;

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::State) {
        StatefulWidget::render(&self, area, buf, state);
    }
}

impl StatefulWidget for &MoleculeVisualizer<'_> {
    type State = MoleculeVisualizerState;

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::State) {
        state.canvas = Some(self.render_inner(area, buf));
    }
}

impl MoleculeVisualizer<'_> {
    /// Depth factor floor; the farthest atom is dimmed to this fraction of
    /// full brightness rather than to black.
    const MIN_DEPTH_BRIGHTNESS: f64 = 0.4;
    const DEPTH_BRIGHTNESS_RANGE: f64 = 1.0 - Self::MIN_DEPTH_BRIGHTNESS;
    /// Number of distinct brightness steps `shade` quantizes to.
    const SHADE_LEVELS: f64 = 5.0;
    /// Uniform gray used for bonds, independent of the bonded atoms' CPK colors.
    const BOND_COLOR: CpkColor = CpkColor {
        r: 120,
        g: 120,
        b: 120,
    };
    /// Perpendicular offset, in braille dots, of the extra lines a double or
    /// triple bond gets from the bond axis. Kept below the minimum atom disk
    /// radius so the line ends stay hidden under the atom disks.
    const BOND_PARALLEL_OFFSET_DOTS: f64 = 0.75;
    /// Projected bond lengths below this (in braille dots) are drawn as a
    /// single line: the bond points almost at the viewer, so a perpendicular
    /// offset is just noise.
    const BOND_MIN_PARALLEL_LENGTH_DOTS: f64 = 2.0;
    /// Fallback marker color when the highlight style has no foreground set.
    const DEFAULT_HIGHLIGHT_COLOR: ratatui::style::Color = ratatui::style::Color::White;
    /// Gap, in braille dots, between an atom's drawn disk and its highlight ring.
    const HIGHLIGHT_RING_GAP_DOTS: f64 = 1.5;
    /// Number of points sampled around the highlight ring.
    const HIGHLIGHT_RING_STEPS: u32 = 48;

    /// The color the highlight markers are drawn in, or `None` when the style
    /// suppresses them entirely.
    fn marker_color(&self) -> Option<ratatui::style::Color> {
        Some(
            self.highlight_style?
                .fg
                .unwrap_or(Self::DEFAULT_HIGHLIGHT_COLOR),
        )
    }

    /// The highlighted atoms that actually exist, in highlight order, with
    /// repeats collapsed onto their first occurrence.
    ///
    /// The order is contractual — the middle of three is the angle's vertex, the
    /// middle two of four the dihedral's axis — so this deduplicates without
    /// reordering, which a set would not.
    fn highlighted_indices(&self) -> Vec<AtomIndex> {
        let mut selected = Vec::with_capacity(self.highlight.len());
        for &index in &self.highlight {
            if self.molecule.get(index).is_some() && !selected.contains(&index) {
                selected.push(index);
            }
        }
        selected
    }

    /// The radius of the highlight ring around atom `i`, in braille dots: the
    /// drawn disk plus the gap that keeps the ring off it. The measurement
    /// overlay starts its clearance here, so both read it from one place.
    fn ring_radius_dots(&self, canvas: &MoleculeCanvas, i: AtomIndex) -> f64 {
        canvas.atom_radius_dots(self.molecule.atoms()[i.get()].covalent_radius())
            + Self::HIGHLIGHT_RING_GAP_DOTS
    }

    /// The marker rings for `selected`, in canvas-data coords, flattened into
    /// one point list: they share a color, so a single draw keeps the terminal's
    /// color escapes down.
    fn highlight_rings(
        &self,
        selected: &[AtomIndex],
        proj: &[(f64, f64, f64)],
        canvas: &MoleculeCanvas,
    ) -> Vec<(f64, f64)> {
        let dot = canvas.dot(); // one braille dot, in world units
        let steps = Self::HIGHLIGHT_RING_STEPS;
        let mut pts = Vec::with_capacity(selected.len() * steps as usize);
        for &i in selected {
            let r_ring = self.ring_radius_dots(canvas, i) * dot;
            let (px, py, _) = proj[i.get()];
            pts.extend((0..steps).map(|k| {
                let theta = std::f64::consts::TAU * f64::from(k) / f64::from(steps);
                let (s, c) = theta.sin_cos();
                (px + r_ring * c, py + r_ring * s)
            }));
        }
        pts
    }

    /// The measurement annotation for `selected` — dashed connectors, an arc for
    /// an angle, and the value as a label — or `None` when the overlay is off,
    /// the selection is not two to four atoms, or the quantity is undefined.
    fn measurement_overlay(
        &self,
        selected: &[AtomIndex],
        proj: &[(f64, f64, f64)],
        canvas: &MoleculeCanvas,
        color: ratatui::style::Color,
    ) -> Option<overlay::Overlay> {
        if !self.show_measurement {
            return None;
        }

        // The same call a consumer makes for their own status line, so the
        // canvas label and the status line cannot disagree.
        let value = Measurement::of(self.molecule, selected).ok().flatten()?;

        let dot = canvas.dot();
        let (bx, by) = canvas.half_bounds();
        let anchors: Vec<overlay::Anchor> = selected
            .iter()
            .map(|&i| {
                // Keep clear of the drawn disk, the ring around it, and a gap.
                let clearance = self.ring_radius_dots(canvas, i) + overlay::RING_CLEARANCE_DOTS;
                let (x, y, _) = proj[i.get()];
                overlay::Anchor {
                    x,
                    y,
                    clearance: clearance * dot,
                }
            })
            .collect();

        overlay::measurement(overlay::Metrics { dot, bx, by }, &anchors, &value, color)
    }

    /// A color key for the elements actually in the molecule (each element's
    /// symbol drawn in its CPK color), so the structure is readable without already
    /// knowing the palette. Empty when the molecule has no atoms.
    fn draw_molecule_legend(&self) -> Line<'static> {
        let mut seen = HashSet::new();
        let spans = self
            .molecule
            .atoms()
            .iter()
            .filter(|atom| seen.insert(atom.element()))
            .map(|atom| {
                let c = atom.cpk();
                Span::styled(
                    format!(" {} ", atom.element().symbol()),
                    Style::default()
                        .fg(ratatui::style::Color::Rgb(c.r, c.g, c.b))
                        .bold(),
                )
            })
            .collect::<Vec<_>>();
        Line::from(spans).centered()
    }

    // The depth is clamped so that if a single atom or a molecule lies flat
    // on the screen, it will not be dimmed at all.
    const fn shade_depth(z: f64, zmin: f64, zmax: f64) -> f64 {
        let zspan = (zmax - zmin).max(1e-5);
        let flat = zspan <= 1e-5;
        if flat {
            1.0
        } else {
            Self::MIN_DEPTH_BRIGHTNESS + Self::DEPTH_BRIGHTNESS_RANGE * ((z - zmin) / zspan)
        }
    }

    fn back_to_front_order(depths: &[f64]) -> Vec<usize> {
        let mut order: Vec<usize> = (0..depths.len()).collect();
        order.sort_by(|&a, &b| depths[a].total_cmp(&depths[b]));
        order
    }

    /// Dim a CPK color by a depth factor. The factor is quantized to a few levels so
    /// neighbouring cells share a color and the terminal can run-length batch the
    /// color escapes. Only the dimming is stepped, not the hue.
    #[must_use]
    fn shade(color: CpkColor, f: f64) -> ratatui::style::Color {
        let f = (f.clamp(0.0, 1.0) * Self::SHADE_LEVELS).round() / Self::SHADE_LEVELS;
        ratatui::style::Color::Rgb(
            (f64::from(color.r) * f) as u8,
            (f64::from(color.g) * f) as u8,
            (f64::from(color.b) * f) as u8,
        )
    }

    /// The canvas lines for every bond, in the bond color shaded at the
    /// midpoint of the bonded atoms' depths. Single bonds are one line; double
    /// and triple bonds get one or two extra lines parallel to the bond axis.
    /// Empty when bonds are hidden.
    fn bond_lines(
        &self,
        proj: &[(f64, f64, f64)],
        canvas: &MoleculeCanvas,
        zmin: f64,
        zmax: f64,
    ) -> Vec<CanvasLine> {
        if !self.show_bonds {
            return Vec::new();
        }

        // One braille dot, in world units.
        let dot = canvas.dot();
        let mut lines = Vec::with_capacity(self.molecule.bonds().len() * 3);
        for &bond in self.molecule.bonds() {
            let (s, e) = (bond.start().get(), bond.end().get());
            let color = Self::shade(
                Self::BOND_COLOR,
                // Nearer bonds are brighter
                Self::shade_depth(f64::midpoint(proj[s].2, proj[e].2), zmin, zmax),
            );
            let (x1, y1) = (proj[s].0, proj[s].1);
            let (x2, y2) = (proj[e].0, proj[e].1);
            let (dx, dy) = (x2 - x1, y2 - y1);
            let len = (dx * dx + dy * dy).sqrt();
            let order = bond.order();
            let mut push_line = |ax, ay, bx, by| {
                lines.push(CanvasLine {
                    x1: ax,
                    y1: ay,
                    x2: bx,
                    y2: by,
                    color,
                });
            };

            // When the bond points almost at the viewer (projected length
            // below a couple of dots) a perpendicular offset is noise, so
            // it collapses to a single line.
            let parallel = len >= Self::BOND_MIN_PARALLEL_LENGTH_DOTS * dot;
            if parallel && order != BondOrder::Single {
                let off = Self::BOND_PARALLEL_OFFSET_DOTS * dot;
                let (nx, ny) = (-dy / len * off, dx / len * off);
                for (ox, oy) in [(nx, ny), (-nx, -ny)] {
                    push_line(x1 + ox, y1 + oy, x2 + ox, y2 + oy);
                }
            }
            // A double bond's offset lines stand in for the axis; every
            // other case keeps the central line.
            if order != BondOrder::Double || !parallel {
                push_line(x1, y1, x2, y2);
            }
        }
        lines
    }

    /// Sets the widget style, draws the optional block, and renders the molecule
    /// into the inner area. Returns the canvas mapping used, for hit-testing.
    fn render_inner(&self, area: Rect, buf: &mut Buffer) -> MoleculeCanvas {
        buf.set_style(area, self.style);

        let molecule_area = self.render_outer_block(area, buf);
        self.render_molecule(molecule_area, buf)
    }

    fn render_outer_block(&self, area: Rect, buf: &mut Buffer) -> Rect {
        let block = match (self.block.clone(), self.show_molecule_legend) {
            // No user block and no legend: nothing to wrap, draw on the raw area.
            (None, false) => return area,
            (None, true) => Block::bordered().title_bottom(self.draw_molecule_legend()),
            (Some(block), true) => block.title_bottom(self.draw_molecule_legend()),
            (Some(block), false) => block,
        };

        let inner = block.inner(area);
        block.render(area, buf);
        inner
    }

    fn render_molecule(&self, area: Rect, buf: &mut Buffer) -> MoleculeCanvas {
        let canvas = MoleculeCanvas::new(area, self.molecule.radius(), self.camera.zoom());
        if area.is_empty() {
            return canvas;
        }

        let proj: Vec<(f64, f64, f64)> = self
            .molecule
            .atoms()
            .iter()
            .map(|atom| {
                let [x, y, z] = atom.position();
                self.camera.project_point(x, y, z)
            })
            .collect();

        let proj_depths: Vec<f64> = proj.iter().map(|p| p.2).collect();
        let zmin = proj_depths.iter().copied().fold(f64::INFINITY, f64::min);
        let zmax = proj_depths
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        // One braille dot, in world units.
        let dot = canvas.dot();

        // Bond lines are drawn before the atoms, so the atom disks still
        // occlude the bond ends.
        let bond_lines = self.bond_lines(&proj, &canvas, zmin, zmax);

        // Atoms as small screen-space disks, drawn back-to-front (painter's
        // algorithm) by projected depth so nearer atoms occlude farther ones.
        // Consecutive atoms sharing a shaded color are merged into one Points
        // call to keep the draw count and the terminal's color escapes low.
        let order = Self::back_to_front_order(&proj_depths);

        let mut groups: Vec<(ratatui::style::Color, Vec<(f64, f64)>)> = Vec::new();
        for i in order {
            let atom = &self.molecule.atoms()[i];
            // Nearer atoms are brighter.
            let color = Self::shade(atom.cpk(), Self::shade_depth(proj_depths[i], zmin, zmax));
            if groups.last().map(|(c, _)| *c) != Some(color) {
                groups.push((color, Vec::new()));
            }
            let pts = &mut groups.last_mut().expect("just pushed").1;
            let r_dots = canvas.atom_radius_dots(atom.covalent_radius());
            let n = r_dots.ceil() as i32;
            for di in -n..=n {
                for dj in -n..=n {
                    if f64::from(di * di + dj * dj) <= r_dots * r_dots {
                        pts.push((
                            proj[i].0 + f64::from(di) * dot,
                            proj[i].1 + f64::from(dj) * dot,
                        ));
                    }
                }
            }
        }
        // Drawn last, on top of the atoms they mark, so the selection stays
        // visible. An empty ring list stays `None` so that an unhighlighted
        // render issues no extra draw call at all.
        let selected = self.highlighted_indices();
        let marker_color = self.marker_color();
        let highlight_rings = marker_color
            .map(|color| (self.highlight_rings(&selected, &proj, &canvas), color))
            .filter(|(pts, _)| !pts.is_empty());
        let measurement =
            marker_color.and_then(|c| self.measurement_overlay(&selected, &proj, &canvas, c));

        let drawing_canvas = Canvas::default()
            .background_color(self.style.bg.unwrap_or(ratatui::style::Color::Reset))
            .x_bounds([-canvas.bx, canvas.bx])
            .y_bounds([-canvas.by, canvas.by])
            .paint(move |ctx| {
                for line in &bond_lines {
                    ctx.draw(line);
                }

                for (color, pts) in &groups {
                    ctx.draw(&Points {
                        coords: pts,
                        color: *color,
                    });
                }

                if let Some((pts, color)) = &highlight_rings {
                    ctx.draw(&Points {
                        coords: pts,
                        color: *color,
                    });
                }

                if let Some(measurement) = &measurement {
                    if !measurement.points.is_empty() {
                        ctx.draw(&Points {
                            coords: &measurement.points,
                            color: measurement.color,
                        });
                    }
                    // The paint closure is `Fn`, not `FnOnce`, so the label is
                    // cloned rather than moved out.
                    let (x, y, line) = &measurement.label;
                    ctx.print(*x, *y, line.clone());
                }
            });

        drawing_canvas.render(area, buf);

        canvas
    }
}

#[cfg(test)]
mod tests {
    use crate::molecule::{Atom, AtomIndex, Bond};

    use super::*;

    use ratatui::style::{Color, Modifier};

    fn buffer_lines(buffer: &Buffer) -> Vec<String> {
        let area = *buffer.area();

        (0..area.height)
            .map(|y| {
                (0..area.width)
                    .map(|x| buffer[(area.x + x, area.y + y)].symbol())
                    .collect::<String>()
            })
            .collect()
    }

    fn create_molecule() -> Molecule {
        let atoms = vec![
            Atom::new(Element::C, [1.0, 0.0, 0.0]),
            Atom::new(Element::C, [0.0, 1.0, 0.0]),
            Atom::new(Element::C, [-1.0, 0.0, 0.0]),
            Atom::new(Element::C, [0.0, -1.0, 0.0]),
        ];
        atoms.into_iter().collect()
    }

    fn painted_cells(buffer: &Buffer) -> usize {
        buffer_lines(buffer)
            .iter()
            .flat_map(|line| line.chars())
            .filter(|c| !c.is_whitespace())
            .count()
    }

    /// Unicode codepoint of the first braille cell (`⠂` is the offset from it).
    const BRAILLE_BASE: u32 = 0x2800;

    /// The number of braille dots lit, counting the bits inside each braille
    /// cell rather than the cells themselves: a double bond's offset lines can
    /// sit in a different dot row of the *same* cell as its single-bond
    /// counterpart, which a cell count cannot see.
    fn lit_dots(buffer: &Buffer) -> usize {
        buffer_lines(buffer)
            .iter()
            .flat_map(|line| line.chars())
            .filter_map(|c| u32::from(c).checked_sub(BRAILLE_BASE))
            .filter(|pattern| *pattern < 0x100)
            .map(|pattern| pattern.count_ones() as usize)
            .sum()
    }

    fn render_to_buffer(viz: &MoleculeVisualizer<'_>) -> Buffer {
        render_to_buffer_at(viz, Rect::new(0, 0, 20, 10))
    }

    fn render_to_buffer_at(viz: &MoleculeVisualizer<'_>, area: Rect) -> Buffer {
        let mut buffer = Buffer::empty(area);
        Widget::render(viz, buffer.area, &mut buffer);
        buffer
    }

    #[test]
    fn highlight_adds_a_marker_ring() {
        let mol = create_molecule();
        let plain = render_to_buffer(&MoleculeVisualizer::new(&mol).show_bonds(false));
        let marked = render_to_buffer(
            &MoleculeVisualizer::new(&mol)
                .show_bonds(false)
                .highlight(Some(AtomIndex::new(0))),
        );

        assert!(
            painted_cells(&marked) > painted_cells(&plain),
            "highlighting an atom should paint additional marker cells"
        );
    }

    #[test]
    fn highlight_style_none_suppresses_the_marker() {
        let mol = create_molecule();
        let no_highlight = render_to_buffer(&MoleculeVisualizer::new(&mol).show_bonds(false));
        let suppressed = render_to_buffer(
            &MoleculeVisualizer::new(&mol)
                .show_bonds(false)
                .highlight(Some(AtomIndex::new(0)))
                .highlight_style(None),
        );

        assert_eq!(
            no_highlight, suppressed,
            "highlight_style(None) should draw exactly as if no atom were highlighted"
        );
    }

    #[test]
    fn out_of_range_highlight_is_ignored() {
        let mol = create_molecule();
        let no_highlight = render_to_buffer(&MoleculeVisualizer::new(&mol).show_bonds(false));
        let out_of_range = render_to_buffer(
            &MoleculeVisualizer::new(&mol)
                .show_bonds(false)
                .highlight(Some(AtomIndex::new(999))),
        );

        assert_eq!(no_highlight, out_of_range);
    }

    /// The default [`create_molecule`] diamond's expected buffer. Its edges
    /// perceive as double bonds, so each edge draws a pair of parallel lines.
    fn diamond_expected() -> Vec<String> {
        vec![
            "┌──────────────────┐".to_string(),
            "│      ⣀⣿⣿⣿⣿       │".to_string(),
            "│ ⢀  ⣠⠮⠊  ⠁⠣⡱⡀     │".to_string(),
            "│⣿⣿⣿⣿⠁      ⠘⢌⢆    │".to_string(),
            "│⣿⣿⣿⣿⠁       ⠈⢢⢱⣀⣀⣀│".to_string(),
            "│⠉⠉⠉⢏⠢⡀       ⢀⣿⣿⣿⣿│".to_string(),
            "│    ⠱⡑⡄      ⢀⣿⣿⣿⣿│".to_string(),
            "│     ⠈⢎⢆ ⡀ ⡠⡲⠋  ⠁ │".to_string(),
            "│       ⣿⣿⣿⣿⠉      │".to_string(),
            "└─────── C ────────┘".to_string(),
        ]
    }

    #[test]
    fn mol_gets_drawn() {
        let mol = create_molecule();
        let viz = MoleculeVisualizer::new(&mol).show_bonds(true);

        let buffer = render_to_buffer(&viz);

        assert_eq!(buffer_lines(&buffer), diamond_expected());
    }

    #[test]
    fn empty_mol_draws_empty_canvas() {
        let empty_mol = Molecule::from_atoms(Vec::new());
        let viz = MoleculeVisualizer::new(&empty_mol).show_bonds(true);

        let buffer = render_to_buffer(&viz);

        let expected = vec![
            "┌──────────────────┐".to_string(),
            "│                  │".to_string(),
            "│                  │".to_string(),
            "│                  │".to_string(),
            "│                  │".to_string(),
            "│                  │".to_string(),
            "│                  │".to_string(),
            "│                  │".to_string(),
            "│                  │".to_string(),
            "└──────────────────┘".to_string(),
        ];

        assert_eq!(buffer_lines(&buffer), expected);
    }

    #[test]
    fn mol_gets_drawn_without_bonds() {
        let mol = create_molecule();
        let viz = MoleculeVisualizer::new(&mol).show_bonds(false);

        let buffer = render_to_buffer(&viz);

        let expected = vec![
            "┌──────────────────┐".to_string(),
            "│       ⢿⣿⣿⡿       │".to_string(),
            "│ ⢀       ⠁        │".to_string(),
            "│⣿⣿⣿⣷              │".to_string(),
            "│⣿⣿⣿⣿⠁         ⢀⣀⣀⣀│".to_string(),
            "│⠉⠉⠉⠁         ⢀⣿⣿⣿⣿│".to_string(),
            "│              ⢿⣿⣿⣿│".to_string(),
            "│         ⡀      ⠁ │".to_string(),
            "│       ⣾⣿⣿⣷       │".to_string(),
            "└─────── C ────────┘".to_string(),
        ];
        assert_eq!(buffer_lines(&buffer), expected);
    }

    #[test]
    fn mol_gets_drawn_without_legend() {
        let mol = create_molecule();
        let viz = MoleculeVisualizer::new(&mol)
            .show_bonds(false)
            .show_molecule_legend(false)
            .block(Block::bordered());

        let buffer = render_to_buffer_at(&viz, Rect::new(0, 0, 10, 5));

        let expected = vec![
            "┌────────┐".to_string(),
            "│⣠⣤⡀⠲⠖   │".to_string(),
            "│⠻⠿⠃  ⢠⣶⣦│".to_string(),
            "│   ⠴⠦⠈⠛⠋│".to_string(),
            "└────────┘".to_string(),
        ];
        assert_eq!(buffer_lines(&buffer), expected);
    }

    #[test]
    fn setting_style_changes_border() {
        let empty_mol = Molecule::from_atoms(Vec::new());
        let viz = MoleculeVisualizer::new(&empty_mol)
            .show_bonds(false)
            .style(Style::new().red())
            .block(Block::bordered())
            .show_molecule_legend(true);

        let buffer = render_to_buffer_at(&viz, Rect::new(0, 0, 10, 5));

        let mut expected = Buffer::with_lines([
            "┌────────┐",
            "│        │",
            "│        │",
            "│        │",
            "└────────┘",
        ]);
        expected.set_style(buffer.area, Style::new().red());

        assert_eq!(buffer, expected);
    }

    #[test]
    fn dont_double_draw_block() {
        let mol = create_molecule();
        let viz = MoleculeVisualizer::new(&mol)
            .show_bonds(false)
            .block(Block::bordered().title("user"))
            .show_molecule_legend(true);

        let buffer = render_to_buffer_at(&viz, Rect::new(0, 0, 10, 5));

        let expected = vec![
            "┌user────┐".to_string(),
            "│⣠⣤⡀⠲⠖   │".to_string(),
            "│⠻⠿⠃  ⢠⣶⣦│".to_string(),
            "│   ⠴⠦⠈⠛⠋│".to_string(),
            "└── C ───┘".to_string(),
        ];
        assert_eq!(buffer_lines(&buffer), expected);
    }

    #[test]
    fn render_in_minimal_buffer() {
        let mol = create_molecule();
        let viz = MoleculeVisualizer::new(&mol);

        // This should not panic, even if the buffer is too small to render.
        let buffer = render_to_buffer_at(&viz, Rect::new(0, 0, 1, 1));
        assert_eq!(buffer, Buffer::with_lines(["┌"]));
    }

    #[test]
    fn render_in_zero_size_buffer() {
        let mol = create_molecule();
        let viz = MoleculeVisualizer::new(&mol);

        // This should not panic, even if the buffer has zero size.
        render_to_buffer_at(&viz, Rect::ZERO);
    }

    #[test]
    fn atoms_have_color() {
        let molecule = vec![Atom::new(Element::N, [0.0, 0.0, 0.0])]
            .into_iter()
            .collect();
        let viz = MoleculeVisualizer::new(&molecule);

        let buffer = render_to_buffer(&viz);

        let mut expected = Buffer::with_lines([
            "┌──────────────────┐",
            "│                  │",
            "│                  │",
            "│         ⡀        │",
            "│       ⣾⣿⣿⣷       │",
            "│      ⠈⢿⣿⣿⡿⠁      │",
            "│         ⠁        │",
            "│                  │",
            "│                  │",
            "└─────── N ────────┘",
        ]);

        expected[(10, 3)].set_fg(Color::Rgb(143, 143, 255));
        expected[(8, 4)].set_fg(Color::Rgb(143, 143, 255));
        expected[(9, 4)].set_fg(Color::Rgb(143, 143, 255));
        expected[(10, 4)].set_fg(Color::Rgb(143, 143, 255));
        expected[(11, 4)].set_fg(Color::Rgb(143, 143, 255));
        expected[(7, 5)].set_fg(Color::Rgb(143, 143, 255));
        expected[(8, 5)].set_fg(Color::Rgb(143, 143, 255));
        expected[(9, 5)].set_fg(Color::Rgb(143, 143, 255));
        expected[(10, 5)].set_fg(Color::Rgb(143, 143, 255));
        expected[(11, 5)].set_fg(Color::Rgb(143, 143, 255));
        expected[(12, 5)].set_fg(Color::Rgb(143, 143, 255));
        expected[(10, 6)].set_fg(Color::Rgb(143, 143, 255));
        for col in [8, 9, 10] {
            expected[(col, 9)].set_style(
                Style::default()
                    .fg(Color::Rgb(143, 143, 255))
                    .add_modifier(Modifier::BOLD),
            );
        }

        assert_eq!(buffer, expected);
    }

    #[test]
    fn back_to_front_order_draws_nearest_last() {
        let depths = [-2.0, 1.0, 0.0];

        assert_eq!(MoleculeVisualizer::back_to_front_order(&depths), [0, 2, 1]);
    }

    #[test]
    fn depth_factor_brightens_nearer_depths() {
        assert!(
            MoleculeVisualizer::shade_depth(2.0, -1.0, 3.0)
                > MoleculeVisualizer::shade_depth(-1.0, -1.0, 3.0)
        );
    }

    #[test]
    fn dont_shade_no_depth() {
        assert_eq!(MoleculeVisualizer::shade_depth(1.0, 1.0, 1.000001), 1.0);
    }

    fn atom(x: f64, y: f64, z: f64) -> Atom {
        Atom::new(Element::C, [x, y, z])
    }

    #[test]
    fn pick_atom_maps_center_cell_to_origin_atom() {
        let molecule: Molecule = vec![atom(0.0, 0.0, 0.0)].into_iter().collect();
        let canvas = MoleculeCanvas::new(Rect::new(0, 0, 20, 10), molecule.radius(), 1.0);
        let camera = Camera::new(0.0, 0.0, 1.0);

        // Clicking the middle of the canvas hits the atom sitting at the origin.
        assert_eq!(
            canvas.pick_atom(camera, &molecule, (10, 5)),
            Some(AtomIndex::new(0))
        );

        // A corner click lands on empty space.
        assert_eq!(canvas.pick_atom(camera, &molecule, (0, 0)), None);

        // A click outside the canvas rect is rejected outright.
        assert_eq!(canvas.pick_atom(camera, &molecule, (99, 99)), None);
    }

    #[test]
    fn cell_to_data_maps_cells_to_the_canvas_bounds() {
        let canvas = MoleculeCanvas::new(Rect::new(2, 3, 20, 10), 1.0, 1.0);
        let by = MoleculeCanvas::EDGE_PADDING;
        let bx = by; // 20x10 packs 2 dots/cell across and 4 down, a square grid

        // The top-left cell's center samples near (-bx, by); the bottom-right
        // cell's near (bx, -by).
        let (x, y) = canvas.cell_to_data((2, 3)).unwrap();
        assert!((x + bx - 0.5 * 2.0 * bx / 20.0).abs() < 1e-9);
        assert!((y - by + 0.5 * 2.0 * by / 10.0).abs() < 1e-9);
        let (x, y) = canvas.cell_to_data((21, 12)).unwrap();
        assert!((x - bx + 0.5 * 2.0 * bx / 20.0).abs() < 1e-9);
        assert!((y + by - 0.5 * 2.0 * by / 10.0).abs() < 1e-9);

        // Cells outside the canvas area are rejected.
        assert_eq!(canvas.cell_to_data((1, 3)), None);
        assert_eq!(canvas.cell_to_data((22, 3)), None);
    }

    #[test]
    fn panning_moves_the_atom_by_the_same_cells() {
        // The user-facing round trip: a click hit-tests, a drag converts cell
        // deltas to world units, and the next click lands where the molecule
        // moved to.
        let molecule: Molecule = vec![atom(0.0, 0.0, 0.0)].into_iter().collect();
        let camera = Camera::new(0.0, 0.0, 1.0);
        // Odd dimensions so cell (10, 5) is exactly centered on the origin.
        let canvas = MoleculeCanvas::new(Rect::new(0, 0, 21, 11), molecule.radius(), 1.0);

        let (cx, cy) = (10, 5);
        assert_eq!(
            canvas.pick_atom(camera, &molecule, (cx, cy)),
            Some(AtomIndex::new(0))
        );

        // Drag 2 cells right, 1 cell down.
        let mut panned = camera;
        let (dx, dy) = canvas.cell_delta_to_world(2, 1);
        panned.translate(dx, dy);

        assert_eq!(
            canvas.pick_atom(panned, &molecule, (cx + 2, cy + 1)),
            Some(AtomIndex::new(0))
        );
        assert_eq!(canvas.pick_atom(panned, &molecule, (cx, cy)), None);
    }

    #[test]
    fn canvas_reports_whether_a_cell_is_inside_its_area() {
        let canvas = MoleculeCanvas::new(Rect::new(2, 3, 5, 4), 1.0, 1.0);

        assert!(canvas.contains_cell((2, 3)));
        assert!(canvas.contains_cell((6, 6)));
        assert!(!canvas.contains_cell((7, 6)));
        assert!(!canvas.contains_cell((6, 7)));
    }

    #[test]
    fn pick_atom_prefers_the_front_atom_on_overlap() {
        // Two atoms at the same projected (x, y) but different depth; the one
        // nearer the viewer (larger projected z) must win.
        let molecule: Molecule = vec![atom(0.0, 0.0, 2.0), atom(0.0, 0.0, -2.0)]
            .into_iter()
            .collect();
        let canvas = MoleculeCanvas::new(Rect::new(0, 0, 20, 10), molecule.radius(), 1.0);
        let camera = Camera::new(0.0, 0.0, 1.0);

        assert_eq!(
            canvas.pick_atom(camera, &molecule, (10, 5)),
            Some(AtomIndex::new(0))
        );
    }

    const CAMERA_ROTATION_STEP: f64 = 0.12;

    #[test]
    fn zoom_camera() {
        let mol = create_molecule();
        let mut viz = MoleculeVisualizer::new(&mol);

        viz.camera.zoom_by(2.0);
        let buffer = render_to_buffer(&viz);

        let expected = vec![
            "┌──────────────────┐".to_string(),
            "│              ⠈⢎⢆ │".to_string(),
            "│                ⠣⡣│".to_string(),
            "│                 ⠑│".to_string(),
            "│                  │".to_string(),
            "│                  │".to_string(),
            "│⢄                 │".to_string(),
            "│⢪⢢                │".to_string(),
            "│ ⠱⡱⡀              │".to_string(),
            "└─────── C ────────┘".to_string(),
        ];

        assert_eq!(buffer_lines(&buffer), expected);
    }

    #[test]
    fn rotate_camera() {
        let mol = create_molecule();
        let mut viz = MoleculeVisualizer::new(&mol);

        viz.camera
            .rotate(6.0 * CAMERA_ROTATION_STEP, -CAMERA_ROTATION_STEP * 6.0);
        let buffer = render_to_buffer(&viz);

        let expected = vec![
            "┌──────────────────┐".to_string(),
            "│       ⢶⣾⣿⡶       │".to_string(),
            "│       ⢀⢿⣹⣼⣦⣤⡀    │".to_string(),
            "│       ⡸⡧⣿⣿⣿⣿⡧    │".to_string(),
            "│      ⢀⢧⠃⠻⡿⡿⠿⠃    │".to_string(),
            "│    ⢠⣶⣾⣾⣦⢠⢳⠁      │".to_string(),
            "│    ⢺⣿⣿⣿⣿⢺⡎       │".to_string(),
            "│    ⠈⠛⠻⡟⡏⣷⠁       │".to_string(),
            "│       ⠾⣿⡿⠷       │".to_string(),
            "└─────── C ────────┘".to_string(),
        ];

        assert_eq!(buffer_lines(&buffer), expected);
    }

    #[test]
    fn reset_camera() {
        let mol = create_molecule();
        let mut viz = MoleculeVisualizer::new(&mol);

        viz.camera
            .rotate(6.0 * CAMERA_ROTATION_STEP, -CAMERA_ROTATION_STEP * 6.0);
        viz.camera.zoom_by(2.0);
        viz.camera.reset();
        let buffer = render_to_buffer(&viz);

        assert_eq!(buffer_lines(&buffer), diamond_expected());
    }

    #[test]
    fn bonds_can_be_supplied_as_tuples() {
        let atoms = vec![
            Atom::new(Element::C, [0.0, 0.0, 0.0]),
            Atom::new(Element::C, [1.5, 0.0, 0.0]),
            Atom::new(Element::C, [3.0, 0.0, 0.0]),
        ];
        let mol = Molecule::from_atoms_with_bonds(atoms, [(0, 1), (1, 2)]);

        assert_eq!(mol.bonds(), vec![Bond::from((0, 1)), Bond::from((1, 2))]);
        assert_eq!(
            mol.bonds()[0],
            Bond::new(AtomIndex::new(0), AtomIndex::new(1))
        );
    }

    /// A two-carbon molecule with an explicit bond `order`, the atoms `d` (Å)
    /// apart along the x-axis.
    fn diatomic(order: BondOrder, d: f64) -> Molecule {
        Molecule::from_atoms_with_bonds(
            [
                Atom::new(Element::C, [0.0, 0.0, 0.0]),
                Atom::new(Element::C, [d, 0.0, 0.0]),
            ],
            [Bond::new(AtomIndex::new(0), AtomIndex::new(1)).with_order(order)],
        )
    }

    #[test]
    fn higher_bond_orders_light_more_dots() {
        let light = |order: BondOrder| {
            let mol = diatomic(order, 1.34);
            lit_dots(&render_to_buffer(&MoleculeVisualizer::new(&mol)))
        };

        let single = light(BondOrder::Single);
        let double = light(BondOrder::Double);
        let triple = light(BondOrder::Triple);

        assert!(
            double > single,
            "a double bond ({double} dots) should light more than a single ({single})"
        );
        assert!(
            triple > double,
            "a triple bond ({triple} dots) should light more than a double ({double})"
        );
    }

    #[test]
    fn head_on_bond_renders_without_panic() {
        // With the identity camera the viewing axis is world z, so a bond
        // along z projects to a point — the degenerate case for parallel
        // lines must collapse to a single line rather than panic.
        let mol = Molecule::from_atoms_with_bonds(
            [
                Atom::new(Element::C, [0.0, 0.0, -0.67]),
                Atom::new(Element::C, [0.0, 0.0, 0.67]),
            ],
            [Bond::new(AtomIndex::new(0), AtomIndex::new(1)).with_order(BondOrder::Double)],
        );
        let viz = MoleculeVisualizer::new(&mol).camera(Camera::new(0.0, 0.0, 1.0));

        let buffer = render_to_buffer(&viz);
        let expected = vec![
            "┌──────────────────┐".to_string(),
            "│                  │".to_string(),
            "│                  │".to_string(),
            "│         ⡀        │".to_string(),
            "│       ⣾⣿⣿⣷       │".to_string(),
            "│      ⠈⢿⣿⣿⡿⠁      │".to_string(),
            "│         ⠁        │".to_string(),
            "│                  │".to_string(),
            "│                  │".to_string(),
            "└─────── C ────────┘".to_string(),
        ];

        assert_eq!(buffer_lines(&buffer), expected);
    }

    /// The showcase example's caffeine, kept in sync with
    /// `examples/showcase.rs` so the feature is verified on the coordinates
    /// the README gif actually renders.
    fn caffeine() -> Molecule {
        Molecule::from_atoms([
            Atom::new(Element::C, [0.000, 1.402, 0.000]),
            Atom::new(Element::N, [1.214, 0.701, 0.060]),
            Atom::new(Element::C, [1.214, -0.701, -0.020]),
            Atom::new(Element::N, [0.000, -1.402, 0.080]),
            Atom::new(Element::C, [-1.214, -0.701, -0.030]),
            Atom::new(Element::C, [-1.214, 0.701, 0.020]),
            Atom::new(Element::O, [0.000, 2.620, -0.080]),
            Atom::new(Element::O, [2.300, -1.290, 0.050]),
            Atom::new(Element::N, [-2.420, -1.360, 0.060]),
            Atom::new(Element::N, [2.420, 1.360, -0.050]),
            Atom::new(Element::C, [-3.620, -0.600, -0.120]),
            Atom::new(Element::C, [3.620, 0.600, 0.120]),
            Atom::new(Element::C, [0.000, -2.820, -0.140]),
            Atom::new(Element::H, [-4.460, -1.290, -0.030]),
            Atom::new(Element::H, [-3.560, -0.170, -1.130]),
            Atom::new(Element::H, [-3.760, 0.210, 0.610]),
            Atom::new(Element::H, [4.460, 1.290, 0.030]),
            Atom::new(Element::H, [3.560, 0.170, 1.130]),
            Atom::new(Element::H, [3.760, -0.210, -0.610]),
            Atom::new(Element::H, [-0.900, -3.390, -0.030]),
            Atom::new(Element::H, [0.880, -3.400, 0.030]),
            Atom::new(Element::H, [0.020, -2.930, -1.230]),
        ])
    }

    #[test]
    fn caffeine_carbonyls_perceive_as_double_bonds() {
        let mol = caffeine();

        let carbonyls = mol
            .bonds()
            .iter()
            .filter(|bond| {
                let (a, b) = (
                    &mol.atoms()[bond.start().get()],
                    &mol.atoms()[bond.end().get()],
                );
                matches!(
                    (a.element(), b.element()),
                    (Element::C, Element::O) | (Element::O, Element::C)
                )
            })
            .collect::<Vec<_>>();

        assert_eq!(carbonyls.len(), 2, "expected exactly two C–O bonds");
        for bond in carbonyls {
            assert_eq!(
                bond.order(),
                BondOrder::Double,
                "carbonyl bond {}–{} should be double",
                bond.start(),
                bond.end()
            );
        }
    }

    /// The whole buffer as one string, for asserting on label *text* rather
    /// than on braille art, which any dot-level change would shift.
    fn buffer_text(buffer: &Buffer) -> String {
        buffer_lines(buffer).join("\n")
    }

    /// Every braille cell's position and glyph. Comparing these between two
    /// renders isolates "did a dot move" from "did a label appear".
    fn braille_cells(buffer: &Buffer) -> Vec<(u16, u16, String)> {
        let area = *buffer.area();
        (0..area.height)
            .flat_map(|y| (0..area.width).map(move |x| (x, y)))
            .filter_map(|(x, y)| {
                let symbol = buffer[(area.x + x, area.y + y)].symbol();
                let braille = symbol
                    .chars()
                    .next()
                    .and_then(|c| u32::from(c).checked_sub(BRAILLE_BASE))
                    .is_some_and(|pattern| pattern < 0x100);
                braille.then(|| (x, y, symbol.to_string()))
            })
            .collect()
    }

    /// A canvas with room for a measurement label beside the molecule.
    fn wide() -> Rect {
        Rect::new(0, 0, 44, 14)
    }

    /// A camera looking straight down the world z axis, so the projection is
    /// the x-y plane and test coordinates land where they read.
    fn head_on() -> Camera {
        Camera::new(0.0, 0.0, 1.0)
    }

    fn highlighted_with(
        mol: &Molecule,
        atoms: impl IntoIterator<Item = usize>,
        measure: bool,
    ) -> Buffer {
        render_to_buffer_at(
            &MoleculeVisualizer::new(mol)
                .camera(head_on())
                .highlight(atoms.into_iter().map(AtomIndex::new))
                .show_measurement(measure),
            wide(),
        )
    }

    fn highlighted(mol: &Molecule, atoms: impl IntoIterator<Item = usize>) -> Buffer {
        highlighted_with(mol, atoms, true)
    }

    /// The label may *cover* braille cells — `Context::print` writes whole cells
    /// over every layer — so the check is that the overlay lights no cell that
    /// was not already lit, which a stray corner dot would.
    fn assert_lights_no_new_cell(before: &Buffer, after: &Buffer, what: &str) {
        let already_lit: Vec<(u16, u16)> = braille_cells(before)
            .into_iter()
            .map(|(x, y, _)| (x, y))
            .collect();
        for (x, y, glyph) in braille_cells(after) {
            assert!(
                already_lit.contains(&(x, y)),
                "{what} lit a new braille cell {glyph:?} at ({x}, {y})"
            );
        }
    }

    #[test]
    fn an_empty_highlight_takes_no_overlay_code_path() {
        // Every literal-art snapshot in this module renders without a
        // highlight. This pins the reason they are allowed to stay literal:
        // with nothing highlighted, the ring and overlay code must not reach
        // the canvas at all.
        let mol = create_molecule();
        let plain = render_to_buffer(&MoleculeVisualizer::new(&mol));
        let measured = render_to_buffer(&MoleculeVisualizer::new(&mol).show_measurement(true));
        let unmeasured = render_to_buffer(&MoleculeVisualizer::new(&mol).show_measurement(false));

        assert_eq!(plain, measured);
        assert_eq!(plain, unmeasured);
        assert_eq!(buffer_lines(&plain), diamond_expected());
    }

    #[test]
    fn highlight_draws_a_ring_on_every_selected_atom() {
        let mol = create_molecule();
        let rings = |atoms: &[usize]| {
            painted_cells(&render_to_buffer(
                &MoleculeVisualizer::new(&mol)
                    .show_bonds(false)
                    .show_measurement(false)
                    .highlight(atoms.iter().copied().map(AtomIndex::new)),
            ))
        };

        assert!(rings(&[0]) > rings(&[]), "one ring should paint more cells");
        assert!(
            rings(&[0, 1]) > rings(&[0]),
            "a second ring should paint more still"
        );
    }

    #[test]
    fn highlighting_the_same_atom_twice_draws_one_ring() {
        let mol = create_molecule();

        assert_eq!(
            highlighted(&mol, [0]),
            highlighted(&mol, [0, 0]),
            "a repeated atom must collapse, not measure itself against itself"
        );
    }

    #[test]
    fn an_empty_highlight_list_is_the_unhighlighted_render() {
        let mol = create_molecule();
        let plain = render_to_buffer(&MoleculeVisualizer::new(&mol));

        assert_eq!(
            plain,
            render_to_buffer(&MoleculeVisualizer::new(&mol).highlight([]))
        );
        assert_eq!(
            plain,
            render_to_buffer(&MoleculeVisualizer::new(&mol).highlight(None::<AtomIndex>))
        );
    }

    #[test]
    fn out_of_range_indices_are_dropped_from_a_mixed_selection() {
        let mol = create_molecule();

        // Only atom 0 survives, so this is a lone ring and no measurement.
        assert_eq!(highlighted(&mol, [0, 999]), highlighted(&mol, [0]));
    }

    #[test]
    fn highlight_style_none_suppresses_rings_and_overlay() {
        let mol = create_molecule();
        let plain = render_to_buffer(&MoleculeVisualizer::new(&mol));
        let suppressed = render_to_buffer(
            &MoleculeVisualizer::new(&mol)
                .highlight([AtomIndex::new(0), AtomIndex::new(1)])
                .highlight_style(None),
        );

        assert_eq!(
            plain, suppressed,
            "highlight_style(None) must kill the whole marker apparatus"
        );
    }

    /// Two atoms 1.8 Å apart in the screen plane, well clear of each other.
    fn pair() -> Molecule {
        Molecule::from_atoms([
            Atom::new(Element::C, [0.0, 0.9, 0.0]),
            Atom::new(Element::O, [0.0, -0.9, 0.0]),
        ])
    }

    #[test]
    fn two_highlighted_atoms_draw_a_measurement() {
        let mol = pair();
        let with = |measure| painted_cells(&highlighted_with(&mol, [0, 1], measure));

        assert!(
            with(true) > with(false),
            "the overlay should paint connectors and a label"
        );
    }

    #[test]
    fn one_highlighted_atom_draws_no_measurement() {
        let mol = pair();
        let one = |measure| highlighted_with(&mol, [0], measure);

        assert_eq!(one(true), one(false));
    }

    #[test]
    fn five_highlighted_atoms_draw_rings_but_no_measurement() {
        let mol = caffeine();
        let five = |measure| highlighted_with(&mol, 0..5, measure);

        assert_eq!(five(true), five(false), "five atoms measure nothing");
        let text = buffer_text(&five(true));
        assert!(!text.contains('°') && !text.contains('Å'), "{text}");
    }

    #[test]
    fn the_distance_label_prints_the_true_value() {
        let text = buffer_text(&highlighted(&pair(), [0, 1]));

        assert!(
            text.contains("1.800 Å"),
            "expected the distance in:\n{text}"
        );
    }

    #[test]
    fn the_distance_label_is_the_3d_value_not_the_projected_one() {
        // Both atoms lie on the viewing axis, so they project to a single point
        // and there is no connector to draw — but the distance between them is
        // still 1.340 Å, and that is what the reader needs.
        let along_view = Molecule::from_atoms([
            Atom::new(Element::C, [0.0, 0.0, -0.67]),
            Atom::new(Element::C, [0.0, 0.0, 0.67]),
        ]);
        let text = buffer_text(&highlighted(&along_view, [0, 1]));

        assert!(
            text.contains("1.340 Å"),
            "expected the 3-D distance in:\n{text}"
        );
    }

    /// A right angle at atom 1, with atom 0 out along x and atom 2 up along y.
    fn bent() -> Molecule {
        Molecule::from_atoms([
            Atom::new(Element::C, [1.4, 0.0, 0.0]),
            Atom::new(Element::O, [0.0, 0.0, 0.0]),
            Atom::new(Element::C, [0.0, 1.4, 0.0]),
        ])
    }

    #[test]
    fn three_highlighted_atoms_label_the_angle() {
        let text = buffer_text(&highlighted(&bent(), [0, 1, 2]));

        assert!(text.contains("90.0°"), "expected the angle in:\n{text}");
    }

    #[test]
    fn the_angle_vertex_is_the_middle_highlighted_atom() {
        let mol = bent();

        // Same three atoms, different vertex: 90 degrees at atom 1, 45 at atom 0.
        assert!(buffer_text(&highlighted(&mol, [0, 1, 2])).contains("90.0°"));
        assert!(buffer_text(&highlighted(&mol, [1, 0, 2])).contains("45.0°"));
    }

    #[test]
    fn four_highlighted_atoms_label_the_dihedral() {
        let chain = Molecule::from_atoms([
            Atom::new(Element::C, [0.0, 1.0, 0.0]),
            Atom::new(Element::C, [0.0, 0.0, 0.0]),
            Atom::new(Element::C, [1.5, 0.0, 0.0]),
            Atom::new(Element::C, [1.5, 0.0, 1.0]),
        ]);
        let text = buffer_text(&highlighted(&chain, [0, 1, 2, 3]));

        assert!(text.contains("90.0°"), "expected the dihedral in:\n{text}");
    }

    #[test]
    fn a_degenerate_measurement_moves_no_braille_dot() {
        // `Painter::get_point` bounds-checks a coordinate with four comparisons
        // that are all false for NaN, then saturates the cast — so a NaN
        // reaching `Points` paints a dot in the canvas's top-left cell instead
        // of panicking. Comparing only the braille cells, and letting the label
        // differ, is what catches that stray dot.
        let along_view = Molecule::from_atoms([
            Atom::new(Element::C, [0.0, 0.0, -0.67]),
            Atom::new(Element::C, [0.0, 0.0, 0.67]),
        ]);
        let viz = |measure| highlighted_with(&along_view, [0, 1], measure);

        assert_lights_no_new_cell(
            &viz(false),
            &viz(true),
            "the overlay, though the atoms project to a single point,",
        );

        // And it must not have passed by drawing nothing: the value is the part
        // the reader actually needs.
        assert!(buffer_text(&viz(true)).contains("1.340 Å"));
    }

    #[test]
    fn coincident_atoms_draw_no_connectors() {
        // Every separation, arm, and axis is degenerate at once. Nothing may be
        // drawn between the atoms, and nothing may panic on the way to deciding
        // that — for any size of selection.
        let stacked = Molecule::from_atoms([Atom::new(Element::C, [0.0; 3]); 4]);

        for count in 1..=5 {
            let render = |measure| highlighted_with(&stacked, 0..count, measure);

            assert_lights_no_new_cell(
                &render(false),
                &render(true),
                &format!("{count} coincident atoms"),
            );
        }
    }

    #[test]
    fn highlighting_an_empty_molecule_is_a_no_op() {
        let empty = Molecule::from_atoms(Vec::new());

        assert_eq!(
            render_to_buffer_at(&MoleculeVisualizer::new(&empty), wide()),
            highlighted(&empty, 0..4),
            "indices must be range-checked before anything indexes the atoms"
        );
    }

    #[test]
    fn measurement_overlay_renders_in_a_minimal_buffer() {
        let mol = bent();
        let viz = MoleculeVisualizer::new(&mol).highlight((0..3).map(AtomIndex::new));

        // This should not panic, even with no room to draw the annotation.
        let buffer = render_to_buffer_at(&viz, Rect::new(0, 0, 1, 1));
        assert_eq!(buffer, Buffer::with_lines(["┌"]));
    }

    #[test]
    fn measurement_overlay_renders_in_a_zero_size_buffer() {
        let mol = bent();
        let viz = MoleculeVisualizer::new(&mol).highlight((0..3).map(AtomIndex::new));

        render_to_buffer_at(&viz, Rect::ZERO);
    }
}
