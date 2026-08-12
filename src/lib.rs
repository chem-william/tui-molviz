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
//! * `examples/quickstart.rs` is a simple example plotting a small organic molecule.
//! * `examples/showcase.rs` is a more complex example that showcases zoom, rotation, and selection of atoms

pub mod camera;
pub mod molecule;
use std::collections::HashSet;

use crate::camera::Camera;
use crate::molecule::Molecule;

pub use mendeleev::Color as CpkColor;
pub use mendeleev::Element;
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
    fn new(inner: Rect, radius: f64, zoom: f64) -> Self {
        let w = f64::from(inner.width.max(1));
        let h = f64::from(inner.height.max(1));
        let (rx, ry) = (2.0 * w, 4.0 * h);
        let by = (radius * Self::EDGE_PADDING) / zoom;
        let bx = by * (rx / ry);
        let dpu = ry / (2.0 * by);
        Self { inner, bx, by, dpu }
    }

    /// Drawn radius of an atom, in braille dots.
    fn atom_radius_dots(&self, cov: f64) -> f64 {
        (cov * Self::ATOM_RADIUS_SCALE * self.dpu)
            .clamp(Self::MIN_ATOM_RADIUS_DOTS, Self::MAX_ATOM_RADIUS_DOTS)
    }

    /// Inverse of the canvas mapping: the atom whose drawn disk a clicked
    /// terminal cell lands in (front-most on overlap), or `None` for empty
    /// space. The canvas maps data x in `[-bx, bx]` left→right and data y in
    /// `[-by, by]` bottom→top, so the row axis is flipped relative to screen rows.
    ///
    /// `camera` and `molecule` must match what the last render drew so the
    /// projection lines up with the pixels on screen.
    #[must_use]
    pub fn pick_atom(
        &self,
        camera: Camera,
        molecule: &Molecule,
        position: impl Into<Position>,
    ) -> Option<usize> {
        let position = position.into();
        if !self.inner.contains(position) {
            return None;
        }
        let (col, row) = (position.x, position.y);

        // Cell -> canvas-data coords, sampling the cell's center; y is flipped.
        let fx = (f64::from(col - self.inner.x) + 0.5) / f64::from(self.inner.width);
        let fy = (f64::from(row - self.inner.y) + 0.5) / f64::from(self.inner.height);
        let px = -self.bx + fx * 2.0 * self.bx;
        let py = self.by - fy * 2.0 * self.by;

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
            .map(|(i, _)| i)
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
    /// The camera used to display the molecule. Used to control rotation and zooming
    camera: Camera,
    /// Atom index to draw a highlight marker on, if any. Out-of-range indices
    /// are ignored at render time. Default is `None`.
    highlight: Option<usize>,
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
            highlight: None,
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

    /// Highlights the atom at the given index by drawing a marker ring around it,
    /// or clears the highlight with `None`. The index is typically one returned by
    /// [`MoleculeCanvas::pick_atom`]; out-of-range indices are ignored at render time.
    ///
    /// This is a fluent setter method which must be chained or used as it consumes self
    #[must_use = "method moves the value of self and returns the modified value"]
    pub const fn highlight(mut self, highlight: Option<usize>) -> Self {
        self.highlight = highlight;
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
    /// Fallback marker color when the highlight style has no foreground set.
    const DEFAULT_HIGHLIGHT_COLOR: ratatui::style::Color = ratatui::style::Color::White;
    /// Gap, in braille dots, between an atom's drawn disk and its highlight ring.
    const HIGHLIGHT_RING_GAP_DOTS: f64 = 1.5;
    /// Number of points sampled around the highlight ring.
    const HIGHLIGHT_RING_STEPS: u32 = 48;

    /// The marker ring for the highlighted atom, if one is set, its style has a
    /// marker color, and the index is in range. Returns the ring's braille points
    /// (in canvas-data coords) and color, for drawing on top of the molecule.
    fn highlight_ring(
        &self,
        proj: &[(f64, f64, f64)],
        canvas: &MoleculeCanvas,
    ) -> Option<(Vec<(f64, f64)>, ratatui::style::Color)> {
        let i = self.highlight.filter(|&i| i < proj.len())?;
        let color = self
            .highlight_style?
            .fg
            .unwrap_or(Self::DEFAULT_HIGHLIGHT_COLOR);

        let dot = 1.0 / canvas.dpu; // one braille dot, in world units
        let r_dots = canvas.atom_radius_dots(self.molecule.atoms()[i].covalent_radius());
        let r_ring = (r_dots + Self::HIGHLIGHT_RING_GAP_DOTS) * dot;
        let pts = (0..Self::HIGHLIGHT_RING_STEPS)
            .map(|k| {
                let theta =
                    std::f64::consts::TAU * f64::from(k) / f64::from(Self::HIGHLIGHT_RING_STEPS);
                let (s, c) = theta.sin_cos();
                (proj[i].0 + r_ring * c, proj[i].1 + r_ring * s)
            })
            .collect();
        Some((pts, color))
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

    fn visible_depth(projected_z: f64) -> f64 {
        projected_z
    }

    fn depth_factor(z: f64, zmin: f64, zspan: f64) -> f64 {
        Self::MIN_DEPTH_BRIGHTNESS + Self::DEPTH_BRIGHTNESS_RANGE * ((z - zmin) / zspan)
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
    fn shade(color: &CpkColor, f: f64) -> ratatui::style::Color {
        let f = (f.clamp(0.0, 1.0) * Self::SHADE_LEVELS).round() / Self::SHADE_LEVELS;
        ratatui::style::Color::Rgb(
            (f64::from(color.r) * f) as u8,
            (f64::from(color.g) * f) as u8,
            (f64::from(color.b) * f) as u8,
        )
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
            (None, true) => Block::bordered().title_bottom(self.draw_molecule_legend()),
            (None, false) => {
                return area;
            }
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

        let proj_depths: Vec<f64> = proj.iter().map(|p| Self::visible_depth(p.2)).collect();
        let zmin = proj_depths.iter().copied().fold(f64::INFINITY, f64::min);
        let zmax = proj_depths
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let zspan = (zmax - zmin).max(1e-6);
        // Nearer atoms are brighter.
        let depth = |z: f64| Self::depth_factor(z, zmin, zspan);

        let bond_lines: Vec<(f64, f64, f64, f64, ratatui::style::Color)> = if self.show_bonds {
            // Bonds split at their midpoint so each half takes its own atom's depth.
            self.molecule
                .bonds()
                .iter()
                .map(|&bond| {
                    let color = Self::shade(
                        &Self::BOND_COLOR,
                        depth(f64::midpoint(
                            proj_depths[bond.start()],
                            proj_depths[bond.end()],
                        )),
                    );
                    (
                        proj[bond.start()].0,
                        proj[bond.start()].1,
                        proj[bond.end()].0,
                        proj[bond.end()].1,
                        color,
                    )
                })
                .collect()
        } else {
            Vec::new()
        };

        // Atoms as small screen-space disks, drawn back-to-front (painter's
        // algorithm) by projected depth so nearer atoms occlude farther ones.
        // Consecutive atoms sharing a shaded color are merged into one Points
        // call to keep the draw count and the terminal's color escapes low.
        let dot = 1.0 / canvas.dpu; // one braille dot, in world units
        let order = Self::back_to_front_order(&proj_depths);

        let mut groups: Vec<(ratatui::style::Color, Vec<(f64, f64)>)> = Vec::new();
        for i in order {
            let atom = &self.molecule.atoms()[i];
            let color = Self::shade(&atom.cpk(), depth(proj_depths[i]));
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
        // Drawn last, on top of the atom it marks, so the selection stays visible.
        let highlight_ring = self.highlight_ring(&proj, &canvas);

        let drawing_canvas = Canvas::default()
            .background_color(self.style.bg.unwrap_or(ratatui::style::Color::Reset))
            .x_bounds([-canvas.bx, canvas.bx])
            .y_bounds([-canvas.by, canvas.by])
            .paint(move |ctx| {
                for &(x1, y1, x2, y2, color) in &bond_lines {
                    ctx.draw(&CanvasLine {
                        x1,
                        y1,
                        x2,
                        y2,
                        color,
                    });
                }

                for (color, pts) in &groups {
                    ctx.draw(&Points {
                        coords: pts,
                        color: *color,
                    });
                }

                if let Some((pts, color)) = &highlight_ring {
                    ctx.draw(&Points {
                        coords: pts,
                        color: *color,
                    });
                }
            });

        drawing_canvas.render(area, buf);

        canvas
    }
}

#[cfg(test)]
mod tests {
    use crate::molecule::{Atom, Bond};

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

    fn render_to_buffer(viz: &MoleculeVisualizer<'_>) -> Buffer {
        let mut buffer = Buffer::empty(Rect::new(0, 0, 20, 10));
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
                .highlight(Some(0)),
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
                .highlight(Some(0))
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
                .highlight(Some(999)),
        );

        assert_eq!(no_highlight, out_of_range);
    }

    #[test]
    fn mol_gets_drawn() {
        let mol = create_molecule();
        let viz = MoleculeVisualizer::new(&mol).show_bonds(true);

        let area = Rect::new(0, 0, 20, 10);
        let mut buffer = Buffer::empty(area);

        Widget::render(&viz, buffer.area, &mut buffer);

        let expected = vec![
            "┌──────────────────┐".to_string(),
            "│      ⢀⣿⣿⣿⡿       │".to_string(),
            "│ ⢀  ⢀⠔⠁  ⠁⠈⢆      │".to_string(),
            "│⣿⣿⣿⣷⠁       ⠱⡀    │".to_string(),
            "│⣿⣿⣿⣿⠁        ⠈⢆⣀⣀⣀│".to_string(),
            "│⠉⠉⠉⠱⡀        ⢀⣿⣿⣿⣿│".to_string(),
            "│    ⠈⢆       ⢀⢿⣿⣿⣿│".to_string(),
            "│      ⠱⡀ ⡀ ⢀⠔⠁  ⠁ │".to_string(),
            "│       ⣾⣿⣿⣿⠁      │".to_string(),
            "└─────── C ────────┘".to_string(),
        ];

        assert_eq!(buffer_lines(&buffer), expected);
    }

    #[test]
    fn empty_mol_draws_empty_canvas() {
        let empty_mol = Molecule::from_atoms(Vec::new());
        let viz = MoleculeVisualizer::new(&empty_mol).show_bonds(true);

        let area = Rect::new(0, 0, 20, 10);
        let mut buffer = Buffer::empty(area);

        Widget::render(&viz, buffer.area, &mut buffer);

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

        let area = Rect::new(0, 0, 20, 10);
        let mut buffer = Buffer::empty(area);
        Widget::render(&viz, buffer.area, &mut buffer);

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

        let area = Rect::new(0, 0, 10, 5);
        let mut buffer = Buffer::empty(area);
        Widget::render(&viz, buffer.area, &mut buffer);

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

        let area = Rect::new(0, 0, 10, 5);
        let mut buffer = Buffer::empty(area);
        Widget::render(&viz, buffer.area, &mut buffer);

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

        let area = Rect::new(0, 0, 10, 5);
        let mut buffer = Buffer::empty(area);
        Widget::render(&viz, buffer.area, &mut buffer);

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
        let chart = MoleculeVisualizer::new(&mol);

        let mut buffer = Buffer::empty(Rect::new(0, 0, 1, 1));
        // This should not panic, even if the buffer is too small to render the chart.
        Widget::render(&chart, buffer.area, &mut buffer);
        assert_eq!(buffer, Buffer::with_lines(["┌"]));
    }

    #[test]
    fn render_in_zero_size_buffer() {
        let mol = create_molecule();
        let chart = MoleculeVisualizer::new(&mol);

        let mut buffer = Buffer::empty(Rect::ZERO);
        // This should not panic, even if the buffer has zero size.
        Widget::render(&chart, buffer.area, &mut buffer);
    }

    #[test]
    fn atoms_have_color() {
        let molecule = vec![Atom::new(Element::N, [0.0, 0.0, 0.0])]
            .into_iter()
            .collect();
        let viz = MoleculeVisualizer::new(&molecule);

        let area = Rect::new(0, 0, 20, 10);
        let mut buffer = Buffer::empty(area);
        Widget::render(&viz, buffer.area, &mut buffer);

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

        expected[(10, 3)].set_fg(Color::Rgb(57, 57, 102));
        expected[(8, 4)].set_fg(Color::Rgb(57, 57, 102));
        expected[(9, 4)].set_fg(Color::Rgb(57, 57, 102));
        expected[(10, 4)].set_fg(Color::Rgb(57, 57, 102));
        expected[(11, 4)].set_fg(Color::Rgb(57, 57, 102));
        expected[(7, 5)].set_fg(Color::Rgb(57, 57, 102));
        expected[(8, 5)].set_fg(Color::Rgb(57, 57, 102));
        expected[(9, 5)].set_fg(Color::Rgb(57, 57, 102));
        expected[(10, 5)].set_fg(Color::Rgb(57, 57, 102));
        expected[(11, 5)].set_fg(Color::Rgb(57, 57, 102));
        expected[(12, 5)].set_fg(Color::Rgb(57, 57, 102));
        expected[(10, 6)].set_fg(Color::Rgb(57, 57, 102));
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
        let depths = [
            MoleculeVisualizer::visible_depth(-2.0),
            MoleculeVisualizer::visible_depth(1.0),
            MoleculeVisualizer::visible_depth(0.0),
        ];

        assert_eq!(MoleculeVisualizer::back_to_front_order(&depths), [0, 2, 1]);
    }

    #[test]
    fn depth_factor_brightens_nearer_depths() {
        assert!(
            MoleculeVisualizer::depth_factor(2.0, -1.0, 3.0)
                > MoleculeVisualizer::depth_factor(-1.0, -1.0, 3.0)
        );
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
        assert_eq!(canvas.pick_atom(camera, &molecule, (10, 5)), Some(0));

        // A corner click lands on empty space.
        assert_eq!(canvas.pick_atom(camera, &molecule, (0, 0)), None);

        // A click outside the canvas rect is rejected outright.
        assert_eq!(canvas.pick_atom(camera, &molecule, (99, 99)), None);
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

        assert_eq!(canvas.pick_atom(camera, &molecule, (10, 5)), Some(0));
    }

    const CAMERA_ROTATION_STEP: f64 = 0.12;

    #[test]
    fn zoom_camera() {
        let mol = create_molecule();
        let mut viz = MoleculeVisualizer::new(&mol);

        let area = Rect::new(0, 0, 20, 10);
        let mut buffer = Buffer::empty(area);

        viz.camera.zoom_by(2.0);
        Widget::render(&viz, buffer.area, &mut buffer);

        let expected = vec![
            "┌──────────────────┐".to_string(),
            "│               ⠱⡀ │".to_string(),
            "│                ⠈⢆│".to_string(),
            "│                  │".to_string(),
            "│                  │".to_string(),
            "│                  │".to_string(),
            "│                  │".to_string(),
            "│⠱⡀                │".to_string(),
            "│ ⠈⢆               │".to_string(),
            "└─────── C ────────┘".to_string(),
        ];

        assert_eq!(buffer_lines(&buffer), expected);
    }

    #[test]
    fn rotate_camera() {
        let mol = create_molecule();
        let mut viz = MoleculeVisualizer::new(&mol);

        let area = Rect::new(0, 0, 20, 10);
        let mut buffer = Buffer::empty(area);

        viz.camera
            .rotate(6.0 * CAMERA_ROTATION_STEP, -CAMERA_ROTATION_STEP * 6.0);
        Widget::render(&viz, buffer.area, &mut buffer);

        let expected = vec![
            "┌──────────────────┐".to_string(),
            "│       ⢶⣾⣷⡶       │".to_string(),
            "│        ⡇⣡⣧⣦⣤⡀    │".to_string(),
            "│       ⢸⠠⣿⣿⣿⣿⡧    │".to_string(),
            "│       ⡜ ⠻⢿⡿⠿⠃    │".to_string(),
            "│    ⢠⣶⣾⣷⣦ ⡜       │".to_string(),
            "│    ⢺⣿⣿⣿⣿⠂⡇       │".to_string(),
            "│    ⠈⠛⠻⢻⠋⣸        │".to_string(),
            "│       ⠾⢿⡿⠷       │".to_string(),
            "└─────── C ────────┘".to_string(),
        ];

        assert_eq!(buffer_lines(&buffer), expected);
    }

    #[test]
    fn reset_camera() {
        let mol = create_molecule();
        let mut viz = MoleculeVisualizer::new(&mol);

        let area = Rect::new(0, 0, 20, 10);
        let mut buffer = Buffer::empty(area);

        viz.camera
            .rotate(6.0 * CAMERA_ROTATION_STEP, -CAMERA_ROTATION_STEP * 6.0);
        viz.camera.zoom_by(2.0);
        viz.camera.reset();
        Widget::render(&viz, buffer.area, &mut buffer);

        let expected = vec![
            "┌──────────────────┐".to_string(),
            "│      ⢀⣿⣿⣿⡿       │".to_string(),
            "│ ⢀  ⢀⠔⠁  ⠁⠈⢆      │".to_string(),
            "│⣿⣿⣿⣷⠁       ⠱⡀    │".to_string(),
            "│⣿⣿⣿⣿⠁        ⠈⢆⣀⣀⣀│".to_string(),
            "│⠉⠉⠉⠱⡀        ⢀⣿⣿⣿⣿│".to_string(),
            "│    ⠈⢆       ⢀⢿⣿⣿⣿│".to_string(),
            "│      ⠱⡀ ⡀ ⢀⠔⠁  ⠁ │".to_string(),
            "│       ⣾⣿⣿⣿⠁      │".to_string(),
            "└─────── C ────────┘".to_string(),
        ];

        assert_eq!(buffer_lines(&buffer), expected);
    }

    #[test]
    fn bonds_can_be_supplied_as_tuples() {
        let atoms = vec![
            Atom::new(Element::C, [0.0, 0.0, 0.0]),
            Atom::new(Element::C, [1.5, 0.0, 0.0]),
            Atom::new(Element::C, [3.0, 0.0, 0.0]),
        ];
        let mol = Molecule::from_atoms_with_bonds(atoms, [(0, 1), (1, 2)]);

        assert_eq!(mol.bonds(), vec![Bond::new(0, 1), Bond::new(1, 2)]);
    }
}
