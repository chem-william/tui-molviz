//! # A molecular visualizer for Ratatui
//!
//! [Ratatui](https://ratatui.rs/) is an immediate-mode terminal user interface (TUI) library.
//! `tui-molviz` allows you to show molecules in a Ratatui app.
//!
//! # Quick start
//! ```rust,no_run
//! use ratatui::crossterm::event;
//! use ratatui::Frame;
//!
//! fn main() -> color_eyre::Result<()> {
//!     color_eyre::install()?;
//!
//!     ratatui::run(|terminal| loop {
//!         terminal.draw(render)?;
//!
//!         if event::read()?.is_key_press() {
//!             break Ok(());
//!         }
//!     })
//! }
//!
//! fn render(frame: &mut Frame<'_>) {}
//! ```

use mendeleev::Picometer;
pub use mendeleev::{Color, Element};
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

/// View orientation (radians) and zoom factor.
#[derive(Debug, Clone, Copy)]
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
        self.zoom *= factor;
    }
}

#[must_use]
pub fn cpk(elem: Element) -> Color {
    elem.cpk_color().unwrap_or(Color {
        r: 255,
        g: 110,
        b: 180,
    })
}

/// Orthographic projection of a world point under the camera. Rotates by yaw
/// (about vertical Y) then pitch (about horizontal X); the projected (x, y) are
/// the screen plane and the surviving z is the depth used for shading/occlusion.
#[must_use]
pub fn project_point(camera: Camera, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let (sy, cy) = camera.yaw.sin_cos();
    let (sp, cp) = camera.pitch.sin_cos();

    let x1 = x * cy + z * sy;
    let z1 = -x * sy + z * cy;
    let y2 = y * cp - z1 * sp;
    let z2 = y * sp + z1 * cp;
    (x1, y2, z2)
}

/// The braille canvas a molecule is drawn on: the screen rect inside the block
/// border, the world-space half-extents the canvas is bounded to (`bx`/`by`),
/// and the braille-dots-per-world-unit scale (`dpu`). Owning the mapping in one
/// place keeps drawing (`render_molecule`) and hit-testing (`pick_atom`) in sync.
#[derive(Debug, Clone, Copy, Default)]
pub struct MoleculeCanvas {
    inner: Rect,
    bx: f64,
    by: f64,
    dpu: f64,
}

impl MoleculeCanvas {
    #[must_use]
    pub fn contains_cell(&self, col: u16, row: u16) -> bool {
        self.inner.contains(Position { x: col, y: row })
    }

    /// Fits a molecule of the given radius into `inner` at the camera's zoom.
    /// Braille packs 2 dots per cell across and 4 down.
    fn new(inner: Rect, radius: f64, zoom: f64) -> Self {
        let w = f64::from(inner.width.max(1));
        let h = f64::from(inner.height.max(1));
        let (rx, ry) = (2.0 * w, 4.0 * h);
        let by = (radius * 1.15) / zoom;
        let bx = by * (rx / ry);
        let dpu = ry / (2.0 * by);
        Self { inner, bx, by, dpu }
    }

    /// Drawn radius of an atom, in braille dots.
    fn atom_radius_dots(&self, cov: f64) -> f64 {
        (cov * 0.55 * self.dpu).clamp(1.5, 5.0)
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
        col: u16,
        row: u16,
    ) -> Option<usize> {
        if !self.inner.contains(Position { x: col, y: row }) {
            return None;
        }

        // Cell -> canvas-data coords, sampling the cell's center; y is flipped.
        let fx = (f64::from(col - self.inner.x) + 0.5) / f64::from(self.inner.width);
        let fy = (f64::from(row - self.inner.y) + 0.5) / f64::from(self.inner.height);
        let px = -self.bx + fx * 2.0 * self.bx;
        let py = self.by - fy * 2.0 * self.by;

        molecule
            .atoms
            .iter()
            .enumerate()
            .filter_map(|(i, atom)| {
                let p = project_point(
                    camera,
                    atom.position()[0],
                    atom.position()[1],
                    atom.position()[2],
                );
                let d2 = (p.0 - px).powi(2) + (p.1 - py).powi(2);
                let r_world = self.atom_radius_dots(atom.covalent_radius()) / self.dpu;
                // On overlap, prefer the front-most atom (largest -z).
                (d2 <= r_world * r_world).then_some((i, -p.2))
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
    }
}

/// State handed back by rendering so a later mouse event can hit-test: the
/// widget writes the canvas mapping it just drew with, and the caller reads it
/// in `pick_atom`.
#[derive(Debug, Clone, Copy, Default)]
pub struct MolecularVisualizerState {
    pub canvas: Option<MoleculeCanvas>,
}

#[derive(Debug, Clone, Copy)]
pub struct Atom {
    element: Element,
    position: [f64; 3],
    covalent_radius: f64, // bonding radius (Å)
}

impl Atom {
    #[must_use]
    pub fn new(element: Element, position: [f64; 3]) -> Self {
        Atom {
            element,
            position,
            covalent_radius: Self::bond_radius(element),
        }
    }

    #[must_use]
    pub fn with_covalent_radius(mut self, radius: f64) -> Self {
        self.covalent_radius = radius;
        self
    }

    #[must_use]
    pub fn element(&self) -> Element {
        self.element
    }

    #[must_use]
    pub fn position(&self) -> [f64; 3] {
        self.position
    }

    #[must_use]
    pub fn covalent_radius(&self) -> f64 {
        self.covalent_radius
    }

    fn bond_radius(elem: mendeleev::Element) -> f64 {
        elem.atomic_radius()
            .unwrap_or_else(|| Picometer(f64::from(elem.atomic_number()) * 10.0))
            .0
            / 100.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Bond {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Default, Clone)]
pub struct Molecule {
    pub atoms: Vec<Atom>,
    pub bonds: Vec<Bond>,
    pub radius: f64, // greatest distance of any atom from the centroid
}

impl Molecule {
    fn perceive_bonds(atoms: &[Atom]) -> Vec<Bond> {
        let mut bonds = Vec::new();
        for i in 0..atoms.len() {
            for j in (i + 1)..atoms.len() {
                let (a, b) = (&atoms[i], &atoms[j]);
                let d = ((a.position()[0] - b.position()[0]).powi(2)
                    + (a.position()[1] - b.position()[1]).powi(2)
                    + (a.position()[2] - b.position()[2]).powi(2))
                .sqrt();
                if d > 0.4 && d <= (a.covalent_radius() + b.covalent_radius()) * 1.3 {
                    bonds.push(Bond { start: i, end: j });
                }
            }
        }
        bonds
    }

    fn recenter(atoms: &mut [Atom]) {
        if atoms.is_empty() {
            return;
        }

        let n = atoms.len() as f64;
        let (mut cx, mut cy, mut cz) = (0.0, 0.0, 0.0);
        for a in atoms.iter() {
            cx += a.position()[0];
            cy += a.position()[1];
            cz += a.position()[2];
        }
        cx /= n;
        cy /= n;
        cz /= n;
        for a in atoms.iter_mut() {
            a.position[0] -= cx;
            a.position[1] -= cy;
            a.position[2] -= cz;
        }
    }

    #[must_use]
    pub fn from_atoms(atoms: impl IntoIterator<Item = Atom>) -> Self {
        let mut atoms: Vec<_> = atoms.into_iter().collect();

        Self::recenter(&mut atoms);

        let radius = atoms
            .iter()
            .map(|a| {
                (a.position()[0] * a.position()[0]
                    + a.position()[1] * a.position()[1]
                    + a.position()[2] * a.position()[2])
                    .sqrt()
            })
            .fold(0.0_f64, f64::max)
            .max(1.0);

        let bonds = Self::perceive_bonds(&atoms);
        Molecule {
            atoms,
            bonds,
            radius,
        }
    }

    #[must_use]
    pub fn atoms(&self) -> &[Atom] {
        &self.atoms
    }

    #[must_use]
    pub fn bonds(&self) -> &[Bond] {
        &self.bonds
    }

    #[must_use]
    pub fn radius(&self) -> f64 {
        self.radius
    }
}

pub struct MolecularVisualizer<'a> {
    /// The molecule to visualize
    molecule: &'a Molecule,
    /// Optional block to wrap the molecular visualizer
    block: Option<Block<'a>>,
    /// Base style for the entire widget
    style: Style,
    /// Whether to show molecule legend or not. Default is `true`.
    show_molecule_legend: bool,
    /// Whether to show bonds between atoms. Default is `true`
    show_bonds: bool,
    camera: Camera,
}

impl<'a> MolecularVisualizer<'a> {
    /// Creates a new `MolecularVisualiazer` with the given molecule
    #[must_use]
    pub fn new(molecule: &'a Molecule) -> Self {
        Self {
            molecule,
            block: None,
            style: Style::default(),
            show_molecule_legend: true,
            show_bonds: true,
            camera: Camera::default(),
        }
    }

    /// Wraps the visualizer with the given block.
    #[must_use = "method moves the value of self and returns the modified value"]
    pub fn block(mut self, block: Block<'a>) -> Self {
        self.block = Some(block);
        self
    }

    /// Sets the camera the molecule is drawn from. Hit-testing with
    /// [`MoleculeCanvas::pick_atom`] must use this same camera.
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
    /// `style` accepts any type that is convertible to [`Style`] (e.g. [`Style`], [`Color`], or
    /// your own type that implements [`Into<Style>`]).
    ///
    /// This is a fluent setter method which must be chained or used as it consumes self
    #[must_use]
    pub fn style<S: Into<Style>>(mut self, style: S) -> Self {
        self.style = style.into();
        self
    }
}

impl Styled for MolecularVisualizer<'_> {
    type Item = Self;

    fn style(&self) -> Style {
        self.style
    }

    fn set_style<S: Into<Style>>(self, style: S) -> Self::Item {
        self.style(style)
    }
}

impl Widget for MolecularVisualizer<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        Widget::render(&self, area, buf);
    }
}

impl Widget for &MolecularVisualizer<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let _ = self.render_inner(area, buf);
    }
}

impl StatefulWidget for &MolecularVisualizer<'_> {
    type State = MolecularVisualizerState;

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::State) {
        state.canvas = Some(self.render_inner(area, buf));
    }
}

impl MolecularVisualizer<'_> {
    /// A color key for the elements actually in the molecule (each element's
    /// symbol drawn in its CPK color), so the structure is readable without already
    /// knowing the palette. Empty when the molecule has no atoms.
    fn draw_molecule_legend(&self) -> Line<'static> {
        let mut seen: Vec<mendeleev::Element> = Vec::new();
        for atom in &self.molecule.atoms {
            if !seen.contains(&atom.element) {
                seen.push(atom.element);
            }
        }

        let spans = seen
            .into_iter()
            .map(|elem| {
                let c = cpk(elem);
                Span::styled(
                    format!(" {} ", elem.symbol()),
                    Style::default()
                        .fg(ratatui::style::Color::Rgb(c.r, c.g, c.b))
                        .bold(),
                )
            })
            .collect::<Vec<_>>();
        Line::from(spans).centered()
    }

    fn visible_depth(projected_z: f64) -> f64 {
        -projected_z
    }

    fn depth_factor(z: f64, zmin: f64, zspan: f64) -> f64 {
        0.4 + 0.6 * ((z - zmin) / zspan)
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
    pub fn shade(color: &Color, f: f64) -> ratatui::style::Color {
        let f = (f.clamp(0.0, 1.0) * 5.0).round() / 5.0;
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
        let canvas = MoleculeCanvas::new(area, self.molecule.radius, self.camera.zoom);
        if area.is_empty() {
            return canvas;
        }

        let proj: Vec<(f64, f64, f64)> = self
            .molecule
            .atoms
            .iter()
            .map(|atom| {
                project_point(
                    self.camera,
                    atom.position()[0],
                    atom.position()[1],
                    atom.position()[2],
                )
            })
            .collect();

        let proj_depths: Vec<f64> = proj.iter().map(|p| Self::visible_depth(p.2)).collect();
        let zmin = proj_depths.iter().copied().fold(f64::INFINITY, f64::min);
        let zmax = proj_depths
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let zspan = (zmax - zmin).max(1e-6);
        // Depth factor in [0.4, 1.0]; nearer atoms are brighter.
        let depth = |z: f64| Self::depth_factor(z, zmin, zspan);

        let bond_lines: Vec<(f64, f64, f64, f64, ratatui::style::Color)> = if self.show_bonds {
            // Bonds split at their midpoint so each half takes its own atom's depth.
            self.molecule
                .bonds
                .iter()
                .map(|&bond| {
                    let color = Self::shade(
                        &mendeleev::Color {
                            r: 120,
                            g: 120,
                            b: 120,
                        },
                        depth(f64::midpoint(
                            proj_depths[bond.start],
                            proj_depths[bond.end],
                        )),
                    );
                    (
                        proj[bond.start].0,
                        proj[bond.start].1,
                        proj[bond.end].0,
                        proj[bond.end].1,
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
            let atom = &self.molecule.atoms[i];
            let color = Self::shade(&cpk(atom.element), depth(proj_depths[i]));
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
            });

        drawing_canvas.render(area, buf);

        canvas
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ratatui::style::Modifier;

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

    fn create_h2_molecule() -> Molecule {
        let atoms = vec![
            Atom::new(mendeleev::Element::H, [0.0, 0.0, 0.0]),
            Atom::new(mendeleev::Element::H, [0.0, 0.0, 1.0]),
        ];
        Molecule {
            atoms,
            bonds: vec![Bond { start: 0, end: 1 }],
            radius: 1.1,
        }
    }

    #[test]
    fn h2_gets_drawn() {
        let h2 = create_h2_molecule();
        let viz = MolecularVisualizer::new(&h2);

        let area = Rect::new(0, 0, 20, 10);
        let mut buffer = Buffer::empty(area);
        viz.render(buffer.area, &mut buffer);

        let expected = vec![
            "┌──────────────────┐".to_string(),
            "│                  │".to_string(),
            "│                  │".to_string(),
            "│                  │".to_string(),
            "│        ⣠⣄        │".to_string(),
            "│        ⠙⠛⠢⢄⡀⣀    │".to_string(),
            "│            ⠸⣿⠇   │".to_string(),
            "│                  │".to_string(),
            "│                  │".to_string(),
            "└─────── H ────────┘".to_string(),
        ];

        assert_eq!(buffer_lines(&buffer), expected);
    }

    #[test]
    fn h2_gets_drawn_without_bonds() {
        let h2 = create_h2_molecule();
        let viz = MolecularVisualizer::new(&h2).show_bonds(false);

        let area = Rect::new(0, 0, 20, 10);
        let mut buffer = Buffer::empty(area);
        viz.render(buffer.area, &mut buffer);

        let expected = vec![
            "┌──────────────────┐".to_string(),
            "│                  │".to_string(),
            "│                  │".to_string(),
            "│                  │".to_string(),
            "│        ⣠⣄        │".to_string(),
            "│        ⠙⠋   ⣀    │".to_string(),
            "│            ⠸⣿⠇   │".to_string(),
            "│                  │".to_string(),
            "│                  │".to_string(),
            "└─────── H ────────┘".to_string(),
        ];
        assert_eq!(buffer_lines(&buffer), expected);
    }

    #[test]
    fn h2_gets_drawn_without_legend() {
        let h2 = create_h2_molecule();
        let viz = MolecularVisualizer::new(&h2)
            .show_bonds(false)
            .show_molecule_legend(false)
            .block(Block::bordered());

        let area = Rect::new(0, 0, 10, 5);
        let mut buffer = Buffer::empty(area);
        viz.render(buffer.area, &mut buffer);

        let expected = vec![
            "┌────────┐".to_string(),
            "│        │".to_string(),
            "│   ⠰⠆⣤⡄ │".to_string(),
            "│     ⠉⠁ │".to_string(),
            "└────────┘".to_string(),
        ];
        assert_eq!(buffer_lines(&buffer), expected);
    }

    #[test]
    fn dont_double_draw_block() {
        let h2 = create_h2_molecule();
        let viz = MolecularVisualizer::new(&h2)
            .show_bonds(false)
            .block(Block::bordered().title("user"))
            .show_molecule_legend(true);

        let area = Rect::new(0, 0, 10, 5);
        let mut buffer = Buffer::empty(area);
        viz.render(buffer.area, &mut buffer);

        let expected = vec![
            "┌user────┐".to_string(),
            "│        │".to_string(),
            "│   ⠰⠆⣤⡄ │".to_string(),
            "│     ⠉⠁ │".to_string(),
            "└── H ───┘".to_string(),
        ];
        assert_eq!(buffer_lines(&buffer), expected);
    }

    #[test]
    fn render_in_minimal_buffer() {
        let h2 = create_h2_molecule();
        let chart = MolecularVisualizer::new(&h2);

        let mut buffer = Buffer::empty(Rect::new(0, 0, 1, 1));
        // This should not panic, even if the buffer is too small to render the chart.
        chart.render(buffer.area, &mut buffer);
        assert_eq!(buffer, Buffer::with_lines(["┌"]));
    }

    #[test]
    fn render_in_zero_size_buffer() {
        let h2 = create_h2_molecule();
        let chart = MolecularVisualizer::new(&h2);

        let mut buffer = Buffer::empty(Rect::ZERO);
        // This should not panic, even if the buffer has zero size.
        chart.render(buffer.area, &mut buffer);
    }

    #[test]
    fn atoms_have_color() {
        let atoms = vec![Atom::new(mendeleev::Element::N, [0.0, 0.0, 0.0])];
        let molecule = Molecule {
            atoms,
            bonds: Vec::new(),
            radius: 10.1,
        };
        let viz = MolecularVisualizer::new(&molecule);

        let area = Rect::new(0, 0, 20, 10);
        let mut buffer = Buffer::empty(area);
        viz.render(buffer.area, &mut buffer);

        let mut expected = Buffer::with_lines([
            "┌──────────────────┐",
            "│                  │",
            "│                  │",
            "│                  │",
            "│        ⢀⡀        │",
            "│        ⠈⠁        │",
            "│                  │",
            "│                  │",
            "│                  │",
            "└─────── N ────────┘",
        ]);

        expected[(9, 4)].set_fg(ratatui::style::Color::Rgb(57, 57, 102));
        expected[(10, 4)].set_fg(ratatui::style::Color::Rgb(57, 57, 102));
        expected[(9, 5)].set_fg(ratatui::style::Color::Rgb(57, 57, 102));
        expected[(10, 5)].set_fg(ratatui::style::Color::Rgb(57, 57, 102));
        for col in [8, 9, 10] {
            expected[(col, 9)].set_style(
                Style::default()
                    .fg(ratatui::style::Color::Rgb(143, 143, 255))
                    .add_modifier(Modifier::BOLD),
            );
        }

        assert_eq!(buffer, expected);
    }

    #[test]
    fn back_to_front_order_draws_nearest_last() {
        let depths = [
            MolecularVisualizer::visible_depth(-2.0),
            MolecularVisualizer::visible_depth(1.0),
            MolecularVisualizer::visible_depth(0.0),
        ];

        assert_eq!(MolecularVisualizer::back_to_front_order(&depths), [1, 2, 0]);
    }

    #[test]
    fn depth_factor_brightens_nearer_depths() {
        assert!(
            MolecularVisualizer::depth_factor(2.0, -1.0, 3.0)
                > MolecularVisualizer::depth_factor(-1.0, -1.0, 3.0)
        );
    }

    fn atom(x: f64, y: f64, z: f64) -> Atom {
        Atom::new(mendeleev::Element::C, [x, y, z])
    }

    #[test]
    fn pick_atom_maps_center_cell_to_origin_atom() {
        let molecule = Molecule {
            atoms: vec![atom(0.0, 0.0, 0.0)],
            bonds: vec![],
            radius: 1.0,
        };
        let canvas = MoleculeCanvas::new(Rect::new(0, 0, 20, 10), molecule.radius, 1.0);
        let camera = Camera {
            yaw: 0.0,
            pitch: 0.0,
            zoom: 1.0,
        };

        // Clicking the middle of the canvas hits the atom sitting at the origin.
        assert_eq!(canvas.pick_atom(camera, &molecule, 10, 5), Some(0));

        // A corner click lands on empty space.
        assert_eq!(canvas.pick_atom(camera, &molecule, 0, 0), None);

        // A click outside the canvas rect is rejected outright.
        assert_eq!(canvas.pick_atom(camera, &molecule, 99, 99), None);
    }

    #[test]
    fn canvas_reports_whether_a_cell_is_inside_its_area() {
        let canvas = MoleculeCanvas::new(Rect::new(2, 3, 5, 4), 1.0, 1.0);

        assert!(canvas.contains_cell(2, 3));
        assert!(canvas.contains_cell(6, 6));
        assert!(!canvas.contains_cell(7, 6));
        assert!(!canvas.contains_cell(6, 7));
    }

    #[test]
    fn pick_atom_prefers_the_front_atom_on_overlap() {
        // Two atoms at the same projected (x, y) but different depth; the one
        // nearer the viewer (smaller projected z) must win.
        let molecule = Molecule {
            atoms: vec![atom(0.0, 0.0, 2.0), atom(0.0, 0.0, -2.0)],
            bonds: vec![],
            radius: 2.0,
        };
        let canvas = MoleculeCanvas::new(Rect::new(0, 0, 20, 10), molecule.radius, 1.0);
        let camera = Camera {
            yaw: 0.0,
            pitch: 0.0,
            zoom: 1.0,
        };

        assert_eq!(canvas.pick_atom(camera, &molecule, 10, 5), Some(1));
    }

    const CAMERA_ROTATION_STEP: f64 = 0.12;
    #[test]
    fn rotate_camera_applies_yaw_and_pitch_deltas() {
        let mut camera = Camera::default();
        let yaw = camera.yaw;
        let pitch = camera.pitch;

        camera.rotate(CAMERA_ROTATION_STEP, -CAMERA_ROTATION_STEP);

        assert_eq!(camera.yaw, yaw + CAMERA_ROTATION_STEP);
        assert_eq!(camera.pitch, pitch - CAMERA_ROTATION_STEP);
    }
}
