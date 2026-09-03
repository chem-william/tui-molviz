use std::io::stdout;
use std::time::{Duration, Instant};

use ratatui::{
    DefaultTerminal, Frame,
    crossterm::{
        event::{
            self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind,
            KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
        },
        execute,
    },
    layout::{Constraint, Layout},
    style::{Color, Style},
    text::Line,
    widgets::{Block, Paragraph},
};
use tui_molviz::camera::Camera;
use tui_molviz::molecule::{Atom, Molecule};
use tui_molviz::{AtomIndex, Element, MoleculeVisualizer, MoleculeVisualizerState};

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;

    let mut app = App::new();
    ratatui::run(|terminal| app.run(terminal))
}

struct App {
    molecule: Molecule,
    camera: Camera,
    auto_spin: bool,
    last_tick: Instant,
    should_quit: bool,
    /// Atom the user last clicked, highlighted in the view.
    selected: Option<AtomIndex>,
    /// Canvas mapping from the last render, used to hit-test mouse clicks.
    viz_state: MoleculeVisualizerState,
    /// The cell a left-button press started on, while the button is down.
    press_start: Option<(u16, u16)>,
    /// The last mouse position, to compute drag deltas from.
    last_mouse: Option<(u16, u16)>,
    /// Total cells moved since the press started. A press that moves less
    /// than `DRAG_THRESHOLD` cells is a click, not a pan.
    drag_distance: u32,
}

impl App {
    /// Total cells a press may move and still count as a click.
    const DRAG_THRESHOLD: u32 = 2;
    /// Cells the camera pans per shift+arrow keypress.
    const PAN_STEP_CELLS: i32 = 2;

    fn new() -> Self {
        Self {
            molecule: caffeine(),
            camera: Camera::default(),
            auto_spin: true,
            last_tick: Instant::now(),
            should_quit: false,
            selected: None,
            viz_state: MoleculeVisualizerState::default(),
            press_start: None,
            last_mouse: None,
            drag_distance: 0,
        }
    }

    fn run(&mut self, terminal: &mut DefaultTerminal) -> color_eyre::Result<()> {
        // Mouse capture isn't part of ratatui's default terminal setup, so the
        // example opts in (and out) itself.
        execute!(stdout(), EnableMouseCapture)?;

        let result = (|| {
            while !self.should_quit {
                self.tick();
                terminal.draw(|frame| self.render(frame))?;
                self.handle_events()?;
            }
            Ok(())
        })();

        execute!(stdout(), DisableMouseCapture)?;
        result
    }

    fn render(&mut self, frame: &mut Frame<'_>) {
        let [molecule_area, controls_area] =
            Layout::vertical([Constraint::Min(8), Constraint::Length(5)]).areas(frame.area());

        let visualizer = MoleculeVisualizer::new(&self.molecule)
            .camera(self.camera)
            .highlight(self.selected)
            .block(Block::bordered().title("tui-molviz showcase"))
            .style(Style::default().bg(Color::Black));

        // Rendering statefully hands back the canvas mapping in `viz_state`,
        // which `handle_events` reads to turn a click into an atom index.
        frame.render_stateful_widget(&visualizer, molecule_area, &mut self.viz_state);

        let spin = if self.auto_spin { "on" } else { "off" };
        let selected = match self.selected {
            Some(i) => {
                let atom = &self.molecule.atoms()[i.get()];
                format!("{} (#{i})", atom.element().symbol())
            }
            None => "none".to_string(),
        };
        let (tx, ty) = self.camera.offset();
        let offset = if (tx, ty) == (0.0, 0.0) {
            "center".to_string()
        } else {
            format!("{tx:+.2}, {ty:+.2}")
        };
        let controls = Paragraph::new(vec![
            Line::from(
                "arrows rotate   + zoom in   - zoom out   drag pan   shift+arrows pan",
            ),
            Line::from("c center   r reset   space spin   click select   q quit"),
            Line::from(format!(
                "yaw {:+.2}   pitch {:+.2}   zoom {:.2}   offset {offset}   spin {spin}   selected {selected}",
                self.camera.yaw(), self.camera.pitch(), self.camera.zoom()
            )),
        ])
        .block(Block::bordered().title("controls"));

        frame.render_widget(controls, controls_area);
    }

    fn tick(&mut self) {
        if !self.auto_spin || self.last_tick.elapsed() < Duration::from_millis(50) {
            return;
        }

        self.camera.rotate(0.025, 0.006);
        self.last_tick = Instant::now();
    }

    fn handle_events(&mut self) -> color_eyre::Result<()> {
        if !event::poll(Duration::from_millis(16))? {
            return Ok(());
        }

        match event::read()? {
            Event::Key(key) if key.kind == KeyEventKind::Press => self.handle_key(key),
            Event::Mouse(mouse) => self.handle_mouse(mouse),
            _ => {}
        }

        Ok(())
    }

    fn handle_key(&mut self, key: KeyEvent) {
        // Shift+arrows pan; plain arrows rotate.
        if key.modifiers.contains(KeyModifiers::SHIFT) {
            let (dcol, drow) = match key.code {
                KeyCode::Left => (-Self::PAN_STEP_CELLS, 0),
                KeyCode::Right => (Self::PAN_STEP_CELLS, 0),
                KeyCode::Up => (0, -Self::PAN_STEP_CELLS),
                KeyCode::Down => (0, Self::PAN_STEP_CELLS),
                _ => return,
            };
            self.pan_by_cells(dcol, drow);
            return;
        }

        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Left => self.camera.rotate(-0.12, 0.0),
            KeyCode::Right => self.camera.rotate(0.12, 0.0),
            KeyCode::Up => self.camera.rotate(0.0, -0.12),
            KeyCode::Down => self.camera.rotate(0.0, 0.12),
            KeyCode::Char('+') | KeyCode::Char('=') => self.camera.zoom_by(1.1),
            KeyCode::Char('-') => self.camera.zoom_by(1.0 / 1.1),
            KeyCode::Char('c') => self.camera.recenter(),
            KeyCode::Char('r') => self.camera.reset(),
            KeyCode::Char(' ') => self.auto_spin = !self.auto_spin,
            _ => {}
        }
    }

    /// Routes mouse events: a left-button press that stays put is a click
    /// (atom selection), one that moves is a pan.
    fn handle_mouse(&mut self, mouse: MouseEvent) {
        match mouse.kind {
            MouseEventKind::Down(MouseButton::Left) => {
                let cell = (mouse.column, mouse.row);
                self.press_start = Some(cell);
                self.last_mouse = Some(cell);
                self.drag_distance = 0;
            }
            // `Drag` is motion with a button held; crossterm reports absolute
            // cells, so the delta against the last event is the pan step.
            MouseEventKind::Drag(MouseButton::Left) => {
                let Some((last_col, last_row)) = self.last_mouse else {
                    // No press this app saw to delta against; just record it.
                    self.last_mouse = Some((mouse.column, mouse.row));
                    return;
                };
                let (dcol, drow) = (
                    mouse.column as i32 - last_col as i32,
                    mouse.row as i32 - last_row as i32,
                );
                self.last_mouse = Some((mouse.column, mouse.row));
                if (dcol, drow) != (0, 0) {
                    self.drag_distance += (dcol.abs() + drow.abs()) as u32;
                    self.pan_by_cells(dcol, drow);
                }
            }
            MouseEventKind::Moved => {
                self.last_mouse = Some((mouse.column, mouse.row));
            }
            MouseEventKind::Up(MouseButton::Left) => {
                // Only a press this app saw down can be a click; a press that
                // moved too far is the end of a pan.
                let is_click =
                    self.press_start.is_some() && self.drag_distance < Self::DRAG_THRESHOLD;
                self.press_start = None;
                self.last_mouse = None;
                if is_click {
                    self.handle_click(mouse.column, mouse.row);
                }
            }
            _ => {}
        }
    }

    /// Pan the camera by a terminal-cell delta; `dcol` and `drow` follow the
    /// terminal axes (right and down positive). A no-op until the first render
    /// has produced a canvas mapping.
    fn pan_by_cells(&mut self, dcol: i32, drow: i32) {
        if let Some(canvas) = self.viz_state.canvas() {
            let (dx, dy) = canvas.cell_delta_to_world(dcol, drow);
            self.camera.translate(dx, dy);
        }
    }

    /// Turn a terminal cell (from any source — here a left click) into an atom
    /// selection. `pick_atom` only needs the raw column/row and the same camera
    /// the last frame was drawn with, so the widget stays event-source agnostic.
    fn handle_click(&mut self, col: u16, row: u16) {
        if let Some(canvas) = self.viz_state.canvas() {
            let hit = canvas.pick_atom(self.camera, &self.molecule, (col, row));
            // A click on empty space clears the selection.
            self.selected = hit;
        }
    }
}

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
