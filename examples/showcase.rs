use std::time::{Duration, Instant};

use mendeleev::Element;
use ratatui::{
    DefaultTerminal, Frame,
    crossterm::event::{self, Event, KeyCode, KeyEventKind},
    layout::{Constraint, Layout},
    style::{Color, Style},
    text::Line,
    widgets::{Block, Paragraph},
};
use tui_molviz::{Atom, Camera, MolecularVisualizer, Molecule};

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
}

impl App {
    fn new() -> Self {
        Self {
            molecule: caffeine(),
            camera: Camera::default(),
            auto_spin: true,
            last_tick: Instant::now(),
            should_quit: false,
        }
    }

    fn run(&mut self, terminal: &mut DefaultTerminal) -> color_eyre::Result<()> {
        while !self.should_quit {
            self.tick();
            terminal.draw(|frame| self.render(frame))?;
            self.handle_events()?;
        }

        Ok(())
    }

    fn render(&self, frame: &mut Frame<'_>) {
        let [molecule_area, controls_area] =
            Layout::vertical([Constraint::Min(8), Constraint::Length(3)]).areas(frame.area());

        let visualizer = MolecularVisualizer::new(&self.molecule)
            .camera(self.camera)
            .block(Block::bordered().title("tui-molviz showcase"))
            .style(Style::default().bg(Color::Black));

        frame.render_widget(visualizer, molecule_area);

        let spin = if self.auto_spin { "on" } else { "off" };
        let controls = Paragraph::new(vec![
            Line::from("arrows rotate   + zoom in   - zoom out   r reset   space spin   q quit"),
            Line::from(format!(
                "yaw {:+.2}   pitch {:+.2}   zoom {:.2}   spin {spin}",
                self.camera.yaw, self.camera.pitch, self.camera.zoom
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

        let Event::Key(key) = event::read()? else {
            return Ok(());
        };

        if key.kind != KeyEventKind::Press {
            return Ok(());
        }

        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Left => self.camera.rotate(-0.12, 0.0),
            KeyCode::Right => self.camera.rotate(0.12, 0.0),
            KeyCode::Up => self.camera.rotate(0.0, -0.12),
            KeyCode::Down => self.camera.rotate(0.0, 0.12),
            KeyCode::Char('+') | KeyCode::Char('=') => self.camera.zoom_by(1.1),
            KeyCode::Char('-') => self.camera.zoom_by(1.0 / 1.1),
            KeyCode::Char('r') => self.camera.reset(),
            KeyCode::Char(' ') => self.auto_spin = !self.auto_spin,
            _ => {}
        }

        Ok(())
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
