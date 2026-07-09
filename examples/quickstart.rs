use ratatui::{Frame, crossterm::event, style::Style, widgets::Block};
use tui_molviz::{Atom, Element, MolecularVisualizer, Molecule};

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;

    let molecule = Molecule::from_atoms([
        Atom::new(Element::O, [0.0000, 0.0000, 0.0000]),
        Atom::new(Element::H, [0.9572, 0.0000, 0.0000]),
        Atom::new(Element::H, [-0.2390, 0.9270, 0.0000]),
    ]);

    ratatui::run(|terminal| {
        loop {
            terminal.draw(|frame| render(frame, &molecule))?;

            if event::read()?.as_key_press_event().is_some() {
                break Ok(());
            }
        }
    })
}

fn render(frame: &mut Frame<'_>, molecule: &Molecule) {
    let visualizer =
        MolecularVisualizer::new(molecule).block(Block::bordered().title("Water molecule"));

    frame.render_widget(visualizer, frame.area());
}
