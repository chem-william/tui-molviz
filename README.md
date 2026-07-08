# tui-molviz

A molecular visualizer widget for [Ratatui](https://crates.io/crates/ratatui)

![Example molecule in TUI](https://github.com/chem-william/tui-molviz/blob/main/assets/showcase.gif)

## Quickstart

Run the minimal example:

```sh
cargo run --example quickstart
```

Press any key to quit.

The main usage is to visualize a `Molecule` in your terminal using `MolecularVisualizer` by rendering it using
Ratatui:

```rust
use mendeleev::Element;
use ratatui::{Frame, widgets::Block};
use tui_molviz::{Atom, MolecularVisualizer, Molecule};

fn render(frame: &mut Frame<'_>) {
    let water = Molecule::from_atoms([
        Atom::new(Element::O, [0.0000, 0.0000, 0.0000]),
        Atom::new(Element::H, [0.9572, 0.0000, 0.0000]),
        Atom::new(Element::H, [-0.2390, 0.9270, 0.0000]),
    ]);

    let widget = MolecularVisualizer::new(&water).block(Block::bordered().title("Water"));

    frame.render_widget(widget, frame.area());
}
```

For a larger interactive example with rotation and zoom:

```sh
cargo run --example showcase
```
