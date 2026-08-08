[![Docs Badge]][Docs]
[![Coverage Badge]][Coverage]
[![Dependency Badge]][Dependency Status]

# tui-molviz

A molecular visualizer widget for [Ratatui](https://crates.io/crates/ratatui)

![Example molecule in TUI](https://github.com/chem-william/tui-molviz/blob/main/assets/showcase.gif)

## Installation

```bash
cargo add tui-molviz
```

## Quickstart

Run the minimal example:

```sh
cargo run --example quickstart
```

Press any key to quit.

The main usage is to visualize a `Molecule` in your terminal using `MoleculeVisualizer` by rendering it using
Ratatui:

```rust
use tui_molviz::Element;
use ratatui::{Frame, widgets::Block};
use tui_molviz::molecule::{Atom, Molecule};
use tui_molviz::MoleculeVisualizer;

fn render(frame: &mut Frame<'_>) {
    let water = Molecule::from_atoms([
        Atom::new(Element::O, [0.0000, 0.0000, 0.0000]),
        Atom::new(Element::H, [0.9572, 0.0000, 0.0000]),
        Atom::new(Element::H, [-0.2390, 0.9270, 0.0000]),
    ]);

    let widget = MoleculeVisualizer::new(&water).block(Block::bordered().title("Water"));

    frame.render_widget(widget, frame.area());
}
```

For a larger interactive example with rotation and zoom:

```sh
cargo run --example showcase
```

[Docs]: https://docs.rs/tui-molviz
[Coverage]: https://codecov.io/gh/chem-william/tui-molviz
[Dependency Status]: https://deps.rs/repo/github/chem-william/tui-molviz
[Docs Badge]: https://img.shields.io/docsrs/tui-molviz?logo=rust&style=flat
[Coverage Badge]: https://codecov.io/github/chem-william/tui-molviz/coverage.svg?branch=main
[Dependency Badge]: https://deps.rs/repo/github/chem-william/tui-molviz/status.svg
