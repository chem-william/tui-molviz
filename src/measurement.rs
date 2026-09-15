//! Geometric measurements over a selection of atoms.

use std::fmt;

use mendeleev::Element;
use thiserror::Error;

use crate::molecule::{Atom, AtomIndex, Molecule};

/// A measurement could not be taken over the given atoms.
#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MeasurementError {
    /// An atom index did not name an atom of the molecule.
    #[error("atom {index} is outside the {atom_count}-atom molecule")]
    AtomOutOfRange { index: AtomIndex, atom_count: usize },
    /// The atoms are coincident, or collinear where the measurement needs a
    /// plane, so the quantity is undefined rather than merely extreme.
    #[error("the selected atoms are coincident or collinear, so the measurement is undefined")]
    Degenerate,
}

/// A quantity measured over two, three, or four atoms.
///
/// Build one with [`Measurement::of`] and print it with [`Display`](fmt::Display):
/// each variant names the atoms it was taken over, so the readout is
/// self-describing.
///
/// # Example
///
/// ```rust
/// use tui_molviz::molecule::{Atom, AtomIndex, Molecule};
/// use tui_molviz::{Element, Measurement};
///
/// let carbonyl = Molecule::from_atoms([
///     Atom::new(Element::C, [0.00, 0.0, 0.0]),
///     Atom::new(Element::O, [1.21, 0.0, 0.0]),
/// ]);
///
/// let measured = Measurement::of(&carbonyl, [AtomIndex::new(0), AtomIndex::new(1)])?
///     .expect("two atoms measure a distance");
/// assert_eq!(measured.to_string(), "C0–O1  1.210 Å");
///
/// // Fewer than two atoms is not an error — there is just nothing to measure.
/// assert_eq!(Measurement::of(&carbonyl, [AtomIndex::new(0)])?, None);
/// # Ok::<(), tui_molviz::MeasurementError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Measurement {
    /// The distance between two atoms.
    Distance {
        atoms: [AtomIndex; 2],
        elements: [Element; 2],
        angstroms: f64,
    },
    /// The angle at the middle atom of three, in radians over `[0, PI]`.
    Angle {
        atoms: [AtomIndex; 3],
        elements: [Element; 3],
        radians: f64,
    },
    /// The signed dihedral of four atoms about the middle pair's axis, in
    /// radians over `(-PI, PI]`.
    Dihedral {
        atoms: [AtomIndex; 4],
        elements: [Element; 4],
        radians: f64,
    },
}

impl Measurement {
    /// Measures `atoms` against `molecule`: two atoms give a distance, three an
    /// angle at the middle atom, and four a dihedral about the middle pair.
    ///
    /// The order of `atoms` is what picks the vertex and the torsion axis, so
    /// `[a, b, c]` and `[b, a, c]` are different angles.
    ///
    /// Positions are measured as the molecule stores them. [`Molecule`]
    /// recenters its atoms on their centroid at construction, so the results
    /// match the coordinates that were handed in.
    ///
    /// # Errors
    ///
    /// [`MeasurementError::AtomOutOfRange`] if an index does not name an atom of
    /// `molecule`, and [`MeasurementError::Degenerate`] if the atoms are too
    /// close together, or too nearly collinear, for the quantity to be defined.
    ///
    /// Zero, one, or more than four atoms is `Ok(None)` rather than an error:
    /// it is a selection still being built, not a mistake.
    pub fn of(
        molecule: &Molecule,
        atoms: impl AsRef<[AtomIndex]>,
    ) -> Result<Option<Self>, MeasurementError> {
        let indices = atoms.as_ref();
        let measured = match *indices {
            [a, b] => Self::Distance {
                atoms: [a, b],
                elements: [element(molecule, a)?, element(molecule, b)?],
                angstroms: molecule.distance(a, b)?,
            },
            [a, vertex, c] => Self::Angle {
                atoms: [a, vertex, c],
                elements: [
                    element(molecule, a)?,
                    element(molecule, vertex)?,
                    element(molecule, c)?,
                ],
                radians: molecule.angle(a, vertex, c)?,
            },
            [a, b, c, d] => Self::Dihedral {
                atoms: [a, b, c, d],
                elements: [
                    element(molecule, a)?,
                    element(molecule, b)?,
                    element(molecule, c)?,
                    element(molecule, d)?,
                ],
                radians: molecule.dihedral(a, b, c, d)?,
            },
            _ => return Ok(None),
        };
        Ok(Some(measured))
    }

    /// The atoms the measurement was taken over, in the order they were given.
    #[must_use]
    pub fn atoms(&self) -> &[AtomIndex] {
        match self {
            Self::Distance { atoms, .. } => atoms,
            Self::Angle { atoms, .. } => atoms,
            Self::Dihedral { atoms, .. } => atoms,
        }
    }

    /// The elements of [`atoms`](Self::atoms), in the same order.
    #[must_use]
    pub fn elements(&self) -> &[Element] {
        match self {
            Self::Distance { elements, .. } => elements,
            Self::Angle { elements, .. } => elements,
            Self::Dihedral { elements, .. } => elements,
        }
    }

    /// The measured value: distance units for a distance, radians for an angle or a
    /// dihedral. Use [`degrees`](Self::degrees) for the angular variants.
    #[must_use]
    pub fn value(&self) -> f64 {
        match *self {
            Self::Distance { angstroms, .. } => angstroms,
            Self::Angle { radians, .. } | Self::Dihedral { radians, .. } => radians,
        }
    }

    /// Just the value, formatted as it appears after the atom names and as the
    /// on-canvas overlay label, so the two can never disagree on precision.
    pub(crate) fn value_label(&self) -> String {
        match self.degrees() {
            Some(degrees) => format!("{degrees:.1}°"),
            None => format!("{:.3}", self.value()),
        }
    }

    /// The measured angle in degrees, or `None` for a distance.
    #[must_use]
    pub fn degrees(&self) -> Option<f64> {
        match *self {
            Self::Distance { .. } => None,
            Self::Angle { radians, .. } | Self::Dihedral { radians, .. } => {
                Some(radians.to_degrees())
            }
        }
    }
}

impl fmt::Display for Measurement {
    /// Writes the atoms then the value, e.g. `C0–N1–C2  120.4°`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, (index, element)) in self.atoms().iter().zip(self.elements()).enumerate() {
            if i > 0 {
                f.write_str("–")?;
            }
            write!(f, "{}{index}", element.symbol())?;
        }
        write!(f, "  {}", self.value_label())
    }
}

fn atom(molecule: &Molecule, index: AtomIndex) -> Result<&Atom, MeasurementError> {
    molecule.get(index).ok_or(MeasurementError::AtomOutOfRange {
        index,
        atom_count: molecule.atoms().len(),
    })
}

fn element(molecule: &Molecule, index: AtomIndex) -> Result<Element, MeasurementError> {
    Ok(atom(molecule, index)?.element())
}

#[cfg(test)]
mod tests {
    use crate::molecule::Atom;

    use super::*;

    fn indices(raw: impl IntoIterator<Item = usize>) -> Vec<AtomIndex> {
        raw.into_iter().map(AtomIndex::new).collect()
    }

    /// Four carbons forming a right-angled torsion: 0-1-2 is a right angle at
    /// atom 1, and 0-1-2-3 is a 90 degree dihedral.
    fn chain() -> Molecule {
        Molecule::from_atoms([
            Atom::new(Element::C, [0.0, 1.0, 0.0]),
            Atom::new(Element::C, [0.0, 0.0, 0.0]),
            Atom::new(Element::C, [1.5, 0.0, 0.0]),
            Atom::new(Element::C, [1.5, 0.0, 1.0]),
        ])
    }

    #[test]
    fn two_atoms_measure_a_distance() {
        let measured = Measurement::of(&chain(), indices([1, 2])).unwrap().unwrap();

        assert!(matches!(measured, Measurement::Distance { .. }));
        assert!((measured.value() - 1.5).abs() < 1e-12);
        assert_eq!(measured.degrees(), None);
        assert_eq!(measured.atoms(), indices([1, 2]));
    }

    #[test]
    fn three_atoms_measure_the_angle_at_the_middle_one() {
        let measured = Measurement::of(&chain(), indices([0, 1, 2]))
            .unwrap()
            .unwrap();

        assert!(matches!(measured, Measurement::Angle { .. }));
        assert!((measured.degrees().unwrap() - 90.0).abs() < 1e-9);
    }

    #[test]
    fn four_atoms_measure_a_dihedral() {
        let measured = Measurement::of(&chain(), indices([0, 1, 2, 3]))
            .unwrap()
            .unwrap();

        assert!(matches!(measured, Measurement::Dihedral { .. }));
        assert!((measured.degrees().unwrap() - 90.0).abs() < 1e-9);
    }

    #[test]
    fn the_order_of_the_atoms_picks_the_vertex() {
        let mol = chain();
        let at_one = Measurement::of(&mol, indices([0, 1, 2])).unwrap().unwrap();
        let at_zero = Measurement::of(&mol, indices([1, 0, 2])).unwrap().unwrap();

        assert_ne!(
            at_one.degrees(),
            at_zero.degrees(),
            "moving the vertex must change the angle"
        );
    }

    #[test]
    fn too_few_or_too_many_atoms_is_not_an_error() {
        let mol = chain();

        for count in [0, 1, 5, 6] {
            assert_eq!(
                Measurement::of(&mol, indices(0..count)).unwrap(),
                None,
                "{count} atoms should measure nothing"
            );
        }
    }

    #[test]
    fn an_index_outside_the_molecule_is_rejected() {
        let err = Measurement::of(&chain(), indices([0, 99])).unwrap_err();

        assert_eq!(
            err,
            MeasurementError::AtomOutOfRange {
                index: AtomIndex::new(99),
                atom_count: 4,
            }
        );
    }

    #[test]
    fn coincident_atoms_are_degenerate_rather_than_nan() {
        let stacked = Molecule::from_atoms([
            Atom::new(Element::C, [0.0, 0.0, 0.0]),
            Atom::new(Element::C, [0.0, 0.0, 0.0]),
            Atom::new(Element::C, [1.0, 0.0, 0.0]),
        ]);

        assert_eq!(
            Measurement::of(&stacked, indices([0, 1, 2])),
            Err(MeasurementError::Degenerate)
        );
        // A distance between coincident atoms is still perfectly well defined.
        let d = Measurement::of(&stacked, indices([0, 1])).unwrap().unwrap();
        assert_eq!(d.value(), 0.0);
    }

    #[test]
    fn display_names_the_atoms_then_the_value() {
        let mol = Molecule::from_atoms([
            Atom::new(Element::C, [0.00, 0.0, 0.0]),
            Atom::new(Element::O, [1.21, 0.0, 0.0]),
            Atom::new(Element::N, [1.21, 1.00, 0.0]),
        ]);

        assert_eq!(
            Measurement::of(&mol, indices([0, 1]))
                .unwrap()
                .unwrap()
                .to_string(),
            "C0–O1  1.210 Å"
        );
        assert_eq!(
            Measurement::of(&mol, indices([0, 1, 2]))
                .unwrap()
                .unwrap()
                .to_string(),
            "C0–O1–N2  90.0°"
        );
    }

    #[test]
    fn a_dihedral_displays_its_sign() {
        let mirrored = Molecule::from_atoms([
            Atom::new(Element::C, [0.0, 1.0, 0.0]),
            Atom::new(Element::C, [0.0, 0.0, 0.0]),
            Atom::new(Element::C, [1.5, 0.0, 0.0]),
            Atom::new(Element::C, [1.5, 0.0, -1.0]),
        ]);

        assert_eq!(
            Measurement::of(&mirrored, indices([0, 1, 2, 3]))
                .unwrap()
                .unwrap()
                .to_string(),
            "C0–C1–C2–C3  -90.0°"
        );
    }
}
