use std::{fmt, slice, vec};

use mendeleev::Color as CpkColor;
use mendeleev::{Element, Picometer};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Atom {
    element: Element,
    position: [f64; 3],
    covalent_radius: f64, // bonding radius (Å)
}

impl Atom {
    /// Creates a new `Atom` from the given [`Element`] and `position`.
    #[must_use]
    pub fn new(element: Element, position: [f64; 3]) -> Self {
        Self {
            element,
            position,
            covalent_radius: Self::bond_radius(element),
        }
    }

    /// Computes the [`CpkColor`] from the [`Element`] of [`Self`].
    #[must_use]
    pub fn cpk(&self) -> CpkColor {
        self.element.cpk_color().unwrap_or(CpkColor {
            r: 255,
            g: 110,
            b: 180,
        })
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

    fn bond_radius(elem: Element) -> f64 {
        elem.atomic_radius()
            .unwrap_or_else(|| Picometer(f64::from(elem.atomic_number()) * 10.0))
            .0
            / 100.0
    }
}

/// The position of an atom in a [`Molecule`]'s atom list.
///
/// An `AtomIndex` is what [`Bond::start`] and [`Bond::end`] name, and what
/// [`MoleculeCanvas::pick_atom`](crate::MoleculeCanvas::pick_atom) returns, so
/// a picked atom can be highlighted without any further conversion.
///
/// [`AtomIndex::new`] does not check the position against any molecule's atom
/// count.
///
/// # Example
///
/// ```rust
/// use tui_molviz::molecule::AtomIndex;
///
/// let index = AtomIndex::new(2);
/// assert_eq!(index.get(), 2);
/// assert_eq!(format!("{index}"), "2");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AtomIndex {
    index: usize,
}

impl AtomIndex {
    /// Creates a new [`Self`] from a raw position without checking it.
    ///
    /// Use [`Molecule::try_from_atoms_with_bonds`] when positions come from
    /// untrusted input, e.g. a parsed file.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tui_molviz::molecule::AtomIndex;
    ///
    /// let index: AtomIndex = 2.into();
    /// assert_eq!(index.get(), 2);
    /// ```
    #[must_use]
    pub const fn new(index: usize) -> Self {
        Self { index }
    }

    /// The raw position of the atom in the atom list.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tui_molviz::molecule::AtomIndex;
    ///
    /// let elements = ["O", "H", "H"];
    /// let index = AtomIndex::new(1);
    /// assert_eq!(elements[index.get()], "H");
    /// ```
    #[must_use]
    pub const fn get(self) -> usize {
        self.index
    }
}

impl From<usize> for AtomIndex {
    fn from(index: usize) -> Self {
        Self::new(index)
    }
}

impl From<AtomIndex> for usize {
    fn from(index: AtomIndex) -> Self {
        index.get()
    }
}

impl fmt::Display for AtomIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.index, f)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Bond {
    start: AtomIndex,
    end: AtomIndex,
}

impl Bond {
    /// Creates a new `Bond` that begins at `start` and ends at `end`.
    #[must_use]
    pub fn new(start: AtomIndex, end: AtomIndex) -> Self {
        Self { start, end }
    }

    /// The atom the bond begins at.
    #[must_use]
    pub fn start(&self) -> AtomIndex {
        self.start
    }

    /// The atom the bond ends at.
    #[must_use]
    pub fn end(&self) -> AtomIndex {
        self.end
    }
}

impl From<(usize, usize)> for Bond {
    /// Converts a `(start, end)` pair of raw positions into a [`Bond`].
    fn from((start, end): (usize, usize)) -> Self {
        Self::new(AtomIndex::from(start), AtomIndex::from(end))
    }
}

/// A bond referenced an atom index outside the molecule's atom list.
#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
#[error(
    "bond ({}, {}) references an atom outside the {atom_count}-atom molecule",
    bond.start(),
    bond.end()
)]
pub struct InvalidBondError {
    bond: Bond,
    atom_count: usize,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Molecule {
    atoms: Vec<Atom>,
    bonds: Vec<Bond>,
    radius: f64, // greatest distance of any atom from the centroid
}

impl Molecule {
    /// Below this separation (Å), atoms are treated as coincident (e.g.
    /// duplicate input) rather than bonded.
    const MIN_BOND_DISTANCE: f64 = 0.4;
    /// A bond is perceived when interatomic distance is within this multiple
    /// of the atoms' summed covalent radii.
    const BOND_DISTANCE_TOLERANCE: f64 = 1.3;

    /// Perceives the bonds between `atoms`. Scales as O(n^2) so is computationally expensive for many atoms.
    fn perceive_bonds(atoms: &[Atom]) -> Vec<Bond> {
        let mut bonds = Vec::new();
        for i in 0..atoms.len() {
            for j in (i + 1)..atoms.len() {
                let (a, b) = (&atoms[i], &atoms[j]);
                let d = ((a.position()[0] - b.position()[0]).powi(2)
                    + (a.position()[1] - b.position()[1]).powi(2)
                    + (a.position()[2] - b.position()[2]).powi(2))
                .sqrt();
                let bond_cutoff =
                    (a.covalent_radius() + b.covalent_radius()) * Self::BOND_DISTANCE_TOLERANCE;
                if d > Self::MIN_BOND_DISTANCE && d <= bond_cutoff {
                    bonds.push(Bond::from((i, j)));
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

    // The floor is 1.0 as a single-atom molecule has radius zero and would
    // divide by zero in `MoleculeCanvas::new`.
    fn bounding_radius(atoms: &[Atom]) -> f64 {
        atoms
            .iter()
            .map(|a| {
                (a.position()[0] * a.position()[0]
                    + a.position()[1] * a.position()[1]
                    + a.position()[2] * a.position()[2])
                    .sqrt()
            })
            .fold(0.0_f64, f64::max)
            .max(1.0)
    }

    /// Creates a new [`Self`] from `atoms`.
    ///
    /// Bonds are perceived from interatomic distances.
    #[must_use]
    pub fn from_atoms(atoms: impl IntoIterator<Item = Atom>) -> Self {
        let atoms: Vec<_> = atoms.into_iter().collect();
        let bonds = Self::perceive_bonds(&atoms);

        Self::from_atoms_with_bonds(atoms, bonds)
    }

    /// Creates a new [`Self`] from `atoms` and `bonds`.
    /// This method might be preferred if bonds can be obtained from elsewhere
    /// as perceiving bonds scales as O(n^2) which is computationally expensive for many atoms.
    ///
    /// # Panics
    ///
    /// If any bond references an atom index outside `atoms`.
    #[must_use]
    pub fn from_atoms_with_bonds(
        atoms: impl IntoIterator<Item = Atom>,
        bonds: impl IntoIterator<Item = impl Into<Bond>>,
    ) -> Self {
        match Self::try_from_atoms_with_bonds(atoms, bonds) {
            Ok(molecule) => molecule,
            Err(err) => panic!("{err}"),
        }
    }

    /// Fallible version of [`Molecule::from_atoms_with_bonds`], for bonds
    /// coming from e.g. a parsed file rather than
    /// hand-written call sites.
    ///
    /// # Errors
    ///
    /// Returns an error if either `bond.start >= atoms.len()` or `bond.end >= atoms.len()`.
    pub fn try_from_atoms_with_bonds(
        atoms: impl IntoIterator<Item = Atom>,
        bonds: impl IntoIterator<Item = impl Into<Bond>>,
    ) -> Result<Self, InvalidBondError> {
        let mut atoms: Vec<_> = atoms.into_iter().collect();
        let bonds: Vec<_> = bonds.into_iter().map(Into::into).collect();

        if let Some(&bond) = bonds
            .iter()
            .find(|bond| bond.start().get() >= atoms.len() || bond.end().get() >= atoms.len())
        {
            return Err(InvalidBondError {
                bond,
                atom_count: atoms.len(),
            });
        }

        Self::recenter(&mut atoms);
        let radius = Self::bounding_radius(&atoms);
        Ok(Molecule {
            atoms,
            bonds,
            radius,
        })
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

    /// Returns an iterator over the molecule's atoms.
    ///
    /// Atoms are yielded in the order they were supplied at construction, so an
    /// atom's position in this iteration is the [`AtomIndex`] that [`Bond::start`],
    /// [`Bond::end`], and [`pick_atom`](crate::MoleculeCanvas::pick_atom) refer
    /// to.
    ///
    /// There is deliberately no `iter_mut`: the centroid the atoms are recentered
    /// on, the cached [`radius`](Self::radius), and the perceived [`bonds`](Self::bonds)
    /// are all derived from the positions at construction, so handing out
    /// `&mut Atom` would silently invalidate them. Build a new [`Molecule`] from
    /// the edited atoms instead.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tui_molviz::molecule::{Atom, Molecule};
    /// use tui_molviz::Element;
    ///
    /// let water = Molecule::from_atoms([
    ///     Atom::new(Element::O, [0.0000, 0.0000, 0.0000]),
    ///     Atom::new(Element::H, [0.9572, 0.0000, 0.0000]),
    ///     Atom::new(Element::H, [-0.2390, 0.9270, 0.0000]),
    /// ]);
    ///
    /// let elements: Vec<_> = water.iter().map(Atom::element).collect();
    /// assert_eq!(elements, [Element::O, Element::H, Element::H]);
    /// ```
    pub fn iter(&self) -> slice::Iter<'_, Atom> {
        self.atoms.iter()
    }
}

impl FromIterator<Atom> for Molecule {
    /// Bonds are perceived from interatomic distances, same as [`Molecule::from_atoms`].
    fn from_iter<T: IntoIterator<Item = Atom>>(iter: T) -> Self {
        Self::from_atoms(iter)
    }
}

impl IntoIterator for Molecule {
    type Item = Atom;
    type IntoIter = vec::IntoIter<Atom>;

    /// Consumes the molecule and yields its atoms, dropping the bonds.
    ///
    /// Collecting the result back into a [`Molecule`] re-perceives bonds so a
    /// molecule built with explicit bonds via [`Molecule::from_atoms_with_bonds`]
    /// drops the supplied bonds.
    fn into_iter(self) -> Self::IntoIter {
        self.atoms.into_iter()
    }
}

impl<'a> IntoIterator for &'a Molecule {
    type Item = &'a Atom;
    type IntoIter = slice::Iter<'a, Atom>;

    /// Equivalent to [`Molecule::iter`].
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use mendeleev::Element;

    use super::*;

    fn create_molecule() -> Molecule {
        let atoms = vec![
            Atom::new(Element::C, [1.0, 0.0, 0.0]),
            Atom::new(Element::C, [0.0, 1.0, 0.0]),
            Atom::new(Element::C, [-1.0, 0.0, 0.0]),
            Atom::new(Element::C, [0.0, -1.0, 0.0]),
        ];
        atoms.into_iter().collect()
    }

    #[test]
    fn molecule_has_properties() {
        let mol = create_molecule();

        assert_eq!(
            mol.atoms(),
            &vec![
                Atom::new(Element::C, [1.0, 0.0, 0.0]),
                Atom::new(Element::C, [0.0, 1.0, 0.0]),
                Atom::new(Element::C, [-1.0, 0.0, 0.0]),
                Atom::new(Element::C, [0.0, -1.0, 0.0]),
            ]
        );

        assert_eq!(mol.radius(), 1.0);

        assert_eq!(
            mol.bonds(),
            vec![
                Bond::from((0, 1)),
                Bond::from((0, 3)),
                Bond::from((1, 2)),
                Bond::from((2, 3)),
            ],
            "molecule had unexpected bonds"
        );
    }

    #[test]
    fn iter_mol_yields_atoms() {
        let mol = create_molecule();

        assert_eq!(mol.iter().copied().collect::<Vec<_>>(), mol.atoms());
        assert_eq!(mol.iter().len(), 4);
    }

    #[test]
    fn atom_index_round_trips_through_usize() {
        let index = AtomIndex::new(7);

        assert_eq!(index.get(), 7);
        assert_eq!(usize::from(index), 7);
        assert_eq!(AtomIndex::from(7), index);
        assert_eq!(format!("{index}"), "7");
    }

    #[test]
    fn out_of_range_bond_is_rejected() {
        let atoms = [Atom::new(Element::C, [0.0, 0.0, 0.0])];
        let bond = Bond::from((0, 1));

        let err = Molecule::try_from_atoms_with_bonds(atoms, [bond]).unwrap_err();

        assert_eq!(
            err,
            InvalidBondError {
                bond,
                atom_count: 1
            }
        );
    }
}
