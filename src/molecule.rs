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

/// The bond order between two atoms.
///
/// The order is what the visualizer draws: single bonds as one line, double
/// bonds as two parallel lines, and triple bonds as three.
///
/// Aromatic bonds are not modeled. Assign them the order that best
/// matches how you want them drawn.
///
/// # Example
///
/// ```rust
/// use tui_molviz::molecule::{Atom, Bond, BondOrder, Molecule};
/// use tui_molviz::Element;
///
/// let atoms = [
///     Atom::new(Element::C, [0.0, 0.0, 0.0]),
///     Atom::new(Element::O, [1.21, 0.0, 0.0]),
/// ];
/// let carbonyl = Bond::new(0.into(), 1.into()).with_order(BondOrder::Double);
/// let molecule = Molecule::from_atoms_with_bonds(atoms, [carbonyl]);
///
/// assert_eq!(molecule.bonds()[0].order(), BondOrder::Double);
/// assert_eq!(format!("{} {} {}", BondOrder::Single, BondOrder::Double, BondOrder::Triple),
///     "single double triple");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive] // leaves room for e.g. Aromatic
pub enum BondOrder {
    /// E.g. C–C.
    Single,
    /// E.g. C=O.
    Double,
    /// E.g. N≡N.
    Triple,
}

impl Default for BondOrder {
    /// Defaults to [`BondOrder::Single`].
    fn default() -> Self {
        Self::Single
    }
}

impl fmt::Display for BondOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let word = match self {
            Self::Single => "single",
            Self::Double => "double",
            Self::Triple => "triple",
        };
        write!(f, "{word}")
    }
}

/// A bond between two atoms of a [`Molecule`], naming the atoms by
/// [`AtomIndex`] and carrying a [`BondOrder`] the visualizer draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Bond {
    start: AtomIndex,
    end: AtomIndex,
    order: BondOrder,
}

impl Bond {
    /// Creates a new single [`Bond`] that begins at `start` and
    /// ends at `end`. Use [`Bond::with_order`] for double and
    /// triple bonds.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tui_molviz::molecule::{Bond, BondOrder};
    ///
    /// let bond = Bond::new(0.into(), 1.into());
    /// assert_eq!(bond.order(), BondOrder::Single);
    /// ```
    #[must_use]
    pub const fn new(start: AtomIndex, end: AtomIndex) -> Self {
        Self {
            start,
            end,
            order: BondOrder::Single,
        }
    }

    /// Returns a copy of the bond with its order replaced by `order`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tui_molviz::molecule::{Bond, BondOrder};
    ///
    /// let bond = Bond::new(0.into(), 1.into()).with_order(BondOrder::Double);
    /// assert_eq!(bond.order(), BondOrder::Double);
    /// ```
    #[must_use]
    pub const fn with_order(self, order: BondOrder) -> Self {
        Self { order, ..self }
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

    /// The order of the bond (single, double, or triple).
    #[must_use]
    pub const fn order(&self) -> BondOrder {
        self.order
    }
}

impl From<(usize, usize)> for Bond {
    /// Converts a `(start, end)` pair of raw positions into a [`Bond`].
    fn from((start, end): (usize, usize)) -> Self {
        Self::new(AtomIndex::from(start), AtomIndex::from(end))
    }
}

impl From<(usize, usize, BondOrder)> for Bond {
    /// Converts a `(start, end, order)` triple of raw positions into a [`Bond`].
    fn from((start, end, order): (usize, usize, BondOrder)) -> Self {
        Self::new(AtomIndex::from(start), AtomIndex::from(end)).with_order(order)
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
    /// Bond-order cutoffs, as a ratio of interatomic distance to the atoms'
    /// summed covalent radii.
    ///
    /// The single-bond edge stays at 1.3 so every bond perceived before
    /// bond orders existed is still perceived, now as a single bond.
    const SINGLE_BOND_RATIO: f64 = 1.3;
    const DOUBLE_BOND_RATIO: f64 = 1.02;
    const TRIPLE_BOND_RATIO: f64 = 0.88;

    /// Perceives the bonds between `atoms`.
    /// Scales as O(n^2) so is computationally expensive for many atoms.
    fn perceive_bonds(atoms: &[Atom]) -> Vec<Bond> {
        let mut bonds = Vec::new();
        for i in 0..atoms.len() {
            for j in (i + 1)..atoms.len() {
                let (a, b) = (&atoms[i], &atoms[j]);
                let d = ((a.position()[0] - b.position()[0]).powi(2)
                    + (a.position()[1] - b.position()[1]).powi(2)
                    + (a.position()[2] - b.position()[2]).powi(2))
                .sqrt();
                if let Some(order) =
                    Self::perceived_order(a.covalent_radius() + b.covalent_radius(), d)
                {
                    bonds.push(Bond::from((i, j, order)));
                }
            }
        }
        bonds
    }

    /// The [`BondOrder`] perceived for two atoms with summed covalent radius
    /// `sum_radii` (Å) separated by `d` (Å), or `None` when the pair is too
    /// close to be distinct atoms or too far apart to be bonded.
    fn perceived_order(sum_radii: f64, d: f64) -> Option<BondOrder> {
        if d <= Self::MIN_BOND_DISTANCE {
            return None;
        }
        let r = d / sum_radii;
        if r <= Self::TRIPLE_BOND_RATIO {
            Some(BondOrder::Triple)
        } else if r <= Self::DOUBLE_BOND_RATIO {
            Some(BondOrder::Double)
        } else if r <= Self::SINGLE_BOND_RATIO {
            Some(BondOrder::Single)
        } else {
            None
        }
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
    /// Bonds — and their [`BondOrder`]s — are perceived from interatomic
    /// distances: a ratio of distance to summed covalent radii, calibrated
    /// against typical bond lengths. It is a heuristic and approximate (it
    /// cannot e.g. tell an aromatic ring bond from a plain single bond); for
    /// precise control, build the molecule with
    /// [`Molecule::from_atoms_with_bonds`] and attach explicit orders via
    /// [`Bond::with_order`].
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

        // The diamond's edges sit at sqrt(2) Å, a double-bond ratio for carbon's
        // covalent radius, so they perceive as double bonds.
        assert_eq!(
            mol.bonds(),
            vec![
                Bond::from((0, 1, BondOrder::Double)),
                Bond::from((0, 3, BondOrder::Double)),
                Bond::from((1, 2, BondOrder::Double)),
                Bond::from((2, 3, BondOrder::Double)),
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

    /// A two-atom molecule with `a` at the origin and `b` a distance `d` (Å)
    /// along the x-axis, using the elements' default covalent radii.
    fn diatomic(a: Element, b: Element, d: f64) -> Molecule {
        Molecule::from_atoms([Atom::new(a, [0.0, 0.0, 0.0]), Atom::new(b, [d, 0.0, 0.0])])
    }

    fn bond_order_of(mol: &Molecule) -> BondOrder {
        assert_eq!(
            mol.bonds().len(),
            1,
            "expected exactly one bond, got {:?}",
            mol.bonds()
        );
        mol.bonds()[0].order()
    }

    #[test]
    fn perceives_bond_orders_at_typical_bond_lengths() {
        assert_eq!(
            bond_order_of(&diatomic(Element::C, Element::C, 1.54)),
            BondOrder::Single
        );
        assert_eq!(
            bond_order_of(&diatomic(Element::C, Element::C, 1.34)),
            BondOrder::Double
        );
        assert_eq!(
            bond_order_of(&diatomic(Element::C, Element::C, 1.20)),
            BondOrder::Triple
        );
        // C=O
        assert_eq!(
            bond_order_of(&diatomic(Element::C, Element::O, 1.21)),
            BondOrder::Double
        );
        // C≡O
        assert_eq!(
            bond_order_of(&diatomic(Element::C, Element::O, 1.13)),
            BondOrder::Triple
        );
        assert_eq!(
            bond_order_of(&diatomic(Element::C, Element::H, 1.09)),
            BondOrder::Single
        );
        // O=O
        assert_eq!(
            bond_order_of(&diatomic(Element::O, Element::O, 1.21)),
            BondOrder::Double
        );
    }

    #[test]
    fn does_not_perceive_bonds_outside_the_cutoffs() {
        // Closer than MIN_BOND_DISTANCE: coincident atoms, not a bond.
        assert!(diatomic(Element::C, Element::C, 0.39).bonds().is_empty());
        // Farther than the single-bond ratio.
        assert!(diatomic(Element::C, Element::C, 1.85).bonds().is_empty());
    }

    #[test]
    fn explicit_bond_orders_are_authoritative() {
        // At 1.21 Å the pair would perceive as a double bond, so pinning it
        // single proves the explicit order wins over the distance heuristic.
        let atoms = [
            Atom::new(Element::C, [0.0, 0.0, 0.0]),
            Atom::new(Element::O, [1.21, 0.0, 0.0]),
        ];
        let mol = Molecule::try_from_atoms_with_bonds(
            atoms,
            [Bond::new(AtomIndex::new(0), AtomIndex::new(1))],
        )
        .unwrap();

        assert_eq!(mol.bonds()[0].order(), BondOrder::Single);
    }

    #[test]
    fn bond_tuples_carry_an_order() {
        let single: Bond = (0, 1).into();
        let double: Bond = (0, 1, BondOrder::Double).into();

        assert_eq!(single.order(), BondOrder::Single);
        assert_eq!(
            double,
            Bond::new(AtomIndex::new(0), AtomIndex::new(1)).with_order(BondOrder::Double)
        );
    }

    #[test]
    fn bond_order_defaults_to_single_and_displays_in_words() {
        assert_eq!(BondOrder::default(), BondOrder::Single);
        assert_eq!(format!("{}", BondOrder::Single), "single");
        assert_eq!(format!("{}", BondOrder::Double), "double");
        assert_eq!(format!("{}", BondOrder::Triple), "triple");
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
