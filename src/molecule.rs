use mendeleev::Color as CpkColor;
use mendeleev::{Element, Picometer};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Atom {
    element: Element,
    position: [f64; 3],
    covalent_radius: f64, // bonding radius (Å)
}

impl Atom {
    /// Creates a new `Atom` from the given [`Element`] and `position`.
    #[must_use]
    pub fn new(element: Element, position: [f64; 3]) -> Self {
        Atom {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Bond {
    start: usize,
    end: usize,
}

impl Bond {
    /// Creates a new `Bond` that begins at `start` and ends at `end`.
    #[must_use]
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    #[must_use]
    pub fn start(&self) -> usize {
        self.start
    }

    #[must_use]
    pub fn end(&self) -> usize {
        self.end
    }
}

impl From<(usize, usize)> for Bond {
    fn from((start, end): (usize, usize)) -> Self {
        Self { start, end }
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

#[derive(Debug, Default, Clone, PartialEq, PartialOrd)]
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
            .find(|bond| bond.start >= atoms.len() || bond.end >= atoms.len())
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
}

impl FromIterator<Atom> for Molecule {
    /// Bonds are perceived from interatomic distances, same as [`Molecule::from_atoms`].
    fn from_iter<T: IntoIterator<Item = Atom>>(iter: T) -> Self {
        Self::from_atoms(iter)
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
                Bond::new(0, 1),
                Bond::new(0, 3),
                Bond::new(1, 2),
                Bond::new(2, 3),
            ],
            "molecule had unexpected bonds"
        );
    }
}
