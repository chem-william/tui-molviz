//! An ordered set of selected atoms.

use std::slice;

use crate::molecule::AtomIndex;

/// The atoms a user has picked, in the order they picked them.
///
/// Order is the point: it is what makes the middle of three atoms an angle's
/// vertex, and the middle two of four a dihedral's axis. Feed a `Selection`
/// straight to [`Measurement::of`](crate::Measurement::of) to read the
/// corresponding quantity, and to
/// [`MoleculeVisualizer::highlight`](crate::MoleculeVisualizer::highlight) to
/// draw it.
///
/// An atom appears at most once. A `Selection` knows nothing about any
/// particular [`Molecule`](crate::molecule::Molecule), so an index that names no
/// atom is caught when it is measured or drawn, not when it is inserted.
///
/// # Example
///
/// ```rust
/// use tui_molviz::{AtomIndex, Selection};
///
/// // Four atoms is a dihedral, so that is a natural place to stop.
/// let mut selection = Selection::with_limit(4);
/// for i in 0..4 {
///     assert!(selection.toggle(AtomIndex::new(i)));
/// }
///
/// // Full: further atoms are rejected until something is deselected.
/// assert!(selection.is_full());
/// assert!(!selection.toggle(AtomIndex::new(9)));
/// assert_eq!(selection.len(), 4);
///
/// // Toggling a selected atom deselects it, making room again.
/// assert!(!selection.toggle(AtomIndex::new(0)));
/// assert!(selection.toggle(AtomIndex::new(9)));
/// assert_eq!(
///     selection.as_slice(),
///     [AtomIndex::new(1), AtomIndex::new(2), AtomIndex::new(3), AtomIndex::new(9)],
/// );
/// ```
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Selection {
    // A `Vec` rather than a set: insertion order is contractual, and a
    // selection is a handful of atoms, so the linear `contains` is free.
    atoms: Vec<AtomIndex>,
    limit: Option<usize>,
}

impl Selection {
    /// Creates an empty selection with no limit on how many atoms it holds.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            atoms: Vec::new(),
            limit: None,
        }
    }

    /// Creates an empty selection that holds at most `limit` atoms.
    ///
    /// Once full it rejects further atoms rather than dropping the oldest, so a
    /// selection never changes under the user except where they clicked.
    #[must_use]
    pub const fn with_limit(limit: usize) -> Self {
        Self {
            atoms: Vec::new(),
            limit: Some(limit),
        }
    }

    /// The greatest number of atoms this selection holds, if it is limited.
    #[must_use]
    pub const fn limit(&self) -> Option<usize> {
        self.limit
    }

    /// Whether the selection is at its limit, so [`insert`](Self::insert) will
    /// reject anything new.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.limit.is_some_and(|limit| self.atoms.len() >= limit)
    }

    /// Selects `index`, returning whether it was added.
    ///
    /// `false` when the atom was already selected, or the selection is full.
    pub fn insert(&mut self, index: AtomIndex) -> bool {
        if self.is_full() || self.contains(index) {
            return false;
        }
        self.atoms.push(index);
        true
    }

    /// Deselects `index`, returning whether it had been selected. The remaining
    /// atoms keep their relative order.
    pub fn remove(&mut self, index: AtomIndex) -> bool {
        let Some(at) = self.atoms.iter().position(|&i| i == index) else {
            return false;
        };
        self.atoms.remove(at);
        true
    }

    /// Deselects `index` if it was selected, otherwise selects it.
    ///
    /// Returns whether `index` is selected *after* the call, so a rejected
    /// insertion and a removal both report `false` — [`is_full`](Self::is_full)
    /// tells them apart.
    pub fn toggle(&mut self, index: AtomIndex) -> bool {
        if self.remove(index) {
            return false;
        }
        self.insert(index)
    }

    /// Deselects every atom, keeping the limit.
    pub fn clear(&mut self) {
        self.atoms.clear();
    }

    /// Whether `index` is selected.
    #[must_use]
    pub fn contains(&self, index: AtomIndex) -> bool {
        self.atoms.contains(&index)
    }

    /// The number of selected atoms.
    #[must_use]
    pub fn len(&self) -> usize {
        self.atoms.len()
    }

    /// Whether no atom is selected.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }

    /// The atom selected first, if any.
    #[must_use]
    pub fn first(&self) -> Option<AtomIndex> {
        self.atoms.first().copied()
    }

    /// The atom selected most recently, if any.
    #[must_use]
    pub fn last(&self) -> Option<AtomIndex> {
        self.atoms.last().copied()
    }

    /// The selected atoms, in selection order.
    #[must_use]
    pub fn as_slice(&self) -> &[AtomIndex] {
        &self.atoms
    }

    /// Iterates the selected atoms, in selection order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = AtomIndex> + '_ {
        self.atoms.iter().copied()
    }
}

impl AsRef<[AtomIndex]> for Selection {
    fn as_ref(&self) -> &[AtomIndex] {
        self.as_slice()
    }
}

impl FromIterator<AtomIndex> for Selection {
    /// Collects into an unlimited selection, skipping repeats.
    fn from_iter<T: IntoIterator<Item = AtomIndex>>(iter: T) -> Self {
        let mut selection = Self::new();
        selection.extend(iter);
        selection
    }
}

impl Extend<AtomIndex> for Selection {
    /// Selects each atom in turn, so the limit and the no-repeats rule both
    /// still hold afterwards.
    fn extend<T: IntoIterator<Item = AtomIndex>>(&mut self, iter: T) {
        for index in iter {
            self.insert(index);
        }
    }
}

impl<'a> IntoIterator for &'a Selection {
    type Item = AtomIndex;
    type IntoIter = std::iter::Copied<slice::Iter<'a, AtomIndex>>;

    /// Yields copies, so `&selection` can be handed straight to
    /// [`MoleculeVisualizer::highlight`](crate::MoleculeVisualizer::highlight).
    fn into_iter(self) -> Self::IntoIter {
        self.atoms.iter().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn indices(selection: &Selection) -> Vec<usize> {
        selection.iter().map(AtomIndex::get).collect()
    }

    fn selection_of(indices: impl IntoIterator<Item = usize>) -> Selection {
        indices.into_iter().map(AtomIndex::new).collect()
    }

    #[test]
    fn selection_keeps_the_order_atoms_were_picked_in() {
        let selection = selection_of([3, 0, 2]);

        assert_eq!(indices(&selection), [3, 0, 2]);
        assert_eq!(selection.first(), Some(AtomIndex::new(3)));
        assert_eq!(selection.last(), Some(AtomIndex::new(2)));
    }

    #[test]
    fn an_atom_is_selected_at_most_once() {
        let mut selection = selection_of([1, 2]);

        assert!(!selection.insert(AtomIndex::new(1)));
        assert_eq!(indices(&selection), [1, 2]);
    }

    #[test]
    fn removing_an_atom_keeps_the_rest_in_order() {
        let mut selection = selection_of([4, 5, 6]);

        assert!(selection.remove(AtomIndex::new(5)));
        assert!(!selection.remove(AtomIndex::new(5)));
        assert_eq!(indices(&selection), [4, 6]);
    }

    #[test]
    fn toggle_reports_whether_the_atom_ends_up_selected() {
        let mut selection = Selection::new();

        assert!(selection.toggle(AtomIndex::new(0)), "first toggle selects");
        assert!(
            !selection.toggle(AtomIndex::new(0)),
            "second toggle deselects"
        );
        assert!(selection.is_empty());
    }

    #[test]
    fn a_full_selection_rejects_new_atoms_rather_than_evicting_old_ones() {
        let mut selection = Selection::with_limit(2);
        selection.extend([AtomIndex::new(0), AtomIndex::new(1)]);

        assert!(selection.is_full());
        assert!(!selection.insert(AtomIndex::new(2)));
        assert!(!selection.toggle(AtomIndex::new(2)));
        assert_eq!(
            indices(&selection),
            [0, 1],
            "the atoms already selected must survive untouched"
        );

        // Deselecting makes room again.
        selection.remove(AtomIndex::new(0));
        assert!(selection.insert(AtomIndex::new(2)));
        assert_eq!(indices(&selection), [1, 2]);
    }

    #[test]
    fn collecting_respects_the_no_repeats_rule() {
        assert_eq!(indices(&selection_of([1, 1, 2, 1])), [1, 2]);
    }

    #[test]
    fn extending_a_limited_selection_stops_at_the_limit() {
        let mut selection = Selection::with_limit(2);
        selection.extend((0..5).map(AtomIndex::new));

        assert_eq!(indices(&selection), [0, 1]);
        assert_eq!(selection.limit(), Some(2));
    }

    #[test]
    fn clearing_keeps_the_limit() {
        let mut selection = Selection::with_limit(3);
        selection.extend([AtomIndex::new(0)]);
        selection.clear();

        assert!(selection.is_empty());
        assert_eq!(selection.limit(), Some(3));
    }

    #[test]
    fn a_selection_reference_iterates_as_plain_atom_indices() {
        // This is what lets `.highlight(&selection)` compile against
        // `impl IntoIterator<Item = AtomIndex>`.
        let selection = selection_of([7, 8]);
        fn takes_indices(atoms: impl IntoIterator<Item = AtomIndex>) -> Vec<AtomIndex> {
            atoms.into_iter().collect()
        }

        assert_eq!(
            takes_indices(&selection),
            [AtomIndex::new(7), AtomIndex::new(8)]
        );
        assert_eq!(selection.as_ref(), selection.as_slice());
    }
}
