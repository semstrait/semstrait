use std::collections::BTreeMap;

use semstrait_common::EntityId;
use serde::{Deserialize, Serialize};

use super::semantic_definition::SemanticDefinition;
use crate::error::CompileError;

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct SemanticBitmap {
    pub entries: BTreeMap<EntityId, SemanticDefinition>,
}

pub fn guard_registry_size(entry_count: usize) -> Result<(), CompileError> {
    if u64::try_from(entry_count).unwrap_or(u64::MAX) > u64::from(u32::MAX) {
        return Err(CompileError::BitmaskValueOverflow {
            registry_len: entry_count,
            bit_width: u32::BITS,
        });
    }
    Ok(())
}

impl SemanticBitmap {
    pub fn freeze(
        entries: impl IntoIterator<Item = SemanticDefinition>,
    ) -> Result<Self, CompileError> {
        let mut by_id: BTreeMap<EntityId, SemanticDefinition> = entries
            .into_iter()
            .map(|def| (def.id.clone(), def))
            .collect();
        guard_registry_size(by_id.len())?;
        for (ordinal, definition) in by_id.values_mut().enumerate() {
            definition.bit_position = u32::try_from(ordinal)
                .expect("guard_registry_size bounds the count to the u32 width");
        }
        Ok(Self { entries: by_id })
    }

    #[must_use]
    pub fn get(&self, id: &str) -> Option<&SemanticDefinition> {
        self.entries.get(id)
    }

    #[must_use]
    pub fn by_name(&self, name: &str) -> Option<&SemanticDefinition> {
        self.entries.values().find(|def| def.name.as_str() == name)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[must_use]
    pub fn mask_of<'i>(&self, ids: impl IntoIterator<Item = &'i EntityId>) -> SemanticBitmask {
        SemanticBitmask::from_positions(
            ids.into_iter()
                .filter_map(|id| self.entries.get(id).map(|def| def.bit_position)),
        )
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct SemanticBitmask {
    pub words: Vec<u64>,
}

impl SemanticBitmask {
    #[must_use]
    pub fn empty() -> Self {
        Self { words: Vec::new() }
    }

    #[must_use]
    pub fn from_positions(positions: impl IntoIterator<Item = u32>) -> Self {
        let mut words: Vec<u64> = Vec::new();
        for n in positions {
            let word = (n / 64) as usize;
            let bit = n % 64;
            if word >= words.len() {
                words.resize(word + 1, 0);
            }
            words[word] |= 1u64 << bit;
        }
        let mut mask = Self { words };
        mask.trim();
        mask
    }

    #[must_use]
    pub fn contains(&self, position: u32) -> bool {
        let word = (position / 64) as usize;
        let bit = position % 64;
        self.words.get(word).is_some_and(|w| w & (1u64 << bit) != 0)
    }

    #[must_use]
    pub fn set_positions(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.count_ones() as usize);
        for (word_idx, &word) in self.words.iter().enumerate() {
            let base = (word_idx as u32) * 64;
            let mut bits = word;
            while bits != 0 {
                let bit = bits.trailing_zeros();
                out.push(base + bit);
                bits &= bits - 1;
            }
        }
        out
    }

    #[must_use]
    pub fn is_canonical(&self) -> bool {
        self.words.last().is_none_or(|&w| w != 0)
    }

    #[must_use]
    pub fn count_ones(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }

    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        let len = self.words.len().max(other.words.len());
        let mut words = Vec::with_capacity(len);
        for i in 0..len {
            let a = self.words.get(i).copied().unwrap_or(0);
            let b = other.words.get(i).copied().unwrap_or(0);
            words.push(a | b);
        }
        let mut mask = Self { words };
        mask.trim();
        mask
    }

    #[must_use]
    pub fn intersection(&self, other: &Self) -> Self {
        let len = self.words.len().min(other.words.len());
        let mut words = Vec::with_capacity(len);
        for i in 0..len {
            words.push(self.words[i] & other.words[i]);
        }
        let mut mask = Self { words };
        mask.trim();
        mask
    }

    #[must_use]
    pub fn difference(&self, other: &Self) -> Self {
        let mut words = Vec::with_capacity(self.words.len());
        for (i, &word) in self.words.iter().enumerate() {
            words.push(word & !other.words.get(i).copied().unwrap_or(0));
        }
        let mut mask = Self { words };
        mask.trim();
        mask
    }

    #[must_use]
    pub fn complement(&self, width: u32) -> Self {
        let whole_words = (width / 64) as usize;
        let tail_bits = width % 64;
        let len = whole_words + usize::from(tail_bits != 0);
        let mut words = Vec::with_capacity(len);
        for i in 0..len {
            let admitted = if i < whole_words {
                u64::MAX
            } else {
                (1u64 << tail_bits) - 1
            };
            words.push(admitted & !self.words.get(i).copied().unwrap_or(0));
        }
        let mut mask = Self { words };
        mask.trim();
        mask
    }

    #[must_use]
    pub fn is_subset_of(&self, other: &Self) -> bool {
        self.words
            .iter()
            .enumerate()
            .all(|(i, &w)| w & !other.words.get(i).copied().unwrap_or(0) == 0)
    }

    fn trim(&mut self) {
        while self.words.last() == Some(&0) {
            self.words.pop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::semantic_definition::{DimensionType, SemanticRoleDefinition};
    use semstrait_common::SemanticsName;
    use semstrait_ir::SemanticRole;
    use semstrait_sql::datatype::DataType;

    fn def(id: &str) -> SemanticDefinition {
        SemanticDefinition::new(
            id,
            SemanticsName::from(id),
            Some(DataType::String),
            SemanticRoleDefinition::Dimension {
                category: DimensionType::categorical(),
            },
        )
    }

    #[test]
    fn the_freeze_assigns_dense_positions_in_canonical_id_order() {
        let bitmap =
            SemanticBitmap::freeze([def("c"), def("a"), def("b")]).expect("small registry");
        assert_eq!(bitmap.len(), 3);
        assert_eq!(bitmap.get("a").map(|d| d.bit_position), Some(0));
        assert_eq!(bitmap.get("b").map(|d| d.bit_position), Some(1));
        assert_eq!(bitmap.get("c").map(|d| d.bit_position), Some(2));

        let positions: Vec<u32> = bitmap.entries.values().map(|d| d.bit_position).collect();
        assert_eq!(positions, vec![0, 1, 2], "no holes, and ascending by key");
    }

    #[test]
    fn the_freeze_overwrites_any_incoming_position() {
        let mut planted = def("a");
        planted.bit_position = 41;
        let bitmap = SemanticBitmap::freeze([planted, def("b")]).expect("small registry");
        assert_eq!(bitmap.get("a").map(|d| d.bit_position), Some(0));
        assert_eq!(bitmap.get("b").map(|d| d.bit_position), Some(1));
    }

    #[test]
    fn the_freeze_is_deterministic() {
        let one = SemanticBitmap::freeze([def("a"), def("b"), def("c")]).expect("freeze");
        let two = SemanticBitmap::freeze([def("c"), def("b"), def("a")]).expect("freeze");
        assert_eq!(one, two);
    }

    #[test]
    fn a_registry_past_the_bit_width_is_refused_rather_than_clamped() {
        assert!(guard_registry_size(0).is_ok());
        assert!(guard_registry_size(1).is_ok());
        assert!(guard_registry_size(u32::MAX as usize).is_ok());

        let over = u32::MAX as usize + 1;
        match guard_registry_size(over) {
            Err(CompileError::BitmaskValueOverflow {
                registry_len,
                bit_width,
            }) => {
                assert_eq!(registry_len, over, "names the size that overflowed");
                assert_eq!(bit_width, 32, "names the width it overflowed");
            }
            other => panic!("expected BitmaskValueOverflow, got {other:?}"),
        }
    }

    #[test]
    fn a_duplicate_id_collapses_and_density_still_holds() {
        let bitmap = SemanticBitmap::freeze([def("a"), def("a"), def("b")]).expect("freeze");
        assert_eq!(bitmap.len(), 2);
        assert_eq!(
            bitmap
                .entries
                .values()
                .map(|d| d.bit_position)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
    }

    #[test]
    fn mask_of_converts_an_id_set_and_ignores_an_unregistered_id() {
        let bitmap = SemanticBitmap::freeze([def("a"), def("b"), def("c")]).expect("freeze");
        let mask = bitmap.mask_of([&"a".to_owned(), &"c".to_owned()]);
        assert_eq!(mask.set_positions(), vec![0, 2]);

        let with_stranger = bitmap.mask_of([&"a".to_owned(), &"absent".to_owned()]);
        assert_eq!(with_stranger.set_positions(), vec![0]);
    }

    #[test]
    fn by_name_finds_an_entry_without_a_persisted_index() {
        let bitmap = SemanticBitmap::freeze([def("a"), def("b")]).expect("freeze");
        assert_eq!(bitmap.by_name("b").map(|d| d.bit_position), Some(1));
        assert!(bitmap.by_name("absent").is_none());
        assert_eq!(
            bitmap.by_name("a").map(|d| d.role()),
            Some(SemanticRole::Dimension)
        );
    }

    #[test]
    fn bitmap_serde_json_round_trip() {
        let bitmap = SemanticBitmap::freeze([def("dim.a"), def("measure.b")]).expect("freeze");
        let json = serde_json::to_string(&bitmap).expect("serialize bitmap");
        let back: SemanticBitmap = serde_json::from_str(&json).expect("deserialize bitmap");
        assert_eq!(bitmap, back);
        assert!(SemanticBitmap::default().is_empty());
    }

    #[test]
    fn from_positions_round_trips_via_set_positions() {
        let positions = [0u32, 1, 5, 63, 64, 130];
        let mask = SemanticBitmask::from_positions(positions);
        assert_eq!(mask.set_positions(), positions);
        assert_eq!(mask.count_ones(), positions.len() as u32);
    }

    #[test]
    fn empty_input_equals_empty_constructor() {
        assert_eq!(
            SemanticBitmask::from_positions([]),
            SemanticBitmask::empty()
        );
        assert!(SemanticBitmask::empty().words.is_empty());
        assert_eq!(SemanticBitmask::empty().count_ones(), 0);
        assert!(SemanticBitmask::empty().set_positions().is_empty());
    }

    #[test]
    fn canonical_trimming_drops_cleared_high_words() {
        let high = SemanticBitmask::from_positions([0u32, 200]);
        assert!(high.is_canonical());
        assert_eq!(high.words.len(), 4);

        let low_only = SemanticBitmask::from_positions([0u32]);
        let trimmed = high.intersection(&low_only);
        assert!(trimmed.is_canonical());
        assert_eq!(trimmed.words, vec![1u64]);
        assert_eq!(trimmed.set_positions(), vec![0]);
    }

    #[test]
    fn contains_matches_set_positions() {
        let mask = SemanticBitmask::from_positions([2u32, 70]);
        assert!(mask.contains(2));
        assert!(mask.contains(70));
        assert!(!mask.contains(0));
        assert!(!mask.contains(3));
        assert!(!mask.contains(10_000));
    }

    #[test]
    fn is_canonical_rejects_trailing_zero_word() {
        let bad = SemanticBitmask { words: vec![1, 0] };
        assert!(!bad.is_canonical());
        let good = SemanticBitmask { words: vec![1] };
        assert!(good.is_canonical());
        assert!(SemanticBitmask { words: vec![] }.is_canonical());
    }

    #[test]
    fn cross_word_boundary_positions() {
        let mask = SemanticBitmask::from_positions([63u32, 64, 65]);
        assert_eq!(mask.words.len(), 2);
        assert_eq!(mask.words[0], 1u64 << 63);
        assert_eq!(mask.words[1], 0b11);
        assert_eq!(mask.set_positions(), vec![63, 64, 65]);
        assert!(mask.contains(63));
        assert!(mask.contains(64));
        assert!(mask.contains(65));
    }

    #[test]
    fn union_intersection_subset_are_correct_and_canonical() {
        let a = SemanticBitmask::from_positions([0u32, 5, 64]);
        let b = SemanticBitmask::from_positions([5u32, 64, 200]);

        let union = a.union(&b);
        assert_eq!(union.set_positions(), vec![0, 5, 64, 200]);
        assert!(union.is_canonical());

        let inter = a.intersection(&b);
        assert_eq!(inter.set_positions(), vec![5, 64]);
        assert!(inter.is_canonical());

        let lo = SemanticBitmask::from_positions([0u32]);
        let hi = SemanticBitmask::from_positions([200u32]);
        assert_eq!(lo.intersection(&hi), SemanticBitmask::empty());

        assert!(inter.is_subset_of(&a));
        assert!(inter.is_subset_of(&b));
        assert!(a.is_subset_of(&union));
        assert!(b.is_subset_of(&union));
        assert!(!a.is_subset_of(&b));
        assert!(SemanticBitmask::empty().is_subset_of(&a));
    }

    #[test]
    fn difference_subtracts_and_stays_canonical() {
        let a = SemanticBitmask::from_positions([0u32, 5, 64, 200]);
        let b = SemanticBitmask::from_positions([5u32, 200]);

        let left = a.difference(&b);
        assert_eq!(left.set_positions(), vec![0, 64]);
        assert!(left.is_canonical(), "the cleared high word is trimmed");
        assert_eq!(left.words.len(), 2);

        assert_eq!(a.difference(&a), SemanticBitmask::empty());
        assert_eq!(
            SemanticBitmask::empty().difference(&a),
            SemanticBitmask::empty()
        );

        assert_eq!(b.difference(&a), SemanticBitmask::empty());
    }

    #[test]
    fn complement_is_bounded_by_the_supplied_width_and_stays_canonical() {
        let mask = SemanticBitmask::from_positions([0u32, 3]);

        let rest = mask.complement(5);
        assert_eq!(rest.set_positions(), vec![1, 2, 4]);
        assert!(rest.is_canonical());
        assert_eq!(rest.complement(5), mask);

        assert_eq!(mask.complement(4).set_positions(), vec![1, 2]);
        assert_eq!(mask.complement(0), SemanticBitmask::empty());
        assert_eq!(
            SemanticBitmask::empty().complement(3).set_positions(),
            vec![0, 1, 2]
        );
        let over = SemanticBitmask::from_positions([1u32, 130]);
        assert_eq!(over.complement(3).set_positions(), vec![0, 2]);
        assert!(over.complement(3).is_canonical());
        assert_eq!(
            SemanticBitmask::from_positions([64u32])
                .complement(66)
                .set_positions()
                .len(),
            65
        );
    }

    #[test]
    fn the_registry_is_the_authority_for_a_complement_width() {
        let bitmap = SemanticBitmap::freeze([def("a"), def("b"), def("c")]).expect("freeze");
        let served = bitmap.mask_of([&"b".to_owned()]);
        let width = u32::try_from(bitmap.len()).expect("a small registry");

        assert_eq!(served.complement(width).set_positions(), vec![0, 2]);
        assert_eq!(
            served.union(&served.complement(width)).count_ones(),
            width,
            "a mask and its complement partition the registry"
        );
        assert_eq!(
            served.intersection(&served.complement(width)),
            SemanticBitmask::empty()
        );
    }

    #[test]
    fn bitmask_serde_json_round_trip() {
        let mask = SemanticBitmask::from_positions([1u32, 63, 64, 130]);
        let json = serde_json::to_string(&mask).expect("serialize mask");
        let back: SemanticBitmask = serde_json::from_str(&json).expect("deserialize mask");
        assert_eq!(mask, back);

        let empty = SemanticBitmask::empty();
        let json = serde_json::to_string(&empty).expect("serialize empty");
        let back: SemanticBitmask = serde_json::from_str(&json).expect("deserialize empty");
        assert_eq!(empty, back);
    }
}
