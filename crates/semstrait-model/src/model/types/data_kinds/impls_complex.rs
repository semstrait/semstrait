use super::base::ComplexExtras;
use super::grainset::{Grainset, GrainsetBody, NestedGrainset};
use super::joinset::{Joinset, JoinsetBody, NestedJoinset};
use super::refs::NestedDataKindRef;
use super::traits::ComplexDataKind;
use super::unionset::{NestedUnionset, UnionMode, Unionset, UnionsetBody};
use super::variants::DataKindVariant;

const GRAINSET_ALLOWED_CHILDREN: &[DataKindVariant] = &[
    DataKindVariant::Dataset,
    DataKindVariant::Unionset,
    DataKindVariant::Joinset,
];

impl ComplexDataKind for Grainset {
    fn extras(&self) -> &ComplexExtras {
        &self.body.base.extras
    }
    fn allowed_child_variants(&self) -> &'static [DataKindVariant] {
        GRAINSET_ALLOWED_CHILDREN
    }
    fn child_count(&self) -> usize {
        grainset_child_count(&self.body)
    }
    fn children_ref(&self) -> Box<dyn Iterator<Item = NestedDataKindRef<'_>> + '_> {
        Box::new(grainset_children_ref(&self.body))
    }
}

impl ComplexDataKind for NestedGrainset {
    fn extras(&self) -> &ComplexExtras {
        &self.body.base.extras
    }
    fn allowed_child_variants(&self) -> &'static [DataKindVariant] {
        GRAINSET_ALLOWED_CHILDREN
    }
    fn child_count(&self) -> usize {
        grainset_child_count(&self.body)
    }
    fn children_ref(&self) -> Box<dyn Iterator<Item = NestedDataKindRef<'_>> + '_> {
        Box::new(grainset_children_ref(&self.body))
    }
}

fn grainset_child_count(body: &GrainsetBody) -> usize {
    body.datasets.len() + body.unionsets.len() + body.joinsets.len()
}

fn grainset_children_ref(body: &GrainsetBody) -> impl Iterator<Item = NestedDataKindRef<'_>> {
    body.datasets
        .iter()
        .map(NestedDataKindRef::Dataset)
        .chain(body.unionsets.iter().map(NestedDataKindRef::Unionset))
        .chain(body.joinsets.iter().map(NestedDataKindRef::Joinset))
}

const UNIONSET_ALLOWED_CHILDREN: &[DataKindVariant] = &[
    DataKindVariant::Dataset,
    DataKindVariant::Grainset,
    DataKindVariant::Joinset,
];

impl ComplexDataKind for Unionset {
    fn extras(&self) -> &ComplexExtras {
        &self.body.base.extras
    }
    fn allowed_child_variants(&self) -> &'static [DataKindVariant] {
        UNIONSET_ALLOWED_CHILDREN
    }
    fn child_count(&self) -> usize {
        unionset_child_count(&self.body)
    }
    fn children_ref(&self) -> Box<dyn Iterator<Item = NestedDataKindRef<'_>> + '_> {
        Box::new(unionset_children_ref(&self.body))
    }
}

impl ComplexDataKind for NestedUnionset {
    fn extras(&self) -> &ComplexExtras {
        &self.body.base.extras
    }
    fn allowed_child_variants(&self) -> &'static [DataKindVariant] {
        UNIONSET_ALLOWED_CHILDREN
    }
    fn child_count(&self) -> usize {
        unionset_child_count(&self.body)
    }
    fn children_ref(&self) -> Box<dyn Iterator<Item = NestedDataKindRef<'_>> + '_> {
        Box::new(unionset_children_ref(&self.body))
    }
}

fn unionset_child_count(body: &UnionsetBody) -> usize {
    body.datasets.len() + body.grainsets.len() + body.joinsets.len()
}

fn unionset_children_ref(body: &UnionsetBody) -> impl Iterator<Item = NestedDataKindRef<'_>> {
    body.datasets
        .iter()
        .map(NestedDataKindRef::Dataset)
        .chain(body.grainsets.iter().map(NestedDataKindRef::Grainset))
        .chain(body.joinsets.iter().map(NestedDataKindRef::Joinset))
}

const JOINSET_ALLOWED_CHILDREN: &[DataKindVariant] = &[
    DataKindVariant::Dataset,
    DataKindVariant::Grainset,
    DataKindVariant::Unionset,
];

impl ComplexDataKind for Joinset {
    fn extras(&self) -> &ComplexExtras {
        &self.body.base.extras
    }
    fn allowed_child_variants(&self) -> &'static [DataKindVariant] {
        JOINSET_ALLOWED_CHILDREN
    }
    fn child_count(&self) -> usize {
        joinset_child_count(&self.body)
    }
    fn children_ref(&self) -> Box<dyn Iterator<Item = NestedDataKindRef<'_>> + '_> {
        Box::new(joinset_children_ref(&self.body))
    }
}

impl ComplexDataKind for NestedJoinset {
    fn extras(&self) -> &ComplexExtras {
        &self.body.base.extras
    }
    fn allowed_child_variants(&self) -> &'static [DataKindVariant] {
        JOINSET_ALLOWED_CHILDREN
    }
    fn child_count(&self) -> usize {
        joinset_child_count(&self.body)
    }
    fn children_ref(&self) -> Box<dyn Iterator<Item = NestedDataKindRef<'_>> + '_> {
        Box::new(joinset_children_ref(&self.body))
    }
}

fn joinset_child_count(body: &JoinsetBody) -> usize {
    body.datasets.len() + body.grainsets.len() + body.unionsets.len()
}

fn joinset_children_ref(body: &JoinsetBody) -> impl Iterator<Item = NestedDataKindRef<'_>> {
    body.datasets
        .iter()
        .map(NestedDataKindRef::Dataset)
        .chain(body.grainsets.iter().map(NestedDataKindRef::Grainset))
        .chain(body.unionsets.iter().map(NestedDataKindRef::Unionset))
}

impl GrainsetBody {
    #[must_use]
    pub fn minimum_children(&self) -> usize {
        1
    }
}

impl UnionsetBody {
    #[must_use]
    pub fn minimum_children(&self) -> usize {
        match self.mode {
            UnionMode::All => 1,
            UnionMode::Unique => 2,
        }
    }
}

impl JoinsetBody {
    #[must_use]
    pub fn minimum_children(&self) -> usize {
        2
    }
}
