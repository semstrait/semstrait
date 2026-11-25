use super::base::{ComplexExtras, LeafExtras};
use super::refs::{
    AnyDataKindRef, ComplexDataKindRef, NestedDataKindRef, PublicDataKindRef, SimpleDataKindRef,
};
use super::sealed;
use super::traits::{ComplexDataKind, DataKind, NestedDataKind, PublicDataKind, SimpleDataKind};
use super::variants::{DataKindForm, DataKindVariant};
use crate::model::types::entities::ai::AiContext;
use crate::model::types::entities::semantic_interface::SemanticInterface;

impl<'a> sealed::Sealed for AnyDataKindRef<'a> {}
impl<'a> DataKind for AnyDataKindRef<'a> {
    fn name(&self) -> &str {
        match self {
            AnyDataKindRef::Dataset(v) => v.name(),
            AnyDataKindRef::NestedDataset(v) => v.name(),
            AnyDataKindRef::Grainset(v) => v.name(),
            AnyDataKindRef::NestedGrainset(v) => v.name(),
            AnyDataKindRef::Unionset(v) => v.name(),
            AnyDataKindRef::NestedUnionset(v) => v.name(),
            AnyDataKindRef::Joinset(v) => v.name(),
            AnyDataKindRef::NestedJoinset(v) => v.name(),
        }
    }
    fn variant(&self) -> DataKindVariant {
        match self {
            AnyDataKindRef::Dataset(_) | AnyDataKindRef::NestedDataset(_) => {
                DataKindVariant::Dataset
            }
            AnyDataKindRef::Grainset(_) | AnyDataKindRef::NestedGrainset(_) => {
                DataKindVariant::Grainset
            }
            AnyDataKindRef::Unionset(_) | AnyDataKindRef::NestedUnionset(_) => {
                DataKindVariant::Unionset
            }
            AnyDataKindRef::Joinset(_) | AnyDataKindRef::NestedJoinset(_) => {
                DataKindVariant::Joinset
            }
        }
    }
    fn form(&self) -> DataKindForm {
        match self {
            AnyDataKindRef::Dataset(_)
            | AnyDataKindRef::Grainset(_)
            | AnyDataKindRef::Unionset(_)
            | AnyDataKindRef::Joinset(_) => DataKindForm::Public,
            AnyDataKindRef::NestedDataset(_)
            | AnyDataKindRef::NestedGrainset(_)
            | AnyDataKindRef::NestedUnionset(_)
            | AnyDataKindRef::NestedJoinset(_) => DataKindForm::Nested,
        }
    }
}

impl<'a> sealed::Sealed for PublicDataKindRef<'a> {}
impl<'a> DataKind for PublicDataKindRef<'a> {
    fn name(&self) -> &str {
        match self {
            Self::Dataset(v) => v.name(),
            Self::Grainset(v) => v.name(),
            Self::Unionset(v) => v.name(),
            Self::Joinset(v) => v.name(),
        }
    }
    fn variant(&self) -> DataKindVariant {
        match self {
            Self::Dataset(_) => DataKindVariant::Dataset,
            Self::Grainset(_) => DataKindVariant::Grainset,
            Self::Unionset(_) => DataKindVariant::Unionset,
            Self::Joinset(_) => DataKindVariant::Joinset,
        }
    }
    fn form(&self) -> DataKindForm {
        DataKindForm::Public
    }
}
impl<'a> PublicDataKind for PublicDataKindRef<'a> {
    fn description(&self) -> Option<&str> {
        match self {
            Self::Dataset(v) => v.description(),
            Self::Grainset(v) => v.description(),
            Self::Unionset(v) => v.description(),
            Self::Joinset(v) => v.description(),
        }
    }
    fn ai_context(&self) -> Option<&AiContext> {
        match self {
            Self::Dataset(v) => v.ai_context(),
            Self::Grainset(v) => v.ai_context(),
            Self::Unionset(v) => v.ai_context(),
            Self::Joinset(v) => v.ai_context(),
        }
    }
    fn semantic_interface(&self) -> &SemanticInterface {
        match self {
            Self::Dataset(v) => v.semantic_interface(),
            Self::Grainset(v) => v.semantic_interface(),
            Self::Unionset(v) => v.semantic_interface(),
            Self::Joinset(v) => v.semantic_interface(),
        }
    }
}

impl<'a> sealed::Sealed for NestedDataKindRef<'a> {}
impl<'a> DataKind for NestedDataKindRef<'a> {
    fn name(&self) -> &str {
        match self {
            Self::Dataset(v) => v.name(),
            Self::Grainset(v) => v.name(),
            Self::Unionset(v) => v.name(),
            Self::Joinset(v) => v.name(),
        }
    }
    fn variant(&self) -> DataKindVariant {
        match self {
            Self::Dataset(_) => DataKindVariant::Dataset,
            Self::Grainset(_) => DataKindVariant::Grainset,
            Self::Unionset(_) => DataKindVariant::Unionset,
            Self::Joinset(_) => DataKindVariant::Joinset,
        }
    }
    fn form(&self) -> DataKindForm {
        DataKindForm::Nested
    }
}
impl<'a> NestedDataKind for NestedDataKindRef<'a> {}

impl<'a> sealed::Sealed for SimpleDataKindRef<'a> {}
impl<'a> DataKind for SimpleDataKindRef<'a> {
    fn name(&self) -> &str {
        match self {
            Self::Public(v) => v.name(),
            Self::Nested(v) => v.name(),
        }
    }
    fn variant(&self) -> DataKindVariant {
        DataKindVariant::Dataset
    }
    fn form(&self) -> DataKindForm {
        match self {
            Self::Public(_) => DataKindForm::Public,
            Self::Nested(_) => DataKindForm::Nested,
        }
    }
}
impl<'a> SimpleDataKind for SimpleDataKindRef<'a> {
    fn extras(&self) -> &LeafExtras {
        match self {
            Self::Public(v) => &v.body.base.extras,
            Self::Nested(v) => &v.body.base.extras,
        }
    }
}

impl<'a> sealed::Sealed for ComplexDataKindRef<'a> {}
impl<'a> DataKind for ComplexDataKindRef<'a> {
    fn name(&self) -> &str {
        match self {
            Self::Grainset(v) => v.name(),
            Self::NestedGrainset(v) => v.name(),
            Self::Unionset(v) => v.name(),
            Self::NestedUnionset(v) => v.name(),
            Self::Joinset(v) => v.name(),
            Self::NestedJoinset(v) => v.name(),
        }
    }
    fn variant(&self) -> DataKindVariant {
        match self {
            Self::Grainset(_) | Self::NestedGrainset(_) => DataKindVariant::Grainset,
            Self::Unionset(_) | Self::NestedUnionset(_) => DataKindVariant::Unionset,
            Self::Joinset(_) | Self::NestedJoinset(_) => DataKindVariant::Joinset,
        }
    }
    fn form(&self) -> DataKindForm {
        match self {
            Self::Grainset(_) | Self::Unionset(_) | Self::Joinset(_) => DataKindForm::Public,
            Self::NestedGrainset(_) | Self::NestedUnionset(_) | Self::NestedJoinset(_) => {
                DataKindForm::Nested
            }
        }
    }
}
impl<'a> ComplexDataKind for ComplexDataKindRef<'a> {
    fn extras(&self) -> &ComplexExtras {
        match self {
            Self::Grainset(v) => ComplexDataKind::extras(*v),
            Self::NestedGrainset(v) => ComplexDataKind::extras(*v),
            Self::Unionset(v) => ComplexDataKind::extras(*v),
            Self::NestedUnionset(v) => ComplexDataKind::extras(*v),
            Self::Joinset(v) => ComplexDataKind::extras(*v),
            Self::NestedJoinset(v) => ComplexDataKind::extras(*v),
        }
    }
    fn allowed_child_variants(&self) -> &'static [DataKindVariant] {
        match self {
            Self::Grainset(v) => v.allowed_child_variants(),
            Self::NestedGrainset(v) => v.allowed_child_variants(),
            Self::Unionset(v) => v.allowed_child_variants(),
            Self::NestedUnionset(v) => v.allowed_child_variants(),
            Self::Joinset(v) => v.allowed_child_variants(),
            Self::NestedJoinset(v) => v.allowed_child_variants(),
        }
    }
    fn child_count(&self) -> usize {
        match self {
            Self::Grainset(v) => v.child_count(),
            Self::NestedGrainset(v) => v.child_count(),
            Self::Unionset(v) => v.child_count(),
            Self::NestedUnionset(v) => v.child_count(),
            Self::Joinset(v) => v.child_count(),
            Self::NestedJoinset(v) => v.child_count(),
        }
    }
    fn children_ref(&self) -> Box<dyn Iterator<Item = NestedDataKindRef<'_>> + '_> {
        match self {
            Self::Grainset(v) => v.children_ref(),
            Self::NestedGrainset(v) => v.children_ref(),
            Self::Unionset(v) => v.children_ref(),
            Self::NestedUnionset(v) => v.children_ref(),
            Self::Joinset(v) => v.children_ref(),
            Self::NestedJoinset(v) => v.children_ref(),
        }
    }
}
