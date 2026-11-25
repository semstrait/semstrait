use super::base::{ComplexExtras, LeafExtras};
use super::refs::NestedDataKindRef;
use super::sealed;
use super::variants::{DataKindForm, DataKindVariant};
use crate::model::types::entities::ai::AiContext;
use crate::model::types::entities::semantic_interface::SemanticInterface;

pub trait DataKind: sealed::Sealed {
    fn name(&self) -> &str;
    fn variant(&self) -> DataKindVariant;
    fn form(&self) -> DataKindForm;
}

pub trait SimpleDataKind: DataKind {
    fn extras(&self) -> &LeafExtras;
}

pub trait ComplexDataKind: DataKind {
    fn extras(&self) -> &ComplexExtras;
    fn allowed_child_variants(&self) -> &'static [DataKindVariant];
    fn child_count(&self) -> usize;
    fn children_ref(&self) -> Box<dyn Iterator<Item = NestedDataKindRef<'_>> + '_>;
}

pub trait PublicDataKind: DataKind {
    fn description(&self) -> Option<&str>;
    fn ai_context(&self) -> Option<&AiContext>;
    fn semantic_interface(&self) -> &SemanticInterface;
}

pub trait NestedDataKind: DataKind {}
