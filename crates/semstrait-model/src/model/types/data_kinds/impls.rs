use super::base::LeafExtras;
use super::dataset::{Dataset, NestedDataset};
use super::grainset::{Grainset, NestedGrainset};
use super::joinset::{Joinset, NestedJoinset};
use super::sealed;
use super::traits::{DataKind, NestedDataKind, PublicDataKind, SimpleDataKind};
use super::unionset::{NestedUnionset, Unionset};
use super::variants::{DataKindForm, DataKindVariant};
use crate::model::types::entities::ai::AiContext;
use crate::model::types::entities::semantic_interface::SemanticInterface;

macro_rules! impl_data_kind {
    (
        ty: $ty:ty,
        variant: $variant:expr,
        form: $form:expr
    ) => {
        impl sealed::Sealed for $ty {}
        impl DataKind for $ty {
            fn name(&self) -> &str {
                &self.body.base.name
            }
            fn variant(&self) -> DataKindVariant {
                $variant
            }
            fn form(&self) -> DataKindForm {
                $form
            }
        }
    };
}

macro_rules! impl_public_data_kind {
    ($ty:ty) => {
        impl PublicDataKind for $ty {
            fn description(&self) -> Option<&str> {
                self.description.as_deref()
            }
            fn ai_context(&self) -> Option<&AiContext> {
                self.ai_context.as_ref()
            }
            fn semantic_interface(&self) -> &SemanticInterface {
                &self.semantic_interface
            }
        }
    };
}

macro_rules! impl_nested_data_kind {
    ($ty:ty) => {
        impl NestedDataKind for $ty {}
    };
}

impl_data_kind!(
    ty: Dataset,
    variant: DataKindVariant::Dataset,
    form: DataKindForm::Public
);
impl_data_kind!(
    ty: NestedDataset,
    variant: DataKindVariant::Dataset,
    form: DataKindForm::Nested
);
impl_public_data_kind!(Dataset);
impl_nested_data_kind!(NestedDataset);

impl SimpleDataKind for Dataset {
    fn extras(&self) -> &LeafExtras {
        &self.body.base.extras
    }
}

impl SimpleDataKind for NestedDataset {
    fn extras(&self) -> &LeafExtras {
        &self.body.base.extras
    }
}

impl_data_kind!(
    ty: Grainset,
    variant: DataKindVariant::Grainset,
    form: DataKindForm::Public
);
impl_data_kind!(
    ty: NestedGrainset,
    variant: DataKindVariant::Grainset,
    form: DataKindForm::Nested
);
impl_public_data_kind!(Grainset);
impl_nested_data_kind!(NestedGrainset);

impl_data_kind!(
    ty: Unionset,
    variant: DataKindVariant::Unionset,
    form: DataKindForm::Public
);
impl_data_kind!(
    ty: NestedUnionset,
    variant: DataKindVariant::Unionset,
    form: DataKindForm::Nested
);
impl_public_data_kind!(Unionset);
impl_nested_data_kind!(NestedUnionset);

impl_data_kind!(
    ty: Joinset,
    variant: DataKindVariant::Joinset,
    form: DataKindForm::Public
);
impl_data_kind!(
    ty: NestedJoinset,
    variant: DataKindVariant::Joinset,
    form: DataKindForm::Nested
);
impl_public_data_kind!(Joinset);
impl_nested_data_kind!(NestedJoinset);
