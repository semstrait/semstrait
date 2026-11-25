use crate::error::validate::ValidateErrorKind;
use crate::model::types::data_kinds::{
    AnyDataKindRef, DataKind, DataKindVariant, Dataset, NestedDataset,
};
use crate::model::types::entities::TemporalShape;
use crate::model::SemanticModel;
use semstrait_common::diagnostic::{Diagnostic, Diagnostics};

pub(super) fn check_temporal(model: &SemanticModel, diags: &mut Diagnostics<ValidateErrorKind>) {
    super::walk_complex(model, |parent, any| match any {
        AnyDataKindRef::Dataset(d) => check_dataset_temporal(d, diags),
        AnyDataKindRef::NestedDataset(nd) => check_nested_dataset_temporal(parent, nd, diags),
        AnyDataKindRef::Grainset(g) => {
            check_complex_extras_grain(g.name(), &g.body.base.extras.temporal, diags);
        }
        AnyDataKindRef::Unionset(u) => {
            check_complex_extras_grain(u.name(), &u.body.base.extras.temporal, diags);
        }
        AnyDataKindRef::Joinset(j) => {
            check_complex_extras_grain(j.name(), &j.body.base.extras.temporal, diags);
        }
        AnyDataKindRef::NestedGrainset(_)
        | AnyDataKindRef::NestedUnionset(_)
        | AnyDataKindRef::NestedJoinset(_) => {}
    });
}

fn check_dataset_temporal(d: &Dataset, diags: &mut Diagnostics<ValidateErrorKind>) {
    if let Some(t) = &d.body.base.extras.temporal {
        if t.granularity.is_none() {
            diags.push(Diagnostic::new(
                ValidateErrorKind::TemporalLeafMissingGranularity {
                    data_kind: d.body.base.name.clone(),
                },
            ));
        }
    }
}

fn check_nested_dataset_temporal(
    parent: Option<AnyDataKindRef<'_>>,
    nd: &NestedDataset,
    diags: &mut Diagnostics<ValidateErrorKind>,
) {
    let parent_variant = parent.map(|p| p.variant());
    let parent_name = parent.map(|p| p.name().to_string()).unwrap_or_default();

    if let Some(t) = &nd.body.base.extras.temporal {
        if t.granularity.is_none() {
            diags.push(Diagnostic::new(
                ValidateErrorKind::TemporalLeafMissingGranularity {
                    data_kind: format!("{}.{}", parent_name, nd.body.base.name),
                },
            ));
        }
    }

    if matches!(parent_variant, Some(DataKindVariant::Grainset)) {
        let axis_without_granularity =
            (nd.body.base.extras.temporal.as_ref()).is_some_and(|t| t.granularity.is_none());
        if axis_without_granularity {
            diags.push(Diagnostic::new(
                ValidateErrorKind::GrainsetChildMissingGranularity {
                    grainset: parent_name,
                    child: nd.body.base.name.clone(),
                },
            ));
        }
    }
}

fn check_complex_extras_grain(
    name: &str,
    temporal: &Option<TemporalShape>,
    diags: &mut Diagnostics<ValidateErrorKind>,
) {
    if let Some(t) = temporal {
        if t.granularity.is_some() {
            diags.push(Diagnostic::new(
                ValidateErrorKind::TemporalGranularityOnComplex {
                    data_kind: name.to_string(),
                },
            ));
        }
    }
}
