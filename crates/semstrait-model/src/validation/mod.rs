mod cycle_check;
pub(crate) mod relationship;
mod semantics;
mod structure;
mod temporal;

use crate::error::validate::ValidateErrorKind;
use crate::model::types::data_kinds::{AnyDataKindRef, GrainsetBody, JoinsetBody, UnionsetBody};
use crate::model::SemanticModel;
use semstrait_common::diagnostic::{split_by_severity, Diagnostics};

pub(crate) const MAX_COMPOSITION_DEPTH: usize = 4;

pub fn validate(
    model: &SemanticModel,
) -> Result<Diagnostics<ValidateErrorKind>, Diagnostics<ValidateErrorKind>> {
    let mut diags: Diagnostics<ValidateErrorKind> = Vec::new();

    structure::check_structure(model, &mut diags);
    temporal::check_temporal(model, &mut diags);
    relationship::check_all_relationships(model, &mut diags);
    semantics::check_all_semantics(model, &mut diags);
    cycle_check::check_all_cycles(model, &mut diags);

    let (errors, warnings) = split_by_severity(diags);
    if errors.is_empty() {
        Ok(warnings)
    } else {
        let mut combined = errors;
        combined.extend(warnings);
        Err(combined)
    }
}

pub(super) fn walk_complex<'a, F>(model: &'a SemanticModel, mut visitor: F)
where
    F: FnMut(Option<AnyDataKindRef<'a>>, AnyDataKindRef<'a>),
{
    for any in model.iter_all() {
        visitor(None, any);
        descend(any, &mut visitor);
    }
}

fn descend<'a, F>(parent: AnyDataKindRef<'a>, visitor: &mut F)
where
    F: FnMut(Option<AnyDataKindRef<'a>>, AnyDataKindRef<'a>),
{
    let children: Vec<AnyDataKindRef<'a>> = match parent {
        AnyDataKindRef::Grainset(g) => grainset_children(&g.body),
        AnyDataKindRef::NestedGrainset(g) => grainset_children(&g.body),
        AnyDataKindRef::Unionset(u) => unionset_children(&u.body),
        AnyDataKindRef::NestedUnionset(u) => unionset_children(&u.body),
        AnyDataKindRef::Joinset(j) => joinset_children(&j.body),
        AnyDataKindRef::NestedJoinset(j) => joinset_children(&j.body),
        _ => return,
    };
    for child in children {
        visitor(Some(parent), child);
        descend(child, visitor);
    }
}

fn grainset_children(body: &GrainsetBody) -> Vec<AnyDataKindRef<'_>> {
    body.datasets
        .iter()
        .map(AnyDataKindRef::NestedDataset)
        .chain(body.unionsets.iter().map(AnyDataKindRef::NestedUnionset))
        .chain(body.joinsets.iter().map(AnyDataKindRef::NestedJoinset))
        .collect()
}

fn unionset_children(body: &UnionsetBody) -> Vec<AnyDataKindRef<'_>> {
    body.datasets
        .iter()
        .map(AnyDataKindRef::NestedDataset)
        .chain(body.grainsets.iter().map(AnyDataKindRef::NestedGrainset))
        .chain(body.joinsets.iter().map(AnyDataKindRef::NestedJoinset))
        .collect()
}

fn joinset_children(body: &JoinsetBody) -> Vec<AnyDataKindRef<'_>> {
    body.datasets
        .iter()
        .map(AnyDataKindRef::NestedDataset)
        .chain(body.grainsets.iter().map(AnyDataKindRef::NestedGrainset))
        .chain(body.unionsets.iter().map(AnyDataKindRef::NestedUnionset))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_model_fires_diagnostic() {
        let m = SemanticModel {
            name: "tiny".to_string(),
            ..SemanticModel::default()
        };
        let err = validate(&m).unwrap_err();
        assert!(err
            .iter()
            .any(|d| matches!(d.kind, ValidateErrorKind::EmptyModel)));
    }
}
