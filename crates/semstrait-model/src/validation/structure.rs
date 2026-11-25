use crate::error::validate::ValidateErrorKind;
use crate::model::types::data_kinds::{
    AnyDataKindRef, ComplexDataKind, DataKind, NestedDataKindRef,
};
use crate::model::SemanticModel;
use semstrait_common::diagnostic::{Diagnostic, Diagnostics};

pub(super) fn check_structure(model: &SemanticModel, diags: &mut Diagnostics<ValidateErrorKind>) {
    check_empty_model(model, diags);
    super::walk_complex(model, |_parent, any| match any {
        AnyDataKindRef::Grainset(g) => {
            check_complex_min_children(g.name(), g.child_count(), g.body.minimum_children(), diags)
        }
        AnyDataKindRef::NestedGrainset(g) => {
            check_complex_min_children(g.name(), g.child_count(), g.body.minimum_children(), diags)
        }
        AnyDataKindRef::Unionset(u) => {
            check_complex_min_children(u.name(), u.child_count(), u.body.minimum_children(), diags)
        }
        AnyDataKindRef::NestedUnionset(u) => {
            check_complex_min_children(u.name(), u.child_count(), u.body.minimum_children(), diags)
        }
        AnyDataKindRef::Joinset(j) => {
            check_complex_min_children(j.name(), j.child_count(), j.body.minimum_children(), diags)
        }
        AnyDataKindRef::NestedJoinset(j) => {
            check_complex_min_children(j.name(), j.child_count(), j.body.minimum_children(), diags)
        }
        _ => {}
    });
    check_composition_depth(model, diags);
}

fn check_empty_model(model: &SemanticModel, diags: &mut Diagnostics<ValidateErrorKind>) {
    if model.is_empty_model() {
        diags.push(Diagnostic::new(ValidateErrorKind::EmptyModel));
    }
}

fn check_complex_min_children(
    name: &str,
    count: usize,
    minimum: usize,
    diags: &mut Diagnostics<ValidateErrorKind>,
) {
    if count < minimum {
        diags.push(Diagnostic::new(
            ValidateErrorKind::ComplexDataKindInsufficientChildren {
                parent: name.to_string(),
                child_count: count,
            },
        ));
    }
}

fn check_composition_depth(model: &SemanticModel, diags: &mut Diagnostics<ValidateErrorKind>) {
    for root in model.iter_complex() {
        let depth = composition_depth(&root);
        if depth > super::MAX_COMPOSITION_DEPTH {
            diags.push(Diagnostic::new(
                ValidateErrorKind::CompositionNestingTooDeep {
                    root: root.name().to_string(),
                    depth,
                },
            ));
        }
    }
}

fn composition_depth(composition: &impl ComplexDataKind) -> usize {
    1 + composition
        .children_ref()
        .filter_map(|child| match child {
            NestedDataKindRef::Dataset(_) => None,
            NestedDataKindRef::Grainset(g) => Some(composition_depth(g)),
            NestedDataKindRef::Unionset(u) => Some(composition_depth(u)),
            NestedDataKindRef::Joinset(j) => Some(composition_depth(j)),
        })
        .max()
        .unwrap_or(0)
}
