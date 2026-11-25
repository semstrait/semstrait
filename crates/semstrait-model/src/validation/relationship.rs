// TODO: the joinset-local relationship walk drops the
// parent-scope context (the previous `scope_parent: Option<String>`
// argument was unused). Re-introduce a scope argument when
// joinset-local endpoint resolution actually fires per-joinset
// diagnostics — at that point `check_root_relationships_endpoints`
// should also fan out into each joinset's locally-known names instead
// of consulting only the root data-kind maps.

use crate::error::validate::ValidateErrorKind;
use crate::model::types::data_kinds::AnyDataKindRef;
use crate::model::types::entities::relationship::cross_with_preserved_side;
use crate::model::types::entities::{Cardinality, CrossFilter, JoinKeySpec, Relationship};
use crate::model::SemanticModel;
use semstrait_common::diagnostic::{Diagnostic, Diagnostics};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) const ROOT_SCOPE: &str = "semantic_model";

pub(super) fn check_all_relationships(
    model: &SemanticModel,
    diags: &mut Diagnostics<ValidateErrorKind>,
) {
    check_relationships(model.relationships.values(), diags);
    check_root_relationships_endpoints(model, diags);
    super::walk_complex(model, |_parent, any| {
        let body = match any {
            AnyDataKindRef::Joinset(j) => &j.body,
            AnyDataKindRef::NestedJoinset(j) => &j.body,
            _ => return,
        };
        check_relationships(body.relationships.iter(), diags);
        check_duplicate_names(
            body.relationships.iter().map(|r| r.name.as_str()),
            body.base.name.as_str(),
            diags,
        );
    });
}

fn check_duplicate_names<'a>(
    names: impl Iterator<Item = &'a str>,
    scope: &str,
    diags: &mut Diagnostics<ValidateErrorKind>,
) {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for name in names {
        *counts.entry(name).or_default() += 1;
    }
    report_duplicate_names(counts.into_iter(), scope, diags);
}

pub(crate) fn report_duplicate_names<'a>(
    counts: impl Iterator<Item = (&'a str, usize)>,
    scope: &str,
    diags: &mut Diagnostics<ValidateErrorKind>,
) {
    for (name, occurrences) in counts {
        if occurrences > 1 {
            diags.push(Diagnostic::new(
                ValidateErrorKind::DuplicateRelationshipName {
                    relationship: name.to_owned(),
                    scope: scope.to_owned(),
                    occurrences,
                },
            ));
        }
    }
}

fn check_relationships<'a>(
    rels: impl Iterator<Item = &'a Relationship>,
    diags: &mut Diagnostics<ValidateErrorKind>,
) {
    for r in rels {
        if r.cardinality.is_none() && !matches!(r.on, JoinKeySpec::Cross) {
            diags.push(Diagnostic::new(
                ValidateErrorKind::RelationshipMissingCardinality {
                    relationship: r.name.clone(),
                },
            ));
        }

        diags.extend(cross_with_preserved_side(r).map(Diagnostic::new));

        if matches!(
            r.cardinality,
            Some(Cardinality::OneToOne | Cardinality::ManyToMany)
        ) {
            if r.optional.is_none() {
                diags.push(Diagnostic::new(
                    ValidateErrorKind::RelationshipSymmetricCardinalityIncomplete {
                        relationship: r.name.clone(),
                        missing: "optional".to_string(),
                    },
                ));
            }
            if r.cross_filter.is_none() {
                diags.push(Diagnostic::new(
                    ValidateErrorKind::RelationshipSymmetricCardinalityIncomplete {
                        relationship: r.name.clone(),
                        missing: "cross_filter".to_string(),
                    },
                ));
            }
        }

        if matches!(r.cardinality, Some(Cardinality::ManyToMany)) {
            if let Some(cf) = r.cross_filter {
                if matches!(cf, CrossFilter::Left | CrossFilter::Right) {
                    diags.push(Diagnostic::new(
                        ValidateErrorKind::RelationshipManyToManyCrossFilterDirectional {
                            relationship: r.name.clone(),
                        },
                    ));
                }
            }
        }
    }
}

fn check_root_relationships_endpoints(
    model: &SemanticModel,
    diags: &mut Diagnostics<ValidateErrorKind>,
) {
    let mut known_names: BTreeSet<&str> = BTreeSet::new();
    known_names.extend(model.datasets.values().map(|d| d.body.base.name.as_str()));
    known_names.extend(model.grainsets.values().map(|g| g.body.base.name.as_str()));
    known_names.extend(model.unionsets.values().map(|u| u.body.base.name.as_str()));
    known_names.extend(model.joinsets.values().map(|j| j.body.base.name.as_str()));

    for r in model.relationships.values() {
        if !known_names.contains(r.from.as_str()) {
            diags.push(Diagnostic::new(
                ValidateErrorKind::RelationshipDanglingEndpoint {
                    relationship: r.name.clone(),
                    side: "from".to_string(),
                    endpoint: r.from.as_str().to_string(),
                },
            ));
        }
        if !known_names.contains(r.to.as_str()) {
            diags.push(Diagnostic::new(
                ValidateErrorKind::RelationshipDanglingEndpoint {
                    relationship: r.name.clone(),
                    side: "to".to_string(),
                    endpoint: r.to.as_str().to_string(),
                },
            ));
        }
    }
}
