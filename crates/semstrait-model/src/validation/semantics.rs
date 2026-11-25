use crate::error::validate::ValidateErrorKind;
use crate::model::types::data_kinds::{DataKind, PublicDataKind};
use crate::model::types::entities::{Metric, SemanticInterface, SemanticSite};
use crate::model::SemanticModel;
use semstrait_common::diagnostic::{Diagnostic, Diagnostics};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn check_all_semantics(
    model: &SemanticModel,
    diags: &mut Diagnostics<ValidateErrorKind>,
) {
    check_orphan_pool_semantics(model, diags);
    check_pool_ref_expressions(model, diags);
    check_inline_shadow_root_pool(model, diags);
    check_identical_keyset_membership(model, diags);
}

fn check_identical_keyset_membership(
    model: &SemanticModel,
    diags: &mut Diagnostics<ValidateErrorKind>,
) {
    for kind in model.iter_public() {
        let mut by_members: BTreeMap<Vec<String>, Vec<String>> = BTreeMap::new();
        for (keyset, members) in kind.semantic_interface().keysets() {
            let members: Vec<String> = members.iter().map(|n| n.as_str().to_owned()).collect();
            by_members.entry(members).or_default().push(keyset);
        }
        for (members, keysets) in by_members {
            if keysets.len() > 1 {
                diags.push(Diagnostic::new(
                    ValidateErrorKind::DataKindKeysetsIdenticalMembership {
                        data_kind: kind.name().to_owned(),
                        keysets,
                        members,
                    },
                ));
            }
        }
    }
}

fn check_orphan_pool_semantics(model: &SemanticModel, diags: &mut Diagnostics<ValidateErrorKind>) {
    let referenced_dimensions =
        collect_referenced_pool_names(model, |i| reference_target_names(i.dimensions.values()));
    let referenced_measures =
        collect_referenced_pool_names(model, |i| reference_target_names(i.measures.values()));
    let referenced_metrics =
        collect_referenced_pool_names(model, |i| reference_target_names(i.metrics.values()));

    for d in model.dimensions.values() {
        let name = d.name().as_str();
        if !referenced_dimensions.contains(name) {
            diags.push(Diagnostic::new(ValidateErrorKind::OrphanSharedSemantics {
                carrier: "Dimension".to_string(),
                name: name.to_string(),
            }));
        }
    }
    for m in model.measures.values() {
        let name = m.name().as_str();
        if !referenced_measures.contains(name) {
            diags.push(Diagnostic::new(ValidateErrorKind::OrphanSharedSemantics {
                carrier: "Measure".to_string(),
                name: name.to_string(),
            }));
        }
    }
    for m in model.metrics.values() {
        let name = m.name().as_str();
        if !referenced_metrics.contains(name) {
            diags.push(Diagnostic::new(ValidateErrorKind::OrphanSharedSemantics {
                carrier: "Metric".to_string(),
                name: name.to_string(),
            }));
        }
    }
}

fn reference_target_names<'a, I, R>(
    sites: impl Iterator<Item = &'a SemanticSite<I, R>>,
) -> Vec<String>
where
    I: crate::model::types::entities::AuthoringBody + 'a,
    R: crate::model::types::entities::AuthoringBody + 'a,
{
    sites
        .filter(|s| matches!(s, SemanticSite::Reference(_)))
        .map(|s| s.name().as_str().to_string())
        .collect()
}

fn collect_referenced_pool_names<F>(model: &SemanticModel, picker: F) -> BTreeSet<String>
where
    F: Fn(&SemanticInterface) -> Vec<String>,
{
    let mut out = BTreeSet::new();
    for d in model.datasets.values() {
        for n in picker(&d.semantic_interface) {
            out.insert(n);
        }
    }
    for g in model.grainsets.values() {
        for n in picker(&g.semantic_interface) {
            out.insert(n);
        }
    }
    for u in model.unionsets.values() {
        for n in picker(&u.semantic_interface) {
            out.insert(n);
        }
    }
    for j in model.joinsets.values() {
        for n in picker(&j.semantic_interface) {
            out.insert(n);
        }
    }
    out
}

fn check_pool_ref_expressions(model: &SemanticModel, diags: &mut Diagnostics<ValidateErrorKind>) {
    let scan_iface = |iface: &SemanticInterface, diags: &mut Diagnostics<ValidateErrorKind>| {
        for site in iface.metrics.values() {
            if let SemanticSite::Reference(r) = site {
                let name = r.name();
                let pool_entry = model.metrics.values().find(|m| m.name() == name);
                let pool_has_expr = pool_entry.and_then(Metric::expr).is_some();
                if !pool_has_expr && r.expr().is_none() {
                    diags.push(Diagnostic::new(
                        ValidateErrorKind::SemanticsRefMissingExpr {
                            carrier: "Metric".to_string(),
                            name: name.as_str().to_string(),
                        },
                    ));
                }
            }
        }
    };

    for d in model.datasets.values() {
        scan_iface(&d.semantic_interface, diags);
    }
    for g in model.grainsets.values() {
        scan_iface(&g.semantic_interface, diags);
    }
    for u in model.unionsets.values() {
        scan_iface(&u.semantic_interface, diags);
    }
    for j in model.joinsets.values() {
        scan_iface(&j.semantic_interface, diags);
    }
}

fn check_inline_shadow_root_pool(
    model: &SemanticModel,
    diags: &mut Diagnostics<ValidateErrorKind>,
) {
    let scan = |iface: &SemanticInterface, diags: &mut Diagnostics<ValidateErrorKind>| {
        for site in iface.dimensions.values() {
            if matches!(site, SemanticSite::Inline(_)) {
                let name = site.name();
                if model.dimensions.values().any(|p| p.name() == name) {
                    diags.push(Diagnostic::new(
                        ValidateErrorKind::SemanticsShadowRootPool {
                            carrier: "Dimension".to_string(),
                            name: name.as_str().to_string(),
                        },
                    ));
                }
            }
        }
        for site in iface.measures.values() {
            if matches!(site, SemanticSite::Inline(_)) {
                let name = site.name();
                if model.measures.values().any(|p| p.name() == name) {
                    diags.push(Diagnostic::new(
                        ValidateErrorKind::SemanticsShadowRootPool {
                            carrier: "Measure".to_string(),
                            name: name.as_str().to_string(),
                        },
                    ));
                }
            }
        }
        for site in iface.metrics.values() {
            if matches!(site, SemanticSite::Inline(_)) {
                let name = site.name();
                if model.metrics.values().any(|p| p.name() == name) {
                    diags.push(Diagnostic::new(
                        ValidateErrorKind::SemanticsShadowRootPool {
                            carrier: "Metric".to_string(),
                            name: name.as_str().to_string(),
                        },
                    ));
                }
            }
        }
    };

    for d in model.datasets.values() {
        scan(&d.semantic_interface, diags);
    }
    for g in model.grainsets.values() {
        scan(&g.semantic_interface, diags);
    }
    for u in model.unionsets.values() {
        scan(&u.semantic_interface, diags);
    }
    for j in model.joinsets.values() {
        scan(&j.semantic_interface, diags);
    }
}
