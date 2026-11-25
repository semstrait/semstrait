use crate::error::validate::ValidateErrorKind;
use crate::model::builder::model::SemanticModelBuilder;
use semstrait_common::diagnostic::{Diagnostic, Diagnostics, Location};
use semstrait_common::identity::EntityNamespace;

pub(super) fn collect_duplicate_data_kinds(
    builder: &SemanticModelBuilder,
    diags: &mut Diagnostics<ValidateErrorKind>,
) {
    emit_duplicate_names(
        builder,
        EntityNamespace::DataKind,
        diags,
        |name, occurrences| ValidateErrorKind::DuplicateDataKindName { name, occurrences },
    );
}

pub(super) fn collect_duplicate_shared_semantics(
    builder: &SemanticModelBuilder,
    diags: &mut Diagnostics<ValidateErrorKind>,
) {
    for (kind, carrier) in [
        (EntityNamespace::Dimension, "dimensions"),
        (EntityNamespace::Measure, "measures"),
        (EntityNamespace::Metric, "metrics"),
    ] {
        emit_duplicate_names(builder, kind, diags, move |name, occurrences| {
            ValidateErrorKind::DuplicateSharedSemanticsName {
                carrier: carrier.to_owned(),
                name,
                occurrences,
            }
        });
    }
}

pub(super) fn collect_repeated_interface_member_paths(
    builder: &SemanticModelBuilder,
    diags: &mut Diagnostics<ValidateErrorKind>,
) {
    for (path, locations) in builder.ledger.claims_in(EntityNamespace::InterfaceMember) {
        if locations.len() < 2 {
            continue;
        }
        let (data_kind, collection, name) = split_member_path(path);
        diags.push(Diagnostic::new(
            ValidateErrorKind::InterfaceMemberPathNotUnique {
                data_kind,
                collection,
                name,
                occurrences: locations.to_vec(),
            },
        ));
    }
}

pub(super) fn collect_duplicate_relationship_names(
    builder: &SemanticModelBuilder,
    diags: &mut Diagnostics<ValidateErrorKind>,
) {
    crate::validation::relationship::report_duplicate_names(
        builder
            .ledger
            .claims_in(EntityNamespace::Relationship)
            .map(|(name, locations)| (name, locations.len())),
        crate::validation::relationship::ROOT_SCOPE,
        diags,
    );
}

pub(super) fn collect_duplicate_entity_ids(
    builder: &SemanticModelBuilder,
    diags: &mut Diagnostics<ValidateErrorKind>,
) {
    for (id, occurrences) in &builder.id_locations {
        if occurrences.len() > 1 {
            diags.push(Diagnostic::new(ValidateErrorKind::DuplicateEntityId {
                id: id.clone(),
                occurrences: occurrences.clone(),
            }));
        }
    }
}

fn emit_duplicate_names(
    builder: &SemanticModelBuilder,
    kind: EntityNamespace,
    diags: &mut Diagnostics<ValidateErrorKind>,
    make: impl Fn(String, Vec<Location>) -> ValidateErrorKind,
) {
    for (name, occurrences) in builder.ledger.claims_in(kind) {
        if occurrences.len() > 1 {
            diags.push(Diagnostic::new(make(name.to_owned(), occurrences.to_vec())));
        }
    }
}

fn split_member_path(path: &str) -> (String, String, String) {
    let mut parts = path.splitn(3, '.');
    let data_kind = parts.next().unwrap_or_default().to_owned();
    let collection = parts.next().unwrap_or_default().to_owned();
    let name = parts.next().unwrap_or_default().to_owned();
    (data_kind, collection, name)
}
