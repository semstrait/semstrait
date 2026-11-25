use std::collections::{BTreeMap, BTreeSet};

use semstrait_common::EntityId;
use semstrait_sql::expr::NamedReference;

use super::assemble::{condition_handles, constituents};
use crate::error::CompileError;
use crate::manifest::expression::{ManifestExpressions, PhysicalExprId};
use crate::manifest::semantic_signature::SemanticSignature;
use crate::manifest::semantic_source::{RoleDefinition, SemanticSource};
use crate::manifest::source_alias::{SourceAlias, SourceAliases};

pub(crate) fn misplaced_columns(
    entry: &SemanticSource,
    semantic_sources: &BTreeMap<EntityId, SemanticSource>,
    signatures: &BTreeMap<EntityId, SemanticSignature>,
    aliases: &SourceAliases,
    pool: &ManifestExpressions,
) -> Vec<CompileError> {
    let (signature_trees, condition) = match &entry.role_definition {
        RoleDefinition::Dataset { source_id, .. } => {
            let Some(alias) = aliases.physical.get(source_id) else {
                return Vec::new();
            };
            (BTreeSet::from([alias]), BTreeSet::new())
        }
        RoleDefinition::Unionset { .. } | RoleDefinition::Grainset { .. } => {
            (semantic(aliases, [&entry.id]), BTreeSet::new())
        }
        RoleDefinition::Joinset { from, to, .. } => (
            semantic(aliases, [from, to]),
            semantic(aliases, constituents(entry, semantic_sources)),
        ),
    };
    let mut faults = Vec::new();
    let mut check = |root: PhysicalExprId, admitted: &BTreeSet<&SourceAlias>| {
        for (expression, column) in pool.columns(&root) {
            let placed = matches!(
                column.parts(),
                [qualifier, _] if admitted.iter().any(|alias| alias.as_str() == qualifier)
            );
            if !placed {
                faults.push(CompileError::ColumnQualifierMismatch {
                    expression,
                    column: column.clone(),
                    expected: admitted.iter().copied().cloned().collect(),
                });
            }
        }
    };
    let signature_roots = entry
        .semantic_signature_ids
        .iter()
        .filter_map(|id| signatures.get(id))
        .flat_map(|signature| {
            std::iter::once(signature.expression_id).chain(signature.filter_expr_id)
        });
    for root in signature_roots {
        check(root, &signature_trees);
    }
    for root in condition_handles(entry) {
        check(root, &condition);
    }
    faults
}

pub(crate) fn qualified_column(
    alias: &str,
    name: &str,
    node: &str,
) -> Result<NamedReference, CompileError> {
    NamedReference::qualified(alias, name).ok_or_else(|| CompileError::EmptyColumnPart {
        node: node.to_owned(),
        parts: vec![alias.to_owned(), name.to_owned()],
    })
}

fn semantic<'a>(
    aliases: &'a SourceAliases,
    ids: impl IntoIterator<Item = &'a EntityId>,
) -> BTreeSet<&'a SourceAlias> {
    ids.into_iter()
        .filter_map(|id| aliases.semantic.get(id))
        .collect()
}
