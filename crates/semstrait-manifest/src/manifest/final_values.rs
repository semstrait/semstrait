use std::collections::BTreeSet;

use semstrait_common::EntityId;
use semstrait_sql::expr::NamedReference;

use super::expression::ManifestExpressions;
use super::expression_layer::ExprLayer;
use super::semantic_definition::{SemanticDefinition, SemanticRoleDefinition};
use super::semantic_source::RoleDefinition;
use super::serving::Serving;
use crate::identity::mint_signature_id;

#[derive(Clone, Copy)]
pub(crate) struct ChildPools<'a> {
    pub serving: Serving<'a>,
    pub expressions: &'a ManifestExpressions,
}

impl<'a> ChildPools<'a> {
    fn layer_stored_by<'s>(
        &self,
        source: &'s EntityId,
        definition: &EntityId,
        seen: &mut BTreeSet<&'s EntityId>,
    ) -> Option<ExprLayer>
    where
        'a: 's,
    {
        if !seen.insert(source) {
            return None;
        }
        let own = self
            .serving
            .signatures()
            .get(&mint_signature_id(source, definition))
            .and_then(|signature| self.expressions.get(&signature.expression_id))
            .map(|entry| entry.layer);
        let RoleDefinition::Grainset {
            grains: children, ..
        } = &self.serving.sources().get(source)?.role_definition
        else {
            return own;
        };
        own.or_else(|| self.layer_stored_by_any(children, definition, seen))
    }

    fn layer_stored_by_any<'s>(
        &self,
        sources: &'s [EntityId],
        definition: &EntityId,
        seen: &mut BTreeSet<&'s EntityId>,
    ) -> Option<ExprLayer>
    where
        'a: 's,
    {
        sources
            .iter()
            .find_map(|source| self.layer_stored_by(source, definition, seen))
    }
}

impl ChildPools<'_> {
    pub(crate) fn final_layer(
        &self,
        children: &[EntityId],
        definition: &SemanticDefinition,
    ) -> ExprLayer {
        let stored = self.layer_stored_by_any(children, &definition.id, &mut BTreeSet::new());
        layer_of_final_value(&definition.role_definition, stored)
    }
}

pub(crate) fn layer_of_final_value(
    role: &SemanticRoleDefinition,
    stored_by_child: Option<ExprLayer>,
) -> ExprLayer {
    match (role, role.aggregation()) {
        (SemanticRoleDefinition::Dimension { .. } | SemanticRoleDefinition::Filter, _) => {
            ExprLayer::Scalar
        }
        (_, Some(_)) => ExprLayer::Aggregate,
        (_, None) if stored_by_child == Some(ExprLayer::Scalar) => ExprLayer::Scalar,
        (_, None) => ExprLayer::Aggregate,
    }
}

pub(crate) fn final_value_layer(
    alias: &str,
    column: &NamedReference,
    layer_of_name: impl FnOnce(&str) -> Option<ExprLayer>,
) -> ExprLayer {
    final_value_name(alias, column)
        .and_then(layer_of_name)
        .unwrap_or(ExprLayer::Scalar)
}

fn final_value_name<'c>(alias: &str, column: &'c NamedReference) -> Option<&'c str> {
    match column.parts() {
        [column_alias, name] if column_alias.as_str() == alias => Some(name.as_str()),
        _ => None,
    }
}
