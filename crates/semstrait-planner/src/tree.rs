use semstrait_common::{Tree, ValidateError};
use semstrait_ir::PhysicalExpr;
use semstrait_sql::expr::{Call, Expr, NamedReference};

use crate::steps::Aggregate;

pub(crate) fn column_leaves(tree: &PhysicalExpr) -> impl Iterator<Item = &NamedReference> {
    tree.pre_order().filter_map(|node| match node {
        Expr::Column(column) => Some(column),
        _ => None,
    })
}

pub(crate) fn replace_columns(
    tree: PhysicalExpr,
    by: &[(NamedReference, PhysicalExpr)],
) -> Result<PhysicalExpr, ValidateError> {
    tree.transform(&mut |node| {
        let Expr::Column(column) = &node else {
            return Ok(node);
        };
        let replaced = (by.iter()).find(|(field, _)| field == column);
        Ok(replaced.map_or(node, |(_, tree)| tree.clone()))
    })
}

pub(crate) fn aggregate_calls(tree: &PhysicalExpr) -> impl Iterator<Item = &Aggregate> {
    tree.pre_order().filter_map(|node| match node {
        Expr::Function(Call::Aggregate(call, _)) => Some(call),
        _ => None,
    })
}
