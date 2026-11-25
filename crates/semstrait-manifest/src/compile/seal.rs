use std::collections::BTreeSet;

use crate::error::CompileError;
use crate::manifest::expression::{ManifestExpressions, PhysicalExprId};

pub(super) fn sweep_orphan_expressions(
    pool: &ManifestExpressions,
    roots: impl IntoIterator<Item = PhysicalExprId>,
) -> Option<CompileError> {
    orphan_expressions(pool, roots)
        .into_iter()
        .next()
        .map(|expr_id| CompileError::OrphanExpression { expr_id })
}

pub(crate) fn orphan_expressions(
    pool: &ManifestExpressions,
    roots: impl IntoIterator<Item = PhysicalExprId>,
) -> Vec<PhysicalExprId> {
    let named: BTreeSet<PhysicalExprId> = roots.into_iter().collect();
    pool.physical
        .keys()
        .filter(|id| !named.contains(id))
        .copied()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use semstrait_ir::PhysicalExpr;
    use semstrait_model::Agg;
    use semstrait_sql::expr::{Expr, NamedReference};

    use crate::compile::expressions::lower::wrap_aggregate;
    use crate::manifest::expression::ExprRoot;

    fn col_leaf(name: &str) -> PhysicalExpr {
        Expr::Column(NamedReference::unqualified(name).expect("a name"))
    }

    fn agg_expr(col: &str) -> PhysicalExpr {
        let sum = Agg::of_name("sum").expect("a catalog aggregate");
        wrap_aggregate(col_leaf(col), (sum.function(), sum.distinct()), "test").expect("sum wraps")
    }

    #[test]
    fn an_entry_that_no_root_names_is_refused() {
        let mut pool = ManifestExpressions::default();
        let referenced = pool
            .emit(
                &col_leaf("region"),
                "sig.a",
                ExprRoot::SignatureExpression,
                [],
            )
            .expect("emit");
        let orphan = pool
            .emit(
                &agg_expr("amount"),
                "sig.b",
                ExprRoot::SignatureExpression,
                [],
            )
            .expect("emit");

        match sweep_orphan_expressions(&pool, [referenced]) {
            Some(CompileError::OrphanExpression { expr_id }) => assert_eq!(expr_id, orphan),
            other => panic!("expected the orphan to be named, got {other:?}"),
        }
    }

    #[test]
    fn a_store_whose_every_entry_a_root_names_passes() {
        let mut pool = ManifestExpressions::default();
        let region = pool
            .emit(
                &col_leaf("region"),
                "sig.a",
                ExprRoot::SignatureExpression,
                [],
            )
            .expect("emit");
        let amount = pool
            .emit(
                &agg_expr("amount"),
                "sig.b",
                ExprRoot::SignatureExpression,
                [],
            )
            .expect("emit");
        assert_eq!(pool.len(), 2, "one entry per tree");
        assert!(sweep_orphan_expressions(&pool, [region, amount]).is_none());
    }
}
