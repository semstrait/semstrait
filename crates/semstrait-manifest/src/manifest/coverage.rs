use semstrait_sql::expr::Expr;

use super::expression::StoredExpr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Coverage {
    Native,
    Derived,
    NullFill,
}

impl Coverage {
    #[must_use]
    pub fn of(realization: Option<&StoredExpr>) -> Self {
        match realization.map(StoredExpr::expr) {
            None => Self::NullFill,
            Some(Expr::Column(_)) => Self::Native,
            Some(_) => Self::Derived,
        }
    }
}

#[cfg(test)]
mod coverage_axis_tests {
    use super::*;
    use semstrait_ir::PhysicalExpr;
    use semstrait_model::Agg;
    use semstrait_sql::datatype::DataType;
    use semstrait_sql::expr::{CastFailure, Literal, NamedReference};

    use crate::compile::expressions::lower::wrap_aggregate;

    use crate::manifest::expression::{ExprRoot, ManifestExpressions};

    fn column() -> PhysicalExpr {
        Expr::Column(NamedReference::unqualified("region").expect("a name"))
    }

    fn root(expr: &PhysicalExpr) -> StoredExpr {
        let mut pool = ManifestExpressions::default();
        let id = pool
            .emit(expr, "sig.probe", ExprRoot::SignatureExpression, [])
            .expect("emit");
        pool.get(&id).expect("the emitted entry").expr.clone()
    }

    #[test]
    fn no_realization_for_the_pair_is_null_fill() {
        assert_eq!(Coverage::of(None), Coverage::NullFill);
    }

    #[test]
    fn a_bare_column_is_native() {
        assert_eq!(Coverage::of(Some(&root(&column()))), Coverage::Native);
    }

    #[test]
    fn anything_that_is_not_a_bare_column_is_derived() {
        let literal = Expr::Literal(Literal::String("eu".to_owned()));
        let cast = Expr::Cast {
            input: Box::new(column()),
            target: DataType::String,
            on_failure: CastFailure::Error,
        };
        let sum = Agg::of_name("sum").expect("a catalog aggregate");
        let aggregate =
            wrap_aggregate(column(), (sum.function(), sum.distinct()), "test").expect("sum wraps");
        for shape in [literal, cast, aggregate] {
            assert_eq!(
                Coverage::of(Some(&root(&shape))),
                Coverage::Derived,
                "not a bare column, so `Derived`: {shape:?}"
            );
        }
    }

    #[test]
    fn the_axis_is_three_way_and_a_metadata_value_is_not_a_fourth_outcome() {
        let metadata_value = Expr::Literal(Literal::String("orders.parquet".to_owned()));
        let authored_constant = Expr::Literal(Literal::String("orders.parquet".to_owned()));
        assert_eq!(
            Coverage::of(Some(&root(&metadata_value))),
            Coverage::of(Some(&root(&authored_constant))),
            "the two are the same bytes in the artifact, so they cannot differ here"
        );

        let outcomes = [Coverage::Native, Coverage::Derived, Coverage::NullFill];
        assert_eq!(
            outcomes
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            3,
            "three distinct outcomes and no fourth"
        );
    }
}
