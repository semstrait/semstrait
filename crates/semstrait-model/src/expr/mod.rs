use semstrait_sql::expr::{Expr, IntoLiteral, NamedReference, PhysicalLeaf, RowExpr, SemanticLeaf};

pub type UnresolvedSemanticExpr = RowExpr<SemanticLeaf>;

pub type UnresolvedPhysicalExpr = RowExpr<PhysicalLeaf>;

pub fn col(name: impl Into<String>) -> Option<UnresolvedSemanticExpr> {
    NamedReference::unqualified(name).map(Expr::Column)
}

pub fn field(name: impl Into<String>) -> Option<UnresolvedSemanticExpr> {
    NamedReference::unqualified(name).map(|name| Expr::Leaf(SemanticLeaf::Field(name)))
}

pub fn dim(name: impl Into<String>) -> Option<UnresolvedSemanticExpr> {
    NamedReference::unqualified(name).map(|name| Expr::Leaf(SemanticLeaf::Dimension(name, None)))
}

pub fn measure(name: impl Into<String>) -> Option<UnresolvedSemanticExpr> {
    NamedReference::unqualified(name).map(|name| Expr::Leaf(SemanticLeaf::Measure(name, None)))
}

pub fn metric(name: impl Into<String>) -> Option<UnresolvedSemanticExpr> {
    NamedReference::unqualified(name).map(|name| Expr::Leaf(SemanticLeaf::Metric(name, None)))
}

pub fn lit(value: impl IntoLiteral) -> UnresolvedSemanticExpr {
    Expr::Literal(value.into_literal())
}

pub fn physical_col(name: impl Into<String>) -> Option<UnresolvedPhysicalExpr> {
    NamedReference::unqualified(name).map(Expr::Column)
}

#[cfg(test)]
mod tests {
    use super::*;
    use semstrait_sql::expr::Literal;

    #[test]
    fn lit_gives_the_tagged_literal_of_the_values_canonical_type() {
        for (built, expected) in [
            (lit(42i64), Literal::BigInt(42)),
            (lit(7i32), Literal::Integer(7)),
            (lit(2.5f64), Literal::Double(2.5)),
            (lit("hello"), Literal::String("hello".into())),
            (lit("hi".to_string()), Literal::String("hi".into())),
            (lit(true), Literal::Boolean(true)),
            (lit(Literal::BigInt(99)), Literal::BigInt(99)),
        ] {
            assert_eq!(built, Expr::Literal(expected));
        }
    }

    #[test]
    fn a_leaf_builder_gives_none_for_an_empty_name() {
        for (builder, built) in [
            ("col", col("")),
            ("field", field("")),
            ("dim", dim("")),
            ("measure", measure("")),
            ("metric", metric("")),
        ] {
            assert_eq!(built, None, "{builder}");
        }
        assert_eq!(physical_col(""), None);
    }
}
