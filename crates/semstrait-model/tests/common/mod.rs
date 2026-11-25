#![allow(dead_code)]

pub mod golden;

use semstrait_common::diagnostic::Diagnostic;
use semstrait_common::Granularity;
use semstrait_model::{
    parse, Agg, ComplexExtras, LeafExtras, ModelBuildErrorKind, NestedDataset, ParseErrorKind,
    SemanticModel, TemporalShape, UnresolvedSemanticExpr, ValidateErrorKind,
};
use semstrait_sql::expr::{Call, Expr, Leaf, Literal, RowExpr, SemanticLeaf, UntypedLiteral};
use semstrait_sql::functions::registry;

pub fn leaf(name: &str) -> NestedDataset {
    NestedDataset::builder(name)
        .extras(LeafExtras::default())
        .build()
}

pub fn leaf_with_granularity(name: &str, granularity: Granularity) -> NestedDataset {
    NestedDataset::builder(name)
        .extras(LeafExtras::default().with_temporal(TemporalShape::events("ts", Some(granularity))))
        .build()
}

pub fn leaf_with_axis_only(name: &str) -> NestedDataset {
    NestedDataset::builder(name)
        .extras(LeafExtras::default().with_temporal(TemporalShape::events("ts", None)))
        .build()
}

pub fn declares(granularity: Granularity) -> ComplexExtras {
    ComplexExtras::default().with_temporal(TemporalShape::events("ts", Some(granularity)))
}

pub fn expression_model(expr: &str) -> String {
    format!(
        r#"
semantic_model:
  name: expressions
  dimensions:
    - name: written
      data_type: string
      type: categorical
      expr: {expr}
  datasets:
    - name: orders
      dimensions:
        - ref: written
"#
    )
}

pub fn built(yaml: &str) -> SemanticModel {
    parse(yaml)
        .unwrap_or_else(|diagnostics| panic!("parse refused:\n{yaml}\n{diagnostics:?}"))
        .build()
        .unwrap_or_else(|diagnostics| panic!("build refused:\n{yaml}\n{diagnostics:?}"))
        .0
}

pub fn expression(model: &SemanticModel) -> UnresolvedSemanticExpr {
    let dimension = model.dimensions.values().next().expect("one Dimension");
    dimension.body.body.expr.clone().expect("an expression")
}

pub fn parsed_expression(expr: &str) -> UnresolvedSemanticExpr {
    expression(&built(&expression_model(expr)))
}

pub fn refused(yaml: &str) -> Vec<ParseErrorKind> {
    let Err(diagnostics) = parse(yaml) else {
        panic!("parse accepted:\n{yaml}");
    };
    diagnostics
        .into_iter()
        .map(|diagnostic| diagnostic.kind)
        .collect()
}

pub fn refused_expression(expr: &str) -> Vec<ParseErrorKind> {
    refused(&expression_model(expr))
}

pub fn untyped(literal: Literal) -> UnresolvedSemanticExpr {
    Expr::Leaf(SemanticLeaf::Untyped(UntypedLiteral(literal)))
}

pub fn call<L: Leaf>(name: &str, args: Vec<RowExpr<L>>) -> RowExpr<L> {
    let function = registry()
        .get(name)
        .expect("a catalog function")
        .try_into()
        .expect("a function that a row-level tree holds");
    Expr::Function(Call::new(function, args).expect("a call that fits its declaration"))
}

pub fn agg(name: &str) -> Agg {
    Agg::of_name(name).expect("a catalog aggregate")
}

pub fn names_variant(
    diagnostics: &[Diagnostic<ModelBuildErrorKind>],
    wanted: &ValidateErrorKind,
) -> bool {
    diagnostics
        .iter()
        .any(|d| matches!(&d.kind, ModelBuildErrorKind::Validate(k) if k == wanted))
}
