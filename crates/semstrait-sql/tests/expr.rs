mod common;

use common::{aggregate_function, ascending, scalar, window_function};
use semstrait_common::{Tree, ValidateError};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{
    AggregateAllowed, AggregateCall, Aggregation, BinaryOpKind, Call, CallCheck, CastFailure,
    DimensionAccessor, Expr, ExprOperationExt, IntoLiteral, Leaf, LikeKind, Literal,
    MeasureAccessor, MetricAccessor, NamedReference, NoAggregate, NoLeaf, NullOrdering, Over,
    PhysicalLeaf, RebuildError, RowExpr, ScalarCall, SemanticLeaf, SortDir, UnaryOpKind,
    UntypedLiteral, WindowBound, WindowCall, WindowFrame, WindowFrameKind,
};

#[test]
fn rust_s_operators_build_the_operation_nodes() {
    let binary = |op, left: i64, right: i64| Expr::BinaryOp {
        op,
        left: boxed(left),
        right: boxed(right),
    };
    let cases = [
        (lit(1) + lit(2), binary(BinaryOpKind::Add, 1, 2)),
        (lit(1) - lit(2), binary(BinaryOpKind::Subtract, 1, 2)),
        (lit(1) * lit(2), binary(BinaryOpKind::Multiply, 1, 2)),
        (lit(1) / lit(2), binary(BinaryOpKind::Divide, 1, 2)),
        (lit(1) % lit(2), binary(BinaryOpKind::Mod, 1, 2)),
        (lit(1) & lit(2), binary(BinaryOpKind::And, 1, 2)),
        (lit(1) | lit(2), binary(BinaryOpKind::Or, 1, 2)),
        (
            -lit(1),
            Expr::UnaryOp {
                op: UnaryOpKind::Negate,
                operand: boxed(1),
            },
        ),
        (
            !lit(1),
            Expr::UnaryOp {
                op: UnaryOpKind::Not,
                operand: boxed(1),
            },
        ),
    ];
    for (built, node) in cases {
        assert_eq!(built, node);
    }
}

#[test]
fn the_comparisons_and_is_null_build_their_operation_nodes() {
    let binary = |op| Expr::BinaryOp {
        op,
        left: boxed(1),
        right: boxed(2),
    };
    let cases = [
        (
            ExprOperationExt::eq(lit(1), lit(2)),
            binary(BinaryOpKind::Eq),
        ),
        (lit(1).neq(lit(2)), binary(BinaryOpKind::NotEq)),
        (lit(1).lt(lit(2)), binary(BinaryOpKind::Lt)),
        (lit(1).lt_eq(lit(2)), binary(BinaryOpKind::LtEq)),
        (lit(1).gt(lit(2)), binary(BinaryOpKind::Gt)),
        (lit(1).gt_eq(lit(2)), binary(BinaryOpKind::GtEq)),
        (lit(1).is_null(), Expr::IsNull(boxed(1))),
    ];
    for (built, node) in cases {
        assert_eq!(built, node);
    }
}

#[test]
fn a_rust_value_is_the_literal_of_its_canonical_data_type() {
    assert_eq!(true.into_literal(), Literal::Boolean(true));
    assert_eq!(1_i32.into_literal(), Literal::Integer(1));
    assert_eq!(1_i64.into_literal(), Literal::BigInt(1));
    assert_eq!(1.5_f64.into_literal(), Literal::Double(1.5));
    assert_eq!("a".into_literal(), Literal::String("a".to_owned()));
    assert_eq!(
        "a".to_owned().into_literal(),
        Literal::String("a".to_owned())
    );
    assert_eq!(
        Literal::Date("2026-05-19".to_owned()).into_literal(),
        Literal::Date("2026-05-19".to_owned())
    );
}

fn tree_parameters<L: Leaf, A: Aggregation>() {}

#[derive(Debug, Clone, PartialEq)]
struct StageLeaf;

impl Leaf for StageLeaf {}

fn lit(value: i64) -> Expr<StageLeaf> {
    Expr::Literal(Literal::BigInt(value))
}

fn boxed(value: i64) -> Box<Expr<StageLeaf>> {
    Box::new(lit(value))
}

fn one_node_of_each_kind(scale: i64) -> Vec<(Expr<StageLeaf>, Vec<i64>)> {
    let operand = |value: i64| lit(value * scale);
    let boxed = |value: i64| Box::new(operand(value));
    vec![
        (lit(1), vec![]),
        (
            Expr::Column(NamedReference::new(["ps0", "amount"]).expect("two parts")),
            vec![],
        ),
        (Expr::Leaf(StageLeaf), vec![]),
        (
            Expr::BinaryOp {
                op: BinaryOpKind::Add,
                left: boxed(1),
                right: boxed(2),
            },
            vec![1, 2],
        ),
        (
            Expr::UnaryOp {
                op: UnaryOpKind::Negate,
                operand: boxed(1),
            },
            vec![1],
        ),
        (
            Expr::Cast {
                input: boxed(1),
                target: DataType::Integer,
                on_failure: CastFailure::Error,
            },
            vec![1],
        ),
        (
            Expr::Case {
                whens: vec![(operand(1), operand(2)), (operand(3), operand(4))],
                else_: Some(boxed(5)),
            },
            vec![1, 2, 3, 4, 5],
        ),
        (
            Expr::Case {
                whens: vec![(operand(1), operand(2))],
                else_: None,
            },
            vec![1, 2],
        ),
        (
            Expr::InList {
                value: boxed(1),
                list: vec![operand(2), operand(3)],
                negated: false,
            },
            vec![1, 2, 3],
        ),
        (
            Expr::Between {
                value: boxed(1),
                low: boxed(2),
                high: boxed(3),
                negated: true,
            },
            vec![1, 2, 3],
        ),
        (
            Expr::Like {
                value: boxed(1),
                pattern: boxed(2),
                kind: LikeKind::ILike,
            },
            vec![1, 2],
        ),
        (Expr::IsNull(boxed(1)), vec![1]),
        (
            Expr::Function(Call::Scalar(
                ScalarCall::builder(scalar("coalesce"), vec![operand(1), operand(2)])
                    .build()
                    .expect("coalesce takes two values"),
            )),
            vec![1, 2],
        ),
        (
            Expr::Function(Call::Aggregate(
                AggregateCall::builder(
                    aggregate_function("string_agg"),
                    vec![operand(1), operand(2)],
                )
                .filter(operand(3))
                .order_by(vec![ascending(operand(4))])
                .build()
                .expect("string_agg takes a FILTER and an ORDER BY"),
                AggregateAllowed,
            )),
            vec![1, 2, 3, 4],
        ),
        (
            Expr::Function(Call::Window(
                WindowCall::builder(window_function("first_value"), vec![operand(1)])
                    .over(Over {
                        partition_by: vec![operand(2)],
                        order_by: vec![ascending(operand(3))],
                        window_frame: None,
                    })
                    .build()
                    .expect("first_value takes a PARTITION BY and an ORDER BY"),
            )),
            vec![1, 2, 3],
        ),
    ]
}

#[test]
fn each_node_kind_gives_its_operands_as_its_children_in_order() {
    for (node, operands) in one_node_of_each_kind(1) {
        let expected: Vec<Expr<StageLeaf>> = operands.into_iter().map(lit).collect();
        assert_eq!(
            node.children(),
            expected.iter().collect::<Vec<_>>(),
            "{node:?}"
        );
    }
}

#[test]
fn each_node_kind_rebuilds_with_new_children() {
    let nodes = one_node_of_each_kind(1).into_iter();
    for ((node, operands), (expected, _)) in nodes.zip(one_node_of_each_kind(10)) {
        let children = operands.iter().map(|value| lit(value * 10)).collect();
        assert_eq!(
            node.clone().with_new_children(children),
            Ok(expected),
            "{node:?}"
        );
    }
}

#[test]
fn each_node_kind_but_a_leaf_rebuilds_over_another_leaf() {
    let nodes = one_node_of_each_kind(1).into_iter();
    for ((node, operands), (expected, _)) in nodes.zip(one_node_of_each_kind(10)) {
        let children = operands.iter().map(|value| lit(value * 10)).collect();
        let rebuilt = node.rebuilt::<StageLeaf>(children, CallCheck::Full);
        match node {
            Expr::Leaf(_) => assert_eq!(rebuilt, Err(RebuildError::Leaf)),
            _ => assert_eq!(rebuilt, Ok(expected), "{node:?}"),
        }
    }
    let sum = Expr::<StageLeaf, NoAggregate>::BinaryOp {
        op: BinaryOpKind::Add,
        left: Box::new(Expr::Leaf(StageLeaf)),
        right: Box::new(Expr::Leaf(StageLeaf)),
    };
    let literal = || Expr::<NoLeaf>::Literal(Literal::BigInt(1));
    assert_eq!(
        sum.rebuilt(vec![literal(), literal()], CallCheck::Full),
        Ok(literal() + literal())
    );
    assert_eq!(
        sum.rebuilt(vec![literal()], CallCheck::Full),
        Err(RebuildError::ChildCount)
    );
}

#[test]
fn a_rebuild_with_another_number_of_children_is_refused() {
    let node = Expr::IsNull(boxed(1));
    assert_eq!(
        node.with_new_children(vec![lit(1), lit(2)]),
        Err(ValidateError::ChildCountMismatch {
            expected: 1,
            got: 2
        })
    );
}

#[test]
fn a_rebuild_refuses_an_empty_in_list_and_a_case_with_no_when() {
    let in_list = Expr::<StageLeaf>::InList {
        value: boxed(1),
        list: vec![],
        negated: false,
    };
    assert_eq!(
        in_list.with_new_children(vec![lit(1)]),
        Err(ValidateError::EmptyInList)
    );
    let case = Expr::<StageLeaf>::Case {
        whens: vec![],
        else_: Some(boxed(1)),
    };
    assert_eq!(
        case.with_new_children(vec![lit(1)]),
        Err(ValidateError::EmptyCase)
    );
}

#[test]
fn a_row_expr_gives_its_children_as_any_tree_does() {
    let row: RowExpr<StageLeaf> = Expr::IsNull(Box::new(Expr::Leaf(StageLeaf)));
    assert_eq!(row.children(), [&Expr::Leaf(StageLeaf)]);
}

#[test]
fn no_leaf_is_a_leaf_under_either_aggregation() {
    tree_parameters::<NoLeaf, AggregateAllowed>();
    tree_parameters::<NoLeaf, NoAggregate>();
}

#[test]
fn a_stage_implements_leaf_for_its_own_leaf() {
    tree_parameters::<StageLeaf, NoAggregate>();
}

#[test]
fn no_leaf_and_no_aggregate_hold_no_value() {
    fn no_leaf(leaf: NoLeaf) -> std::convert::Infallible {
        match leaf {}
    }
    fn no_aggregate(aggregate: NoAggregate) -> std::convert::Infallible {
        match aggregate {}
    }
    let _ = (no_leaf, no_aggregate);
}

fn untyped<L: Leaf>(leaf: &L) -> Option<&UntypedLiteral> {
    leaf.untyped()
}

#[test]
fn an_untyped_literal_s_leaf_gives_its_literal_through_the_leaf_bound() {
    let one = UntypedLiteral(Literal::BigInt(1));
    assert_eq!(untyped(&SemanticLeaf::Untyped(one.clone())), Some(&one));
    let year = UntypedLiteral(Literal::String("year".to_owned()));
    assert_eq!(untyped(&PhysicalLeaf::Untyped(year.clone())), Some(&year));
}

#[test]
fn a_semantic_reference_and_a_stage_s_own_leaf_give_no_untyped_literal() {
    let revenue = NamedReference::new(["revenue"]).expect("one part");
    let references = [
        SemanticLeaf::Field(revenue.clone()),
        SemanticLeaf::Dimension(revenue.clone(), Some(DimensionAccessor::Lag(1))),
        SemanticLeaf::Measure(revenue.clone(), Some(MeasureAccessor::Previous)),
        SemanticLeaf::Metric(revenue, Some(MetricAccessor::PercentChange)),
    ];
    for reference in &references {
        assert_eq!(untyped(reference), None, "{reference:?}");
    }
    assert_eq!(untyped(&StageLeaf), None);
}

fn written<T: serde::Serialize>(values: &[T]) -> Vec<String> {
    values
        .iter()
        .map(|value| serde_json::to_string(value).expect("writes"))
        .collect()
}

fn reads_back<T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug>(
    values: &[T],
) {
    for value in values {
        let form = serde_json::to_string(value).expect("writes");
        assert_eq!(&serde_json::from_str::<T>(&form).expect("reads"), value);
    }
}

#[test]
fn each_operator_kind_writes_today_s_bytes() {
    use BinaryOpKind::*;
    let binary = [
        Add,
        Subtract,
        Multiply,
        Divide,
        SafeDivide,
        Mod,
        Eq,
        NotEq,
        Lt,
        LtEq,
        Gt,
        GtEq,
        IsNotDistinctFrom,
        And,
        Or,
    ];
    assert_eq!(
        written(&binary),
        [
            r#""add""#,
            r#""subtract""#,
            r#""multiply""#,
            r#""divide""#,
            r#""safe_divide""#,
            r#""mod""#,
            r#""eq""#,
            r#""not_eq""#,
            r#""lt""#,
            r#""lt_eq""#,
            r#""gt""#,
            r#""gt_eq""#,
            r#""is_not_distinct_from""#,
            r#""and""#,
            r#""or""#,
        ]
    );
    reads_back(&binary);
    let unary = [UnaryOpKind::Negate, UnaryOpKind::Not];
    assert_eq!(written(&unary), [r#""negate""#, r#""not""#]);
    reads_back(&unary);
    let like = [
        LikeKind::Like,
        LikeKind::NotLike,
        LikeKind::ILike,
        LikeKind::NotILike,
    ];
    assert_eq!(
        written(&like),
        [
            r#""like""#,
            r#""not_like""#,
            r#""i_like""#,
            r#""not_i_like""#
        ]
    );
    reads_back(&like);
    let cast = [CastFailure::Error, CastFailure::Null];
    assert_eq!(written(&cast), [r#""error""#, r#""null""#]);
    reads_back(&cast);
}

#[test]
fn each_window_type_writes_today_s_bytes() {
    let bounds = [
        WindowBound::UnboundedPreceding,
        WindowBound::Preceding(3),
        WindowBound::CurrentRow,
        WindowBound::Following(5),
        WindowBound::UnboundedFollowing,
    ];
    assert_eq!(
        written(&bounds),
        [
            r#"{"type":"unbounded_preceding"}"#,
            r#"{"type":"preceding","content":3}"#,
            r#"{"type":"current_row"}"#,
            r#"{"type":"following","content":5}"#,
            r#"{"type":"unbounded_following"}"#,
        ]
    );
    reads_back(&bounds);
    let kinds = [
        WindowFrameKind::Rows,
        WindowFrameKind::Range,
        WindowFrameKind::Groups,
    ];
    assert_eq!(written(&kinds), [r#""rows""#, r#""range""#, r#""groups""#]);
    reads_back(&kinds);
    let frame = [WindowFrame::new(
        WindowFrameKind::Rows,
        WindowBound::UnboundedPreceding,
        WindowBound::CurrentRow,
    )];
    assert_eq!(
        written(&frame),
        [r#"{"kind":"rows","start":{"type":"unbounded_preceding"},"end":{"type":"current_row"}}"#]
    );
    reads_back(&frame);
    let orders = [
        SortDir::Asc {
            nulls: NullOrdering::First,
        },
        SortDir::Desc {
            nulls: NullOrdering::Last,
        },
        SortDir::Asc {
            nulls: NullOrdering::Unspecified,
        },
    ];
    assert_eq!(
        written(&orders),
        [
            r#"{"asc":{"nulls":"first"}}"#,
            r#"{"desc":{"nulls":"last"}}"#,
            r#"{"asc":{"nulls":"unspecified"}}"#,
        ]
    );
    reads_back(&orders);
}

#[test]
fn a_window_frame_a_bound_and_a_direction_refuse_an_unknown_key() {
    let window_frame = r#"{"kind":"rows","start":{"type":"unbounded_preceding"},"end":{"type":"current_row"},"exclude":"ties"}"#;
    let bound = r#"{"type":"preceding","content":3,"inclusive":true}"#;
    let direction = r#"{"asc":{"nulls":"first","collation":"C"}}"#;
    let reads = [
        (
            "exclude",
            serde_json::from_str::<WindowFrame>(window_frame).map(drop),
        ),
        (
            "inclusive",
            serde_json::from_str::<WindowBound>(bound).map(drop),
        ),
        (
            "collation",
            serde_json::from_str::<SortDir>(direction).map(drop),
        ),
    ];
    for (key, read) in reads {
        let refusal = read.expect_err(key).to_string();
        assert!(refusal.contains(key), "{key}: {refusal}");
    }
}

#[test]
fn a_name_path_s_last_part_is_its_name() {
    let column = NamedReference::new(["ps0", "amount"]).expect("two parts");
    assert_eq!(column.parts(), ["ps0", "amount"]);
    assert_eq!(column.name(), "amount");
    assert_eq!(
        NamedReference::new(["amount"]).expect("one part").name(),
        "amount"
    );
}

#[test]
fn an_unqualified_reference_is_one_part_and_a_qualified_one_two_and_neither_is_cut_at_a_dot() {
    let unqualified = NamedReference::unqualified("a.b").expect("a name");
    assert_eq!(unqualified.parts(), ["a.b"]);
    let qualified = NamedReference::qualified("ssd4", "a.b").expect("an alias and a name");
    assert_eq!(qualified.parts(), ["ssd4", "a.b"]);
    assert_eq!(qualified.name(), "a.b");
}

#[test]
fn an_unqualified_or_a_qualified_reference_with_an_empty_part_is_none() {
    assert_eq!(NamedReference::unqualified(""), None);
    for (qualifier, name) in [("ps0", ""), ("ssd0", ""), ("", "amount"), ("", "")] {
        assert_eq!(
            NamedReference::qualified(qualifier, name),
            None,
            "{qualifier:?}, {name:?}"
        );
    }
}

#[test]
fn a_name_path_with_no_part_is_none() {
    assert_eq!(NamedReference::new(Vec::<String>::new()), None);
}

#[test]
fn a_name_path_with_an_empty_part_is_none_and_its_serde_form_is_refused() {
    for parts in [&["ps0", ""][..], &[""], &["", "amount"]] {
        assert_eq!(
            NamedReference::new(parts.iter().copied()),
            None,
            "{parts:?}"
        );
        let json = serde_json::json!(parts);
        let refusal = serde_json::from_value::<NamedReference>(json)
            .expect_err("an empty part is refused")
            .to_string();
        assert_eq!(
            refusal,
            r#"invalid value: string "", expected a name path of one part or more, none of them empty"#,
            "{parts:?}"
        );
    }
}

#[test]
fn a_name_path_s_serde_form_is_an_array_of_strings() {
    let column = NamedReference::new(["ps0", "amount"]).expect("two parts");
    let json = serde_json::to_value(&column).expect("serializes");
    assert_eq!(json, serde_json::json!(["ps0", "amount"]));
    assert_eq!(
        serde_json::from_value::<NamedReference>(json).expect("reads back"),
        column
    );
    assert!(serde_json::from_str::<NamedReference>("[]").is_err());
}
