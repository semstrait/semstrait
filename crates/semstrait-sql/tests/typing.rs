use semstrait_common::Dialect;
use semstrait_sql::datatype::{
    promote, CanonicalTypeSystem, DataType, SubstraitTypeSystem, TypeFamily, TypeOrFamily,
    TypeSystem,
};
use semstrait_sql::expr::{
    is_condition, type_tree, AggregateCall, BinaryOpKind, Call, CallClause, CallError, CastFailure,
    Expr, ExprOperationExt, FunctionRef, InputTypes, LikeKind, Literal, LiteralError,
    NamedReference, OptionSetting, Over, PhysicalLeaf, ScalarCall, SemanticLeaf, TypeError, Typed,
    UntypedLiteral, WindowBound, WindowCall, WindowFrame, WindowFrameKind,
};
use semstrait_sql::functions::ParameterType;
use semstrait_sql::DocumentDialect;

mod common;

use common::{ascending, function, window_function};

fn column_type(name: &str) -> Typed {
    let data_type = match name {
        "qty" | "bonus" => DataType::Integer,
        "units" => DataType::BigInt,
        "price" | "amount" => decimal(18, 2),
        "rate" => decimal(10, 4),
        "score" => DataType::Double,
        "ratio" => DataType::Real,
        "day" => DataType::Date,
        "region" | "name" | "note" => DataType::String,
        _ => DataType::Unknown,
    };
    Typed::new(data_type, matches!(name, "bonus" | "note"))
}

struct Columns;

impl InputTypes<PhysicalLeaf> for Columns {
    fn column(&self, column: &NamedReference) -> Typed {
        column_type(column.name())
    }

    fn leaf(&self, leaf: &PhysicalLeaf) -> Typed {
        unreachable!("typing types an untyped literal by its context: {leaf:?}")
    }
}

impl InputTypes<SemanticLeaf> for Columns {
    fn column(&self, column: &NamedReference) -> Typed {
        column_type(column.name())
    }

    fn leaf(&self, leaf: &SemanticLeaf) -> Typed {
        match leaf {
            SemanticLeaf::Dimension(dimension, None) => column_type(dimension.name()),
            other => unreachable!("the tests type no other leaf: {other:?}"),
        }
    }
}

type Tree = Expr<PhysicalLeaf>;

const fn decimal(precision: u8, scale: i8) -> DataType {
    DataType::Decimal { precision, scale }
}

fn col(name: &str) -> Tree {
    Expr::Column(NamedReference::new([name]).expect("one part"))
}

fn tagged(literal: Literal) -> Tree {
    Expr::Literal(literal)
}

fn untyped(literal: Literal) -> Tree {
    Expr::Leaf(PhysicalLeaf::Untyped(UntypedLiteral(literal)))
}

fn decimal_literal(value: &str, precision: u8, scale: i8) -> Literal {
    Literal::Decimal {
        value: value.into(),
        precision,
        scale,
    }
}

fn typed_in(tree: Tree, context: Option<&DataType>) -> Result<(Tree, Typed), TypeError> {
    type_tree(
        tree,
        &Columns,
        DocumentDialect::Canonical,
        &CanonicalTypeSystem,
        context,
    )
}

fn typed(tree: Tree, context: Option<&DataType>) -> (Tree, Typed) {
    typed_in(tree, context).unwrap_or_else(|error| panic!("typing refused it: {error}"))
}

fn type_of(tree: Tree) -> DataType {
    typed(tree, None).1.data_type
}

fn tree_in(tree: Tree, context: Option<&DataType>) -> Tree {
    typed(tree, context).0
}

fn tree_of(tree: Tree) -> Tree {
    tree_in(tree, None)
}

fn refusal(tree: Tree) -> TypeError {
    match typed_in(tree, None) {
        Ok(typed) => panic!("typing took it: {typed:?}"),
        Err(error) => error,
    }
}

#[test]
fn qty_plus_units_and_units_plus_qty_are_bigint() {
    assert_eq!(type_of(col("qty") + col("units")), DataType::BigInt);
    assert_eq!(type_of(col("units") + col("qty")), DataType::BigInt);
}

#[test]
fn an_operation_with_a_decimal_operand_takes_the_type_system_s_decimal_rule() {
    let cases = [
        (col("qty") + col("price"), decimal(19, 2)),
        (col("units") * col("price"), decimal(38, 2)),
        (col("price") + col("rate"), decimal(21, 4)),
        (col("price") * col("rate"), decimal(29, 6)),
        (col("price") * col("price"), decimal(37, 4)),
        (col("price") / col("rate"), decimal(38, 12)),
        (col("units") + col("price"), decimal(22, 2)),
    ];
    for (tree, expected) in cases {
        assert_eq!(type_of(tree.clone()), expected, "{tree:?}");
    }
}

#[test]
fn a_remainder_takes_the_decimal_remainder_and_a_negation_its_operand_s_type() {
    assert_eq!(type_of(col("price") % col("rate")), decimal(10, 4));
    assert_eq!(type_of(-col("price")), decimal(18, 2));
}

#[test]
fn score_times_price_has_no_type_because_neither_widens_to_the_other() {
    assert_eq!(type_of(col("score") * col("price")), DataType::Unknown);
}

#[test]
fn units_greater_than_price_is_boolean_because_both_are_numbers() {
    assert_eq!(type_of(col("units").gt(col("price"))), DataType::Boolean);
}

#[test]
fn day_plus_an_interval_of_any_interval_family_is_a_date() {
    let intervals = [
        Literal::Interval("P1D".into()),
        Literal::IntervalYearMonth("P1M".into()),
        Literal::IntervalDaySecond {
            value: "P1D".into(),
            precision: 6,
        },
    ];
    for interval in intervals {
        let tree = col("day") + Expr::Literal(interval);
        assert_eq!(type_of(tree.clone()), DataType::Date, "{tree:?}");
    }
}

#[test]
fn day_minus_day_is_refused_because_no_form_of_subtract_takes_two_dates() {
    assert_eq!(
        refusal(col("day") - col("day")),
        TypeError::Arguments {
            name: "subtract",
            types: vec![DataType::Date.into(), DataType::Date.into()],
        }
    );
}

#[test]
fn an_untyped_literal_beside_a_date_or_an_interval_keeps_its_parse_default_and_is_refused() {
    let interval = || tagged(Literal::Interval("P1D".into()));
    let cases = [
        (
            col("day") + untyped(Literal::String("P1D".into())),
            vec![DataType::Date.into(), DataType::String.into()],
        ),
        (
            untyped(Literal::String("2026-05-19".into())) + interval(),
            vec![DataType::String.into(), DataType::Interval.into()],
        ),
    ];
    for (tree, types) in cases {
        assert_eq!(refusal(tree), TypeError::Arguments { name: "add", types });
    }
}

#[test]
fn region_equal_to_a_number_is_refused_because_they_are_two_families() {
    assert_eq!(
        refusal(ExprOperationExt::eq(
            col("region"),
            tagged(Literal::BigInt(5))
        )),
        TypeError::Arguments {
            name: "eq",
            types: vec![DataType::String.into(), DataType::BigInt.into()],
        }
    );
}

#[test]
fn a_year_month_interval_is_never_compared_with_a_day_time_interval() {
    let month = tagged(Literal::IntervalYearMonth("P1M".into()));
    let days = tagged(Literal::IntervalDaySecond {
        value: "P30D".into(),
        precision: 6,
    });
    assert_eq!(
        refusal(month.lt(days)),
        TypeError::Arguments {
            name: "lt",
            types: vec![
                DataType::IntervalYearMonth.into(),
                DataType::IntervalDaySecond { precision: 6 }.into(),
            ],
        }
    );
}

#[test]
fn an_operand_outside_its_parameter_s_family_is_refused() {
    assert_eq!(
        refusal(!col("qty")),
        TypeError::Argument {
            name: "not",
            parameter: "operand",
            takes: &ParameterType::Family(&TypeFamily::Boolean),
            got: DataType::Integer.into(),
        }
    );
}

#[test]
fn an_untyped_literal_takes_the_other_operand_s_type() {
    let cases = [
        (
            col("amount").gt(untyped(Literal::Double(1.5))),
            col("amount").gt(tagged(decimal_literal("1.5", 18, 2))),
        ),
        (
            col("qty") + untyped(Literal::BigInt(1)),
            col("qty") + tagged(Literal::Integer(1)),
        ),
        (
            col("amount").gt(untyped(Literal::String("5".into()))),
            col("amount").gt(tagged(decimal_literal("5", 18, 2))),
        ),
    ];
    for (authored, typed) in cases {
        assert_eq!(tree_of(authored), typed);
    }
}

#[test]
fn an_untyped_literal_that_its_context_s_type_does_not_hold_is_refused() {
    let cases = [
        (
            col("amount").gt(untyped(Literal::Double(1.125))),
            Literal::Double(1.125),
            decimal(18, 2),
            LiteralError::CutsADigit,
        ),
        (
            ExprOperationExt::eq(col("region"), untyped(Literal::BigInt(5))),
            Literal::BigInt(5),
            DataType::String,
            LiteralError::NoReading,
        ),
    ];
    for (tree, literal, context, error) in cases {
        assert_eq!(
            refusal(tree),
            TypeError::UntypedLiteral {
                literal,
                context,
                error
            }
        );
    }
}

#[test]
fn a_tagged_literal_keeps_its_type_and_is_checked_by_its_tag_s_family() {
    assert_eq!(
        refusal(col("amount").gt(tagged(Literal::String("5".into())))),
        TypeError::Arguments {
            name: "gt",
            types: vec![decimal(18, 2).into(), DataType::String.into()],
        }
    );
}

#[test]
fn where_no_operand_has_a_type_the_node_s_context_passes_down() {
    let integer = Some(&DataType::Integer);
    assert_eq!(
        tree_in(
            untyped(Literal::BigInt(1)) + untyped(Literal::BigInt(2)),
            integer
        ),
        tagged(Literal::Integer(1)) + tagged(Literal::Integer(2))
    );
    assert_eq!(
        tree_of(col("amount").gt(untyped(Literal::BigInt(1)) + untyped(Literal::BigInt(2)))),
        col("amount").gt(tagged(decimal_literal("1", 18, 2)) + tagged(decimal_literal("2", 18, 2)))
    );
    assert_eq!(
        tree_in(
            ExprOperationExt::eq(untyped(Literal::BigInt(1)), untyped(Literal::BigInt(2))),
            Some(&DataType::Boolean)
        ),
        ExprOperationExt::eq(tagged(Literal::BigInt(1)), tagged(Literal::BigInt(2)))
    );
}

#[test]
fn an_unknown_operand_gives_no_context_so_the_literal_keeps_its_parse_default() {
    let integer = Some(&DataType::Integer);
    let authored = col("mystery") + untyped(Literal::BigInt(1));
    assert_eq!(
        tree_in(authored, integer),
        col("mystery") + tagged(Literal::BigInt(1))
    );
}

#[test]
fn a_double_literal_in_a_decimal_context_reads_through_its_shortest_exact_text() {
    assert_eq!(
        tree_of(col("rate").gt(untyped(Literal::Double(0.1)))),
        col("rate").gt(tagged(decimal_literal("0.1", 10, 4)))
    );
}

#[test]
fn a_root_untyped_literal_takes_the_context_s_type_and_a_tagged_one_keeps_its_own() {
    let typed = [
        (
            untyped(Literal::Double(1.5)),
            decimal(18, 2),
            decimal_literal("1.5", 18, 2),
        ),
        (
            untyped(Literal::String("2026-05-19".into())),
            DataType::Date,
            Literal::Date("2026-05-19".into()),
        ),
        (
            tagged(Literal::String("2026-05-19".into())),
            DataType::Date,
            Literal::String("2026-05-19".into()),
        ),
        (
            untyped(Literal::Null(DataType::String)),
            DataType::BigInt,
            Literal::Null(DataType::BigInt),
        ),
    ];
    for (authored, context, literal) in typed {
        assert_eq!(tree_in(authored, Some(&context)), tagged(literal));
    }
    let refused = [
        (
            Literal::Double(1.5),
            DataType::Integer,
            LiteralError::CutsADigit,
        ),
        (
            Literal::String("2026-05-19T10:00:00+02:00".into()),
            DataType::Timestamp { precision: 6 },
            LiteralError::Zoned,
        ),
    ];
    for (literal, context, error) in refused {
        assert_eq!(
            typed_in(untyped(literal.clone()), Some(&context)),
            Err(TypeError::UntypedLiteral {
                literal,
                context,
                error
            })
        );
    }
}

#[test]
fn a_leaf_takes_the_caller_s_type_which_an_untyped_literal_beside_it_takes() {
    let dimension = |name| {
        let reference = NamedReference::new([name]).expect("one part");
        Expr::<SemanticLeaf>::Leaf(SemanticLeaf::Dimension(reference, None))
    };
    let authored =
        dimension("qty") + Expr::Leaf(SemanticLeaf::Untyped(UntypedLiteral(Literal::BigInt(1))));
    let typed = type_tree(
        authored,
        &Columns,
        DocumentDialect::Canonical,
        &CanonicalTypeSystem,
        None,
    );
    let expected = Typed::new(DataType::Integer, false);
    assert_eq!(
        typed,
        Ok((
            dimension("qty") + Expr::Literal(Literal::Integer(1)),
            expected
        ))
    );
}

fn case(whens: Vec<(Tree, Tree)>, else_: Option<Tree>) -> Tree {
    Expr::Case {
        whens,
        else_: else_.map(Box::new),
    }
}

fn more_than_one(name: &str) -> Tree {
    col(name).gt(untyped(Literal::BigInt(1)))
}

#[test]
fn a_case_whose_then_types_differ_is_refused() {
    let tree = case(vec![(more_than_one("qty"), col("qty"))], Some(col("units")));
    assert_eq!(
        refusal(tree),
        TypeError::CaseBranches {
            first: DataType::Integer.into(),
            other: DataType::BigInt.into(),
        }
    );
}

#[test]
fn a_case_whose_when_is_not_a_boolean_is_refused() {
    let tree = case(vec![(col("qty"), col("price"))], None);
    assert_eq!(
        refusal(tree),
        TypeError::CaseWhen {
            got: DataType::Integer.into(),
        }
    );
}

#[test]
fn a_case_takes_its_branches_one_type_and_an_untyped_branch_takes_it_too() {
    let authored = case(
        vec![(more_than_one("qty"), col("price"))],
        Some(untyped(Literal::Null(DataType::String))),
    );
    let typed = case(
        vec![(col("qty").gt(tagged(Literal::Integer(1))), col("price"))],
        Some(tagged(Literal::Null(decimal(18, 2)))),
    );
    assert_eq!(tree_of(authored.clone()), typed);
    assert_eq!(type_of(authored), decimal(18, 2));
}

#[test]
fn a_case_of_untyped_branches_takes_the_context_of_the_operand_it_fills() {
    let branches = |one, two| case(vec![(more_than_one("units"), one)], Some(two));
    let authored = col("qty") + branches(untyped(Literal::BigInt(1)), untyped(Literal::BigInt(2)));
    assert_eq!(
        tree_of(authored),
        col("qty")
            + case(
                vec![(
                    col("units").gt(tagged(Literal::BigInt(1))),
                    tagged(Literal::Integer(1)),
                )],
                Some(tagged(Literal::Integer(2))),
            )
    );
}

fn cast(input: Tree, target: DataType) -> Tree {
    Expr::Cast {
        input: Box::new(input),
        target,
        on_failure: CastFailure::Error,
    }
}

#[test]
fn a_cast_that_the_cast_table_refuses_is_refused() {
    assert_eq!(
        refusal(cast(tagged(Literal::Boolean(true)), DataType::Date)),
        TypeError::Cast {
            from: DataType::Boolean.into(),
            to: DataType::Date,
        }
    );
}

#[test]
fn a_cast_takes_its_target_type_and_a_cast_from_unknown_is_not_checked() {
    assert_eq!(type_of(cast(col("region"), DataType::Date)), DataType::Date);
    assert_eq!(
        type_of(cast(col("mystery"), DataType::Date)),
        DataType::Date
    );
}

#[test]
fn between_and_in_take_operands_of_one_family_and_an_untyped_one_takes_their_type() {
    let between = |low, high| Expr::Between {
        value: Box::new(col("qty")),
        low: Box::new(low),
        high: Box::new(high),
        negated: false,
    };
    let in_list = |list| Expr::InList {
        value: Box::new(col("qty")),
        list,
        negated: true,
    };
    let one = || untyped(Literal::BigInt(1));
    let cases = [
        (
            between(one(), col("units")),
            between(tagged(Literal::Integer(1)), col("units")),
        ),
        (
            in_list(vec![one(), col("units")]),
            in_list(vec![tagged(Literal::Integer(1)), col("units")]),
        ),
    ];
    for (authored, typed) in cases {
        assert_eq!(type_of(authored.clone()), DataType::Boolean);
        assert_eq!(tree_of(authored), typed);
    }
    assert_eq!(
        refusal(in_list(vec![col("region")])),
        TypeError::Arguments {
            name: "in",
            types: vec![DataType::Integer.into(), DataType::String.into()],
        }
    );
}

#[test]
fn like_takes_two_strings_whose_untyped_pattern_keeps_its_parse_default() {
    let like = |value, pattern| Expr::Like {
        value: Box::new(value),
        pattern: Box::new(pattern),
        kind: LikeKind::ILike,
    };
    let pattern = || Literal::String("%a%".into());
    let authored = like(col("region"), untyped(pattern()));
    assert_eq!(type_of(authored.clone()), DataType::Boolean);
    assert_eq!(tree_of(authored), like(col("region"), tagged(pattern())));
    assert_eq!(
        refusal(like(col("qty"), untyped(pattern()))),
        TypeError::Argument {
            name: "ilike",
            parameter: "value",
            takes: &ParameterType::Family(&TypeFamily::String),
            got: DataType::Integer.into(),
        }
    );
}

#[test]
fn is_null_takes_an_operand_of_any_type() {
    assert_eq!(type_of(col("day").is_null()), DataType::Boolean);
}

#[test]
fn is_not_distinct_from_is_a_boolean_over_two_operands_of_one_family() {
    assert_eq!(
        type_of(col("qty").is_not_distinct_from(col("units"))),
        DataType::Boolean
    );
    assert_eq!(
        refusal(col("region").is_not_distinct_from(col("units"))),
        TypeError::Arguments {
            name: "is_not_distinct_from",
            types: vec![DataType::String.into(), DataType::BigInt.into()],
        }
    );
}

fn nullable(tree: Tree) -> bool {
    typed(tree, None).1.nullable
}

#[test]
fn an_operation_s_result_can_be_null_by_its_declaration_s_rule() {
    let flag = |name| col(name).gt(tagged(Literal::Integer(1)));
    let cases = [
        (col("qty") + col("units"), false),
        (col("qty") + col("bonus"), true),
        (flag("qty") & flag("bonus"), true),
        (flag("qty") | flag("units"), false),
        (col("bonus").is_null(), false),
        (col("bonus").is_not_distinct_from(col("qty")), false),
        (ExprOperationExt::eq(col("bonus"), col("qty")), true),
        (col("qty") + tagged(Literal::Null(DataType::Integer)), true),
    ];
    for (tree, expected) in cases {
        assert_eq!(nullable(tree.clone()), expected, "{tree:?}");
    }
}

#[test]
fn a_case_can_be_null_by_its_branches_or_where_it_has_no_else() {
    let when = || col("qty").gt(tagged(Literal::Integer(1)));
    let cases = [
        (case(vec![(when(), col("qty"))], Some(col("qty"))), false),
        (case(vec![(when(), col("qty"))], None), true),
        (case(vec![(when(), col("qty"))], Some(col("bonus"))), true),
    ];
    for (tree, expected) in cases {
        assert_eq!(nullable(tree.clone()), expected, "{tree:?}");
    }
}

#[test]
fn a_cast_can_be_null_by_its_input_or_where_its_failure_gives_null() {
    let try_cast = Expr::Cast {
        input: Box::new(col("region")),
        target: DataType::Date,
        on_failure: CastFailure::Null,
    };
    assert!(!nullable(cast(col("region"), DataType::Date)));
    assert!(nullable(cast(col("bonus"), DataType::BigInt)));
    assert!(nullable(try_cast));
}

#[test]
fn typing_reads_each_operation_s_declaration_in_the_document_s_dialect() {
    let divide = || Expr::BinaryOp {
        op: BinaryOpKind::SafeDivide,
        left: Box::new(col("qty")),
        right: Box::new(col("units")),
    };
    let nullable_in =
        |dialect| match type_tree(divide(), &Columns, dialect, &CanonicalTypeSystem, None) {
            Ok((_, typed)) => typed.nullable,
            Err(error) => panic!("typing refused it: {error}"),
        };
    assert!(nullable_in(DocumentDialect::Canonical));
    assert!(!nullable_in(DocumentDialect::Dialect(Dialect::DataFusion)));
}

fn call(name: &str, args: Vec<Tree>) -> Tree {
    Expr::Function(Call::new(function(name), args).unwrap_or_else(|error| panic!("{error}")))
}

fn type_by(tree: Tree, type_system: &dyn TypeSystem) -> DataType {
    match type_tree(
        tree,
        &Columns,
        DocumentDialect::Canonical,
        type_system,
        None,
    ) {
        Ok((_, typed)) => typed.data_type,
        Err(error) => panic!("typing refused it: {error}"),
    }
}

#[test]
fn a_call_with_a_decimal_argument_takes_the_type_system_s_rule() {
    let cases = [
        (
            call("greatest", vec![col("price"), col("rate")]),
            decimal(20, 4),
        ),
        (call("sum", vec![col("price")]), decimal(38, 2)),
        (call("avg", vec![col("qty")]), DataType::Double),
        (call("avg", vec![col("price")]), decimal(38, 6)),
        (call("ceil", vec![col("price")]), decimal(17, 0)),
        (
            call("round", vec![col("price"), untyped(Literal::BigInt(0))]),
            decimal(19, 2),
        ),
    ];
    for (tree, expected) in cases {
        assert_eq!(type_of(tree.clone()), expected, "{tree:?}");
    }
}

#[test]
fn a_call_whose_result_is_t_is_unknown_where_an_argument_at_t_is_unknown() {
    let one = || untyped(Literal::BigInt(1));
    let coalesce = || call("coalesce", vec![col("x"), one()]);
    let lag_of_x = |offset, default| lag(vec![col("x"), offset, default], Vec::new());
    let cases = [
        coalesce(),
        call("coalesce", vec![col("x"), col("qty")]),
        call("coalesce", vec![col("qty"), col("x")]),
        lag_of_x(one(), untyped(Literal::BigInt(0))),
        case(vec![(col("qty").gt(one()), col("x"))], Some(col("qty"))),
    ];
    for tree in cases {
        assert_eq!(type_of(tree.clone()), DataType::Unknown, "{tree:?}");
    }
    assert_eq!(
        tree_of(coalesce()),
        call("coalesce", vec![col("x"), tagged(Literal::BigInt(1))])
    );
    assert_eq!(
        tree_of(lag_of_x(one(), untyped(Literal::BigInt(0)))),
        lag_of_x(tagged(Literal::Integer(1)), tagged(Literal::BigInt(0)))
    );
}

#[test]
fn an_untyped_literal_at_a_call_s_parameter_takes_the_context_that_binds_it() {
    let measure = Some(&decimal(18, 2));
    let cases = [
        (
            call("coalesce", vec![col("qty"), untyped(Literal::BigInt(0))]),
            None,
            call("coalesce", vec![col("qty"), tagged(Literal::Integer(0))]),
        ),
        (
            call("sum", vec![untyped(Literal::BigInt(1))]),
            measure,
            call("sum", vec![tagged(Literal::BigInt(1))]),
        ),
        (
            call("sum", vec![col("amount") * untyped(Literal::Double(1.5))]),
            measure,
            call(
                "sum",
                vec![col("amount") * tagged(decimal_literal("1.5", 18, 2))],
            ),
        ),
    ];
    for (authored, context, typed) in cases {
        assert_eq!(tree_in(authored, context), typed);
    }
}

#[test]
fn a_call_of_untyped_arguments_whose_result_is_theirs_takes_the_context_of_the_operand_it_fills() {
    let one_and_two = || vec![untyped(Literal::BigInt(1)), untyped(Literal::BigInt(2))];
    let typed = |one, two| vec![tagged(one), tagged(two)];
    assert_eq!(
        tree_of(col("qty") + call("coalesce", one_and_two())),
        col("qty") + call("coalesce", typed(Literal::Integer(1), Literal::Integer(2)))
    );
    assert_eq!(
        tree_of(col("amount").gt(call("greatest", one_and_two()))),
        col("amount").gt(call(
            "greatest",
            typed(decimal_literal("1", 18, 2), decimal_literal("2", 18, 2))
        ))
    );
    let text = || untyped(Literal::String("a".into()));
    assert_eq!(
        tree_of(col("qty") + call("length", vec![text()])),
        col("qty") + call("length", vec![tagged(Literal::String("a".into()))])
    );
}

#[test]
fn a_call_whose_arguments_do_not_fit_its_signature_is_refused() {
    let cases = [
        (
            call("upper", vec![col("amount")]),
            TypeError::Argument {
                name: "upper",
                parameter: "value",
                takes: &ParameterType::Family(&TypeFamily::String),
                got: decimal(18, 2).into(),
            },
        ),
        (
            call(
                "extract",
                vec![untyped(Literal::String("hour".into())), col("day")],
            ),
            TypeError::EnumRefused {
                name: "extract",
                parameter: "component",
                value: "hour".into(),
                got: DataType::Date.into(),
            },
        ),
        (
            call("coalesce", vec![col("qty"), col("region")]),
            TypeError::TypeVariable {
                name: "coalesce",
                bound: DataType::Integer.into(),
                got: DataType::String.into(),
            },
        ),
    ];
    for (tree, error) in cases {
        assert_eq!(refusal(tree), error);
    }
}

#[test]
fn a_parameter_that_an_earlier_argument_s_type_conditions_is_written_only_where_it_holds() {
    let text = |value: &str| tagged(Literal::String(value.into()));
    let timestamp = || cast(col("day"), DataType::timestamp());
    let date_trunc = |args| call("date_trunc", args);
    let in_stockholm = date_trunc(vec![text("day"), timestamp(), text("Europe/Stockholm")]);
    assert_eq!(tree_of(in_stockholm.clone()), in_stockholm);
    assert_eq!(
        typed(date_trunc(vec![text("day"), timestamp()]), None),
        (
            date_trunc(vec![text("day"), timestamp(), text("UTC")]),
            Typed::new(DataType::timestamp(), false)
        )
    );
    let on_a_date = date_trunc(vec![text("week"), col("day"), text("monday")]);
    assert_eq!(
        typed(on_a_date.clone(), None),
        (on_a_date, Typed::new(DataType::Date, false))
    );
    assert_eq!(
        refusal(date_trunc(vec![text("day"), col("day"), text("UTC")])),
        TypeError::ParameterRefused {
            name: "date_trunc",
            parameter: "timezone"
        }
    );
}

fn concat(options: Vec<OptionSetting>) -> Tree {
    let FunctionRef::Scalar(concat) = function("concat") else {
        panic!("`concat` is a scalar function of the catalog");
    };
    let args = vec![col("name"), col("note")];
    let call = ScalarCall::builder(concat, args).options(options).build();
    Expr::Function(Call::Scalar(call.unwrap_or_else(|error| panic!("{error}"))))
}

#[test]
fn a_call_s_result_can_be_null_by_its_declaration_s_rule() {
    let ignore_nulls = OptionSetting {
        name: "null_handling".into(),
        value: "IGNORE_NULLS".into(),
    };
    let cases = [
        (call("upper", vec![col("name")]), false),
        (call("upper", vec![col("note")]), true),
        (call("concat_ws", vec![col("name"), col("note")]), false),
        (call("concat_ws", vec![col("note"), col("name")]), true),
        (call("coalesce", vec![col("bonus"), col("qty")]), false),
        (call("coalesce", vec![col("bonus"), col("bonus")]), true),
        (call("nullif", vec![col("qty"), col("qty")]), true),
        (call("count", vec![col("note")]), false),
        (concat(Vec::new()), true),
        (concat(vec![ignore_nulls]), false),
        (call("sum", vec![col("qty")]), true),
    ];
    for (tree, expected) in cases {
        assert_eq!(nullable(tree.clone()), expected, "{tree:?}");
    }
}

fn aggregate(name: &str, args: Vec<Tree>, filter: Option<Tree>, order_by: Vec<Tree>) -> Tree {
    let FunctionRef::Aggregate(function, aggregation) = function(name) else {
        panic!("`{name}` is no aggregate function of the catalog");
    };
    let order_by = order_by.into_iter().map(ascending).collect();
    let call = AggregateCall::builder(function, args)
        .maybe_filter(filter)
        .order_by(order_by)
        .build();
    Expr::Function(Call::Aggregate(
        call.unwrap_or_else(|error| panic!("{error}")),
        aggregation,
    ))
}

fn lag(args: Vec<Tree>, partition_by: Vec<Tree>) -> Tree {
    let lag = window_function("lag");
    let over = Over {
        partition_by,
        order_by: vec![ascending(col("day"))],
        window_frame: None,
    };
    let call = WindowCall::builder(lag, args).over(over).build();
    Expr::Function(Call::Window(call.unwrap_or_else(|error| panic!("{error}"))))
}

#[test]
fn a_window_call_s_arguments_bind_as_a_scalar_call_s_and_its_over_is_typed() {
    let region = || Literal::String("north".into());
    let authored = lag(
        vec![
            col("price"),
            untyped(Literal::BigInt(1)),
            untyped(Literal::BigInt(0)),
        ],
        vec![untyped(region())],
    );
    let typed = lag(
        vec![
            col("price"),
            tagged(Literal::Integer(1)),
            tagged(decimal_literal("0", 18, 2)),
        ],
        vec![tagged(region())],
    );
    assert_eq!(type_of(authored.clone()), decimal(18, 2));
    assert_eq!(tree_of(authored), typed);
}

#[test]
fn an_aggregate_call_s_filter_is_a_boolean_and_its_within_group_key_has_its_declared_type() {
    let more_than = |name| col(name).gt(untyped(Literal::BigInt(1)));
    let typed_more_than = |name| col(name).gt(tagged(Literal::Integer(1)));
    let sum = |filter| aggregate("sum", vec![col("price")], Some(filter), Vec::new());
    assert_eq!(tree_of(sum(more_than("qty"))), sum(typed_more_than("qty")));
    assert_eq!(
        refusal(sum(col("qty"))),
        TypeError::Clause {
            name: "sum",
            clause: CallClause::Filter,
            takes: &ParameterType::Family(&TypeFamily::Boolean),
            got: DataType::Integer.into(),
        }
    );
    let fraction = || tagged(Literal::Double(0.5));
    let percentile = |key| aggregate("percentile_cont", vec![fraction()], None, vec![key]);
    assert_eq!(type_of(percentile(col("price"))), DataType::Double);
    assert_eq!(
        refusal(percentile(col("region"))),
        TypeError::Clause {
            name: "percentile_cont",
            clause: CallClause::WithinGroup,
            takes: &ParameterType::Family(&TypeFamily::Number),
            got: DataType::String.into(),
        }
    );
}

fn first_value(key: Tree, window_frame: WindowFrame) -> Tree {
    let first_value = window_function("first_value");
    let over = Over {
        partition_by: Vec::new(),
        order_by: vec![ascending(key)],
        window_frame: Some(window_frame),
    };
    let args = vec![col("price")];
    let call = WindowCall::builder(first_value, args).over(over).build();
    Expr::Function(Call::Window(call.unwrap_or_else(|error| panic!("{error}"))))
}

#[test]
fn a_range_window_frame_with_an_offset_takes_no_string_or_boolean_order_key() {
    use WindowBound::{CurrentRow, Preceding, UnboundedPreceding};
    let range_offset = WindowFrame::new(WindowFrameKind::Range, Preceding(5), CurrentRow);
    for (key, family) in [
        (col("region"), TypeFamily::String),
        (col("qty").gt(col("units")), TypeFamily::Boolean),
    ] {
        assert_eq!(
            refusal(first_value(key, range_offset)),
            TypeError::RangeOrderKey {
                name: "first_value",
                family,
            }
        );
    }
    for (key, window_frame) in [
        (col("qty"), range_offset),
        (col("mystery"), range_offset),
        (
            col("region"),
            WindowFrame::new(WindowFrameKind::Range, UnboundedPreceding, CurrentRow),
        ),
        (
            col("region"),
            WindowFrame::new(WindowFrameKind::Rows, Preceding(5), CurrentRow),
        ),
    ] {
        assert_eq!(type_of(first_value(key, window_frame)), decimal(18, 2));
    }
}

#[test]
fn typing_reads_each_call_s_declaration_in_the_document_s_dialect() {
    let in_dialect = |tree, dialect| {
        type_tree(tree, &Columns, dialect, &CanonicalTypeSystem, None).map(|(_, typed)| typed)
    };
    let datafusion = DocumentDialect::Dialect(Dialect::DataFusion);
    let greatest = || call("greatest", vec![col("qty"), col("bonus")]);
    let nullable_in = |dialect| in_dialect(greatest(), dialect).map(|typed| typed.nullable);
    assert_eq!(nullable_in(DocumentDialect::Canonical), Ok(true));
    assert_eq!(nullable_in(datafusion), Ok(false));
    let sign = call("sign", vec![col("price")]);
    let integer = Typed::new(DataType::Integer, false);
    assert_eq!(in_dialect(sign, datafusion), Ok(integer));
    let extract = call(
        "regexp_extract",
        vec![
            col("name"),
            tagged(Literal::String("(a)".into())),
            tagged(Literal::Integer(1)),
        ],
    );
    assert_eq!(
        in_dialect(extract, datafusion),
        Err(TypeError::NotInDialect {
            name: "regexp_extract",
            dialect: Dialect::DataFusion,
        })
    );
    let week = |value: &str| tagged(Literal::String(value.into()));
    let date_trunc = call("date_trunc", vec![week("week"), col("day"), week("monday")]);
    assert_eq!(
        in_dialect(date_trunc, datafusion),
        Err(TypeError::Call(CallError::TooManyArguments {
            function: "date_trunc",
            got: 3,
        }))
    );
}

#[test]
fn a_call_s_type_is_its_return_rule_s_by_the_canonical_and_by_substrait_s_type_system() {
    let cases = [
        ("avg", "qty", DataType::Double, DataType::Integer),
        ("median", "qty", DataType::Double, DataType::Integer),
        ("avg", "ratio", DataType::Double, DataType::Real),
        ("median", "ratio", DataType::Double, DataType::Real),
        ("avg", "score", DataType::Double, DataType::Double),
        ("median", "score", DataType::Double, DataType::Double),
        ("avg", "price", decimal(38, 6), decimal(38, 2)),
        ("upper", "name", DataType::String, DataType::String),
    ]
    .map(|(function, column, canonical, substrait)| {
        (call(function, vec![col(column)]), canonical, substrait)
    });
    for (tree, canonical, substrait) in cases {
        assert_eq!(
            type_by(tree.clone(), &CanonicalTypeSystem),
            canonical,
            "{tree:?}"
        );
        assert_eq!(
            type_by(tree.clone(), &SubstraitTypeSystem),
            substrait,
            "{tree:?}"
        );
    }
}

#[test]
fn promote_gives_the_bigger_type_and_the_decimal_union_with_a_decimal() {
    let cases = [
        ([DataType::Integer, DataType::BigInt], DataType::BigInt),
        ([decimal(18, 2), decimal(10, 4)], decimal(20, 4)),
        ([DataType::BigInt, decimal(18, 2)], decimal(21, 2)),
        ([DataType::Date, DataType::Interval], DataType::Unknown),
    ];
    for (types, promoted) in cases {
        assert_eq!(promote(&SubstraitTypeSystem, &types), promoted, "{types:?}");
    }
}

fn score_times_price() -> Tree {
    col("score") * col("price")
}

#[test]
fn a_result_with_no_type_is_refused_where_its_operands_family_is() {
    let upper = |value| call("upper", vec![value]);
    let abc = || tagged(Literal::String("abc".into()));
    for (value, got) in [
        (col("score"), TypeOrFamily::from(DataType::Double)),
        (score_times_price(), TypeFamily::Number.into()),
    ] {
        assert_eq!(
            refusal(upper(value.clone())),
            TypeError::Argument {
                name: "upper",
                parameter: "value",
                takes: &ParameterType::Family(&TypeFamily::String),
                got: got.clone(),
            }
        );
        assert_eq!(
            refusal(ExprOperationExt::eq(value, abc())),
            TypeError::Arguments {
                name: "eq",
                types: vec![got, DataType::String.into()],
            }
        );
    }
}

#[test]
fn each_result_with_no_type_keeps_its_arguments_one_family() {
    let cases = [
        call("greatest", vec![col("score"), col("price")]),
        call("least", vec![col("score"), col("price")]),
        score_times_price() + col("qty"),
        -score_times_price(),
        call("abs", vec![score_times_price()]),
        call(
            "round",
            vec![score_times_price(), tagged(Literal::Integer(2))],
        ),
        call("coalesce", vec![score_times_price(), col("score")]),
        aggregate("sum", vec![score_times_price()], None, Vec::new()),
        case(
            vec![(more_than_one("qty"), score_times_price())],
            Some(col("score")),
        ),
    ];
    for tree in cases {
        assert_eq!(type_of(tree.clone()), DataType::Unknown, "{tree:?}");
        assert!(
            matches!(
                refusal(call("upper", vec![tree.clone()])),
                TypeError::Argument {
                    name: "upper",
                    got: TypeOrFamily::Family(TypeFamily::Number),
                    ..
                }
            ),
            "{tree:?}"
        );
    }
}

#[test]
fn a_date_or_a_time_with_no_type_keeps_its_family_beside_an_interval() {
    let timestamp = |precision| cast(col("day"), DataType::Timestamp { precision });
    let day_or_timestamp = || call("greatest", vec![col("day"), timestamp(6)]);
    let one_day = || tagged(Literal::Interval("P1D".into()));
    let cases = [
        day_or_timestamp() + one_day(),
        one_day() + day_or_timestamp(),
        day_or_timestamp() - one_day(),
        call("greatest", vec![timestamp(3), timestamp(6)]) + one_day(),
    ];
    for tree in cases {
        assert_eq!(type_of(tree.clone()), DataType::Unknown, "{tree:?}");
        assert!(
            matches!(
                refusal(call("upper", vec![tree.clone()])),
                TypeError::Argument {
                    name: "upper",
                    got: TypeOrFamily::Family(TypeFamily::DateAndTimestamp),
                    ..
                }
            ),
            "{tree:?}"
        );
        assert!(
            matches!(
                refusal(tree.clone().gt(tagged(Literal::Integer(5)))),
                TypeError::Arguments { name: "gt", .. }
            ),
            "{tree:?}"
        );
    }
}

#[test]
fn a_result_with_no_type_fits_a_parameter_of_its_family_and_casts_to_a_number() {
    let cases = [
        (
            aggregate("sum", vec![score_times_price()], None, Vec::new()),
            DataType::Unknown,
        ),
        (call("sqrt", vec![score_times_price()]), DataType::Double),
        (score_times_price().gt(col("qty")), DataType::Boolean),
        (
            cast(score_times_price(), DataType::Double),
            DataType::Double,
        ),
    ];
    for (tree, expected) in cases {
        assert_eq!(type_of(tree.clone()), expected, "{tree:?}");
    }
}

#[test]
fn an_authored_cast_of_a_result_with_no_type_is_checked_by_its_family() {
    let error = refusal(cast(score_times_price(), DataType::Date));
    assert_eq!(
        error,
        TypeError::Cast {
            from: TypeFamily::Number.into(),
            to: DataType::Date,
        }
    );
    assert!(error.to_string().contains("number"), "{error}");
    for target in [DataType::String, DataType::BigInt] {
        assert_eq!(type_of(cast(score_times_price(), target.clone())), target);
    }
    let mystery_times_two = col("mystery") * untyped(Literal::BigInt(2));
    assert_eq!(
        type_of(cast(mystery_times_two, DataType::Date)),
        DataType::Date
    );
}

#[test]
fn a_when_or_a_filter_with_no_type_is_refused_by_its_family() {
    assert_eq!(
        refusal(case(vec![(score_times_price(), col("qty"))], None)),
        TypeError::CaseWhen {
            got: TypeFamily::Number.into(),
        }
    );
    assert_eq!(
        refusal(aggregate(
            "sum",
            vec![col("qty")],
            Some(score_times_price()),
            Vec::new()
        )),
        TypeError::Clause {
            name: "sum",
            clause: CallClause::Filter,
            takes: &ParameterType::Family(&TypeFamily::Boolean),
            got: TypeFamily::Number.into(),
        }
    );
}

#[test]
fn a_condition_is_a_boolean_or_has_no_type_of_the_boolean_family_or_of_none() {
    for (tree, condition) in [
        (more_than_one("qty"), true),
        (col("mystery"), true),
        (score_times_price(), false),
        (col("score") + tagged(Literal::Integer(1)), false),
    ] {
        let (_, got) = typed(tree.clone(), Some(&DataType::Boolean));
        assert_eq!(
            is_condition(&got.data_type, got.family.as_ref()),
            condition,
            "{tree:?}"
        );
    }
    assert!(is_condition(&DataType::Unknown, Some(&TypeFamily::Boolean)));
}

#[test]
fn a_refusal_of_a_node_with_no_type_names_its_family() {
    let refused = [
        case(vec![(score_times_price(), col("qty"))], None),
        aggregate(
            "sum",
            vec![col("qty")],
            Some(score_times_price()),
            Vec::new(),
        ),
        case(
            vec![(more_than_one("qty"), score_times_price())],
            Some(col("region")),
        ),
        call("coalesce", vec![score_times_price(), col("region")]),
        call("upper", vec![score_times_price()]),
        ExprOperationExt::eq(score_times_price(), tagged(Literal::String("a".into()))),
        cast(score_times_price(), DataType::Date),
    ];
    for tree in refused {
        let message = refusal(tree.clone()).to_string();
        assert!(message.contains("number"), "{tree:?}: {message}");
        assert!(!message.contains("unknown"), "{tree:?}: {message}");
    }
}

#[test]
fn a_result_over_a_column_that_compile_did_not_observe_has_no_family() {
    let tree = call("upper", vec![col("mystery") * untyped(Literal::BigInt(2))]);
    assert_eq!(type_of(tree), DataType::String);
}

#[test]
fn a_branch_or_an_argument_at_t_with_no_type_is_refused_beside_another_family() {
    assert_eq!(
        refusal(case(
            vec![(more_than_one("qty"), score_times_price())],
            Some(col("region"))
        )),
        TypeError::CaseBranches {
            first: TypeFamily::Number.into(),
            other: DataType::String.into(),
        }
    );
    assert_eq!(
        refusal(call("coalesce", vec![score_times_price(), col("region")])),
        TypeError::TypeVariable {
            name: "coalesce",
            bound: TypeFamily::Number.into(),
            got: DataType::String.into(),
        }
    );
}
