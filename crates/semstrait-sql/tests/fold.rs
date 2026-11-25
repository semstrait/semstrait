use std::borrow::Cow;

use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{
    BinaryOpKind, Call, CastFailure, Expr, ExprOperationExt, FunctionRef, Leaf, LikeKind, Literal,
    NamedReference,
};

mod common;

#[derive(Debug, Clone, PartialEq)]
struct Ref(&'static str);

impl Leaf for Ref {}

type Tree = Expr<Ref>;

fn literal_of(leaf: &Ref) -> Option<Literal> {
    match leaf.0 {
        "dataset_name" => Some(Literal::String("google_ads".to_owned())),
        "paid" => Some(Literal::Boolean(false)),
        "nothing" => Some(Literal::Null(DataType::String)),
        "two" => Some(Literal::Integer(2)),
        "yes" => Some(Literal::Boolean(true)),
        _ => None,
    }
}

fn link(name: &'static str) -> Tree {
    Expr::Leaf(Ref(name))
}

fn lit(literal: Literal) -> Tree {
    Expr::Literal(literal)
}

fn boolean(value: bool) -> Tree {
    lit(Literal::Boolean(value))
}

fn integer(value: i32) -> Tree {
    lit(Literal::Integer(value))
}

fn string(value: &str) -> Tree {
    lit(Literal::String(value.to_owned()))
}

fn null(data_type: DataType) -> Tree {
    lit(Literal::Null(data_type))
}

fn column(name: &str) -> Tree {
    Expr::Column(NamedReference::new([name]).expect("a name path of one part"))
}

fn between(value: Tree, low: Tree, high: Tree, negated: bool) -> Tree {
    Expr::Between {
        value: Box::new(value),
        low: Box::new(low),
        high: Box::new(high),
        negated,
    }
}

fn in_list(value: Tree, list: Vec<Tree>, negated: bool) -> Tree {
    Expr::InList {
        value: Box::new(value),
        list,
        negated,
    }
}

fn like(value: Tree, pattern: Tree) -> Tree {
    Expr::Like {
        value: Box::new(value),
        pattern: Box::new(pattern),
        kind: LikeKind::Like,
    }
}

fn case(whens: Vec<(Tree, Tree)>, else_: Option<Tree>) -> Tree {
    Expr::Case {
        whens,
        else_: else_.map(Box::new),
    }
}

fn cast(input: Tree, target: DataType) -> Tree {
    Expr::Cast {
        input: Box::new(input),
        target,
        on_failure: CastFailure::Error,
    }
}

fn call(name: &str, args: Vec<Tree>) -> Tree {
    let function = FunctionRef::Scalar(common::scalar(name));
    Expr::Function(Call::new(function, args).expect("the call fits its declaration"))
}

fn assert_folds(cases: impl IntoIterator<Item = (Tree, Tree)>) {
    for (tree, value) in cases {
        assert_eq!(tree.fold(), value);
    }
}

fn assert_stays(trees: impl IntoIterator<Item = Tree>) {
    for tree in trees {
        assert_eq!(tree.clone().fold(), tree);
    }
}

fn assert_folds_with_links(cases: impl IntoIterator<Item = (Tree, Tree)>) {
    for (tree, value) in cases {
        assert_eq!(tree.fold_with(&|l| literal_of(l).map(Cow::Owned)), value);
    }
}

#[test]
fn false_decides_an_and_and_true_an_or_over_a_column() {
    assert_folds([
        (column("x").eq(string("a")) & boolean(false), boolean(false)),
        (column("x").eq(string("a")) | boolean(true), boolean(true)),
    ]);
}

#[test]
fn the_operands_that_are_not_null_decide_a_kleene_operation() {
    assert_folds([
        (boolean(false) & null(DataType::Boolean), boolean(false)),
        (boolean(true) | null(DataType::Boolean), boolean(true)),
        (
            between(integer(5), null(DataType::Integer), integer(3), false),
            boolean(false),
        ),
    ]);
}

#[test]
fn a_null_operand_decides_an_operation_by_its_null_rule() {
    let cast = Expr::Cast {
        input: Box::new(null(DataType::Integer)),
        target: DataType::BigInt,
        on_failure: CastFailure::Error,
    };
    assert_folds([
        (
            column("x").eq(null(DataType::String)),
            null(DataType::Boolean),
        ),
        (
            integer(1) + null(DataType::Integer),
            null(DataType::Integer),
        ),
        (-null(DataType::Integer), null(DataType::Integer)),
        (!null(DataType::Boolean), null(DataType::Boolean)),
        (
            like(column("x"), null(DataType::String)),
            null(DataType::Boolean),
        ),
        (cast, null(DataType::BigInt)),
        (null(DataType::Integer).is_null(), boolean(true)),
    ]);
}

#[test]
fn an_operation_over_literals_gives_its_value() {
    assert_folds([
        (!boolean(true), boolean(false)),
        (-integer(5), integer(-5)),
        (integer(5).is_null(), boolean(false)),
        (
            in_list(integer(5), vec![integer(1), null(DataType::Integer)], false),
            null(DataType::Boolean),
        ),
        (
            in_list(integer(5), vec![integer(5), null(DataType::Integer)], false),
            boolean(true),
        ),
        (
            in_list(integer(5), vec![integer(1), integer(2)], true),
            boolean(true),
        ),
        (
            between(integer(5), integer(1), integer(3), true),
            boolean(true),
        ),
    ]);
}

#[test]
fn a_cast_of_a_literal_gives_the_canonical_cast_s_value() {
    let decimal = |value: &str, precision, scale| {
        lit(Literal::Decimal {
            value: value.to_owned(),
            precision,
            scale,
        })
    };
    let to_decimal = |precision, scale| DataType::Decimal { precision, scale };
    assert_folds([
        (
            cast(lit(Literal::Double(1.7)), DataType::Integer),
            integer(1),
        ),
        (
            cast(lit(Literal::Double(-1.7)), DataType::Integer),
            integer(-1),
        ),
        (
            cast(string("2026-05-19"), DataType::Date),
            lit(Literal::Date("2026-05-19".to_owned())),
        ),
        (
            cast(string("42"), DataType::BigInt),
            lit(Literal::BigInt(42)),
        ),
        (cast(integer(7), DataType::BigInt), lit(Literal::BigInt(7))),
        (
            cast(decimal("2.345", 10, 3), to_decimal(10, 2)),
            decimal("2.35", 10, 2),
        ),
        (
            cast(decimal("-2.345", 10, 3), to_decimal(10, 2)),
            decimal("-2.35", 10, 2),
        ),
        (
            cast(decimal("0.9", 38, 38), to_decimal(38, 0)),
            decimal("1", 38, 0),
        ),
        (
            cast(decimal("-0.9", 38, 38), to_decimal(38, 0)),
            decimal("-1", 38, 0),
        ),
        (cast(decimal("2.7", 10, 1), DataType::Integer), integer(2)),
        (
            cast(lit(Literal::Double(1.5)), to_decimal(18, 2)),
            decimal("1.5", 18, 2),
        ),
        (
            cast(lit(Literal::Double(1.005)), to_decimal(10, 2)),
            decimal("1", 10, 2),
        ),
        (cast(integer(5), to_decimal(18, 2)), decimal("5", 18, 2)),
        (
            cast(integer(5), DataType::Double),
            lit(Literal::Double(5.0)),
        ),
        (
            cast(lit(Literal::Real(1.5)), DataType::Double),
            lit(Literal::Double(1.5)),
        ),
        (cast(boolean(true), DataType::Boolean), boolean(true)),
        (
            cast(lit(Literal::Date("2026-05-19".to_owned())), DataType::Date),
            lit(Literal::Date("2026-05-19".to_owned())),
        ),
    ]);
}

#[test]
fn a_cast_of_a_literal_that_the_canonical_cast_gives_no_value_for_stays() {
    assert_stays([
        cast(string("2026-13-45"), DataType::Date),
        cast(string("1.7"), DataType::Integer),
        cast(string("abc"), DataType::Integer),
        cast(lit(Literal::BigInt(i64::MAX)), DataType::Integer),
        cast(lit(Literal::Double(1e40)), DataType::BigInt),
        cast(
            lit(Literal::Double(123.456)),
            DataType::Decimal {
                precision: 4,
                scale: 2,
            },
        ),
    ]);
}

#[test]
fn a_cast_of_a_literal_to_a_string_stays() {
    assert_stays([
        cast(integer(5), DataType::String),
        cast(
            lit(Literal::Date("2026-05-19".to_owned())),
            DataType::String,
        ),
    ]);
}

#[test]
fn an_operation_that_its_literals_do_not_decide_stays() {
    assert_stays([
        boolean(true) & column("x").eq(string("a")),
        column("x").is_null(),
        in_list(
            column("x"),
            vec![integer(1), null(DataType::Integer)],
            false,
        ),
        like(string("Paid Search"), string("Paid%")),
        column("x") + null(DataType::Integer),
        -null(DataType::Double),
    ]);
}

#[test]
fn a_case_with_literal_conditions_gives_its_branch() {
    assert_folds([
        (
            case(
                vec![(boolean(false), string("a")), (boolean(true), column("x"))],
                Some(string("c")),
            ),
            column("x"),
        ),
        (
            case(
                vec![(integer(1).eq(integer(1)), string("a"))],
                Some(string("b")),
            ),
            string("a"),
        ),
        (
            case(
                vec![(null(DataType::Boolean), string("a"))],
                Some(string("b")),
            ),
            string("b"),
        ),
        (
            case(vec![(boolean(false), string("a"))], None),
            null(DataType::String),
        ),
    ]);
}

#[test]
fn a_case_that_its_literals_do_not_decide_stays() {
    assert_stays([
        case(
            vec![
                (boolean(false), string("a")),
                (column("x").eq(string("a")), string("b")),
                (boolean(true), string("c")),
            ],
            None,
        ),
        case(vec![(boolean(false), column("x"))], None),
    ]);
}

#[test]
fn integer_arithmetic_gives_the_bigger_operand_s_type() {
    assert_folds([
        (
            integer(1) + lit(Literal::BigInt(2)),
            lit(Literal::BigInt(3)),
        ),
        (lit(Literal::SmallInt(6)) * integer(7), integer(42)),
        (integer(7) / integer(2), integer(3)),
        (integer(-7) / integer(2), integer(-3)),
        (integer(-7) % integer(2), integer(-1)),
    ]);
}

#[test]
fn an_integer_sum_that_overflows_stays() {
    assert_stays([
        integer(i32::MAX) + integer(1),
        lit(Literal::SmallInt(i16::MAX)) + lit(Literal::SmallInt(1)),
        -integer(i32::MIN),
        integer(1) / integer(0),
    ]);
}

#[test]
fn a_safe_divide_with_a_null_operand_or_a_divisor_of_0_stays() {
    let safe_divide = |left: Tree, right: Tree| Expr::BinaryOp {
        op: BinaryOpKind::SafeDivide,
        left: Box::new(left),
        right: Box::new(right),
    };
    assert_folds([(safe_divide(integer(-7), integer(2)), integer(-3))]);
    assert_stays([
        safe_divide(integer(7), null(DataType::Integer)),
        safe_divide(null(DataType::Integer), integer(2)),
        safe_divide(integer(7), integer(0)),
    ]);
}

#[test]
fn double_arithmetic_stays() {
    assert_stays([
        lit(Literal::Double(0.1)) + lit(Literal::Double(0.2)),
        lit(Literal::Real(1.5)) * lit(Literal::Real(2.0)),
    ]);
}

#[test]
fn a_function_call_stays() {
    assert_stays([
        call("upper", vec![string("a")]),
        call("abs", vec![integer(-1)]),
        call("current_date", Vec::new()),
        call("random", Vec::new()),
    ]);
}

#[test]
fn a_call_s_arguments_fold() {
    let tree = call("abs", vec![integer(1) - integer(2)]);
    assert_eq!(tree.fold(), call("abs", vec![integer(-1)]));
}

#[test]
fn a_comparison_of_two_string_literals_gives_its_boolean() {
    let predicate = |channel: Tree| channel.eq(string("Paid Social"));
    assert_folds([
        (predicate(string("Paid Search")), boolean(false)),
        (predicate(string("Paid Social")), boolean(true)),
    ]);
}

#[test]
fn a_comparison_of_two_varchar_or_two_date_literals_gives_its_boolean() {
    let varchar = |value: &str| {
        lit(Literal::VarChar {
            value: value.to_owned(),
            length: 20,
        })
    };
    let date = |value: &str| lit(Literal::Date(value.to_owned()));
    assert_folds([
        (
            varchar("Paid Search").eq(varchar("Paid Social")),
            boolean(false),
        ),
        (
            varchar("Paid Social").eq(string("Paid Social")),
            boolean(true),
        ),
        (date("2024-01-31").lt(date("2024-02-01")), boolean(true)),
        (date("2024-01-31").eq(date("2024-02-01")), boolean(false)),
    ]);
}

#[test]
fn a_comparison_of_char_decimal_time_or_timestamp_literals_stays() {
    let char = |value: &str| {
        lit(Literal::FixedChar {
            value: value.to_owned(),
            length: 3,
        })
    };
    let decimal = |value: &str| {
        lit(Literal::Decimal {
            value: value.to_owned(),
            precision: 3,
            scale: 2,
        })
    };
    let timestamp = |value: &str| {
        lit(Literal::Timestamp {
            value: value.to_owned(),
            precision: 6,
        })
    };
    assert_stays([
        char("ab").eq(char("ab ")),
        decimal("1.5").eq(decimal("1.50")),
        timestamp("2024-01-31").eq(timestamp("2024-01-31T00:00:00")),
    ]);
}

#[test]
fn a_comparison_with_a_column_stays() {
    assert_stays([column("channel").eq(string("Paid Social"))]);
}

#[test]
fn a_comparison_with_a_function_call_stays() {
    assert_stays([call("upper", vec![string("Paid Search")]).eq(string("PAID SEARCH"))]);
}

#[test]
fn a_metadata_determined_comparison_gives_false() {
    assert_folds([(string("google_ads").eq(string("google")), boolean(false))]);
}

#[test]
fn an_integer_sum_gives_its_value() {
    assert_folds([(integer(1) + integer(2), integer(3))]);
}

#[test]
fn a_leaf_that_stands_for_a_literal_folds_as_that_literal() {
    assert_folds_with_links([
        (link("dataset_name").eq(string("google")), boolean(false)),
        (column("x").eq(string("a")) & link("paid"), boolean(false)),
        (link("nothing").is_null(), boolean(true)),
        (
            case(
                vec![(link("yes"), link("dataset_name"))],
                Some(string("other")),
            ),
            link("dataset_name"),
        ),
        (link("two") + integer(1), integer(3)),
        (cast(link("nothing"), DataType::Date), null(DataType::Date)),
        (cast(link("two"), DataType::BigInt), lit(Literal::BigInt(2))),
    ]);
}

#[test]
fn every_operand_reads_a_leaf_as_the_literal_it_stands_for() {
    assert_folds_with_links([
        (-link("two"), integer(-2)),
        (!link("paid"), boolean(true)),
        (
            between(link("two"), link("two"), link("two"), false),
            boolean(true),
        ),
        (
            in_list(
                link("dataset_name"),
                vec![string("a"), link("dataset_name")],
                false,
            ),
            boolean(true),
        ),
        (link("nothing").eq(string("x")), null(DataType::Boolean)),
        (like(link("nothing"), string("g%")), null(DataType::Boolean)),
        (like(string("g"), link("nothing")), null(DataType::Boolean)),
        (
            case(vec![(boolean(false), link("dataset_name"))], None),
            null(DataType::String),
        ),
    ]);
}

#[test]
fn a_leaf_under_a_part_the_fold_leaves_stays_a_leaf() {
    assert_folds_with_links(
        [
            call("upper", vec![link("dataset_name")]),
            like(link("dataset_name"), string("g%")),
            cast(link("dataset_name"), DataType::Date),
        ]
        .map(|tree| (tree.clone(), tree)),
    );
}

#[test]
fn fold_reads_no_leaf_as_a_literal() {
    assert_stays([link("dataset_name").eq(string("google"))]);
}
