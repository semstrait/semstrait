mod support;

use std::collections::BTreeSet;
use std::error::Error;

use semstrait_ir::{AggregateNode, PhysicalExpr, PlanNode, ProjectNode, ScanNode};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{BinaryOpKind, Expr, FunctionRef, LikeKind, Literal, UnaryOpKind};
use semstrait_sql::functions::{registry, Category};
use substrait::proto::expression::literal::LiteralType;
use substrait::proto::expression::{RexType, ScalarFunction};
use substrait::proto::function_argument::ArgType;
use substrait::proto::{AggregateFunction, Expression, FunctionOption, Plan};

use support::{
    aggregate, aggregate_rel, call_text, codes, declared_function, leaf, lines, literal, moments,
    name, orders, produce, projected, root_input, scalar, text, Faithful,
};

const ARITHMETIC: &str = "extension:io.substrait:functions_arithmetic";
const ARITHMETIC_DECIMAL: &str = "extension:io.substrait:functions_arithmetic_decimal";
const COMPARISON: &str = "extension:io.substrait:functions_comparison";
const DATETIME: &str = "extension:io.substrait:functions_datetime";
const LOGARITHMIC: &str = "extension:io.substrait:functions_logarithmic";
const ROUNDING: &str = "extension:io.substrait:functions_rounding";
const ROUNDING_DECIMAL: &str = "extension:io.substrait:functions_rounding_decimal";
const STRING: &str = "extension:io.substrait:functions_string";
const SEMSTRAIT: &str = "extension:semstrait:semstrait";

fn projected_calls(plan: &Plan) -> Vec<&ScalarFunction> {
    (projected(plan).iter())
        .map(|expr| match &expr.rex_type {
            Some(RexType::ScalarFunction(function)) => function,
            other => panic!("a call, not {other:?}"),
        })
        .collect()
}

fn projected_texts(plan: &Plan) -> Vec<String> {
    (projected(plan).iter())
        .map(|expr| text(plan, expr))
        .collect()
}

fn functions(plan: &Plan, anchors: impl IntoIterator<Item = u32>) -> Vec<(&str, &str)> {
    (anchors.into_iter())
        .map(|anchor| declared_function(plan, anchor).expect("a declared function"))
        .collect()
}

fn projected_functions(plan: &Plan) -> Vec<(&str, &str)> {
    let calls = projected_calls(plan).into_iter();
    functions(plan, calls.map(|call| call.function_reference))
}

fn measures(plan: &Plan) -> Vec<&AggregateFunction> {
    (aggregate_rel(root_input(plan)).measures.iter())
        .map(|measure| measure.measure.as_ref().expect("a measure"))
        .collect()
}

fn measure_functions(plan: &Plan) -> Vec<(&str, &str)> {
    let calls = measures(plan).into_iter();
    functions(plan, calls.map(|call| call.function_reference))
}

fn options_text(options: &[FunctionOption]) -> Vec<String> {
    (options.iter())
        .map(|option| format!("{}: {}", option.name, option.preference.join(", ")))
        .collect()
}

fn project_lines(exprs: Vec<PhysicalExpr>) -> Result<PlanNode, Box<dyn Error>> {
    project(lines()?, exprs)
}

fn project(input: ScanNode, exprs: Vec<PhysicalExpr>) -> Result<PlanNode, Box<dyn Error>> {
    let expressions = (exprs.into_iter().enumerate())
        .map(|(position, expr)| (name(&format!("value_{position}")), expr))
        .collect();
    Ok(ProjectNode::new(input, expressions)?.into())
}

fn aggregate_on<'a>(
    input: ScanNode,
    calls: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> Result<AggregateNode, Box<dyn Error>> {
    let entries = (calls.into_iter().enumerate())
        .map(|(position, (function, column))| {
            let call = aggregate(function, vec![leaf(&[column])], None);
            (name(&format!("value_{position}")), call)
        })
        .collect();
    Ok(AggregateNode::new(input, Vec::new(), entries)?)
}

fn binary(op: BinaryOpKind, left: &str, right: &str) -> PhysicalExpr {
    Expr::BinaryOp {
        op,
        left: Box::new(leaf(&[left])),
        right: Box::new(leaf(&[right])),
    }
}

#[test]
fn each_math_function_goes_under_its_rows_extension() -> Result<(), Box<dyn Error>> {
    let rows = [
        ("abs", 1, ARITHMETIC, "abs"),
        ("ceil", 1, ROUNDING, "ceil"),
        ("floor", 1, ROUNDING, "floor"),
        ("round", 1, ROUNDING, "round"),
        ("sqrt", 1, ARITHMETIC, "sqrt"),
        ("exp", 1, ARITHMETIC, "exp"),
        ("ln", 1, LOGARITHMIC, "ln"),
        ("log10", 1, LOGARITHMIC, "log10"),
        ("log2", 1, LOGARITHMIC, "log2"),
        ("cbrt", 1, SEMSTRAIT, "cbrt"),
        ("sin", 1, ARITHMETIC, "sin"),
        ("cos", 1, ARITHMETIC, "cos"),
        ("tan", 1, ARITHMETIC, "tan"),
        ("asin", 1, ARITHMETIC, "asin"),
        ("acos", 1, ARITHMETIC, "acos"),
        ("atan", 1, ARITHMETIC, "atan"),
        ("sinh", 1, ARITHMETIC, "sinh"),
        ("cosh", 1, ARITHMETIC, "cosh"),
        ("tanh", 1, ARITHMETIC, "tanh"),
        ("degrees", 1, ARITHMETIC, "degrees"),
        ("radians", 1, ARITHMETIC, "radians"),
        ("power", 2, ARITHMETIC, "power"),
        ("log", 2, LOGARITHMIC, "logb"),
        ("atan2", 2, ARITHMETIC, "atan2"),
        ("sign", 1, ARITHMETIC, "sign"),
        ("width_bucket", 4, SEMSTRAIT, "width_bucket"),
        ("pi", 0, SEMSTRAIT, "pi"),
        ("random", 0, SEMSTRAIT, "random"),
    ];
    let math: BTreeSet<_> = (registry().all())
        .filter_map(|function| match function {
            FunctionRef::Scalar(function)
                if function.canonical.specific.category == Category::Math =>
            {
                Some(function.canonical.name)
            }
            _ => None,
        })
        .collect();
    let named: BTreeSet<_> = rows.iter().map(|(canonical, ..)| *canonical).collect();
    assert_eq!(named, math, "a row for each math function");
    let arguments = |canonical: &str, arity: usize| {
        let mut arguments = vec![leaf(&["score"]); arity];
        if canonical == "width_bucket" {
            arguments[3] = leaf(&["qty"]);
        }
        arguments
    };
    let project = project_lines(
        (rows.iter())
            .map(|(canonical, arity, ..)| scalar(canonical, arguments(canonical, *arity)))
            .collect(),
    )?;

    let plan = produce(project, &Faithful)?;

    let expected: Vec<_> = rows.iter().map(|(_, _, urn, name)| (*urn, *name)).collect();
    assert_eq!(projected_functions(&plan), expected);
    Ok(())
}

#[test]
fn each_comparison_goes_under_functions_comparison() -> Result<(), Box<dyn Error>> {
    let rows = [
        (BinaryOpKind::Eq, "equal"),
        (BinaryOpKind::NotEq, "not_equal"),
        (BinaryOpKind::Lt, "lt"),
        (BinaryOpKind::LtEq, "lte"),
        (BinaryOpKind::Gt, "gt"),
        (BinaryOpKind::GtEq, "gte"),
        (BinaryOpKind::IsNotDistinctFrom, "is_not_distinct_from"),
    ];
    let project = project_lines(
        (rows.iter())
            .map(|(op, _)| binary(*op, "qty", "qty"))
            .collect(),
    )?;

    let plan = produce(project, &Faithful)?;

    let expected: Vec<_> = rows.iter().map(|(_, name)| (COMPARISON, *name)).collect();
    assert_eq!(projected_functions(&plan), expected);
    Ok(())
}

#[test]
fn each_argument_is_cast_to_the_type_of_substrait_s_overload() -> Result<(), Box<dyn Error>> {
    let between = Expr::Between {
        value: Box::new(leaf(&["qty"])),
        low: Box::new(leaf(&["units"])),
        high: Box::new(leaf(&["price"])),
        negated: false,
    };
    let cases = [
        (
            binary(BinaryOpKind::Add, "qty", "price"),
            "add(cast($0 as decimal(10, 0)), $2)",
        ),
        (
            binary(BinaryOpKind::Add, "qty", "units"),
            "add(cast($0 as i64), $1)",
        ),
        (
            binary(BinaryOpKind::Multiply, "price", "rate"),
            "multiply($2, $3)",
        ),
        (
            binary(BinaryOpKind::Gt, "price", "rate"),
            "gt(cast($2 as decimal(20, 4)), cast($3 as decimal(20, 4)))",
        ),
        (
            between,
            "and(gte(cast($0 as i64), $1), lte(cast($0 as decimal(18, 2)), $2))",
        ),
        (
            scalar("greatest", vec![leaf(&["qty"]), leaf(&["units"])]),
            "greatest(cast($0 as i64), $1)",
        ),
        (
            scalar("power", vec![leaf(&["units"]), leaf(&["units"])]),
            "power(cast($1 as fp64), cast($1 as fp64))",
        ),
        (scalar("sin", vec![leaf(&["qty"])]), "sin(cast($0 as fp64))"),
        (scalar("sinh", vec![leaf(&["score"])]), "sinh($4)"),
        (scalar("cbrt", vec![leaf(&["qty"])]), "cbrt($0)"),
    ];
    let (exprs, expected): (Vec<_>, Vec<_>) = cases.into_iter().unzip();
    let project = project_lines(exprs)?;

    let plan = produce(project, &Faithful)?;

    assert_eq!(projected_texts(&plan), expected);
    Ok(())
}

#[test]
fn an_average_casts_its_argument_to_its_canonical_result_type() -> Result<(), Box<dyn Error>> {
    let aggregates = [
        ("avg", "qty", "avg(cast($0 as fp64))"),
        ("avg", "price", "avg(cast($2 as decimal(38, 6)))"),
        ("avg", "score", "avg($4)"),
        ("median", "qty", "median(EXACT, cast($0 as fp64))"),
        ("stddev_samp", "units", "std_dev(cast($1 as fp64))"),
        ("sum", "qty", "sum($0)"),
    ];
    let calls = aggregates
        .iter()
        .map(|&(function, column, _)| (function, column));
    let node = aggregate_on(lines()?, calls)?;

    let plan = produce(node, &Faithful)?;

    let texts: Vec<_> = (measures(&plan).into_iter())
        .map(|call| call_text(&plan, call.function_reference, &call.arguments))
        .collect();
    let expected: Vec<_> = aggregates.iter().map(|(.., text)| *text).collect();
    assert_eq!(texts, expected);
    Ok(())
}

#[test]
fn a_call_of_more_arguments_than_its_row_places_goes_under_semstrait_s_urn(
) -> Result<(), Box<dyn Error>> {
    let (status, pattern) = (leaf(&["status"]), literal("p"));
    let project = project(
        orders()?,
        vec![
            scalar("regexp_instr", vec![status.clone(), pattern.clone()]),
            scalar(
                "regexp_instr",
                vec![status, pattern, literal(1_i32), literal(2_i32)],
            ),
        ],
    )?;

    let plan = produce(project, &Faithful)?;

    assert_eq!(
        projected_texts(&plan),
        [
            "regexp_strpos($2, 'p', 1, 1)",
            "regexp_instr($2, 'p', 1, 2)"
        ]
    );
    assert_eq!(
        projected_functions(&plan),
        [(STRING, "regexp_strpos"), (SEMSTRAIT, "regexp_instr")]
    );
    Ok(())
}

#[test]
fn a_call_goes_under_the_extension_that_holds_its_overload_for_its_type(
) -> Result<(), Box<dyn Error>> {
    let negate = Expr::UnaryOp {
        op: UnaryOpKind::Negate,
        operand: Box::new(leaf(&["price"])),
    };
    let project = project_lines(vec![
        scalar("abs", vec![leaf(&["price"])]),
        scalar("abs", vec![leaf(&["score"])]),
        scalar("ceil", vec![leaf(&["price"])]),
        scalar("floor", vec![leaf(&["price"])]),
        scalar("round", vec![leaf(&["price"]), literal(2_i32)]),
        negate,
        binary(BinaryOpKind::Mod, "score", "score"),
    ])?;
    let datetimes = aggregate_on(
        moments()?,
        [("min", "at"), ("max", "day"), ("min", "clock")],
    )?;
    let strings = aggregate_on(orders()?, [("min", "status")])?;

    let (project, datetimes, strings) = (
        produce(project, &Faithful)?,
        produce(datetimes, &Faithful)?,
        produce(strings, &Faithful)?,
    );

    assert_eq!(
        projected_functions(&project),
        [
            (ARITHMETIC_DECIMAL, "abs"),
            (ARITHMETIC, "abs"),
            (ROUNDING_DECIMAL, "ceil"),
            (ROUNDING_DECIMAL, "floor"),
            (ROUNDING_DECIMAL, "round"),
            (SEMSTRAIT, "negate"),
            (SEMSTRAIT, "mod"),
        ]
    );
    assert_eq!(
        measure_functions(&datetimes),
        [(DATETIME, "min"), (DATETIME, "max"), (SEMSTRAIT, "min")]
    );
    assert_eq!(measure_functions(&strings), [(SEMSTRAIT, "min")]);
    Ok(())
}

#[test]
fn a_time_with_an_interval_or_any_datetime_with_a_compound_interval_goes_under_semstrait_s_urn(
) -> Result<(), Box<dyn Error>> {
    let day_second = || {
        literal(Literal::IntervalDaySecond {
            value: "PT1H".into(),
            precision: 6,
        })
    };
    let year_month = || literal(Literal::IntervalYearMonth("P1M".into()));
    let compound = || literal(Literal::Interval("P1MT1H".into()));
    let null_day_second = literal(Literal::Null(DataType::IntervalDaySecond { precision: 6 }));
    let operation = |op, left: &str, right| Expr::BinaryOp {
        op,
        left: Box::new(leaf(&[left])),
        right: Box::new(right),
    };
    let rows = [
        (BinaryOpKind::Add, "clock", day_second(), SEMSTRAIT, "add"),
        (
            BinaryOpKind::Subtract,
            "clock",
            year_month(),
            SEMSTRAIT,
            "subtract",
        ),
        (BinaryOpKind::Add, "at", compound(), SEMSTRAIT, "add"),
        (
            BinaryOpKind::Subtract,
            "day",
            compound(),
            SEMSTRAIT,
            "subtract",
        ),
        (
            BinaryOpKind::Add,
            "clock",
            null_day_second,
            SEMSTRAIT,
            "add",
        ),
        (BinaryOpKind::Add, "at", day_second(), DATETIME, "add"),
        (BinaryOpKind::Add, "day", year_month(), DATETIME, "add"),
    ];
    let exprs = (rows.iter())
        .map(|(op, left, right, ..)| operation(*op, left, right.clone()))
        .collect();
    let project = project(moments()?, exprs)?;

    let plan = produce(project, &Faithful)?;

    let expected: Vec<_> = rows.iter().map(|(.., urn, name)| (*urn, *name)).collect();
    assert_eq!(projected_functions(&plan), expected);
    Ok(())
}

#[test]
fn a_round_with_no_scale_is_written_with_the_scale_0() -> Result<(), Box<dyn Error>> {
    let project = project_lines(vec![
        scalar("round", vec![leaf(&["score"])]),
        scalar("round", vec![leaf(&["price"]), literal(2_i32)]),
    ])?;

    let plan = produce(project, &Faithful)?;

    assert_eq!(projected_texts(&plan), ["round($4, 0)", "round($2, 2)"]);
    let scale = |round: &ScalarFunction| match round.arguments.get(1)?.arg_type.as_ref()? {
        ArgType::Value(Expression {
            rex_type: Some(RexType::Literal(scale)),
        }) => scale.literal_type.clone(),
        _ => None,
    };
    let scales: Vec<_> = projected_calls(&plan).into_iter().map(scale).collect();
    assert_eq!(
        scales,
        [Some(LiteralType::I32(0)), Some(LiteralType::I32(2))]
    );
    Ok(())
}

#[test]
fn a_trim_or_a_pad_with_no_characters_is_written_with_the_space() -> Result<(), Box<dyn Error>> {
    let column = |name| leaf(&[name]);
    let project = project(
        codes()?,
        vec![
            scalar("trim", vec![column("status")]),
            scalar("ltrim", vec![column("status")]),
            scalar("rtrim", vec![column("status")]),
            scalar("lpad", vec![column("status"), literal(5_i32)]),
            scalar("rpad", vec![column("status"), literal(5_i32)]),
            scalar("trim", vec![column("status"), literal("x")]),
            scalar("trim", vec![column("code")]),
            scalar("trim", vec![column("flag")]),
            scalar("lpad", vec![column("code"), literal(5_i32)]),
        ],
    )?;

    let plan = produce(project, &Faithful)?;

    assert_eq!(
        projected_texts(&plan),
        [
            "trim($0, ' ')",
            "ltrim($0, ' ')",
            "rtrim($0, ' ')",
            "lpad($0, 5, ' ')",
            "rpad($0, 5, ' ')",
            "trim($0, 'x')",
            "trim(cast($1 as string), ' ')",
            "trim(cast($2 as string), ' ')",
            "lpad(cast($1 as string), 5, ' ')",
        ]
    );
    Ok(())
}

#[test]
fn each_option_that_substrait_s_overload_declares_is_written_at_its_canonical_value(
) -> Result<(), Box<dyn Error>> {
    let ilike = Expr::Like {
        value: Box::new(literal("banana")),
        pattern: Box::new(literal("B%")),
        kind: LikeKind::ILike,
    };
    let calls = [
        (
            binary(BinaryOpKind::Add, "qty", "units"),
            &["overflow: ERROR"][..],
        ),
        (binary(BinaryOpKind::Add, "score", "score"), &[]),
        (
            binary(BinaryOpKind::Add, "price", "rate"),
            &["overflow: ERROR"],
        ),
        (
            binary(BinaryOpKind::Divide, "qty", "units"),
            &["overflow: ERROR", "on_division_by_zero: ERROR"],
        ),
        (
            binary(BinaryOpKind::SafeDivide, "score", "score"),
            &["on_division_by_zero: NULL"],
        ),
        (
            binary(BinaryOpKind::Divide, "price", "rate"),
            &["overflow: ERROR"],
        ),
        (
            binary(BinaryOpKind::Mod, "qty", "units"),
            &["overflow: ERROR", "on_domain_error: ERROR"],
        ),
        (
            binary(BinaryOpKind::Mod, "price", "rate"),
            &["overflow: ERROR"],
        ),
        (scalar("abs", vec![leaf(&["qty"])]), &["overflow: ERROR"]),
        (scalar("abs", vec![leaf(&["score"])]), &[]),
        (
            scalar("sqrt", vec![leaf(&["score"])]),
            &["on_domain_error: ERROR"],
        ),
        (scalar("ln", vec![leaf(&["score"])]), &[]),
        (
            scalar("position", vec![literal("a"), literal("banana")]),
            &["case_sensitivity: CASE_SENSITIVE"],
        ),
        (ilike, &["case_sensitivity: CASE_INSENSITIVE"]),
        (
            scalar("regexp_count", vec![literal("banana"), literal("a")]),
            &[
                "case_sensitivity: CASE_SENSITIVE",
                "multiline: MULTILINE_DISABLED",
                "dotall: DOTALL_DISABLED",
            ],
        ),
        (
            scalar("substring", vec![literal("banana"), literal(2_i32)]),
            &["negative_start: LEFT_OF_BEGINNING"],
        ),
        (
            scalar("round", vec![leaf(&["score"])]),
            &["rounding: TIE_AWAY_FROM_ZERO"],
        ),
    ];
    let aggregates = [
        ("sum", "qty", &["overflow: ERROR"][..]),
        ("avg", "score", &["overflow: ERROR"]),
        ("count", "qty", &["overflow: ERROR"]),
        ("min", "qty", &[]),
        ("stddev_samp", "score", &["distribution: SAMPLE"]),
        ("any_value", "qty", &["ignore_nulls: TRUE"]),
    ];
    let project = project_lines(calls.iter().map(|(expr, _)| expr.clone()).collect())?;
    let node = aggregate_on(
        lines()?,
        aggregates
            .iter()
            .map(|&(function, column, _)| (function, column)),
    )?;

    let (project, node) = (produce(project, &Faithful)?, produce(node, &Faithful)?);

    let written: Vec<_> = (projected_calls(&project).into_iter())
        .map(|call| options_text(&call.options))
        .collect();
    let expected: Vec<_> = calls.iter().map(|(_, options)| options.to_vec()).collect();
    assert_eq!(written, expected);
    let written: Vec<_> = (measures(&node).into_iter())
        .map(|call| options_text(&call.options))
        .collect();
    let expected: Vec<_> = aggregates
        .iter()
        .map(|(.., options)| options.to_vec())
        .collect();
    assert_eq!(written, expected);
    Ok(())
}
