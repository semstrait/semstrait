use std::error::Error;

use semstrait_common::diagnostic::Diagnose;
use semstrait_common::Dialect;
use semstrait_sql::datatype::{DataType, TypeFamily};
use semstrait_sql::expr::{
    AggregateInRowExpr, CallClause, CallError, JsonFormError, Literal, LiteralError, OverError,
    RebuildError, TypeError,
};
use semstrait_sql::functions::{Category, ParameterType, SealFault};
use semstrait_sql::DocumentDialect;

static STRING_FAMILY: ParameterType = ParameterType::Family(&TypeFamily::String);
static BOOLEAN_FAMILY: ParameterType = ParameterType::Family(&TypeFamily::Boolean);

fn assert_messages<E: Error>(cases: impl IntoIterator<Item = (E, &'static str)>) {
    for (error, message) in cases {
        assert_eq!(error.to_string(), message, "{error:?}");
    }
}

#[test]
fn each_call_error_names_the_function_and_what_the_call_does_not_fit() {
    assert_messages([
        (
            CallError::TooFewArguments {
                function: "greatest",
                min: 1,
                max: None,
            },
            "`greatest` takes one argument or more",
        ),
        (
            CallError::TooFewArguments {
                function: "coalesce",
                min: 2,
                max: None,
            },
            "`coalesce` takes two arguments or more",
        ),
        (
            CallError::TooFewArguments {
                function: "upper",
                min: 1,
                max: Some(1),
            },
            "`upper` takes one argument",
        ),
        (
            CallError::TooFewArguments {
                function: "substring",
                min: 2,
                max: Some(3),
            },
            "`substring` takes two to three arguments",
        ),
        (
            CallError::MissingArgument {
                function: "date_trunc",
                parameter: "week_start",
            },
            "`date_trunc` has no argument for `week_start`",
        ),
        (
            CallError::TooManyArguments {
                function: "upper",
                got: 2,
            },
            "`upper` does not take 2 arguments",
        ),
        (
            CallError::NotALiteral {
                function: "round",
                parameter: "scale",
            },
            "`round`'s `scale` takes a literal",
        ),
        (
            CallError::EnumValue {
                function: "extract",
                parameter: "component",
                value: Literal::String("yeer".to_owned()),
            },
            "`extract`'s `component` has no value 'yeer'",
        ),
        (
            CallError::EnumValue {
                function: "extract",
                parameter: "component",
                value: Literal::BigInt(5),
            },
            "`extract`'s `component` has no bigint value",
        ),
        (
            CallError::ClauseRefused {
                function: "upper",
                clause: CallClause::Distinct,
            },
            "`upper` takes no DISTINCT clause",
        ),
        (
            CallError::ClauseRequired {
                function: "lag",
                clause: CallClause::OrderBy,
            },
            "`lag` requires its ORDER BY clause",
        ),
        (
            CallError::UnknownOption {
                function: "concat",
                option: "mode".to_owned(),
            },
            "`concat` has no option `mode`",
        ),
        (
            CallError::OptionValue {
                function: "concat",
                option: "null_handling",
                value: "SOME".to_owned(),
            },
            "`concat`'s option `null_handling` has no value `SOME`",
        ),
        (
            CallError::RepeatedOption {
                function: "concat",
                option: "null_handling".to_owned(),
            },
            "`concat` sets its option `null_handling` more than once",
        ),
        (
            CallError::Over {
                function: "first_value",
                error: OverError::EndsBeforeStart,
            },
            "`first_value`'s OVER is refused: a window frame cannot end before it starts",
        ),
    ]);
    let refusal = CallError::Over {
        function: "first_value",
        error: OverError::GroupsWithoutOrderBy,
    };
    let source = refusal.source().map(ToString::to_string);
    assert_eq!(source, Some(OverError::GroupsWithoutOrderBy.to_string()));
}

#[test]
fn each_named_argument_refusal_names_the_function_and_the_parameter() {
    assert_messages([
        (
            CallError::UnknownParameter {
                function: "nullif",
                name: "bogus".to_owned(),
            },
            "`nullif` has no parameter `bogus`",
        ),
        (
            CallError::MissingParameter {
                function: "date_trunc",
                parameter: "week_start",
            },
            "`date_trunc` writes `week_start`, and no argument names it",
        ),
        (
            CallError::UnwrittenParameter {
                function: "date_trunc",
                parameter: "week_start",
            },
            "`date_trunc` does not write `week_start`, and an argument names it",
        ),
    ]);
}

#[test]
fn each_nesting_refusal_names_both_calls_and_the_nesting_that_it_refuses() {
    assert_messages([
        (
            CallError::AggregateInAggregate {
                function: "count",
                nested: "sum",
            },
            "`count` holds the aggregate call `sum`, and an aggregate call holds no aggregate call",
        ),
        (
            CallError::WindowInAggregate {
                function: "sum",
                nested: "first_value",
            },
            "`sum` holds the window call `first_value`, and an aggregate call holds no window call",
        ),
        (
            CallError::WindowInWindow {
                function: "first_value",
                nested: "lag",
            },
            "`first_value` holds the window call `lag`, and a window call holds no window call",
        ),
    ]);
}

#[test]
fn each_over_error_says_which_rule_the_window_frame_breaks() {
    assert_messages([
        (
            OverError::StartsUnboundedFollowing,
            "a window frame cannot start at UNBOUNDED FOLLOWING",
        ),
        (
            OverError::EndsUnboundedPreceding,
            "a window frame cannot end at UNBOUNDED PRECEDING",
        ),
        (
            OverError::EndsBeforeStart,
            "a window frame cannot end before it starts",
        ),
        (
            OverError::RangeOffsetKeys { keys: 2 },
            "a RANGE window frame with an offset takes one ORDER BY key, not 2",
        ),
        (
            OverError::GroupsWithoutOrderBy,
            "a GROUPS window frame requires an ORDER BY",
        ),
    ]);
}

#[test]
fn each_json_form_error_names_the_call_and_the_dialect() {
    assert_messages([
        (
            JsonFormError::UnknownFunction {
                name: "nope".to_owned(),
                dialect: DocumentDialect::Canonical,
            },
            "no function is named `nope` in Canonical",
        ),
        (
            JsonFormError::NotItsFunctionType {
                name: "sum".to_owned(),
                tag: "scalar",
            },
            "`sum` is no scalar function",
        ),
        (
            JsonFormError::NoNameInDialect {
                function: "date_diff",
                dialect: DocumentDialect::Dialect(Dialect::DataFusion),
            },
            "`date_diff` has no function in Dialect(DataFusion)",
        ),
    ]);
}

#[test]
fn each_literal_error_says_why_the_value_does_not_fit() {
    assert_messages([
        (
            LiteralError::InvalidType,
            "the literal's type is not a valid data type",
        ),
        (LiteralError::Malformed, "the value is not text of its type"),
        (
            LiteralError::TooManyDigits,
            "the value has more digits than the greatest decimal precision",
        ),
        (
            LiteralError::CutsADigit,
            "its type would cut a digit of the value",
        ),
        (
            LiteralError::OutOfRange,
            "the value is outside the range of its type",
        ),
        (
            LiteralError::TooLong,
            "the value is longer than its type's length",
        ),
        (LiteralError::Zoned, "a timestamp holds no zone"),
        (
            LiteralError::ElementType,
            "an array value's type is not the array's element",
        ),
        (
            LiteralError::NoReading,
            "the value has no reading as the type",
        ),
    ]);
}

#[test]
fn each_rebuild_error_says_why_no_node_is_built_and_a_refused_call_is_its_source() {
    let refusal = CallError::TooManyArguments {
        function: "upper",
        got: 2,
    };
    assert_messages([
        (RebuildError::Leaf, "a leaf has no operands to replace"),
        (
            RebuildError::ChildCount,
            "the node's new operands are not one per child",
        ),
        (
            RebuildError::Call(refusal.clone()),
            "the call's constructor refuses it: `upper` does not take 2 arguments",
        ),
    ]);
    let source = RebuildError::Call(refusal.clone());
    let source = source.source().map(ToString::to_string);
    assert_eq!(source, Some(refusal.to_string()));
    assert!(RebuildError::Leaf.source().is_none());
}

#[test]
fn an_aggregate_in_a_row_level_tree_says_so() {
    assert_messages([(
        AggregateInRowExpr,
        "a row-level tree holds no aggregate call",
    )]);
}

#[test]
fn each_seal_fault_is_an_error_whose_message_is_its_diagnostic_s() {
    let cases = [
        (
            SealFault::SharedName { name: "upper" },
            "two functions share the name `upper`",
        ),
        (
            SealFault::SharedDialectName {
                dialect: Dialect::DataFusion,
                name: "signum",
            },
            "two functions share the name `signum` in DataFusion",
        ),
        (
            SealFault::OperationName { name: "mod" },
            "the function name `mod` is an operation's name",
        ),
        (
            SealFault::CategoryNotItsFile {
                function: "upper",
                category: Category::Math,
                file: Category::String,
            },
            "`upper`: the category Math is not its file's, String",
        ),
        (
            SealFault::OptionValueOutsideOption {
                function: "concat",
                option: "null_handling",
                value: "SOME",
            },
            "`concat`: the option `null_handling` has no value `SOME`",
        ),
        (
            SealFault::CanonicalValueNotAmongValues {
                function: "concat",
                option: "null_handling",
            },
            "`concat`: the canonical value of `null_handling` is not among its values",
        ),
        (
            SealFault::OptionalBeforeRequired {
                function: "round",
                parameter: "value",
            },
            "`round`: the required `value` follows an optional parameter",
        ),
        (
            SealFault::OptionalWithNoDefault {
                function: "regexp_instr",
                parameter: "start",
            },
            "`regexp_instr`: the optional `start`, which a later parameter follows, declares no default",
        ),
        (
            SealFault::DefaultNotOfItsType {
                function: "regexp_instr",
                parameter: "start",
            },
            "`regexp_instr`: the default of `start` is not a literal of its type",
        ),
        (
            SealFault::ConditionNamesLaterArgument {
                function: "date_trunc",
                parameter: "timezone",
            },
            "`date_trunc`: the condition of `timezone` names a later argument",
        ),
        (
            SealFault::ArgumentIndexOutOfRange {
                function: "abs",
                index: 3,
            },
            "`abs`: the argument index 3 is out of range",
        ),
        (
            SealFault::UnboundTypeVariable { function: "abs" },
            "`abs`: no parameter binds the type variable `T`",
        ),
        (
            SealFault::AggregateNullability { function: "upper" },
            "`upper`: a scalar function has an aggregate's nullability",
        ),
        (
            SealFault::TwoDialectDeclarations {
                function: "sign",
                dialect: Dialect::DataFusion,
            },
            "`sign`: two declarations for DataFusion",
        ),
        (
            SealFault::DialectAliases {
                function: "sign",
                dialect: Dialect::DataFusion,
            },
            "`sign`: the DataFusion declaration has aliases",
        ),
        (
            SealFault::DialectEqualsCanonical {
                function: "upper",
                dialect: Dialect::DataFusion,
            },
            "`upper`: the DataFusion declaration equals the canonical one",
        ),
    ];
    for (fault, message) in &cases {
        assert_eq!(fault.message(), *message, "{fault:?}");
    }
    assert_messages(cases);
}

#[test]
fn each_type_error_names_the_operation_or_the_function_and_the_types() {
    assert_messages([
        (
            TypeError::Argument {
                name: "upper",
                parameter: "value",
                takes: &STRING_FAMILY,
                got: DataType::BigInt.into(),
            },
            "`upper`'s `value` takes the string family, not bigint",
        ),
        (
            TypeError::Arguments {
                name: "add",
                types: vec![DataType::Date.into(), DataType::Date.into()],
            },
            "`add` does not take these arguments: date, date",
        ),
        (
            TypeError::TypeVariable {
                name: "coalesce",
                bound: DataType::BigInt.into(),
                got: DataType::String.into(),
            },
            "`coalesce` takes its arguments at one type, but one is bigint and another is string",
        ),
        (
            TypeError::EnumRefused {
                name: "extract",
                parameter: "component",
                value: "hour".to_owned(),
                got: DataType::Date.into(),
            },
            "`extract`'s `component` takes no 'hour' beside date",
        ),
        (
            TypeError::ParameterRefused {
                name: "date_trunc",
                parameter: "timezone",
            },
            "`date_trunc` takes no `timezone` where an earlier argument's type does not call for it",
        ),
        (
            TypeError::ParameterRequired {
                name: "date_trunc",
                parameter: "timezone",
            },
            "`date_trunc` requires its `timezone` where an earlier argument's type calls for it",
        ),
        (
            TypeError::Clause {
                name: "sum",
                clause: CallClause::Filter,
                takes: &BOOLEAN_FAMILY,
                got: DataType::BigInt.into(),
            },
            "`sum`'s FILTER takes the boolean family, not bigint",
        ),
        (
            TypeError::RangeOrderKey {
                name: "first_value",
                family: TypeFamily::String,
            },
            "`first_value`'s RANGE window frame with an offset takes no ORDER BY key of the string family",
        ),
        (
            TypeError::UntypedLiteral {
                literal: Literal::Double(1.5),
                context: DataType::Integer,
                error: LiteralError::CutsADigit,
            },
            "the literal Double(1.5) does not read as integer: its type would cut a digit of the value",
        ),
        (
            TypeError::CaseWhen {
                got: DataType::BigInt.into(),
            },
            "a CASE's WHEN is bigint, not a boolean",
        ),
        (
            TypeError::CaseBranches {
                first: DataType::BigInt.into(),
                other: DataType::String.into(),
            },
            "a CASE takes its branches at one type, but one is bigint and another is string",
        ),
        (
            TypeError::Cast {
                from: DataType::Date.into(),
                to: DataType::Boolean,
            },
            "date does not cast to boolean",
        ),
        (
            TypeError::NotInDialect {
                name: "date_diff",
                dialect: Dialect::DataFusion,
            },
            "DataFusion has no `date_diff`",
        ),
        (
            TypeError::Call(CallError::TooManyArguments {
                function: "upper",
                got: 2,
            }),
            "in the document's dialect, `upper` does not take 2 arguments",
        ),
    ]);
}

#[test]
fn each_type_error_names_a_node_with_no_type_by_its_family() {
    let number = || TypeFamily::Number.into();
    assert_messages([
        (
            TypeError::Argument {
                name: "upper",
                parameter: "value",
                takes: &STRING_FAMILY,
                got: number(),
            },
            "`upper`'s `value` takes the string family, not a value of the number family with \
             no type",
        ),
        (
            TypeError::Arguments {
                name: "eq",
                types: vec![number(), DataType::String.into()],
            },
            "`eq` does not take these arguments: a value of the number family with no type, \
             string",
        ),
        (
            TypeError::TypeVariable {
                name: "coalesce",
                bound: number(),
                got: DataType::String.into(),
            },
            "`coalesce` takes its arguments at one type, but one is a value of the number \
             family with no type and another is string",
        ),
        (
            TypeError::EnumRefused {
                name: "extract",
                parameter: "component",
                value: "hour".to_owned(),
                got: TypeFamily::DateAndTimestamp.into(),
            },
            "`extract`'s `component` takes no 'hour' beside a value of the date and timestamp \
             family with no type",
        ),
        (
            TypeError::Clause {
                name: "sum",
                clause: CallClause::Filter,
                takes: &BOOLEAN_FAMILY,
                got: number(),
            },
            "`sum`'s FILTER takes the boolean family, not a value of the number family with no \
             type",
        ),
        (
            TypeError::CaseWhen { got: number() },
            "a CASE's WHEN is a value of the number family with no type, not a boolean",
        ),
        (
            TypeError::CaseBranches {
                first: number(),
                other: DataType::String.into(),
            },
            "a CASE takes its branches at one type, but one is a value of the number family \
             with no type and another is string",
        ),
        (
            TypeError::Cast {
                from: number(),
                to: DataType::Date,
            },
            "a value of the number family with no type does not cast to date",
        ),
    ]);
}
