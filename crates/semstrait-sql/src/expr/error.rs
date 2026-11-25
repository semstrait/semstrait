use semstrait_common::Dialect;
use thiserror::Error;

use super::{CallClause, Literal, LiteralError};
use crate::datatype::{DataType, TypeFamily, TypeOrFamily};
use crate::functions::ParameterType;
use crate::DocumentDialect;

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Error)]
pub enum CallError {
    #[error("`{function}` takes {}", arguments_in_words(*.min, *.max))]
    TooFewArguments {
        function: &'static str,
        min: usize,
        max: Option<usize>,
    },
    #[error("`{function}` has no argument for `{parameter}`")]
    MissingArgument {
        function: &'static str,
        parameter: &'static str,
    },
    #[error("`{function}` does not take {got} arguments")]
    TooManyArguments { function: &'static str, got: usize },
    #[error("`{function}`'s `{parameter}` takes a literal")]
    NotALiteral {
        function: &'static str,
        parameter: &'static str,
    },
    #[error("`{function}`'s `{parameter}` has no {}", enum_value(.value))]
    EnumValue {
        function: &'static str,
        parameter: &'static str,
        value: Literal,
    },
    #[error("`{function}` takes no {clause} clause")]
    ClauseRefused {
        function: &'static str,
        clause: CallClause,
    },
    #[error("`{function}` requires its {clause} clause")]
    ClauseRequired {
        function: &'static str,
        clause: CallClause,
    },
    #[error("`{function}` has no option `{option}`")]
    UnknownOption {
        function: &'static str,
        option: String,
    },
    #[error("`{function}`'s option `{option}` has no value `{value}`")]
    OptionValue {
        function: &'static str,
        option: &'static str,
        value: String,
    },
    #[error("`{function}` sets its option `{option}` more than once")]
    RepeatedOption {
        function: &'static str,
        option: String,
    },
    #[error("`{function}` has no parameter `{name}`")]
    UnknownParameter {
        function: &'static str,
        name: String,
    },
    #[error("`{function}` writes `{parameter}`, and no argument names it")]
    MissingParameter {
        function: &'static str,
        parameter: &'static str,
    },
    #[error("`{function}` does not write `{parameter}`, and an argument names it")]
    UnwrittenParameter {
        function: &'static str,
        parameter: &'static str,
    },
    #[error("`{function}`'s OVER is refused: {error}")]
    Over {
        function: &'static str,
        #[source]
        error: OverError,
    },
    #[error("`{function}` holds the aggregate call `{nested}`, and an aggregate call holds no aggregate call")]
    AggregateInAggregate {
        function: &'static str,
        nested: &'static str,
    },
    #[error(
        "`{function}` holds the window call `{nested}`, and an aggregate call holds no window call"
    )]
    WindowInAggregate {
        function: &'static str,
        nested: &'static str,
    },
    #[error(
        "`{function}` holds the window call `{nested}`, and a window call holds no window call"
    )]
    WindowInWindow {
        function: &'static str,
        nested: &'static str,
    },
}

fn arguments_in_words(min: usize, max: Option<usize>) -> String {
    const WORDS: [&str; 9] = [
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    ];
    let word = |count: usize| {
        (count.checked_sub(1).and_then(|index| WORDS.get(index)))
            .map_or_else(|| count.to_string(), |word| (*word).to_owned())
    };
    let noun = if min == 1 { "argument" } else { "arguments" };
    match max {
        Some(max) if max == min => format!("{} {noun}", word(min)),
        Some(max) => format!("{} to {} arguments", word(min), word(max)),
        None => format!("{} {noun} or more", word(min)),
    }
}

fn enum_value(value: &Literal) -> String {
    match value {
        Literal::String(text) => format!("value '{text}'"),
        other => format!("{} value", other.data_type()),
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Error)]
pub enum OverError {
    #[error("a window frame cannot start at UNBOUNDED FOLLOWING")]
    StartsUnboundedFollowing,
    #[error("a window frame cannot end at UNBOUNDED PRECEDING")]
    EndsUnboundedPreceding,
    #[error("a window frame cannot end before it starts")]
    EndsBeforeStart,
    #[error("a RANGE window frame with an offset takes one ORDER BY key, not {keys}")]
    RangeOffsetKeys { keys: usize },
    #[error("a GROUPS window frame requires an ORDER BY")]
    GroupsWithoutOrderBy,
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum JsonFormError {
    #[error("no function is named `{name}` in {dialect:?}")]
    UnknownFunction {
        name: String,
        dialect: DocumentDialect,
    },
    #[error("`{name}` is no {tag} function")]
    NotItsFunctionType { name: String, tag: &'static str },
    #[error("the call's options name `{name}` twice")]
    RepeatedOption { name: String },
    #[error("`{function}` has no function in {dialect:?}")]
    NoNameInDialect {
        function: &'static str,
        dialect: DocumentDialect,
    },
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Error)]
pub enum RebuildError {
    #[error("a leaf has no operands to replace")]
    Leaf,
    #[error("the node's new operands are not one per child")]
    ChildCount,
    #[error("the call's constructor refuses it: {0}")]
    Call(#[source] CallError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Error)]
#[error("a row-level tree holds no aggregate call")]
pub struct AggregateInRowExpr;

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Error)]
pub enum TypeError {
    #[error("`{name}`'s `{parameter}` takes {takes}, not {got}")]
    Argument {
        name: &'static str,
        parameter: &'static str,
        takes: &'static ParameterType,
        got: TypeOrFamily,
    },
    #[error(
        "`{name}` does not take these arguments: {}",
        .types.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ")
    )]
    Arguments {
        name: &'static str,
        types: Vec<TypeOrFamily>,
    },
    #[error("`{name}` takes its arguments at one type, but one is {bound} and another is {got}")]
    TypeVariable {
        name: &'static str,
        bound: TypeOrFamily,
        got: TypeOrFamily,
    },
    #[error("`{name}`'s `{parameter}` takes no '{value}' beside {got}")]
    EnumRefused {
        name: &'static str,
        parameter: &'static str,
        value: String,
        got: TypeOrFamily,
    },
    #[error(
        "`{name}` takes no `{parameter}` where an earlier argument's type does not call for it"
    )]
    ParameterRefused {
        name: &'static str,
        parameter: &'static str,
    },
    #[error("`{name}` requires its `{parameter}` where an earlier argument's type calls for it")]
    ParameterRequired {
        name: &'static str,
        parameter: &'static str,
    },
    #[error("`{name}`'s {clause} takes {takes}, not {got}")]
    Clause {
        name: &'static str,
        clause: CallClause,
        takes: &'static ParameterType,
        got: TypeOrFamily,
    },
    #[error(
        "`{name}`'s RANGE window frame with an offset takes no ORDER BY key of the {family} family"
    )]
    RangeOrderKey {
        name: &'static str,
        family: TypeFamily,
    },
    #[error("the literal {literal:?} does not read as {context}: {error}")]
    UntypedLiteral {
        literal: Literal,
        context: DataType,
        error: LiteralError,
    },
    #[error("a CASE's WHEN is {got}, not a boolean")]
    CaseWhen { got: TypeOrFamily },
    #[error("a CASE takes its branches at one type, but one is {first} and another is {other}")]
    CaseBranches {
        first: TypeOrFamily,
        other: TypeOrFamily,
    },
    #[error("{from} does not cast to {to}")]
    Cast { from: TypeOrFamily, to: DataType },
    #[error("{dialect:?} has no `{name}`")]
    NotInDialect {
        name: &'static str,
        dialect: Dialect,
    },
    #[error("in the document's dialect, {0}")]
    Call(CallError),
}
