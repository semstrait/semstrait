use semstrait_common::ValidateError;
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::LiteralError;
use thiserror::Error;

use crate::error::serde_message;

#[non_exhaustive]
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ParseError {
    #[error("inline expression DSL is a future feature; rewrite to block form: `{expr}`")]
    InlineExprUnsupported { expr: String },

    #[error("{}", serde_message::unknown_tag(.0))]
    UnknownTag(String),

    #[error("tag `{tag}` not allowed at {site} site")]
    TagNotAllowedAtSite { tag: String, site: &'static str },

    #[error("{}", serde_message::aggregate_in_row_expr(.function))]
    AggregateInRowExpr { function: String },

    #[error("{}", serde_message::call_refused(.function, .reason))]
    CallRefused { function: String, reason: String },

    #[error("{}", serde_message::call_refused(.function, &unknown_parameter(.key, .parameters)))]
    UnknownParameter {
        function: String,
        key: String,
        parameters: Vec<&'static str>,
    },

    #[error("{}", serde_message::call_refused(.function, &format!("the mapping leaves out `{}`, which the call writes", .parameter)))]
    MissingParameter {
        function: String,
        parameter: &'static str,
    },

    #[error("{}", serde_message::call_refused(.function, &format!("the mapping names `{}`, which the call does not write", .parameter)))]
    UnwrittenParameter {
        function: String,
        parameter: &'static str,
    },

    #[error("missing required field `{0}`")]
    MissingField(&'static str),

    #[error("invalid value for field `{field}`: {reason}")]
    InvalidValue { field: &'static str, reason: String },

    #[error("{}", serde_message::data_type_refused(.0))]
    InvalidDataType(String),

    #[error("{}", serde_message::literal_refused(&.data_type.to_string(), &.reason.to_string()))]
    InvalidLiteral {
        data_type: DataType,
        reason: LiteralError,
    },

    #[error("invalid token `{raw}`: {reason}")]
    InvalidToken { raw: String, reason: &'static str },

    #[error("invalid semantic_mapping value: {0}")]
    InvalidMappingValue(String),

    #[error("expected single-key tagged map, got {0} keys")]
    AmbiguousTag(usize),

    #[error("unexpected YAML shape: {0}")]
    UnexpectedShape(String),

    #[error("invalid expression: {0}")]
    InvalidExpression(#[from] ValidateError),

    #[error("expression node is not representable in the authoring grammar: {0}")]
    NotAuthorRepresentable(&'static str),
}

fn unknown_parameter(key: &str, parameters: &[&str]) -> String {
    if parameters.is_empty() {
        return format!("no parameter is named `{key}`; the function has none");
    }
    let names: Vec<_> = parameters.iter().map(|name| format!("`{name}`")).collect();
    format!(
        "no parameter is named `{key}`; the parameters are {}",
        names.join(", ")
    )
}
