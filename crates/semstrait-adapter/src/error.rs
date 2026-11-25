use std::fmt;

use semstrait_common::diagnostic::{Diagnose, Severity};
use semstrait_ir::{field_name, IrErrorKind};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{CallError, LiteralError, NamedReference};
use thiserror::Error;

#[non_exhaustive]
#[derive(Debug, Error, Clone, PartialEq)]
pub enum InvalidRewrite {
    #[error("the producer has no lowering for the node {node}")]
    NoLowering { node: &'static str },

    #[error("the producer knows no {part}")]
    UnknownForm { part: VocabularyPart },

    #[error("an aggregate call outside an aggregate entry")]
    AggregateOutsideEntry,

    #[error(
        "the column `{}` binds to no one field of its input",
        field_name(column)
    )]
    UnboundColumn { column: NamedReference },

    #[error("a call does not bind to its declaration: {error}")]
    UnboundCall { error: CallError },

    #[error("the call of {function} does not fit its Substrait form")]
    CallShape { function: &'static str },

    #[error("the rules replaced the node with one of another output schema")]
    SchemaChanged,

    #[error("the rules replaced an expression of type {offered} with one of type {replacement}")]
    TypeChanged {
        offered: DataType,
        replacement: DataType,
    },

    #[error("an expression does not type over its input: {error}")]
    Untyped { error: IrErrorKind },

    #[error("the type {data_type} has no Substrait form")]
    NoSubstraitType { data_type: DataType },

    #[error("a literal does not fit its type: {error}")]
    UnfitLiteral { error: LiteralError },

    #[error("the position {position} is past Substrait's 32-bit numbers")]
    PositionOutOfRange { position: usize },
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VocabularyPart {
    JoinType,
    SetOperation,
    Reader,
    Expression,
    CastFailure,
    SortDirection,
    WindowFrame,
    Literal,
}

impl fmt::Display for VocabularyPart {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::JoinType => "join type",
            Self::SetOperation => "set operation",
            Self::Reader => "reader",
            Self::Expression => "expression",
            Self::CastFailure => "cast failure",
            Self::SortDirection => "sort direction",
            Self::WindowFrame => "window frame",
            Self::Literal => "literal",
        })
    }
}

pub(crate) fn number<N: TryFrom<usize>>(position: usize) -> Result<N, InvalidRewrite> {
    N::try_from(position).map_err(|_| InvalidRewrite::PositionOutOfRange { position })
}

impl Diagnose for InvalidRewrite {
    fn message(&self) -> String {
        self.to_string()
    }

    fn severity_default(&self) -> Severity {
        Severity::Error
    }
}
