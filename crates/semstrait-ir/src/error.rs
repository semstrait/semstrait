use semstrait_common::diagnostic::{Diagnose, Severity};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{LiteralError, NamedReference, TypeError};
use thiserror::Error;

use crate::expr::ParameterKey;
use crate::plan::Anchor;
use crate::types::{field_name, SchemaColumn};

#[non_exhaustive]
#[derive(Debug, Error, Clone, PartialEq)]
pub enum IrErrorKind {
    #[error("`{}` binds to no one field of [{}]", field_name(leaf), fields.join(", "))]
    UnboundLeaf {
        leaf: NamedReference,
        fields: Vec<String>,
    },

    #[error("typing refuses the expression: {error}")]
    TypeRefused { error: TypeError },

    #[error("the field `{field}` has the type {data_type}, which is not valid")]
    InvalidType { field: String, data_type: DataType },

    #[error("the sort key at position {key} has the type {data_type}, which is not valid")]
    InvalidKeyType { key: usize, data_type: DataType },

    #[error("a filter of type {got} is not a condition")]
    NotACondition { got: DataType },

    #[error("the row {row} holds {got} values for {fields} fields")]
    RowArity {
        row: usize,
        fields: usize,
        got: usize,
    },

    #[error("the row {row} holds a {got} for the field `{field}` of type {expected}")]
    RowType {
        row: usize,
        field: String,
        expected: DataType,
        got: DataType,
    },

    #[error("the row {row} holds a value that can be NULL for the field `{field}`, which is not nullable")]
    RowNull { row: usize, field: String },

    #[error("the emit form maps index {index}, past the natural output's {fields} fields")]
    EmitFormIndexOutOfRange { index: usize, fields: usize },

    #[error("the emit form leaves out the computed field at index {index}")]
    EmitFormLeavesOutComputedField { index: usize },

    #[error("the output schema is not the emit form over the natural output at field {index}")]
    OutputSchemaMismatch { index: usize },

    #[error(
        "the aggregate `{}` of type {} is output as `{}` of type {}",
        expected.name, expected.data_type, got.name, got.data_type
    )]
    AggregateFieldMismatch {
        expected: SchemaColumn,
        got: SchemaColumn,
    },

    #[error("the union has {got} inputs, fewer than two")]
    MultisetArity { got: usize },

    #[error("the union's branch {branch} is not its first branch at field {index}")]
    BranchFieldMismatch { branch: usize, index: usize },

    #[error("the computed field at index {index} has no name")]
    UnnamedField { index: usize },

    #[error("the name `{name}` repeats")]
    RepeatedName { name: String },

    #[error("the fetch bound {bound} is negative")]
    NegativeBound { bound: i64 },

    #[error("the sort key at position {key} leaves its NULL order unspecified")]
    UnspecifiedNullOrdering { key: usize },

    #[error("the order key at position {key} of `{function}` leaves its NULL order unspecified")]
    UnspecifiedWindowNullOrdering { function: &'static str, key: usize },

    #[error(
        "the ORDER BY key at position {key} of `{function}` leaves its NULL order unspecified"
    )]
    UnspecifiedAggregateNullOrdering { function: &'static str, key: usize },

    #[error("the anchor {} repeats", anchor.0)]
    RepeatedAnchor { anchor: Anchor },

    #[error(
        "the output names [{}] are not the root's field names [{}]",
        output_names.join(", "),
        fields.join(", ")
    )]
    OutputNamesMismatch {
        output_names: Vec<String>,
        fields: Vec<String>,
    },

    #[error("the root name `{name}` starts with a reserved prefix")]
    ReservedRootName { name: String },

    #[error("a TopNNode has no lowering yet")]
    TopNNotLowered,

    #[error("the parameter {key:?} is not bound")]
    UnboundParameter { key: ParameterKey },

    #[error("the aggregate call `{function}` is no aggregate node's entry")]
    MisplacedAggregate { function: &'static str },

    #[error("the call of `{function}` carries a null treatment, which Substrait cannot hold")]
    CallWithNullTreatment { function: &'static str },

    #[error("the window call `{function}` sits in a filter")]
    WindowInFilter { function: &'static str },

    #[error("the window call `{function}` sits in a sort key")]
    WindowInSortKey { function: &'static str },

    #[error("the window call `{function}` sits in an aggregate node")]
    WindowInAggregate { function: &'static str },

    #[error("the window call `{function}` sits in a reader's rows")]
    WindowInReader { function: &'static str },

    #[error("the window call `{function}` sits in a join condition or a post-join filter")]
    WindowInJoin { function: &'static str },

    #[error(
        "the window call `{function}` has a GROUPS window frame or an offset past `i64`, \
         which Substrait cannot hold"
    )]
    UnfitWindowFrame { function: &'static str },

    #[error("a literal does not fit its type: {error}")]
    UnfitLiteral { error: LiteralError },
}

impl Diagnose for IrErrorKind {
    fn message(&self) -> String {
        self.to_string()
    }

    fn severity_default(&self) -> Severity {
        Severity::Error
    }
}
