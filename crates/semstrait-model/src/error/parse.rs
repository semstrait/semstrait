use semstrait_common::diagnostic::{Diagnose, Severity};

use crate::error::serde_message;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ParseErrorKind {
    YamlSyntax { message: String },
    UnsetEnvVar { var: String },
    UnknownField { field: String, parent: String },

    InvalidIdentifier { raw: String, reason: String },
    InvalidEntityId { raw: String, reason: String },

    MeasureMissingAgg { carrier: String, name: String },
    UnknownAggregate { name: String },
    SemanticsMissingDataType { carrier: String, name: String },
    InvalidDataType { spelling: String },
    InvalidLiteral { data_type: String, reason: String },
    UnknownTag { tag: String },
    CallRefused { function: String, reason: String },
    AggregateInRowExpr { function: String },
    SemanticsSiteNameWithRef { carrier: String },

    SemanticsSiteKeysetBothSpellings { carrier: String },
    SemanticsSiteKeysetWithoutIsKey { carrier: String, spelling: String },
    SemanticsSiteKeysetListEmpty { carrier: String },
    SemanticsSiteKeysetNameBlank { carrier: String, spelling: String },

    DataKindKeysBlockRetired { data_kind: String },
    NestedDataKindKeynessIgnored { data_kind: String, spelling: String },

    RelationshipJoinKeysEmpty,
    RelationshipJoinKeysUnknownSpelling { spelling: String },
    RelationshipJoinKeyDuplicated { key: String },
}

impl Diagnose for ParseErrorKind {
    fn message(&self) -> String {
        use ParseErrorKind::*;
        match self {
            YamlSyntax { message } => format!("YAML syntax error: {}", message),
            UnsetEnvVar { var } => {
                format!("environment variable `{}` is not set", var)
            }
            UnknownField { field, parent } => {
                format!("unknown field `{}` in `{}`", field, parent)
            }
            InvalidIdentifier { raw, reason } => {
                format!("invalid identifier `{}`: {}", raw, reason)
            }
            InvalidEntityId { raw, reason } => {
                format!("invalid entity id `{}`: {}", raw, reason)
            }
            MeasureMissingAgg { carrier, name } => {
                format!("{} `{}` is missing required `agg:`", carrier, name)
            }
            UnknownAggregate { name } => serde_message::unknown_aggregate(name),
            SemanticsMissingDataType { carrier, name } => {
                format!("{} `{}` is missing required `data_type:`", carrier, name)
            }
            InvalidDataType { spelling } => serde_message::data_type_refused(spelling),
            InvalidLiteral { data_type, reason } => {
                serde_message::literal_refused(data_type, reason)
            }
            UnknownTag { tag } => serde_message::unknown_tag(tag),
            CallRefused { function, reason } => serde_message::call_refused(function, reason),
            AggregateInRowExpr { function } => serde_message::aggregate_in_row_expr(function),
            SemanticsSiteNameWithRef { carrier } => format!(
                "a {} site is either an inline declaration (`name:`) or a \
                 reference (`ref:`), never both",
                carrier
            ),
            SemanticsSiteKeysetBothSpellings { carrier } => format!(
                "a {} site declares both `keyset:` and `keysets:`; keep either the \
                 singular or the plural",
                carrier
            ),
            SemanticsSiteKeysetWithoutIsKey { carrier, spelling } => format!(
                "a {} site names `{}:` without `is_key: true`, so the membership would \
                 silently do nothing",
                carrier, spelling
            ),
            SemanticsSiteKeysetListEmpty { carrier } => format!(
                "a {} site declares `keysets: []`, naming no keyset; write bare \
                 `is_key: true` for `default` membership",
                carrier
            ),
            SemanticsSiteKeysetNameBlank { carrier, spelling } => format!(
                "a {} site names a blank keyset under `{}:`; a keyset name cannot be empty",
                carrier, spelling
            ),
            DataKindKeysBlockRetired { data_kind } => format!(
                "data kind `{}`: the `keys:` block is retired and reaches nothing; mark \
                 each identifying dimension site `is_key: true` instead",
                data_kind
            ),
            NestedDataKindKeynessIgnored {
                data_kind,
                spelling,
            } => format!(
                "nested data kind `{}`: `{}:` reaches nothing — a nested form has no \
                 semantic interface, so keyness is authored on the dimension sites of \
                 the enclosing Public data kind",
                data_kind, spelling
            ),
            RelationshipJoinKeysEmpty => {
                "relationship `on: []` declares an empty join basis; write \
                 `on: cross` for no dimension-name basis"
                    .to_string()
            }
            RelationshipJoinKeysUnknownSpelling { spelling } => format!(
                "unknown relationship `on:` spelling `{}`; expected a list of \
                 dimension names, `auto`, or `cross`",
                spelling
            ),
            RelationshipJoinKeyDuplicated { key } => {
                format!("relationship `on:` repeats the name `{}`", key)
            }
        }
    }

    fn severity_default(&self) -> Severity {
        use ParseErrorKind::*;
        match self {
            DataKindKeysBlockRetired { .. } | NestedDataKindKeynessIgnored { .. } => {
                Severity::Warning
            }
            _ => Severity::Error,
        }
    }
}
