use std::error::Error;
use std::fmt;

use semstrait_common::{DataKindName, SemanticsName};
use semstrait_common::{Diagnose, Granularity, Severity, ValidateError};
use semstrait_ir::{IrErrorKind, SemanticRole, SemiAdditivityStrategy};
use semstrait_manifest::RoleDefinition;
use semstrait_sql::datatype::{DataType, TypeOrFamily};
use semstrait_sql::expr::{NamedReference, TypeError};

use crate::frame_path::{at, FramePath};
use crate::DecisionValue;

pub(crate) type Lacking = (DataKindName, Vec<SemanticsName>);

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum PlanError {
    EmptySelect,
    RepeatedSelection {
        name: SemanticsName,
    },
    SelectionRoleMismatch {
        name: SemanticsName,
        selected: SemanticRole,
        role: SemanticRole,
    },
    NotSelectable {
        name: SemanticsName,
        role: SemanticRole,
    },
    OrderByNotSelected {
        name: SemanticsName,
    },
    FilterInRequestPredicate {
        name: SemanticsName,
    },
    ColumnInRequestPredicate {
        column: NamedReference,
    },
    AccessorInRequestPredicate {
        name: SemanticsName,
    },
    PredicateRoleMismatch {
        name: SemanticsName,
        stated: SemanticRole,
        role: SemanticRole,
    },
    PredicateTypeError {
        error: TypeError,
    },
    PredicateNotACondition {
        got: TypeOrFamily,
    },
    MixedPlacement {
        dimension: SemanticsName,
        aggregated: SemanticsName,
    },
    GranularityOnNonTemporalDimension {
        name: SemanticsName,
        requested: Granularity,
    },
    GranularityFinerThanType {
        name: SemanticsName,
        requested: Granularity,
        data_type: DataType,
    },

    UnknownName {
        name: String,
        looked_in: LookedIn,
    },
    NoCandidate {
        nearest: Vec<Lacking>,
    },

    GrainNotServable {
        path: FramePath,
        source: DataKindName,
        requested: Option<Granularity>,
        source_granularity: Option<Granularity>,
        nearest: Vec<Lacking>,
    },
    StrategyNotFoundForSemanticSource {
        path: FramePath,
        semantic_source: DataKindName,
        role_definition: RoleDefinition,
    },
    UnresolvedType {
        path: FramePath,
        relation: String,
        column: String,
    },
    BrokenSource {
        path: FramePath,
        semantic_source: DataKindName,
    },
    BrokenTree {
        path: FramePath,
        semantic_source: DataKindName,
        semantics: SemanticsName,
    },
    NoSignature {
        path: FramePath,
        semantic_source: DataKindName,
        semantics: SemanticsName,
    },
    UnboundParameter {
        path: FramePath,
        semantics: SemanticsName,
    },
    AmbiguousOperandCondition {
        path: FramePath,
        semantics: SemanticsName,
        semantic_source: DataKindName,
    },
    UngroupedMetricFormula {
        path: FramePath,
        metric: SemanticsName,
        name: SemanticsName,
    },
    FanOutMarkerNotPlaceable {
        semantics: SemanticsName,
        path: FramePath,
    },
    DecisionNotAllowed {
        path: FramePath,
        semantic_source: DataKindName,
        value: DecisionValue,
    },
    NodeRefused {
        path: FramePath,
        error: IrErrorKind,
    },
    ConstructionRefused {
        path: FramePath,
        error: ValidateError,
    },
    Unbuildable {
        path: FramePath,
        semantic_source: DataKindName,
        state: UnbuildableState,
    },
    PlanRefused {
        error: IrErrorKind,
    },
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum UnbuildableState {
    SemiAdditiveStrategy(SemiAdditivityStrategy),
    PartialAggregateNotCombined { function: &'static str },
}

impl fmt::Display for UnbuildableState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SemiAdditiveStrategy(strategy) => {
                write!(f, "the semi-additive strategy `{strategy:?}` has no marker")
            }
            Self::PartialAggregateNotCombined { function } => {
                write!(
                    f,
                    "a partial aggregate of `{function}` does not combine again"
                )
            }
        }
    }
}

impl PlanError {
    #[must_use]
    pub fn path(&self) -> Option<&FramePath> {
        match self {
            Self::GrainNotServable { path, .. }
            | Self::StrategyNotFoundForSemanticSource { path, .. }
            | Self::UnresolvedType { path, .. }
            | Self::BrokenSource { path, .. }
            | Self::BrokenTree { path, .. }
            | Self::NoSignature { path, .. }
            | Self::UnboundParameter { path, .. }
            | Self::AmbiguousOperandCondition { path, .. }
            | Self::UngroupedMetricFormula { path, .. }
            | Self::FanOutMarkerNotPlaceable { path, .. }
            | Self::DecisionNotAllowed { path, .. }
            | Self::NodeRefused { path, .. }
            | Self::ConstructionRefused { path, .. }
            | Self::Unbuildable { path, .. } => Some(path),
            Self::EmptySelect
            | Self::RepeatedSelection { .. }
            | Self::SelectionRoleMismatch { .. }
            | Self::NotSelectable { .. }
            | Self::OrderByNotSelected { .. }
            | Self::FilterInRequestPredicate { .. }
            | Self::ColumnInRequestPredicate { .. }
            | Self::AccessorInRequestPredicate { .. }
            | Self::PredicateRoleMismatch { .. }
            | Self::PredicateTypeError { .. }
            | Self::PredicateNotACondition { .. }
            | Self::MixedPlacement { .. }
            | Self::GranularityOnNonTemporalDimension { .. }
            | Self::GranularityFinerThanType { .. }
            | Self::UnknownName { .. }
            | Self::NoCandidate { .. }
            | Self::PlanRefused { .. } => None,
        }
    }

    fn refusal(&self) -> String {
        match self {
            Self::EmptySelect => "the request selects nothing".to_owned(),
            Self::RepeatedSelection { name } => format!("`{name}` is selected twice"),
            Self::SelectionRoleMismatch {
                name,
                selected,
                role,
            } => format!("`{name}` is selected as a {selected:?}, but it is a {role:?}"),
            Self::NotSelectable { name, role } => {
                format!("`{name}` is a {role:?}, which a request does not select")
            }
            Self::OrderByNotSelected { name } => {
                format!("`order_by` names `{name}`, which the request does not select")
            }
            Self::FilterInRequestPredicate { name } => {
                format!("the request predicate reads the Filter `{name}`: activate it in `filters`")
            }
            Self::ColumnInRequestPredicate { column } => format!(
                "the request predicate reads the column `{}`: it reads semantics only",
                column.parts().join(".")
            ),
            Self::AccessorInRequestPredicate { name } => {
                format!("the request predicate reads `{name}` through an accessor")
            }
            Self::PredicateRoleMismatch { name, stated, role } => format!(
                "the request predicate reads `{name}` as a {stated:?}, but it is a {role:?}"
            ),
            Self::PredicateTypeError { error } => {
                format!("the request predicate does not type: {error}")
            }
            Self::PredicateNotACondition { got } => {
                format!("the request predicate is {got}, not a boolean")
            }
            Self::MixedPlacement {
                dimension,
                aggregated,
            } => format!(
                "the request predicate reads the Dimension `{dimension}` beside `{aggregated}`, \
                 which is read after aggregation"
            ),
            Self::GranularityOnNonTemporalDimension { name, requested } => {
                format!("`{name}` is not temporal, so it takes no granularity `{requested}`")
            }
            Self::GranularityFinerThanType {
                name,
                requested,
                data_type,
            } => format!("`{name}` is a {data_type}, which holds no granularity `{requested}`"),
            Self::UnknownName { name, looked_in } => format!("`{name}` is not in {looked_in}"),
            Self::NoCandidate { nearest } => format!(
                "no public semantic source serves the request; nearest: {}",
                lacking_list(nearest)
            ),
            Self::GrainNotServable {
                source,
                requested,
                source_granularity,
                nearest,
                ..
            } => {
                let held = source_granularity
                    .map_or_else(String::new, |granularity| format!(" at `{granularity}`"));
                let asked = requested.map_or_else(
                    || "the request's grain".to_owned(),
                    |granularity| format!("the requested granularity `{granularity}`"),
                );
                let named = if nearest.is_empty() {
                    String::new()
                } else {
                    format!("; nearest: {}", lacking_list(nearest))
                };
                format!("the semantic source `{source}`{held} cannot serve {asked}{named}")
            }
            Self::StrategyNotFoundForSemanticSource {
                semantic_source, ..
            } => format!("no Strategy plans the semantic source `{semantic_source}`"),
            Self::UnresolvedType {
                relation, column, ..
            } => format!(
                "the column `{column}` of `{relation}` has no type: no catalog observed one, \
                 and no semantics with a type maps it"
            ),
            Self::BrokenSource {
                semantic_source, ..
            } => format!(
                "the store does not hold the semantic source `{semantic_source}` or the \
                 physical source it reads"
            ),
            Self::BrokenTree {
                semantic_source,
                semantics,
                ..
            } => format!("the stored tree of `{semantics}` on `{semantic_source}` is broken"),
            Self::NoSignature {
                semantic_source,
                semantics,
                ..
            } => format!(
                "the semantic source `{semantic_source}` has no signature for `{semantics}`"
            ),
            Self::UnboundParameter { semantics, .. } => {
                format!("the tree of `{semantics}` holds a parameter, which nothing binds")
            }
            Self::AmbiguousOperandCondition {
                semantics,
                semantic_source,
                ..
            } => format!(
                "both operands of `{semantic_source}` map `{semantics}`, and an operand's \
                 condition for it has no own filter of `{semantic_source}` to replace it"
            ),
            Self::UngroupedMetricFormula { metric, name, .. } => format!(
                "the Metric `{metric}` reads `{name}` outside an aggregate, and its formula is \
                 not numeric: group the request by `{name}`"
            ),
            Self::FanOutMarkerNotPlaceable { semantics, .. } => format!(
                "the fan-out marker of `{semantics}` has no place above the join that repeats \
                 its rows"
            ),
            Self::DecisionNotAllowed {
                semantic_source,
                value,
                ..
            } => format!("the body of `{semantic_source}` may not use {value}"),
            Self::NodeRefused { error, .. } => format!("a plan node is refused: {error}"),
            Self::ConstructionRefused { error, .. } => {
                format!("a name or a tree of the plan is refused: {error}")
            }
            Self::Unbuildable {
                semantic_source,
                state,
                ..
            } => format!(
                "the build of `{semantic_source}` meets a state that it has no step for: {state}"
            ),
            Self::PlanRefused { error } => format!("the plan is refused: {error}"),
        }
    }
}

impl Diagnose for PlanError {
    fn message(&self) -> String {
        match self.path() {
            Some(path) => at(path, self.refusal()),
            None => self.refusal(),
        }
    }

    fn severity_default(&self) -> Severity {
        Severity::Error
    }

    fn cause(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::PredicateTypeError { error } => Some(error),
            Self::NodeRefused { error, .. } | Self::PlanRefused { error } => Some(error),
            Self::ConstructionRefused { error, .. } => Some(error),
            _ => None,
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LookedIn {
    PublicSources,
    SemanticDefinitions,
    NamedFilters,
    Source(DataKindName),
}

impl fmt::Display for LookedIn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PublicSources => f.write_str("the public semantic sources"),
            Self::SemanticDefinitions => f.write_str("the semantic definitions"),
            Self::NamedFilters => f.write_str("the named filters"),
            Self::Source(source) => write!(f, "the semantic source `{source}`"),
        }
    }
}

pub(crate) fn quoted(names: &[impl fmt::Display]) -> String {
    let names: Vec<String> = names.iter().map(|name| format!("`{name}`")).collect();
    names.join(", ")
}

fn lacking_list(nearest: &[Lacking]) -> String {
    let nearest: Vec<String> = (nearest.iter())
        .map(|(source, missing)| {
            if missing.is_empty() {
                format!("`{source}` lacks no name")
            } else {
                format!("`{source}` lacks {}", quoted(missing))
            }
        })
        .collect();
    nearest.join("; ")
}

pub(crate) fn call_refused(error: impl fmt::Display) -> ValidateError {
    ValidateError::CallRefused {
        reason: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use semstrait_common::Diagnostic;
    use semstrait_manifest::Selector;
    use semstrait_sql::expr::TypeError;

    use super::*;
    use crate::Combine;

    fn path() -> FramePath {
        FramePath(vec!["orders".into(), "customers".into()])
    }

    fn pathless() -> Vec<(PlanError, Vec<&'static str>)> {
        vec![
            (PlanError::EmptySelect, vec![]),
            (
                PlanError::RepeatedSelection {
                    name: "channel".into(),
                },
                vec!["channel"],
            ),
            (
                PlanError::SelectionRoleMismatch {
                    name: "revenue".into(),
                    selected: SemanticRole::Dimension,
                    role: SemanticRole::Measure,
                },
                vec!["revenue", "Dimension", "Measure"],
            ),
            (
                PlanError::NotSelectable {
                    name: "eu_only".into(),
                    role: SemanticRole::Filter,
                },
                vec!["eu_only", "Filter"],
            ),
            (
                PlanError::OrderByNotSelected {
                    name: "region".into(),
                },
                vec!["region"],
            ),
            (
                PlanError::FilterInRequestPredicate {
                    name: "eu_only".into(),
                },
                vec!["eu_only"],
            ),
            (
                PlanError::ColumnInRequestPredicate {
                    column: NamedReference::unqualified("region_code").expect("a name"),
                },
                vec!["region_code"],
            ),
            (
                PlanError::AccessorInRequestPredicate {
                    name: "revenue".into(),
                },
                vec!["revenue"],
            ),
            (
                PlanError::PredicateRoleMismatch {
                    name: "region".into(),
                    stated: SemanticRole::Metric,
                    role: SemanticRole::Dimension,
                },
                vec!["region", "Metric", "Dimension"],
            ),
            (
                PlanError::PredicateTypeError {
                    error: TypeError::ParameterRefused {
                        name: "date_trunc",
                        parameter: "timezone",
                    },
                },
                vec!["date_trunc", "timezone"],
            ),
            (
                PlanError::PredicateNotACondition {
                    got: TypeOrFamily::Type(DataType::String),
                },
                vec!["string", "boolean"],
            ),
            (
                PlanError::MixedPlacement {
                    dimension: "region".into(),
                    aggregated: "revenue".into(),
                },
                vec!["region", "revenue"],
            ),
            (
                PlanError::GranularityOnNonTemporalDimension {
                    name: "region".into(),
                    requested: Granularity::Month,
                },
                vec!["region", "month"],
            ),
            (
                PlanError::GranularityFinerThanType {
                    name: "ordered_on".into(),
                    requested: Granularity::Hour,
                    data_type: DataType::Date,
                },
                vec!["ordered_on", "hour", "date"],
            ),
            (
                PlanError::UnknownName {
                    name: "refunds".into(),
                    looked_in: LookedIn::Source("orders".into()),
                },
                vec!["refunds", "orders"],
            ),
            (
                PlanError::NoCandidate {
                    nearest: vec![
                        ("orders".into(), vec!["refunds".into()]),
                        ("returns".into(), vec!["channel".into(), "region".into()]),
                    ],
                },
                vec!["orders", "refunds", "returns", "channel", "region"],
            ),
        ]
    }

    fn in_a_frame() -> Vec<(PlanError, Vec<&'static str>)> {
        vec![
            (
                PlanError::GrainNotServable {
                    path: path(),
                    source: "spend_daily".into(),
                    requested: Some(Granularity::Hour),
                    source_granularity: Some(Granularity::Day),
                    nearest: vec![("spend_hourly".into(), vec!["clicks".into()])],
                },
                vec!["spend_daily", "hour", "day", "spend_hourly", "clicks"],
            ),
            (
                PlanError::StrategyNotFoundForSemanticSource {
                    path: path(),
                    semantic_source: "customers".into(),
                    role_definition: RoleDefinition::dataset("customers-source", Selector::Table),
                },
                vec!["customers"],
            ),
            (
                PlanError::UnresolvedType {
                    path: path(),
                    relation: "s3://lake/customers/".into(),
                    column: "qty".into(),
                },
                vec!["s3://lake/customers/", "qty"],
            ),
            (
                PlanError::BrokenSource {
                    path: path(),
                    semantic_source: "customers".into(),
                },
                vec!["customers"],
            ),
            (
                PlanError::BrokenTree {
                    path: path(),
                    semantic_source: "customers".into(),
                    semantics: "region".into(),
                },
                vec!["customers", "region"],
            ),
            (
                PlanError::NoSignature {
                    path: path(),
                    semantic_source: "customers".into(),
                    semantics: "region".into(),
                },
                vec!["customers", "region"],
            ),
            (
                PlanError::UnboundParameter {
                    path: path(),
                    semantics: "revenue_lag".into(),
                },
                vec!["revenue_lag"],
            ),
            (
                PlanError::AmbiguousOperandCondition {
                    path: path(),
                    semantics: "revenue".into(),
                    semantic_source: "orders_customers".into(),
                },
                vec!["revenue", "orders_customers"],
            ),
            (
                PlanError::UngroupedMetricFormula {
                    path: path(),
                    metric: "region_label".into(),
                    name: "region".into(),
                },
                vec!["region_label", "region"],
            ),
            (
                PlanError::FanOutMarkerNotPlaceable {
                    semantics: "balance".into(),
                    path: path(),
                },
                vec!["balance"],
            ),
            (
                PlanError::DecisionNotAllowed {
                    path: path(),
                    semantic_source: "customers".into(),
                    value: DecisionValue::Combine(Combine::Join),
                },
                vec!["customers", "join"],
            ),
            (
                PlanError::DecisionNotAllowed {
                    path: path(),
                    semantic_source: "customers".into(),
                    value: DecisionValue::NoPhysicalRead,
                },
                vec!["customers", "no physical read"],
            ),
            (
                PlanError::NodeRefused {
                    path: path(),
                    error: IrErrorKind::RepeatedName {
                        name: "region".into(),
                    },
                },
                vec!["region"],
            ),
            (
                PlanError::ConstructionRefused {
                    path: path(),
                    error: ValidateError::EmptyName,
                },
                vec!["empty"],
            ),
            (
                PlanError::Unbuildable {
                    path: path(),
                    semantic_source: "customers".into(),
                    state: UnbuildableState::PartialAggregateNotCombined { function: "count" },
                },
                vec!["customers", "count"],
            ),
            (
                PlanError::Unbuildable {
                    path: path(),
                    semantic_source: "balances".into(),
                    state: UnbuildableState::SemiAdditiveStrategy(SemiAdditivityStrategy::Latest),
                },
                vec!["balances", "Latest"],
            ),
        ]
    }

    fn message_naming(error: PlanError, named: &[&str]) -> String {
        let diagnostic = Diagnostic::new(error);
        assert_eq!(diagnostic.severity, Severity::Error);
        let message = diagnostic.kind.message();
        for name in named {
            assert!(message.contains(name), "`{message}` names `{name}`");
        }
        message
    }

    #[test]
    fn every_decide_and_build_error_names_its_path() {
        let path = path();
        for (error, named) in in_a_frame() {
            assert_eq!(error.path(), Some(&path), "{error:?}");
            let message = message_naming(error, &named);
            assert!(message.contains(&path.to_string()), "{message}");
        }
    }

    #[test]
    fn every_binding_and_resolver_error_names_no_path() {
        for (error, named) in pathless() {
            assert_eq!(error.path(), None, "{error:?}");
            message_naming(error, &named);
        }
    }

    #[test]
    fn grain_not_servable_with_no_granularity_renders_an_error() {
        let error = PlanError::GrainNotServable {
            path: FramePath(vec!["spend".into()]),
            source: "spend".into(),
            requested: None,
            source_granularity: None,
            nearest: Vec::new(),
        };

        let message = message_naming(error, &["spend"]);
        assert!(!message.contains("None"), "{message}");
    }

    #[test]
    fn a_child_that_lacks_no_name_says_so_in_the_nearest_list() {
        let error = PlanError::GrainNotServable {
            path: FramePath(vec!["all_spend3".into(), "all_spend3.spend_g3".into()]),
            source: "all_spend3.spend_g3".into(),
            requested: Some(Granularity::Day),
            source_granularity: Some(Granularity::Month),
            nearest: vec![
                ("all_spend3.spend_g3.g3_monthly".into(), Vec::new()),
                ("all_spend3.spend_g3.g3_daily".into(), vec!["budget".into()]),
            ],
        };

        let message = message_naming(error, &[]);
        assert!(
            message.contains(
                "nearest: `all_spend3.spend_g3.g3_monthly` lacks no name; \
                 `all_spend3.spend_g3.g3_daily` lacks `budget`"
            ),
            "{message}"
        );
    }

    #[test]
    fn a_predicate_type_error_gives_typing_s_error_as_its_cause() {
        let typing = TypeError::ParameterRefused {
            name: "date_trunc",
            parameter: "timezone",
        };
        let error = PlanError::PredicateTypeError {
            error: typing.clone(),
        };
        let cause = error.cause().map(ToString::to_string);
        assert_eq!(cause, Some(typing.to_string()));
    }
}
