use std::error::Error;
use std::fmt::Display;

use semstrait_common::{DataKindName, Granularity, SemanticsName};
use semstrait_common::{Diagnose, Severity};
use semstrait_model::model::serde::{SemanticExpr, Unresolved};
use semstrait_model::UnresolvedSemanticExpr;
use semstrait_planner::{
    DimensionModifier, Direction, OrderKey, Request, RequestContext, Selection, Timezone,
    UnknownTimezone, Weekday,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestPart {
    from: Option<DataKindName>,
    select: Vec<Selected>,
    #[serde(
        rename = "where",
        default,
        skip_serializing_if = "Option::is_none",
        with = "Unresolved::<SemanticExpr>"
    )]
    predicate: Option<UnresolvedSemanticExpr>,
    #[serde(default)]
    filters: Vec<SemanticsName>,
    #[serde(default)]
    order_by: Vec<Key>,
    limit: Option<u64>,
    offset: Option<u64>,
    #[serde(default)]
    context: ContextPart,
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
pub enum Selected {
    Field(SemanticsName),
    Dimension {
        dimension: SemanticsName,
        granularity: Option<Granularity>,
    },
    Measure {
        measure: SemanticsName,
    },
    Metric {
        metric: SemanticsName,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Key {
    name: SemanticsName,
    direction: Order,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Order {
    Asc,
    Desc,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextPart {
    timezone: Option<String>,
    week_start: Option<WeekStart>,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WeekStart {
    Monday,
    Sunday,
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Error)]
pub enum WireError {
    #[error("the Request document is refused: {message}")]
    Document { message: String },
    #[error("{0}")]
    Timezone(#[source] UnknownTimezone),
}

impl WireError {
    fn document(error: impl Display) -> Self {
        Self::Document {
            message: error.to_string(),
        }
    }
}

impl Diagnose for WireError {
    fn message(&self) -> String {
        self.to_string()
    }

    fn severity_default(&self) -> Severity {
        Severity::Error
    }

    fn cause(&self) -> Option<&(dyn Error + 'static)> {
        self.source()
    }
}

impl RequestPart {
    pub fn read_yaml(text: &str) -> Result<(Request, RequestContext), WireError> {
        serde_yaml::from_str::<Self>(text)
            .map_err(WireError::document)?
            .read()
    }

    pub fn read_json(text: &str) -> Result<(Request, RequestContext), WireError> {
        serde_json::from_str::<Self>(text)
            .map_err(WireError::document)?
            .read()
    }

    fn read(self) -> Result<(Request, RequestContext), WireError> {
        let request = Request::new(self.select.into_iter().map(Selection::from).collect())
            .with_filters(self.filters)
            .with_order_by(self.order_by.into_iter().map(OrderKey::from).collect());
        let request = self.from.into_iter().fold(request, Request::with_from);
        let request = self.limit.into_iter().fold(request, Request::with_limit);
        let request = self.offset.into_iter().fold(request, Request::with_offset);
        let request = self
            .predicate
            .into_iter()
            .fold(request, Request::with_predicate);
        Ok((request, self.context.read()?))
    }
}

impl ContextPart {
    fn read(self) -> Result<RequestContext, WireError> {
        let week_start = self.week_start.map(Weekday::from);
        let mut context = (week_start.into_iter())
            .fold(RequestContext::default(), RequestContext::with_week_start);
        if let Some(timezone) = &self.timezone {
            context = context.with_timezone(Timezone::new(timezone).map_err(WireError::Timezone)?);
        }
        Ok(context)
    }
}

impl From<Selected> for Selection {
    fn from(selected: Selected) -> Self {
        match selected {
            Selected::Field(name) => Self::Field(name),
            Selected::Dimension {
                dimension,
                granularity,
            } => Self::Dimension {
                name: dimension,
                modifier: granularity.map(DimensionModifier::Granularity),
            },
            Selected::Measure { measure } => Self::Measure { name: measure },
            Selected::Metric { metric } => Self::Metric { name: metric },
        }
    }
}

impl From<Key> for OrderKey {
    fn from(key: Key) -> Self {
        Self::new(key.name, key.direction.into())
    }
}

impl From<Order> for Direction {
    fn from(order: Order) -> Self {
        match order {
            Order::Asc => Self::Asc,
            Order::Desc => Self::Desc,
        }
    }
}

impl From<WeekStart> for Weekday {
    fn from(week_start: WeekStart) -> Self {
        match week_start {
            WeekStart::Monday => Self::Monday,
            WeekStart::Sunday => Self::Sunday,
        }
    }
}
