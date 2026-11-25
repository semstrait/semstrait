use semstrait_common::{Granularity, SemanticsName};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "schemars",
    schemars(
        description = "The `temporal:` block: a shape such as `events` or `snapshot`, plus an optional `granularity:`."
    )
)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct TemporalShape {
    #[serde(flatten)]
    pub kind: TemporalShapeKind,

    #[cfg_attr(
        feature = "schemars",
        schemars(
            description = "The Dataset's time granularity; required when `temporal:` is authored on a leaf; on a composition, only a constituent of another may declare it."
        )
    )]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub granularity: Option<Granularity>,
}

impl TemporalShape {
    pub fn new(kind: TemporalShapeKind) -> Self {
        Self {
            kind,
            granularity: None,
        }
    }

    pub fn with_granularity(mut self, granularity: Granularity) -> Self {
        self.granularity = Some(granularity);
        self
    }

    pub fn timeseries(
        occurred_at: impl Into<SemanticsName>,
        granularity: Option<Granularity>,
    ) -> Self {
        Self {
            kind: TemporalShapeKind::Timeseries(TimeseriesBody {
                occurred_at: occurred_at.into(),
            }),
            granularity,
        }
    }

    pub fn events(event_time: impl Into<SemanticsName>, granularity: Option<Granularity>) -> Self {
        Self {
            kind: TemporalShapeKind::Events(EventsBody {
                event_time: event_time.into(),
            }),
            granularity,
        }
    }

    pub fn snapshot(
        snapshotted_at: impl Into<SemanticsName>,
        granularity: Option<Granularity>,
    ) -> Self {
        Self {
            kind: TemporalShapeKind::Snapshot(SnapshotBody {
                snapshotted_at: snapshotted_at.into(),
            }),
            granularity,
        }
    }

    pub fn scd(
        scd_type: ScdType,
        valid_from: impl Into<SemanticsName>,
        valid_to: impl Into<SemanticsName>,
        granularity: Option<Granularity>,
    ) -> Self {
        Self {
            kind: TemporalShapeKind::Scd(ScdBody {
                scd_type,
                valid_from: valid_from.into(),
                valid_to: valid_to.into(),
            }),
            granularity,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum TemporalShapeKind {
    Timeseries(TimeseriesBody),
    Events(EventsBody),
    Snapshot(SnapshotBody),
    Scd(ScdBody),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct TimeseriesBody {
    pub occurred_at: SemanticsName,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct EventsBody {
    pub event_time: SemanticsName,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct SnapshotBody {
    pub snapshotted_at: SemanticsName,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct ScdBody {
    pub scd_type: ScdType,
    pub valid_from: SemanticsName,
    pub valid_to: SemanticsName,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ScdType {
    Type1,
    Type2,
}
