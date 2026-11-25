use semstrait_common::{EntityId, Granularity};
use serde::{Deserialize, Serialize};

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScdType {
    Type1,
    Type2,
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalShape {
    pub temporal_type: TemporalType,
    pub granularity: Option<Granularity>,
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "content")]
pub enum TemporalType {
    Timeseries {
        occurred_at: EntityId,
    },
    Events {
        event_time: EntityId,
    },
    Snapshot {
        snapshotted_at: EntityId,
    },
    Scd {
        scd_type: ScdType,
        valid_from: EntityId,
        valid_to: EntityId,
    },
}

impl TemporalShape {
    #[must_use]
    pub fn new(temporal_type: TemporalType, granularity: Option<Granularity>) -> Self {
        Self {
            temporal_type,
            granularity,
        }
    }
}

impl TemporalType {
    #[must_use]
    pub fn rank(&self) -> u8 {
        match self {
            Self::Scd { .. } => 0,
            Self::Timeseries { .. } => 1,
            Self::Snapshot { .. } => 2,
            Self::Events { .. } => 3,
        }
    }

    #[must_use]
    pub fn timeseries(occurred_at: impl Into<EntityId>) -> Self {
        Self::Timeseries {
            occurred_at: occurred_at.into(),
        }
    }

    #[must_use]
    pub fn events(event_time: impl Into<EntityId>) -> Self {
        Self::Events {
            event_time: event_time.into(),
        }
    }

    #[must_use]
    pub fn snapshot(snapshotted_at: impl Into<EntityId>) -> Self {
        Self::Snapshot {
            snapshotted_at: snapshotted_at.into(),
        }
    }

    #[must_use]
    pub fn scd(
        scd_type: ScdType,
        valid_from: impl Into<EntityId>,
        valid_to: impl Into<EntityId>,
    ) -> Self {
        Self::Scd {
            scd_type,
            valid_from: valid_from.into(),
            valid_to: valid_to.into(),
        }
    }

    #[must_use]
    pub fn time_axis(&self) -> Option<&EntityId> {
        match self {
            Self::Timeseries { occurred_at } => Some(occurred_at),
            Self::Events { event_time } => Some(event_time),
            Self::Snapshot { snapshotted_at } => Some(snapshotted_at),
            Self::Scd { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variant_constructors_build_each_shape() {
        assert!(matches!(
            TemporalType::timeseries("sem.occurred_at"),
            TemporalType::Timeseries { occurred_at } if occurred_at == "sem.occurred_at"
        ));
        assert!(matches!(
            TemporalType::events("sem.event_time"),
            TemporalType::Events { event_time } if event_time == "sem.event_time"
        ));
        assert!(matches!(
            TemporalType::snapshot("sem.snap_at"),
            TemporalType::Snapshot { snapshotted_at } if snapshotted_at == "sem.snap_at"
        ));
        assert!(matches!(
            TemporalType::scd(ScdType::Type2, "sem.from", "sem.to"),
            TemporalType::Scd { scd_type: ScdType::Type2, valid_from, valid_to }
                if valid_from == "sem.from" && valid_to == "sem.to"
        ));
    }

    #[test]
    fn time_axis_is_the_one_dimension_each_shape_names_scd_none() {
        assert_eq!(
            TemporalType::timeseries("sem.occurred_at").time_axis(),
            Some(&EntityId::from("sem.occurred_at"))
        );
        assert_eq!(
            TemporalType::events("sem.event_time").time_axis(),
            Some(&EntityId::from("sem.event_time"))
        );
        assert_eq!(
            TemporalType::snapshot("sem.snap_at").time_axis(),
            Some(&EntityId::from("sem.snap_at"))
        );
        assert_eq!(
            TemporalType::scd(ScdType::Type2, "sem.from", "sem.to").time_axis(),
            None,
            "an Scd validity range is not one time axis"
        );
    }

    #[test]
    fn shape_carries_granularity_some_and_none() {
        let some = TemporalShape::new(TemporalType::timeseries("sem.t"), Some(Granularity::Day));
        assert_eq!(some.granularity, Some(Granularity::Day));
        let none = TemporalShape::new(TemporalType::events("sem.t"), None);
        assert_eq!(none.granularity, None);
    }

    #[test]
    fn temporal_type_adjacent_tag_wire_form() {
        let json = serde_json::to_string(&TemporalType::timeseries("sem.t")).expect("serialize");
        assert!(
            json.contains("\"type\":\"timeseries\""),
            "expected adjacent `type` tag, got {json}"
        );
        assert!(
            json.contains("\"content\""),
            "expected `content` envelope, got {json}"
        );

        let scd_json =
            serde_json::to_string(&TemporalType::scd(ScdType::Type1, "f", "t")).expect("serialize");
        assert!(
            scd_json.contains("\"type\":\"scd\""),
            "expected `scd` tag, got {scd_json}"
        );
    }

    #[test]
    fn temporal_type_serde_round_trip_all_variants() {
        for t in [
            TemporalType::timeseries("sem.a"),
            TemporalType::events("sem.b"),
            TemporalType::snapshot("sem.c"),
            TemporalType::scd(ScdType::Type1, "sem.d", "sem.e"),
            TemporalType::scd(ScdType::Type2, "sem.f", "sem.g"),
        ] {
            let json = serde_json::to_string(&t).expect("serialize");
            let back: TemporalType = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(t, back);
        }
    }

    #[test]
    fn temporal_shape_serde_round_trip() {
        let shape = TemporalShape::new(
            TemporalType::scd(ScdType::Type2, "sem.from", "sem.to"),
            Some(Granularity::Month),
        );
        let json = serde_json::to_string(&shape).expect("serialize");
        let back: TemporalShape = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(shape, back);
    }
}
