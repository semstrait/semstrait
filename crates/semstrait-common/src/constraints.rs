#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MeasureConstraints {
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub dimensions: Option<DimensionConstraints>,

    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub aggregations: Option<AggregationConstraints>,
}

impl MeasureConstraints {
    pub fn none() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.dimensions.is_none() && self.aggregations.is_none()
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DimensionConstraints {
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Vec::is_empty")
    )]
    pub one_of: Vec<String>,

    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Vec::is_empty")
    )]
    pub none_of: Vec<String>,

    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Vec::is_empty")
    )]
    pub all: Vec<String>,
}

impl DimensionConstraints {
    pub fn none() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.one_of.is_empty() && self.none_of.is_empty() && self.all.is_empty()
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AggregationConstraints {
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Vec::is_empty")
    )]
    pub allowed: Vec<String>,

    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Vec::is_empty")
    )]
    pub prohibited: Vec<String>,
}

impl AggregationConstraints {
    pub fn none() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.allowed.is_empty() && self.prohibited.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measure_constraints_none_is_empty() {
        let m = MeasureConstraints::none();
        assert!(m.is_empty());
        assert!(m.dimensions.is_none());
        assert!(m.aggregations.is_none());
    }

    #[test]
    fn measure_constraints_with_only_dimensions_is_not_empty() {
        let m = MeasureConstraints {
            dimensions: Some(DimensionConstraints::none()),
            aggregations: None,
        };
        assert!(!m.is_empty());
    }

    #[test]
    fn measure_constraints_with_only_aggregations_is_not_empty() {
        let m = MeasureConstraints {
            dimensions: None,
            aggregations: Some(AggregationConstraints::none()),
        };
        assert!(!m.is_empty());
    }

    #[test]
    fn dimension_constraints_none_is_empty() {
        let d = DimensionConstraints::none();
        assert!(d.is_empty());
        assert!(d.one_of.is_empty());
        assert!(d.none_of.is_empty());
        assert!(d.all.is_empty());
    }

    #[test]
    fn dimension_constraints_with_one_of_is_not_empty() {
        let d = DimensionConstraints {
            one_of: vec!["date".into()],
            none_of: Vec::new(),
            all: Vec::new(),
        };
        assert!(!d.is_empty());
    }

    #[test]
    fn dimension_constraints_with_none_of_is_not_empty() {
        let d = DimensionConstraints {
            one_of: Vec::new(),
            none_of: vec!["user".into()],
            all: Vec::new(),
        };
        assert!(!d.is_empty());
    }

    #[test]
    fn dimension_constraints_with_all_is_not_empty() {
        let d = DimensionConstraints {
            one_of: Vec::new(),
            none_of: Vec::new(),
            all: vec!["region".into()],
        };
        assert!(!d.is_empty());
    }

    #[test]
    fn aggregation_constraints_none_is_empty() {
        let a = AggregationConstraints::none();
        assert!(a.is_empty());
    }

    #[test]
    fn aggregation_constraints_with_allowed_is_not_empty() {
        let a = AggregationConstraints {
            allowed: vec!["SUM".into()],
            prohibited: Vec::new(),
        };
        assert!(!a.is_empty());
    }

    #[test]
    fn aggregation_constraints_with_prohibited_is_not_empty() {
        let a = AggregationConstraints {
            allowed: Vec::new(),
            prohibited: vec!["COUNT_DISTINCT".into()],
        };
        assert!(!a.is_empty());
    }

    #[test]
    fn aggregation_constraints_accept_uppercase_tokens() {
        let a = AggregationConstraints {
            allowed: vec!["COUNT_DISTINCT".into(), "SUM".into()],
            prohibited: Vec::new(),
        };
        assert!(a.allowed.contains(&"COUNT_DISTINCT".to_string()));
        assert!(a.allowed.contains(&"SUM".to_string()));
    }

    #[test]
    fn non_exhaustive_struct_literal_construction_compiles_in_test() {
        let _ = MeasureConstraints {
            dimensions: None,
            aggregations: None,
        };
        let _ = DimensionConstraints {
            one_of: Vec::new(),
            none_of: Vec::new(),
            all: Vec::new(),
        };
        let _ = AggregationConstraints {
            allowed: Vec::new(),
            prohibited: Vec::new(),
        };
    }

    #[cfg(feature = "serde")]
    #[test]
    fn measure_constraints_serde_roundtrip() {
        let m = MeasureConstraints {
            dimensions: Some(DimensionConstraints {
                one_of: vec!["date".into()],
                none_of: Vec::new(),
                all: vec!["region".into()],
            }),
            aggregations: Some(AggregationConstraints {
                allowed: vec!["SUM".into()],
                prohibited: Vec::new(),
            }),
        };
        let json = serde_json::to_string(&m).unwrap();
        let back: MeasureConstraints = serde_json::from_str(&json).unwrap();
        assert_eq!(m, back);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn measure_constraints_serde_skips_empty_optionals() {
        let m = MeasureConstraints {
            dimensions: Some(DimensionConstraints {
                one_of: vec!["date".into()],
                none_of: Vec::new(),
                all: Vec::new(),
            }),
            aggregations: None,
        };
        let json = serde_json::to_string(&m).unwrap();
        assert!(!json.contains("none_of"));
        assert!(!json.contains("all"));
        assert!(!json.contains("aggregations"));
    }
}
