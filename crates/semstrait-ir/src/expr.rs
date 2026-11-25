use serde::{Deserialize, Serialize};

use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{Expr, Leaf};

pub type PhysicalExpr = Expr<Parameter>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Parameter {
    pub key: ParameterKey,
    pub data_type: DataType,
}

impl Leaf for Parameter {}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterKey {
    RequestDimensionsMinusTemporal,
    RequestTemporalAxis,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_carries_data_type_for_each_canonical_kind() {
        for dt in [
            DataType::Integer,
            DataType::Double,
            DataType::String,
            DataType::Boolean,
            DataType::Date,
            DataType::Binary,
        ] {
            let p = Parameter {
                key: ParameterKey::RequestTemporalAxis,
                data_type: dt.clone(),
            };
            assert_eq!(p.data_type, dt);
        }
    }

    #[test]
    fn parameter_key_wire_form_is_bare_snake_case() {
        assert_eq!(
            serde_json::to_string(&ParameterKey::RequestTemporalAxis).unwrap(),
            r#""request_temporal_axis""#
        );
        assert_eq!(
            serde_json::to_string(&ParameterKey::RequestDimensionsMinusTemporal).unwrap(),
            r#""request_dimensions_minus_temporal""#
        );
    }

    #[test]
    fn parameter_round_trips_for_each_key() {
        for key in [
            ParameterKey::RequestTemporalAxis,
            ParameterKey::RequestDimensionsMinusTemporal,
        ] {
            let p = Parameter {
                key,
                data_type: DataType::Date,
            };
            let json = serde_json::to_string(&p).expect("serialize");
            let back: Parameter = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(p, back);
        }
    }
}
