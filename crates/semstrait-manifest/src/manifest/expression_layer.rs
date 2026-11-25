use serde::{Deserialize, Serialize};

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExprLayer {
    Scalar,
    Aggregate,
    PostAggregate,
}

impl ExprLayer {
    #[must_use]
    pub fn of_node(is_aggregate: bool, child_layers: impl IntoIterator<Item = ExprLayer>) -> Self {
        if is_aggregate {
            Self::Aggregate
        } else if child_layers.into_iter().any(|layer| layer != Self::Scalar) {
            Self::PostAggregate
        } else {
            Self::Scalar
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_round_trips_every_variant() {
        for layer in [
            ExprLayer::Scalar,
            ExprLayer::Aggregate,
            ExprLayer::PostAggregate,
        ] {
            let json = serde_json::to_string(&layer).expect("serialize");
            let back: ExprLayer = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back, layer);
        }
    }

    #[test]
    fn serde_wire_form_is_snake_case() {
        assert_eq!(
            serde_json::to_string(&ExprLayer::PostAggregate).unwrap(),
            "\"post_aggregate\""
        );
        assert_eq!(
            serde_json::to_string(&ExprLayer::Scalar).unwrap(),
            "\"scalar\""
        );
    }
}
