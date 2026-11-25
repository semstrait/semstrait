use semstrait_common::EntityId;
use serde::{Deserialize, Serialize};

use super::expression::PhysicalExprId;

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticSignature {
    pub id: EntityId,

    pub definition_id: EntityId,

    pub expression_id: PhysicalExprId,

    pub filter_expr_id: Option<PhysicalExprId>,

    pub keysets: Vec<EntityId>,
}

impl SemanticSignature {
    #[must_use]
    pub fn new(
        id: impl Into<EntityId>,
        definition_id: impl Into<EntityId>,
        expression_id: PhysicalExprId,
    ) -> Self {
        Self {
            id: id.into(),
            definition_id: definition_id.into(),
            expression_id,
            filter_expr_id: None,
            keysets: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_keysets(mut self, keysets: Vec<EntityId>) -> Self {
        self.keysets = keysets;
        self
    }

    #[must_use]
    pub fn with_filter(mut self, filter_expr_id: PhysicalExprId) -> Self {
        self.filter_expr_id = Some(filter_expr_id);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(raw: u128) -> PhysicalExprId {
        PhysicalExprId::new(uuid::Uuid::from_u128(raw))
    }

    #[test]
    fn a_new_signature_declares_no_filter_and_no_keyset() {
        let signature = SemanticSignature::new("sig.a", "sem.region", id(1));
        assert!(signature.filter_expr_id.is_none());
        assert!(signature.keysets.is_empty());
    }

    #[test]
    fn keysets_attach_through_the_builder() {
        let signature = SemanticSignature::new("sig.a", "sem.region", id(1))
            .with_keysets(vec!["ks.natural".to_owned()]);
        assert_eq!(signature.keysets, ["ks.natural".to_owned()]);
    }

    #[test]
    fn a_filter_root_attaches_through_the_builder() {
        let signature = SemanticSignature::new("sig.a", "sem.revenue", id(1)).with_filter(id(2));
        assert_eq!(signature.filter_expr_id, Some(id(2)));
        assert_ne!(signature.filter_expr_id, Some(signature.expression_id));
    }

    #[test]
    fn a_signature_round_trips_through_json() {
        let signature = SemanticSignature::new("sig.a", "sem.region", id(7))
            .with_filter(id(8))
            .with_keysets(vec!["ks.natural".to_owned()]);
        let json = serde_json::to_string(&signature).expect("serialize signature");
        let back: SemanticSignature = serde_json::from_str(&json).expect("deserialize signature");
        assert_eq!(back, signature);
    }
}
