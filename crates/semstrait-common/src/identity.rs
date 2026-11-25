use uuid::Uuid;

pub type EntityId = String;

pub const ENTITY_ROOT_NAMESPACE: Uuid = Uuid::from_u128(0x3b1e_9d44_7c25_5f86_b9a3_0e17_4d62_8c5f);

pub const MODEL_ROOT_NAMESPACE: Uuid = Uuid::from_u128(0xb329_1129_820b_5d6f_9079_bc1f_79f5_b8e0);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum EntityNamespace {
    DataKind,
    Dimension,
    Measure,
    Metric,
    Relationship,
    InterfaceMember,

    Source,
    Binding,
    Interface,
    Semantic,
    Model,
    Keyset,
}

impl EntityNamespace {
    const fn label(self) -> &'static str {
        match self {
            Self::DataKind => "data_kind",
            Self::Dimension => "dimension",
            Self::Measure => "measure",
            Self::Metric => "metric",
            Self::Relationship => "relationship",
            Self::InterfaceMember => "interface_member",
            Self::Source => "source",
            Self::Binding => "binding",
            Self::Interface => "interface",
            Self::Semantic => "semantic",
            Self::Model => "model",
            Self::Keyset => "keyset",
        }
    }

    pub const ALL: &'static [Self] = &[
        Self::DataKind,
        Self::Dimension,
        Self::Measure,
        Self::Metric,
        Self::Relationship,
        Self::InterfaceMember,
        Self::Source,
        Self::Binding,
        Self::Interface,
        Self::Semantic,
        Self::Model,
        Self::Keyset,
    ];
}

#[must_use]
pub fn entity_namespace(kind: EntityNamespace) -> Uuid {
    derive_uuid_v5(&ENTITY_ROOT_NAMESPACE, kind.label().as_bytes())
}

#[must_use]
pub fn derive_uuid_v5(namespace: &Uuid, seed: &[u8]) -> Uuid {
    Uuid::new_v5(namespace, seed)
}

#[must_use]
pub fn mint_entity_id(namespace: &Uuid, seed: &[u8]) -> EntityId {
    derive_uuid_v5(namespace, seed).to_string()
}

#[must_use]
pub fn mint_named_entity_id(kind: EntityNamespace, scope_qualified_name: &str) -> EntityId {
    mint_entity_id(&entity_namespace(kind), scope_qualified_name.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_root_namespace_matches_its_documented_derivation() {
        assert_eq!(
            MODEL_ROOT_NAMESPACE,
            Uuid::new_v5(&Uuid::NAMESPACE_URL, b"urn:semstrait:model-root"),
            "MODEL_ROOT_NAMESPACE no longer matches its documented derivation"
        );
    }

    #[test]
    fn entity_root_namespace_is_pinned() {
        assert_eq!(
            ENTITY_ROOT_NAMESPACE.to_string(),
            "3b1e9d44-7c25-5f86-b9a3-0e174d628c5f"
        );
    }

    #[test]
    fn every_class_is_pinned_and_pairwise_distinct() {
        let expected: [(EntityNamespace, &str, &str); 12] = [
            (
                EntityNamespace::DataKind,
                "data_kind",
                "ecff2dd9-7f70-5ee8-b94a-8429fc7e1c0f",
            ),
            (
                EntityNamespace::Dimension,
                "dimension",
                "91845416-af9e-5b32-a6db-eeb4645f13f2",
            ),
            (
                EntityNamespace::Measure,
                "measure",
                "7653b79c-3387-5573-b69b-3fb42024e42e",
            ),
            (
                EntityNamespace::Metric,
                "metric",
                "627dd0a0-e73c-5b6f-8aa4-147c1c25108f",
            ),
            (
                EntityNamespace::Relationship,
                "relationship",
                "21181b13-27a2-5d40-a7f7-2a31bc248253",
            ),
            (
                EntityNamespace::InterfaceMember,
                "interface_member",
                "f2b2cef8-58e2-57e0-ad7c-6d7bd67ca306",
            ),
            (
                EntityNamespace::Source,
                "source",
                "4277c498-4e97-5283-903c-ec2279443e7a",
            ),
            (
                EntityNamespace::Binding,
                "binding",
                "8bedefae-f592-5aaa-b67b-4f38940e0152",
            ),
            (
                EntityNamespace::Interface,
                "interface",
                "8b7b2be4-fc99-5d4e-9bed-656f5dc5c9aa",
            ),
            (
                EntityNamespace::Semantic,
                "semantic",
                "bce6cadf-4ec2-5410-8786-d60c9f2e2b52",
            ),
            (
                EntityNamespace::Model,
                "model",
                "4172dddd-3970-5c82-acf3-c25c24468150",
            ),
            (
                EntityNamespace::Keyset,
                "keyset",
                "08a960db-3f1e-52ec-af57-f6d6ddc6cd89",
            ),
        ];
        assert_eq!(
            expected.len(),
            EntityNamespace::ALL.len(),
            "a class was added without pinning its namespace"
        );
        for (kind, label, pinned) in expected {
            assert_eq!(
                entity_namespace(kind).to_string(),
                pinned,
                "{kind:?}'s namespace moved — every id in that class re-keys"
            );
            assert_eq!(
                entity_namespace(kind),
                Uuid::new_v5(&ENTITY_ROOT_NAMESPACE, label.as_bytes()),
                "{kind:?}'s label drifted from `{label}`"
            );
        }
        for (i, a) in EntityNamespace::ALL.iter().enumerate() {
            for b in &EntityNamespace::ALL[i + 1..] {
                assert_ne!(
                    entity_namespace(*a),
                    entity_namespace(*b),
                    "{a:?} and {b:?} must not share a namespace"
                );
            }
        }
    }

    #[test]
    fn a_scope_qualified_name_mints_a_pinned_id() {
        for (kind, seed, pinned) in [
            (
                EntityNamespace::DataKind,
                "orders",
                "cd3e4c16-3f42-5527-9ebf-35556686870a",
            ),
            (
                EntityNamespace::Dimension,
                "region",
                "f65ecd2b-c70b-5bec-8016-535387ea2ce5",
            ),
            (
                EntityNamespace::Measure,
                "revenue",
                "b329389e-9648-54c8-8729-3d31610c3d90",
            ),
            (
                EntityNamespace::Metric,
                "conversion",
                "8d8b373c-f1db-5f68-800e-f1da37dee15a",
            ),
            (
                EntityNamespace::Relationship,
                "orders_to_customers",
                "a67a3d70-bc88-5d46-8da3-e06d1bc538cf",
            ),
            (
                EntityNamespace::InterfaceMember,
                "orders.dimensions.region",
                "e4b46afd-b2fb-55cc-979b-0dbe527f0420",
            ),
        ] {
            assert_eq!(mint_named_entity_id(kind, seed), pinned, "{kind:?}");
        }
    }

    #[test]
    fn a_scope_local_name_seeds_a_distinct_id_from_a_root_one() {
        let root_level = mint_named_entity_id(EntityNamespace::Relationship, "orders_to_customers");
        let joinset_local = mint_named_entity_id(
            EntityNamespace::Relationship,
            "shipments.orders_to_customers",
        );
        assert_ne!(
            root_level, joinset_local,
            "a Joinset-local relationship re-using a root-level name must not \
             collide with it (ADR 0029)"
        );
        assert_eq!(
            joinset_local, "ac1c50ff-3cd3-5d41-8787-cbd783c8ac34",
            "the scope-qualified seed is `<joinset>.<name>`"
        );
    }

    #[test]
    fn one_class_covers_four_interface_collections_because_the_path_carries_one() {
        assert_ne!(
            mint_named_entity_id(EntityNamespace::InterfaceMember, "orders.dimensions.region"),
            mint_named_entity_id(EntityNamespace::InterfaceMember, "orders.filters.region"),
        );
        assert_ne!(
            mint_named_entity_id(EntityNamespace::InterfaceMember, "orders.dimensions.region"),
            mint_named_entity_id(
                EntityNamespace::InterfaceMember,
                "shipments.dimensions.region"
            ),
        );
    }

    #[test]
    fn the_same_name_in_two_classes_mints_two_ids() {
        assert_ne!(
            mint_named_entity_id(EntityNamespace::Dimension, "volume"),
            mint_named_entity_id(EntityNamespace::Measure, "volume"),
        );
    }

    #[test]
    fn the_mint_is_deterministic_and_seed_sensitive() {
        let ns = entity_namespace(EntityNamespace::Source);
        assert_eq!(mint_entity_id(&ns, b"seed"), mint_entity_id(&ns, b"seed"));
        assert_ne!(mint_entity_id(&ns, b"a"), mint_entity_id(&ns, b"b"));
    }

    #[test]
    fn a_minted_id_is_canonical_uuid_text_of_version_five() {
        let id = mint_named_entity_id(EntityNamespace::Model, "x");
        assert_eq!(id.len(), 36, "canonical hyphenated UUID is 36 chars");
        let hyphens: Vec<usize> = id.match_indices('-').map(|(i, _)| i).collect();
        assert_eq!(hyphens, vec![8, 13, 18, 23], "hyphen positions");
        assert_eq!(id, id.to_lowercase(), "canonical form is lowercase");
        let parsed = Uuid::parse_str(&id).expect("must re-parse as a UUID");
        assert_eq!(parsed.get_version_num(), 5, "the lane holds v5 throughout");
        assert!(!parsed.is_nil());
    }
}
