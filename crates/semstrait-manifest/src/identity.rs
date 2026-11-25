use std::fmt::Debug;

use semstrait_catalog::FileFormat;
use semstrait_common::identity::{
    derive_uuid_v5, entity_namespace, mint_entity_id, mint_named_entity_id, EntityId,
    EntityNamespace,
};
use semstrait_ir::SemanticRole;
use serde::Serialize;
use uuid::Uuid;

use crate::manifest::expression::{ExprRoot, PhysicalExprId};

pub const EXPR_NAMESPACE: Uuid = Uuid::from_u128(0x6f9c_2e7a_4b13_5d8e_a1c0_7f42_9e36_b58d);

fn seed_bytes<T: Serialize + Debug>(value: &T) -> Vec<u8> {
    serde_json::to_vec(value).unwrap_or_else(|_| format!("{value:?}").into_bytes())
}

#[must_use]
pub fn mint_model_id(model_hash: &[u8; 32]) -> EntityId {
    mint_entity_id(&entity_namespace(EntityNamespace::Model), model_hash)
}

#[must_use]
pub fn mint_table_source_id(locator: &str) -> EntityId {
    let seed = seed_bytes(&("table", locator));
    mint_entity_id(&entity_namespace(EntityNamespace::Source), &seed)
}

#[must_use]
pub fn mint_path_source_id(base_path: &str, format: Option<FileFormat>) -> EntityId {
    let seed = seed_bytes(&(base_path, format));
    mint_entity_id(&entity_namespace(EntityNamespace::Source), &seed)
}

#[must_use]
pub fn mint_reduction_leaf_id(dataset_id: &str, physical_source_id: &str) -> EntityId {
    let seed = seed_bytes(&(dataset_id, physical_source_id));
    mint_entity_id(&entity_namespace(EntityNamespace::DataKind), &seed)
}

#[must_use]
pub fn mint_signature_id(semantic_source_id: &str, definition_id: &str) -> EntityId {
    let seed = seed_bytes(&(semantic_source_id, definition_id));
    mint_entity_id(&entity_namespace(EntityNamespace::Binding), &seed)
}

#[must_use]
pub fn mint_semantic_id(role: SemanticRole, name: &str) -> EntityId {
    let namespace = match role {
        SemanticRole::Dimension => EntityNamespace::Dimension,
        SemanticRole::Measure => EntityNamespace::Measure,
        SemanticRole::Metric => EntityNamespace::Metric,
        _ => EntityNamespace::Semantic,
    };
    mint_named_entity_id(namespace, name)
}

#[must_use]
pub fn mint_keyset_id(semantic_source_id: &str, name: &str) -> EntityId {
    let seed = seed_bytes(&(semantic_source_id, name));
    mint_entity_id(&entity_namespace(EntityNamespace::Keyset), &seed)
}

#[must_use]
pub fn mint_semantic_source_id(qualified_name: &str) -> EntityId {
    mint_named_entity_id(EntityNamespace::DataKind, qualified_name)
}

#[must_use]
pub fn mint_expr_id(owner_id: &str, root: ExprRoot) -> PhysicalExprId {
    let seed = seed_bytes(&(owner_id, root));
    PhysicalExprId::new(derive_uuid_v5(&EXPR_NAMESPACE, &seed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_expression_namespace_is_pinned() {
        assert_eq!(
            EXPR_NAMESPACE.to_string(),
            "6f9c2e7a-4b13-5d8e-a1c0-7f429e36b58d"
        );
        assert_eq!(
            derive_uuid_v5(&EXPR_NAMESPACE, b"probe").to_string(),
            "3d9630f6-ceee-5561-beae-ce6f89f8320b"
        );
        for kind in EntityNamespace::ALL {
            assert_ne!(EXPR_NAMESPACE, entity_namespace(*kind), "{kind:?}");
        }
    }

    #[test]
    fn model_id_is_stable_and_keyed_on_hash() {
        let a = mint_model_id(&[0u8; 32]);
        let b = mint_model_id(&[0u8; 32]);
        assert_eq!(a, b, "same model_hash ⇒ same model_id");
        let mut other = [0u8; 32];
        other[0] = 1;
        assert_ne!(a, mint_model_id(&other), "different hash ⇒ different id");
    }

    #[test]
    fn source_id_keys_on_every_seed_field() {
        let base = mint_table_source_id("warehouse.orders");
        assert_eq!(base, mint_table_source_id("warehouse.orders"));
        assert_ne!(base, mint_table_source_id("warehouse.other"));
        let orders = mint_path_source_id("s3://lake/orders/", Some(FileFormat::Parquet));
        assert_ne!(
            orders,
            mint_path_source_id("s3://lake/orders/", Some(FileFormat::Csv))
        );
        assert_ne!(
            orders,
            mint_path_source_id("s3://lake/returns/", Some(FileFormat::Parquet))
        );
    }

    #[test]
    fn the_source_ids_are_pinned() {
        assert_eq!(
            mint_table_source_id("warehouse.orders"),
            "bf575591-d8c6-59c7-b3aa-5fbdf5673b12"
        );
        assert_eq!(
            mint_path_source_id("s3://lake/orders/", Some(FileFormat::Parquet)),
            "e96bfa77-0a5a-5377-8360-418e4659395a"
        );
    }

    #[test]
    fn a_reduction_leaf_id_keys_on_its_dataset_and_its_physical_source() {
        let leaf = mint_reduction_leaf_id("dk.sales", "src.eu");
        assert_eq!(leaf, mint_reduction_leaf_id("dk.sales", "src.eu"));
        assert_ne!(leaf, mint_reduction_leaf_id("dk.sales", "src.us"));
        assert_ne!(leaf, mint_reduction_leaf_id("src.eu", "dk.sales"));
    }

    #[test]
    fn signature_id_is_exactly_the_ordered_pair_it_is_unique_over() {
        let base = mint_signature_id("dk.orders", "sem.region");
        assert_eq!(base, mint_signature_id("dk.orders", "sem.region"));
        assert_ne!(base, mint_signature_id("dk.returns", "sem.region"));
        assert_ne!(base, mint_signature_id("dk.orders", "sem.channel"));
        assert_ne!(
            base,
            mint_signature_id("sem.region", "dk.orders"),
            "the seed is `(source, definition)` in that order"
        );
    }

    #[test]
    fn a_semantic_id_is_the_bare_name_under_its_roles_namespace() {
        let pooled = mint_semantic_id(SemanticRole::Dimension, "region");
        assert_eq!(
            pooled,
            mint_named_entity_id(EntityNamespace::Dimension, "region"),
            "the seed is the bare name a pooled declaration would use"
        );
        assert_ne!(pooled, mint_semantic_id(SemanticRole::Measure, "region"));
        assert_ne!(pooled, mint_semantic_id(SemanticRole::Metric, "region"));
        assert_ne!(
            mint_semantic_id(SemanticRole::Filter, "recent"),
            mint_semantic_id(SemanticRole::Filter, "stale")
        );
    }

    #[test]
    fn keyset_id_seeds_the_owning_source_and_the_name() {
        let base = mint_keyset_id("dk.orders", "natural");
        assert_eq!(base, mint_keyset_id("dk.orders", "natural"));
        assert_ne!(base, mint_keyset_id("dk.orders", "surrogate"));
        assert_ne!(base, mint_keyset_id("dk.returns", "natural"));
    }

    #[test]
    fn keyset_id_carries_no_membership() {
        assert_eq!(
            mint_keyset_id("dk.orders", "default"),
            mint_keyset_id("dk.orders", "default")
        );
    }

    #[test]
    fn a_semantic_source_id_is_its_scope_qualified_name() {
        let top = mint_semantic_source_id("orders");
        assert_eq!(
            top,
            mint_named_entity_id(EntityNamespace::DataKind, "orders"),
            "the seed is the bare name the authoring stage would use, through the same mint"
        );
        assert_ne!(
            mint_semantic_source_id("all_orders.recent"),
            mint_semantic_source_id("all_returns.recent")
        );
        assert_ne!(mint_semantic_source_id("all_orders.recent"), {
            mint_semantic_source_id("recent")
        });
    }

    #[test]
    fn distinct_minters_avoid_collision_on_shared_content() {
        let src = mint_table_source_id("x");
        let semantic = mint_semantic_id(SemanticRole::Filter, "x");
        assert_ne!(src, semantic);
    }

    mod expr_id {
        use super::super::mint_expr_id;
        use crate::manifest::expression::ExprRoot;

        #[test]
        fn the_expression_id_is_minted_from_the_owner_and_the_root_marker() {
            for (owner, expected) in [
                (
                    "cc2ebad2-649d-5ef6-8d87-e64b6e911518",
                    "253641c7-819a-5b60-a3dc-c276bfceb36d",
                ),
                (
                    "393c8b97-2f6b-5f64-95a2-397c723e324b",
                    "cb8b6c6b-b311-5484-9a64-5966edb7851d",
                ),
                (
                    "2147ff70-826e-545d-bf02-09d9bf52cab9",
                    "15184182-69e5-55a9-9292-b09b5731f128",
                ),
                (
                    "88d12c7b-16c9-570c-8bc3-ca37eacd221a",
                    "09712050-49b8-534f-98f1-34d64cf76e3c",
                ),
            ] {
                assert_eq!(
                    mint_expr_id(owner, ExprRoot::SignatureExpression).to_string(),
                    expected,
                    "the expression entry of {owner}"
                );
            }
            let revenue = "cc2ebad2-649d-5ef6-8d87-e64b6e911518";
            let id = mint_expr_id(revenue, ExprRoot::SignatureExpression);
            assert_ne!(
                id,
                mint_expr_id(
                    "393c8b97-2f6b-5f64-95a2-397c723e324b",
                    ExprRoot::SignatureExpression
                ),
                "another owner mints another id"
            );
            for marker in [
                ExprRoot::SignatureFilter,
                ExprRoot::JoinsetOn,
                ExprRoot::JoinsetFilter,
            ] {
                assert_ne!(
                    id,
                    mint_expr_id(revenue, marker),
                    "another marker mints another id: {marker:?}"
                );
            }
        }

        #[test]
        fn result_is_a_versioned_uuidv5() {
            let id = mint_expr_id("sig.a", ExprRoot::SignatureExpression);
            assert_eq!(id.uuid().get_version_num(), 5);
        }
    }
}
