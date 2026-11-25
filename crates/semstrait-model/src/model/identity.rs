use std::collections::BTreeMap;

use semstrait_common::diagnostic::Location;
use semstrait_common::identity::{
    mint_entity_id, mint_named_entity_id, EntityNamespace, MODEL_ROOT_NAMESPACE,
};
use semstrait_common::EntityId;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum IdentityProfile {
    #[default]
    ConvenienceGenerateMissing,
    StrictRequireProvided,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct MintLedger {
    missing_id: Vec<String>,
    claims: BTreeMap<(EntityNamespace, String), Vec<Location>>,
}

impl MintLedger {
    pub(crate) fn missing_ids(&self) -> &[String] {
        &self.missing_id
    }

    pub(crate) fn claims_in(
        &self,
        kind: EntityNamespace,
    ) -> impl Iterator<Item = (&str, &[Location])> {
        self.claims
            .range((kind, String::new())..)
            .take_while(move |((claimed, _), _)| *claimed == kind)
            .map(|((_, path), locations)| (path.as_str(), locations.as_slice()))
    }
}

pub(crate) fn resolve_entity_id(
    id: &mut Option<EntityId>,
    kind: EntityNamespace,
    profile: IdentityProfile,
    path: &str,
    location: &Location,
    ledger: &mut MintLedger,
) -> EntityId {
    ledger
        .claims
        .entry((kind, path.to_owned()))
        .or_default()
        .push(location.clone());
    if id.is_none() {
        if profile == IdentityProfile::StrictRequireProvided {
            ledger.missing_id.push(path.to_owned());
        }
        *id = Some(mint_named_entity_id(kind, path));
    }
    id.clone().expect("resolve_entity_id leaves id Some")
}

#[must_use]
pub(crate) fn interface_member_path(data_kind: &str, collection: &str, name: &str) -> String {
    format!("{data_kind}.{collection}.{name}")
}

#[must_use]
pub fn derive_root_id(name: &str) -> EntityId {
    mint_entity_id(&MODEL_ROOT_NAMESPACE, name.as_bytes())
}

pub(crate) fn validate_authored_id(raw: &str) -> Result<(), String> {
    let parsed = Uuid::parse_str(raw).map_err(|e| format!("not UUID text: {e}"))?;
    if parsed.is_nil() {
        return Err("nil UUID is not a valid entity id".to_owned());
    }
    if raw != parsed.hyphenated().to_string() {
        return Err("not canonical lowercase hyphenated form".to_owned());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_V7: &str = "0197cb6e-a63a-7d53-8d6f-35f4b1d67d4e";
    const VALID_V4: &str = "550e8400-e29b-41d4-a716-446655440000";
    const NIL: &str = "00000000-0000-0000-0000-000000000000";

    #[test]
    fn a_minted_id_passes_sr11() {
        for kind in EntityNamespace::ALL {
            let id = mint_named_entity_id(*kind, "some.name");
            assert!(
                validate_authored_id(&id).is_ok(),
                "{kind:?} minted an id SR-11 rejects: {id}"
            );
        }
    }

    #[test]
    fn accepts_canonical_v7() {
        assert!(validate_authored_id(VALID_V7).is_ok());
    }

    #[test]
    fn accepts_any_version_nibble() {
        assert!(validate_authored_id(VALID_V4).is_ok());
        assert!(validate_authored_id(&derive_root_id("anything")).is_ok());
        assert!(validate_authored_id(VALID_V7).is_ok());
    }

    #[test]
    fn rejects_uppercase() {
        assert!(validate_authored_id(&VALID_V7.to_uppercase()).is_err());
    }

    #[test]
    fn rejects_nil() {
        assert!(validate_authored_id(NIL).is_err());
    }

    #[test]
    fn rejects_non_uuid() {
        assert!(validate_authored_id("not-a-uuid").is_err());
    }

    #[test]
    fn root_derivation_is_pinned_to_frozen_values() {
        assert_eq!(
            derive_root_id("analytics-v1"),
            "c0176300-5341-5e46-8349-4ebc6f19747c"
        );
        assert_eq!(
            derive_root_id("analytics-v2"),
            "97909e45-6829-5bf5-aa2d-017dcd52b9e2"
        );
    }

    #[test]
    fn an_interface_member_path_carries_its_data_kind_and_collection() {
        assert_eq!(
            interface_member_path("orders", "dimensions", "region"),
            "orders.dimensions.region"
        );
        assert_ne!(
            mint_named_entity_id(
                EntityNamespace::InterfaceMember,
                &interface_member_path("orders", "dimensions", "region")
            ),
            mint_named_entity_id(
                EntityNamespace::InterfaceMember,
                &interface_member_path("orders", "filters", "region")
            ),
        );
    }

    fn nowhere() -> Location {
        Location::new(
            semstrait_common::diagnostic::SourceId::unknown(),
            semstrait_common::diagnostic::Span::unknown(),
        )
    }

    #[test]
    fn resolve_entity_id_reports_only_under_the_strict_profile() {
        let mut ledger = MintLedger::default();
        let mut absent = None;
        let minted = resolve_entity_id(
            &mut absent,
            EntityNamespace::InterfaceMember,
            IdentityProfile::ConvenienceGenerateMissing,
            "orders.dimensions.country",
            &nowhere(),
            &mut ledger,
        );
        assert_eq!(absent.as_deref(), Some(minted.as_str()));
        assert!(
            ledger.missing_ids().is_empty(),
            "convenience must mint silently"
        );

        let mut absent = None;
        resolve_entity_id(
            &mut absent,
            EntityNamespace::InterfaceMember,
            IdentityProfile::StrictRequireProvided,
            "orders.dimensions.country",
            &nowhere(),
            &mut ledger,
        );
        assert_eq!(ledger.missing_ids(), ["orders.dimensions.country"]);
        assert!(
            absent.is_some(),
            "strict still mints so the collection has a key"
        );
    }

    #[test]
    fn the_ledger_records_every_claim_under_its_class() {
        let mut ledger = MintLedger::default();
        for _ in 0..2 {
            resolve_entity_id(
                &mut None,
                EntityNamespace::DataKind,
                IdentityProfile::ConvenienceGenerateMissing,
                "orders",
                &nowhere(),
                &mut ledger,
            );
        }
        resolve_entity_id(
            &mut None,
            EntityNamespace::Dimension,
            IdentityProfile::ConvenienceGenerateMissing,
            "orders",
            &nowhere(),
            &mut ledger,
        );
        let data_kinds: Vec<(&str, usize)> = ledger
            .claims_in(EntityNamespace::DataKind)
            .map(|(path, locations)| (path, locations.len()))
            .collect();
        assert_eq!(data_kinds, [("orders", 2)]);
        let dimensions: Vec<(&str, usize)> = ledger
            .claims_in(EntityNamespace::Dimension)
            .map(|(path, locations)| (path, locations.len()))
            .collect();
        assert_eq!(dimensions, [("orders", 1)]);
    }

    #[test]
    fn resolve_entity_id_derives_from_the_path_and_the_class() {
        let mint = |kind, path| {
            let mut id = None;
            resolve_entity_id(
                &mut id,
                kind,
                IdentityProfile::ConvenienceGenerateMissing,
                path,
                &nowhere(),
                &mut MintLedger::default(),
            )
        };
        assert_eq!(
            mint(EntityNamespace::Dimension, "region"),
            mint(EntityNamespace::Dimension, "region")
        );
        assert_ne!(
            mint(EntityNamespace::Dimension, "region"),
            mint(EntityNamespace::Dimension, "territory")
        );
        assert_ne!(
            mint(EntityNamespace::Dimension, "volume"),
            mint(EntityNamespace::Measure, "volume")
        );
    }

    #[test]
    fn resolve_entity_id_leaves_an_authored_id_alone() {
        let mut ledger = MintLedger::default();
        let mut authored = Some(VALID_V7.to_owned());
        let resolved = resolve_entity_id(
            &mut authored,
            EntityNamespace::Dimension,
            IdentityProfile::StrictRequireProvided,
            "orders.dimensions.country",
            &nowhere(),
            &mut ledger,
        );
        assert_eq!(resolved, VALID_V7);
        assert!(ledger.missing_ids().is_empty());
    }

    #[test]
    fn default_profile_is_convenience() {
        assert_eq!(
            IdentityProfile::default(),
            IdentityProfile::ConvenienceGenerateMissing
        );
    }
}
