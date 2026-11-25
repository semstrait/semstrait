use std::collections::BTreeMap;
use std::time::SystemTime;

use semstrait_common::EntityId;
use semstrait_sql::DocumentDialect;
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

use super::bitmap::SemanticBitmap;
use super::coverage::Coverage;
use super::expression::ManifestExpressions;
use super::keyset::Keyset;
use super::metadata::SemanticManifestMetadata;
use super::physical_source::PhysicalSource;
use super::semantic_signature::SemanticSignature;
use super::semantic_source::SemanticSource;
use crate::error::CompileError;

pub(crate) const MANIFEST_DIALECT: DocumentDialect = DocumentDialect::Canonical;

fn read_dialect<'de, D: Deserializer<'de>>(deserializer: D) -> Result<DocumentDialect, D::Error> {
    let written = Value::deserialize(deserializer)?;
    match DocumentDialect::deserialize(&written) {
        Ok(MANIFEST_DIALECT) => Ok(MANIFEST_DIALECT),
        _ => Err(D::Error::custom(CompileError::ManifestDialect {
            dialect: written
                .as_str()
                .map_or_else(|| written.to_string(), str::to_owned),
        })),
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ManifestProvenance {
    pub created_at: SystemTime,
    pub updated_at: SystemTime,
}

impl ManifestProvenance {
    #[must_use]
    pub const fn new(created_at: SystemTime, updated_at: SystemTime) -> Self {
        Self {
            created_at,
            updated_at,
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticManifest {
    pub manifest_version: u64,

    pub created_at: SystemTime,

    pub updated_at: SystemTime,

    pub model_hash: [u8; 32],

    pub model_id: EntityId,

    #[serde(deserialize_with = "read_dialect")]
    pub dialect: DocumentDialect,

    pub semantic_definitions: SemanticBitmap,

    pub semantic_signatures: BTreeMap<EntityId, SemanticSignature>,

    pub semantic_sources: BTreeMap<EntityId, SemanticSource>,
    pub sources: BTreeMap<EntityId, PhysicalSource>,
    pub expressions: ManifestExpressions,
    pub keysets: BTreeMap<EntityId, Keyset>,

    pub metadata: SemanticManifestMetadata,
}

impl SemanticManifest {
    #[must_use]
    pub fn new(
        manifest_version: u64,
        provenance: ManifestProvenance,
        model_hash: [u8; 32],
        model_id: impl Into<EntityId>,
        metadata: SemanticManifestMetadata,
    ) -> Self {
        Self {
            manifest_version,
            created_at: provenance.created_at,
            updated_at: provenance.updated_at,
            model_hash,
            model_id: model_id.into(),
            dialect: MANIFEST_DIALECT,
            semantic_definitions: SemanticBitmap::default(),
            semantic_signatures: BTreeMap::new(),
            semantic_sources: BTreeMap::new(),
            sources: BTreeMap::new(),
            expressions: ManifestExpressions::default(),
            keysets: BTreeMap::new(),
            metadata,
        }
    }

    #[must_use]
    pub fn coverage_of(&self, semantic_id: &EntityId, data_kind_id: &EntityId) -> Coverage {
        let realization = self
            .semantic_sources
            .get(data_kind_id)
            .into_iter()
            .flat_map(|source| self.signatures_of(source))
            .find(|signature| &signature.definition_id == semantic_id)
            .and_then(|signature| self.expressions.linked_root(&signature.expression_id));
        Coverage::of(realization)
    }

    pub fn signatures_of<'a>(
        &'a self,
        source: &'a SemanticSource,
    ) -> impl Iterator<Item = &'a SemanticSignature> {
        (source.semantic_signature_ids.iter()).filter_map(|id| self.semantic_signatures.get(id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::Keyset;
    use std::time::{Duration, UNIX_EPOCH};

    fn fixed_provenance() -> ManifestProvenance {
        ManifestProvenance::new(
            UNIX_EPOCH + Duration::from_secs(1_700_000_000),
            UNIX_EPOCH + Duration::from_secs(1_800_000_000),
        )
    }

    fn fixed_metadata() -> SemanticManifestMetadata {
        SemanticManifestMetadata::new("0.1.0", None, BTreeMap::new())
    }

    fn empty_manifest() -> SemanticManifest {
        SemanticManifest::new(
            1,
            fixed_provenance(),
            [0u8; 32],
            "model.root",
            fixed_metadata(),
        )
    }

    #[test]
    fn new_initialises_the_identity_block_and_empty_collections() {
        let m = empty_manifest();
        assert_eq!(m.manifest_version, 1);
        assert_eq!(m.created_at, fixed_provenance().created_at);
        assert_eq!(m.updated_at, fixed_provenance().updated_at);
        assert_eq!(m.model_id, "model.root");
        assert!(m.semantic_definitions.is_empty());
        assert!(m.semantic_signatures.is_empty());
        assert!(m.semantic_sources.is_empty());
        assert!(m.sources.is_empty());
        assert!(m.keysets.is_empty());
    }

    #[test]
    fn empty_manifest_serde_json_round_trip() {
        let m = empty_manifest();
        let json = serde_json::to_string(&m).expect("serialize empty manifest");
        let back: SemanticManifest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(m, back);
    }

    #[test]
    fn populated_keysets_map_round_trips_keyed_by_the_minted_id() {
        let mut m = empty_manifest();
        let keyset = Keyset::new("dk.orders", "natural");
        let id = keyset.id.clone();
        m.keysets.insert(id.clone(), keyset);

        let json = serde_json::to_string(&m).expect("serialize populated manifest");
        assert!(
            json.contains(&format!("\"{id}\"")),
            "expected the keyset map key to be emitted, got {json}"
        );
        let back: SemanticManifest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(m, back);
        assert_eq!(back.keysets[&id].name.as_str(), "natural");
    }
}
