use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticManifestMetadata {
    pub semstrait_version: String,
    pub source_model_uri: Option<String>,
    pub annotations: BTreeMap<String, String>,
}

impl SemanticManifestMetadata {
    #[must_use]
    pub fn new(
        semstrait_version: impl Into<String>,
        source_model_uri: Option<String>,
        annotations: BTreeMap<String, String>,
    ) -> Self {
        Self {
            semstrait_version: semstrait_version.into(),
            source_model_uri,
            annotations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_carries_its_fields_verbatim() {
        let meta = SemanticManifestMetadata::new(
            "0.1.0",
            Some("file:///model.yaml".to_string()),
            BTreeMap::new(),
        );
        assert_eq!(meta.semstrait_version, "0.1.0");
        assert_eq!(meta.source_model_uri.as_deref(), Some("file:///model.yaml"));
        assert!(meta.annotations.is_empty());
    }

    #[test]
    fn metadata_serde_json_round_trip() {
        let mut annotations = BTreeMap::new();
        annotations.insert("git_sha".to_string(), "deadbeef".to_string());
        let meta = SemanticManifestMetadata::new("0.1.0", None, annotations);
        let json = serde_json::to_string(&meta).expect("serialize metadata");
        let back: SemanticManifestMetadata =
            serde_json::from_str(&json).expect("deserialize metadata");
        assert_eq!(meta, back);
    }
}
