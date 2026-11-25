use std::borrow::Cow;

use bytes::Bytes;
use semstrait_common::io::{Fingerprintable, Persistable};
use semstrait_common::EntityId;

use crate::model::SemanticModel;

#[derive(serde::Serialize)]
struct Persisted<'a> {
    semantic_model: &'a SemanticModel,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PersistedOwned {
    semantic_model: SemanticModel,
}

impl Persistable for SemanticModel {
    fn entity_id(&self) -> &EntityId {
        self.id.as_ref().expect("model root id resolved by build")
    }

    fn integrity_check(&self) -> Result<(), Cow<'static, str>> {
        if self.id.is_none() {
            return Err(Cow::Borrowed(
                "a model root id must be resolved before the model can be persisted",
            ));
        }
        Ok(())
    }

    fn extension() -> &'static str {
        "yaml"
    }

    fn to_bytes(&self) -> Result<Bytes, Cow<'static, str>> {
        serde_yaml::to_string(&Persisted {
            semantic_model: self,
        })
        .map(Bytes::from)
        .map_err(|e| Cow::Owned(format!("unserializable model: {e}")))
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, Cow<'static, str>> {
        let PersistedOwned { semantic_model } =
            serde_yaml::from_slice(bytes).map_err(|e| Cow::Owned(format!("undecodable: {e}")))?;
        if semantic_model.id.is_none() {
            return Err(Cow::Borrowed(
                "persisted model bytes must carry a resolved root `id`",
            ));
        }
        Ok(semantic_model)
    }
}

impl Fingerprintable for SemanticModel {
    fn fingerprint_preimage(&self) -> Vec<u8> {
        let mut projected = self.clone();
        projected.id = None;
        serde_json::to_vec(&projected).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_MODEL_YAML: &str = r#"
semantic_model:
  name: analytics-v1
  datasets:
    - name: orders
      extras:
        semantic_mapping: auto
"#;

    fn built_model(yaml: &str) -> SemanticModel {
        crate::parse::parse(yaml)
            .expect("parse ok")
            .build()
            .expect("build ok")
            .0
    }

    #[test]
    fn round_trip_through_the_wrapper_is_equal_and_byte_identical_on_reencode() {
        let model = built_model(MINIMAL_MODEL_YAML);
        let bytes = model.to_bytes().expect("encode");
        let decoded = SemanticModel::from_bytes(&bytes).expect("decode");

        assert_eq!(decoded, model);
        assert_eq!(decoded.to_bytes().expect("re-encode"), bytes);
    }

    #[test]
    fn bytes_are_authoring_shaped_yaml_not_json() {
        let model = built_model(MINIMAL_MODEL_YAML);
        let bytes = model.to_bytes().expect("encode");
        let text = std::str::from_utf8(&bytes).expect("utf8");

        assert!(text.starts_with("semantic_model:"), "{text}");
        assert!(!text.trim_start().starts_with('{'), "{text}");
    }

    #[test]
    fn extension_is_yaml() {
        assert_eq!(SemanticModel::extension(), "yaml");
    }

    #[test]
    fn entity_id_returns_the_built_models_own_id() {
        let model = built_model(MINIMAL_MODEL_YAML);
        let id = model.id.clone().expect("built model carries a resolved id");

        assert_eq!(model.entity_id(), &id);
    }

    #[test]
    #[should_panic(expected = "model root id resolved by build")]
    fn entity_id_panics_on_a_residual_none_root_id() {
        let model = SemanticModel {
            name: "no-id".to_owned(),
            ..Default::default()
        };

        let _ = model.entity_id();
    }

    #[test]
    fn the_fingerprint_excludes_the_root_id_and_covers_the_name() {
        let model = built_model(MINIMAL_MODEL_YAML);

        let mut reidentified = model.clone();
        reidentified.id = Some("2f8b9c14-6a3d-5e70-9b2f-41c8d7e60a35".to_owned());
        assert_eq!(
            reidentified.fingerprint(),
            model.fingerprint(),
            "the root id is the persistence key, not model content"
        );

        let mut renamed = model.clone();
        renamed.name = "analytics-v2".to_owned();
        assert_ne!(
            renamed.fingerprint(),
            model.fingerprint(),
            "a rename is content, so it moves the fingerprint"
        );
    }

    #[test]
    fn the_preimage_is_not_the_stored_bytes() {
        let model = built_model(MINIMAL_MODEL_YAML);

        assert_ne!(
            model.fingerprint_preimage(),
            model.to_bytes().expect("encode").to_vec(),
            "the digest preimage must not inherit the stored form's codec"
        );
    }

    #[test]
    fn from_bytes_rejects_a_residual_none_root_id() {
        let idless = "semantic_model:\n  name: analytics-v1\n";

        let err = SemanticModel::from_bytes(idless.as_bytes()).expect_err("id-less document");

        assert_eq!(err, "persisted model bytes must carry a resolved root `id`");
    }

    #[test]
    fn from_bytes_rejects_a_document_with_one_entity_id_removed() {
        let one_id_missing = r#"
semantic_model:
  id: "0197cb6e-a63a-7d53-8d6f-35f4b1d67d4e"
  name: analytics-v1
  datasets:
    - name: orders
      extras:
        semantic_mapping: auto
    - id: "0197cb6e-a63a-7d53-8d6f-35f4b1d67d5e"
      name: refunds
      extras:
        semantic_mapping: auto
"#;

        let err = SemanticModel::from_bytes(one_id_missing.as_bytes())
            .expect_err("entity missing its id");

        assert!(err.contains("missing an `id`"), "{err}");
    }

    #[test]
    fn from_bytes_rejects_a_document_with_a_duplicated_entity_id() {
        let duplicated_id = r#"
semantic_model:
  id: "0197cb6e-a63a-7d53-8d6f-35f4b1d67d4e"
  name: analytics-v1
  datasets:
    - id: "0197cb6e-a63a-7d53-8d6f-35f4b1d67d5e"
      name: orders
      extras:
        semantic_mapping: auto
    - id: "0197cb6e-a63a-7d53-8d6f-35f4b1d67d5e"
      name: refunds
      extras:
        semantic_mapping: auto
"#;

        let err = SemanticModel::from_bytes(duplicated_id.as_bytes())
            .expect_err("two entities sharing one id");

        assert!(err.contains("duplicate entity id"), "{err}");
    }

    #[test]
    fn from_bytes_rejects_a_document_with_no_wrapper() {
        let bare = "id: \"0197cb6e-a63a-7d53-8d6f-35f4b1d67d4e\"\nname: analytics-v1\n";

        let err = SemanticModel::from_bytes(bare.as_bytes()).expect_err("no semantic_model: root");

        assert!(err.contains("semantic_model"), "{err}");
    }

    #[test]
    fn from_bytes_rejects_a_document_with_an_extra_top_level_key() {
        let extra_key = "semantic_model:\n  id: \"0197cb6e-a63a-7d53-8d6f-35f4b1d67d4e\"\n  name: analytics-v1\nunexpected_key: surprise\n";

        let err = SemanticModel::from_bytes(extra_key.as_bytes()).expect_err("extra top-level key");

        assert!(err.contains("unexpected_key"), "{err}");
    }

    #[test]
    fn from_bytes_does_not_run_validate_even_though_the_decoded_model_would_fail_it() {
        let serde_valid_but_semantically_invalid =
            "semantic_model:\n  id: \"0197cb6e-a63a-7d53-8d6f-35f4b1d67d4e\"\n  name: analytics-v1\n";

        let model = SemanticModel::from_bytes(serde_valid_but_semantically_invalid.as_bytes())
            .expect("serde-valid despite failing validate");

        assert!(
            crate::validate(&model).is_err(),
            "expected EmptyModel to fail validate, proving from_bytes skipped it"
        );
    }
}
