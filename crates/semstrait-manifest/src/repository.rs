use std::borrow::Cow;

use bytes::Bytes;

use semstrait_common::io::Persistable;
use semstrait_common::EntityId;

use crate::manifest::SemanticManifest;

impl Persistable for SemanticManifest {
    fn entity_id(&self) -> &EntityId {
        &self.model_id
    }

    fn extension() -> &'static str {
        "json"
    }

    fn to_bytes(&self) -> Result<Bytes, Cow<'static, str>> {
        serde_json::to_vec(self)
            .map(Bytes::from)
            .map_err(|e| Cow::Owned(format!("unserializable manifest: {e}")))
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, Cow<'static, str>> {
        serde_json::from_slice(bytes).map_err(|e| Cow::Owned(format!("undecodable: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use semstrait_common::diagnostic::{Diagnose, Severity};
    use semstrait_common::io::{
        FileSystem, FileSystemRepository, InMemoryFileSystem, IoErrorKind, Path, Repository,
        RepositoryErrorKind,
    };
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use std::time::{Duration, UNIX_EPOCH};

    use crate::manifest::{ManifestProvenance, SemanticManifestMetadata};

    const BASE: &str = "state";

    fn _assert_repo_object_safe(_: &dyn Repository<SemanticManifest>) {}

    fn boxed_repo() -> Box<dyn Repository<SemanticManifest>> {
        Box::new(FileSystemRepository::new(
            Arc::new(InMemoryFileSystem::new()),
            Path::new(BASE),
        ))
    }

    fn fixed_metadata() -> SemanticManifestMetadata {
        SemanticManifestMetadata::new("0.1.0", None, BTreeMap::new())
    }

    fn fixed_provenance() -> ManifestProvenance {
        ManifestProvenance::new(
            UNIX_EPOCH + Duration::from_secs(1_700_000_000),
            UNIX_EPOCH + Duration::from_secs(1_800_000_000),
        )
    }

    fn manifest_with_id(model_id: &str) -> SemanticManifest {
        SemanticManifest::new(1, fixed_provenance(), [0u8; 32], model_id, fixed_metadata())
    }

    fn repo() -> FileSystemRepository<SemanticManifest> {
        FileSystemRepository::new(Arc::new(InMemoryFileSystem::new()), Path::new(BASE))
    }

    fn expected_key(id: &str) -> Path {
        Path::new(format!("{BASE}/{id}.{}", SemanticManifest::extension()))
    }

    #[tokio::test]
    async fn save_then_load_round_trips() {
        let repo = repo();
        let m = manifest_with_id("11111111-1111-5111-8111-111111111111");
        repo.save(&m).await.expect("save");
        let back = repo.load(&m.model_id).await.expect("load");
        assert_eq!(m, back);
    }

    #[tokio::test]
    async fn load_missing_is_io_not_found() {
        let repo = repo();
        let err = repo
            .load(&"22222222-2222-5222-8222-222222222222".to_owned())
            .await
            .unwrap_err();
        match err {
            RepositoryErrorKind::Io(IoErrorKind::NotFound { .. }) => {}
            other => panic!("expected Io(NotFound), got {other:?}"),
        }
    }

    #[tokio::test]
    async fn exists_reflects_save() {
        let repo = repo();
        let m = manifest_with_id("33333333-3333-5333-8333-333333333333");
        assert!(!repo.exists(&m.model_id).await.unwrap());
        repo.save(&m).await.expect("save");
        assert!(repo.exists(&m.model_id).await.unwrap());
    }

    #[tokio::test]
    async fn save_is_byte_deterministic_for_equal_manifests() {
        let fs_a = InMemoryFileSystem::new();
        let fs_b = InMemoryFileSystem::new();
        let m = manifest_with_id("44444444-4444-5444-8444-444444444444");
        let key = expected_key(&m.model_id);

        FileSystemRepository::new(Arc::new(fs_a.clone()), Path::new(BASE))
            .save(&m)
            .await
            .expect("save a");
        FileSystemRepository::new(Arc::new(fs_b.clone()), Path::new(BASE))
            .save(&m)
            .await
            .expect("save b");

        let bytes_a = fs_a.read(&key).await.expect("read a");
        let bytes_b = fs_b.read(&key).await.expect("read b");
        assert_eq!(
            bytes_a, bytes_b,
            "equal manifests must serialize identically"
        );
    }

    #[tokio::test]
    async fn resave_identical_manifest_is_idempotent() {
        let repo = repo();
        let m = manifest_with_id("55555555-5555-5555-8555-555555555555");
        repo.save(&m).await.expect("first save");
        repo.save(&m).await.expect("idempotent re-save");
        assert_eq!(repo.load(&m.model_id).await.unwrap(), m);
    }

    #[tokio::test]
    async fn saving_different_manifest_under_same_id_overwrites() {
        let repo = repo();
        let id = "66666666-6666-5666-8666-666666666666";
        let first = manifest_with_id(id);
        repo.save(&first).await.expect("first save");

        let mut second = manifest_with_id(id);
        second.manifest_version = 2;
        repo.save(&second).await.expect("overwrite");

        assert_eq!(repo.load(&id.to_owned()).await.unwrap().manifest_version, 2);
    }

    #[tokio::test]
    async fn load_detects_misfiled_object() {
        let fs = InMemoryFileSystem::new();
        let real_id = "77777777-7777-5777-8777-777777777777";
        let wrong_id = "88888888-8888-5888-8888-888888888888";
        let m = manifest_with_id(real_id);
        let bytes = serde_json::to_vec(&m).unwrap();
        let repo: FileSystemRepository<SemanticManifest> =
            FileSystemRepository::new(Arc::new(fs.clone()), Path::new(BASE));
        let wrong_key = expected_key(wrong_id);
        fs.write(&wrong_key, Bytes::from(bytes)).await.unwrap();

        let err = repo.load(&wrong_id.to_owned()).await.unwrap_err();
        assert!(
            matches!(err, RepositoryErrorKind::IntegrityCheckFailed { .. }),
            "expected IntegrityCheckFailed, got {err:?}"
        );
    }

    #[tokio::test]
    async fn load_undecodable_bytes_is_encoding_mismatch() {
        let fs = InMemoryFileSystem::new();
        let id = "99999999-9999-5999-8999-999999999999";
        let repo: FileSystemRepository<SemanticManifest> =
            FileSystemRepository::new(Arc::new(fs.clone()), Path::new(BASE));
        let key = expected_key(id);
        fs.write(&key, Bytes::from_static(b"not json"))
            .await
            .unwrap();
        let err = repo.load(&id.to_owned()).await.unwrap_err();
        assert!(
            matches!(err, RepositoryErrorKind::EncodingMismatch { .. }),
            "expected EncodingMismatch, got {err:?}"
        );
    }

    #[test]
    fn diagnose_message_io_wraps_inner() {
        let e = RepositoryErrorKind::Io(IoErrorKind::NotFound {
            describe: "manifests/x.json".into(),
        });
        assert_eq!(e.message(), "repository I/O: not found: manifests/x.json");
        assert_eq!(e.severity_default(), Severity::Error);
    }

    #[test]
    fn diagnose_cause_chains_through_io() {
        #[derive(Debug)]
        struct Inner;
        impl std::fmt::Display for Inner {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("inner")
            }
        }
        impl std::error::Error for Inner {}

        let e = RepositoryErrorKind::Io(IoErrorKind::Network {
            describe: "fs".into(),
            reason: "refused".into(),
            source: Some(Box::new(Inner)),
        });
        assert!(
            e.cause().is_some(),
            "Io cause must chain to the inner source"
        );

        let e2 = RepositoryErrorKind::IntegrityCheckFailed { reason: "x".into() };
        assert!(e2.cause().is_none());
    }

    #[test]
    fn object_safe_witnesses_hold() {
        let repo = repo();
        _assert_repo_object_safe(&repo);
        let _boxed: Box<dyn Repository<SemanticManifest>> = boxed_repo();
    }

    #[test]
    fn key_scheme_is_deterministic_and_extension_qualified() {
        let id = "aaaaaaaa-aaaa-5aaa-8aaa-aaaaaaaaaaaa".to_owned();
        assert_eq!(
            expected_key(&id).as_str(),
            "state/aaaaaaaa-aaaa-5aaa-8aaa-aaaaaaaaaaaa.json"
        );
    }
}
