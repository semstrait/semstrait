use std::borrow::Cow;
use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use sha2::{Digest, Sha256};

use crate::diagnostic::{Diagnose, Severity};
use crate::identity::EntityId;
use crate::io::{FileSystem, FileSystemErrorKind, IoErrorKind, Path};

pub trait Persistable: Sized {
    fn entity_id(&self) -> &EntityId;

    fn integrity_check(&self) -> Result<(), Cow<'static, str>> {
        Ok(())
    }

    fn extension() -> &'static str;

    fn to_bytes(&self) -> Result<Bytes, Cow<'static, str>>;

    fn from_bytes(bytes: &[u8]) -> Result<Self, Cow<'static, str>>;
}

pub trait Fingerprintable {
    fn fingerprint_preimage(&self) -> Vec<u8>;

    fn fingerprint(&self) -> [u8; 32] {
        Sha256::digest(self.fingerprint_preimage()).into()
    }
}

#[derive(Debug)]
#[non_exhaustive]
pub enum RepositoryErrorKind {
    Io(IoErrorKind),

    EncodingMismatch { actual: Cow<'static, str> },

    IntegrityCheckFailed { reason: Cow<'static, str> },
}

impl Diagnose for RepositoryErrorKind {
    fn message(&self) -> String {
        match self {
            Self::Io(inner) => format!("repository I/O: {}", inner.message()),
            Self::EncodingMismatch { actual } => {
                format!("encoding mismatch: {actual}")
            }
            Self::IntegrityCheckFailed { reason } => {
                format!("integrity check failed: {reason}")
            }
        }
    }

    fn severity_default(&self) -> Severity {
        Severity::Error
    }

    fn cause(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(inner) => inner.cause(),
            _ => None,
        }
    }
}

impl From<IoErrorKind> for RepositoryErrorKind {
    fn from(inner: IoErrorKind) -> Self {
        Self::Io(inner)
    }
}

impl From<FileSystemErrorKind> for RepositoryErrorKind {
    fn from(err: FileSystemErrorKind) -> Self {
        Self::Io(match err {
            FileSystemErrorKind::NotFound { path } => IoErrorKind::NotFound {
                describe: path.to_string(),
            },
            FileSystemErrorKind::PermissionDenied { path } => IoErrorKind::PermissionDenied {
                describe: path.to_string(),
            },
            FileSystemErrorKind::AuthFailed => IoErrorKind::PermissionDenied {
                describe: "authentication failed".to_owned(),
            },
            FileSystemErrorKind::UnsupportedScheme { scheme } => IoErrorKind::Unsupported {
                describe: scheme,
                reason: "no FileSystem backend handles this scheme".into(),
            },
            FileSystemErrorKind::ConnectionFailed { msg } => IoErrorKind::Network {
                describe: "filesystem".to_owned(),
                reason: Cow::Owned(msg),
                source: None,
            },
            FileSystemErrorKind::InvalidPath { path, msg } => IoErrorKind::Malformed {
                describe: path.to_string(),
                reason: Cow::Owned(msg),
            },
            FileSystemErrorKind::Internal { msg } => IoErrorKind::Malformed {
                describe: "filesystem".to_owned(),
                reason: Cow::Owned(msg),
            },
        })
    }
}

#[allow(clippy::double_must_use)]
#[async_trait]
pub trait Repository<T: Persistable>: Send + Sync {
    async fn save(&self, value: &T) -> Result<(), RepositoryErrorKind>;
    async fn load(&self, id: &EntityId) -> Result<T, RepositoryErrorKind>;
    async fn exists(&self, id: &EntityId) -> Result<bool, RepositoryErrorKind>;
}

#[derive(Debug)]
pub struct FileSystemRepository<T> {
    fs: Arc<dyn FileSystem>,
    base: Path,
    _t: PhantomData<fn() -> T>,
}

#[cfg(feature = "io-aws")]
fn extract_bucket(base: &str) -> String {
    let after_scheme = base.split_once("://").map(|(_, rest)| rest).unwrap_or(base);
    after_scheme.split('/').next().unwrap_or("").to_owned()
}

fn canonical_base(base: Path) -> Path {
    match base.as_str().strip_prefix("s3a://") {
        Some(rest) => Path::new(format!("s3://{rest}")),
        None => base,
    }
}

fn join_under(base: &str, rest: &str) -> String {
    if base.ends_with('/') {
        format!("{base}{rest}")
    } else {
        format!("{base}/{rest}")
    }
}

impl<T: Persistable> FileSystemRepository<T> {
    pub fn new(fs: Arc<dyn FileSystem>, base: Path) -> Self {
        Self {
            fs,
            base: canonical_base(base),
            _t: PhantomData,
        }
    }

    pub fn open(base: Path) -> Result<Self, RepositoryErrorKind> {
        let scheme = base.as_str().split_once("://").map(|(s, _)| s);
        match scheme {
            Some(s) if crate::io::fs::is_drive_letter(s) => Ok(Self::new(
                Arc::new(crate::io::backends::local_fs::LocalFileSystem::new(
                    base.as_str(),
                )),
                Path::new(""),
            )),
            #[cfg(feature = "io-aws")]
            Some("s3") => {
                let bucket = extract_bucket(base.as_str());
                Ok(Self::new(
                    Arc::new(crate::io::backends::s3_fs::S3FileSystem::new(bucket)),
                    base,
                ))
            }
            #[cfg(feature = "io-aws")]
            Some("s3a") => {
                let bucket = extract_bucket(base.as_str());
                Ok(Self::new(
                    Arc::new(crate::io::backends::s3_fs::S3FileSystem::new(bucket)),
                    base,
                ))
            }
            Some("file") => {
                let root = base
                    .as_str()
                    .strip_prefix("file://")
                    .unwrap_or(base.as_str());
                Ok(Self::new(
                    Arc::new(crate::io::backends::local_fs::LocalFileSystem::new(root)),
                    Path::new(""),
                ))
            }
            None => Ok(Self::new(
                Arc::new(crate::io::backends::local_fs::LocalFileSystem::new(
                    base.as_str(),
                )),
                Path::new(""),
            )),
            Some(other) => Err(FileSystemErrorKind::UnsupportedScheme {
                scheme: other.to_owned(),
            }
            .into()),
        }
    }

    fn key_for(&self, id: &EntityId) -> Result<Path, RepositoryErrorKind> {
        if id.is_empty() || id.contains('/') || id.contains('\\') {
            return Err(RepositoryErrorKind::IntegrityCheckFailed {
                reason: format!("id {id:?} is empty or contains a path separator").into(),
            });
        }
        Ok(Path::new(join_under(
            self.base.as_str(),
            &format!("{}.{}", id.to_ascii_lowercase(), T::extension()),
        )))
    }

    fn encoding_mismatch(reason: Cow<'static, str>) -> RepositoryErrorKind {
        RepositoryErrorKind::EncodingMismatch { actual: reason }
    }

    fn decode(&self, bytes: &[u8]) -> Result<T, RepositoryErrorKind> {
        T::from_bytes(bytes).map_err(Self::encoding_mismatch)
    }
}

#[async_trait]
impl<T: Persistable + Send + Sync> Repository<T> for FileSystemRepository<T> {
    async fn save(&self, value: &T) -> Result<(), RepositoryErrorKind> {
        value
            .integrity_check()
            .map_err(|reason| RepositoryErrorKind::IntegrityCheckFailed { reason })?;

        let key = self.key_for(value.entity_id())?;
        let bytes = value.to_bytes().map_err(Self::encoding_mismatch)?;
        Ok(self.fs.write(&key, bytes).await?)
    }

    async fn load(&self, id: &EntityId) -> Result<T, RepositoryErrorKind> {
        let key = self.key_for(id)?;
        let value = self.decode(&self.fs.read(&key).await?)?;

        value
            .integrity_check()
            .map_err(|reason| RepositoryErrorKind::IntegrityCheckFailed { reason })?;

        if !value.entity_id().eq_ignore_ascii_case(id) {
            return Err(RepositoryErrorKind::IntegrityCheckFailed {
                reason: format!(
                    "stored id {} does not match requested id {id}",
                    value.entity_id()
                )
                .into(),
            });
        }
        Ok(value)
    }

    async fn exists(&self, id: &EntityId) -> Result<bool, RepositoryErrorKind> {
        Ok(self.fs.exists(&self.key_for(id)?).await?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::InMemoryFileSystem;

    #[derive(Debug, Clone, PartialEq)]
    struct Sealed {
        id: EntityId,
        payload: u32,
    }

    impl Persistable for Sealed {
        fn entity_id(&self) -> &EntityId {
            &self.id
        }
        fn extension() -> &'static str {
            "json"
        }
        fn to_bytes(&self) -> Result<Bytes, Cow<'static, str>> {
            Ok(Bytes::from(format!("{}|{}", self.id, self.payload)))
        }
        fn from_bytes(bytes: &[u8]) -> Result<Self, Cow<'static, str>> {
            let text = std::str::from_utf8(bytes).map_err(|e| Cow::Owned(e.to_string()))?;
            let (id, payload) = text
                .split_once('|')
                .ok_or(Cow::Borrowed("missing separator"))?;
            Ok(Self {
                id: id.to_owned(),
                payload: payload.parse().map_err(|_| Cow::Borrowed("bad payload"))?,
            })
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    struct Draft {
        id: EntityId,
        payload: u32,
    }

    impl Persistable for Draft {
        fn entity_id(&self) -> &EntityId {
            &self.id
        }
        fn extension() -> &'static str {
            "yaml"
        }
        fn to_bytes(&self) -> Result<Bytes, Cow<'static, str>> {
            Ok(Bytes::from(format!("{}|{}", self.id, self.payload)))
        }
        fn from_bytes(bytes: &[u8]) -> Result<Self, Cow<'static, str>> {
            Sealed::from_bytes(bytes).map(|s| Self {
                id: s.id,
                payload: s.payload,
            })
        }
    }

    #[derive(Debug, PartialEq)]
    struct Unencodable(EntityId);

    impl Persistable for Unencodable {
        fn entity_id(&self) -> &EntityId {
            &self.0
        }
        fn extension() -> &'static str {
            "bin"
        }
        fn to_bytes(&self) -> Result<Bytes, Cow<'static, str>> {
            Err(Cow::Borrowed("encoder always fails"))
        }
        fn from_bytes(_: &[u8]) -> Result<Self, Cow<'static, str>> {
            Err(Cow::Borrowed("decoder always fails"))
        }
    }

    const ID_A: &str = "11111111-1111-5111-8111-111111111111";
    const ID_B: &str = "22222222-2222-5222-8222-222222222222";
    const ID_HEX: &str = "aaaaaaaa-bbbb-5ccc-8ddd-eeeeeeeeeeee";

    fn store() -> InMemoryFileSystem {
        InMemoryFileSystem::new()
    }

    fn repo_over<T: Persistable>(fs: &InMemoryFileSystem, base: &str) -> FileSystemRepository<T> {
        FileSystemRepository::new(Arc::new(fs.clone()), Path::new(base))
    }

    fn sealed(id: &str, payload: u32) -> Sealed {
        Sealed {
            id: id.to_owned(),
            payload,
        }
    }

    #[tokio::test]
    async fn save_then_load_round_trips() {
        let fs = store();
        let repo = repo_over::<Sealed>(&fs, "mem:");
        let value = sealed(ID_A, 7);

        repo.save(&value).await.expect("save");
        let loaded = repo.load(&value.id).await.expect("load");

        assert_eq!(loaded, value);
        assert_eq!(
            loaded.to_bytes().expect("re-encode"),
            value.to_bytes().expect("encode")
        );
    }

    #[tokio::test]
    async fn exists_reflects_save() {
        let fs = store();
        let repo = repo_over::<Sealed>(&fs, "mem:");
        let value = sealed(ID_A, 7);

        assert!(!repo.exists(&value.id).await.expect("exists before"));
        repo.save(&value).await.expect("save");
        assert!(repo.exists(&value.id).await.expect("exists after"));
    }

    #[tokio::test]
    async fn resave_replaces_the_whole_object_including_shrinking_it() {
        let fs = store();
        let repo = repo_over::<Draft>(&fs, "mem:");
        let long = Draft {
            id: ID_A.to_owned(),
            payload: u32::MAX,
        };
        let short = Draft {
            id: ID_A.to_owned(),
            payload: 2,
        };

        repo.save(&long).await.expect("first save");
        repo.save(&short).await.expect("second save");

        assert_eq!(repo.load(&ID_A.to_owned()).await.expect("load"), short);
        assert_eq!(
            fs.read(&Path::new(format!("mem:/{ID_A}.yaml")))
                .await
                .expect("stored bytes"),
            short.to_bytes().expect("encode"),
        );
    }

    #[test]
    fn persistable_members_are_callable_without_a_store() {
        let value = sealed(ID_A, 42);

        let bytes = value.to_bytes().expect("encode");

        assert_eq!(Sealed::from_bytes(&bytes).expect("decode"), value);
        assert_eq!(value.entity_id(), ID_A);
        assert_eq!(Sealed::extension(), "json");
    }

    #[tokio::test]
    async fn key_is_base_id_and_the_dtos_own_extension() {
        let fs = store();
        repo_over::<Sealed>(&fs, "mem:")
            .save(&sealed(ID_A, 1))
            .await
            .expect("save sealed");
        repo_over::<Draft>(&fs, "mem:")
            .save(&Draft {
                id: ID_A.to_owned(),
                payload: 1,
            })
            .await
            .expect("save draft");

        let mut keys: Vec<String> = fs
            .list(&Path::new(""))
            .await
            .expect("list")
            .into_iter()
            .map(|e| e.path.as_str().to_owned())
            .collect();
        keys.sort();

        assert_eq!(
            keys,
            vec![format!("mem:/{ID_A}.json"), format!("mem:/{ID_A}.yaml")]
        );
    }

    #[tokio::test]
    async fn different_bases_do_not_collide_on_equal_ids() {
        let fs = store();
        let tenant_a = repo_over::<Draft>(&fs, "tenants/a");
        let tenant_b = repo_over::<Draft>(&fs, "tenants/b");

        tenant_a
            .save(&Draft {
                id: ID_A.to_owned(),
                payload: 1,
            })
            .await
            .expect("save a");
        tenant_b
            .save(&Draft {
                id: ID_A.to_owned(),
                payload: 2,
            })
            .await
            .expect("save b");

        assert_eq!(tenant_a.load(&ID_A.to_owned()).await.unwrap().payload, 1);
        assert_eq!(tenant_b.load(&ID_A.to_owned()).await.unwrap().payload, 2);
    }

    #[tokio::test]
    async fn a_trailing_slash_on_base_does_not_double_up() {
        let fs = store();
        repo_over::<Draft>(&fs, "root/")
            .save(&Draft {
                id: ID_A.to_owned(),
                payload: 1,
            })
            .await
            .expect("save");

        let keys: Vec<String> = fs
            .list(&Path::new(""))
            .await
            .expect("list")
            .into_iter()
            .map(|e| e.path.as_str().to_owned())
            .collect();
        assert_eq!(keys, vec![format!("root/{ID_A}.yaml")]);
    }

    #[tokio::test]
    async fn an_id_bearing_a_separator_is_refused_before_it_becomes_a_key() {
        let fs = store();
        let repo = repo_over::<Draft>(&fs, "mem:");

        for id in ["../../../etc/passwd", "a/b", "a\\b", ""] {
            let err = repo
                .save(&Draft {
                    id: id.to_owned(),
                    payload: 1,
                })
                .await
                .expect_err("a separator or empty id must be refused");
            match err {
                RepositoryErrorKind::IntegrityCheckFailed { reason } => {
                    assert!(reason.contains(&format!("{id:?}")), "{reason}");
                }
                other => panic!("expected IntegrityCheckFailed for {id:?}, got {other:?}"),
            }
        }
        assert!(fs.list(&Path::new("")).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_dot_id_under_an_empty_extension_composes_a_dotdot_segment() {
        #[derive(Debug, PartialEq)]
        struct Extensionless(EntityId);

        impl Persistable for Extensionless {
            fn entity_id(&self) -> &EntityId {
                &self.0
            }
            fn extension() -> &'static str {
                ""
            }
            fn to_bytes(&self) -> Result<Bytes, Cow<'static, str>> {
                Ok(Bytes::from(self.0.clone()))
            }
            fn from_bytes(bytes: &[u8]) -> Result<Self, Cow<'static, str>> {
                Ok(Self(String::from_utf8_lossy(bytes).into_owned()))
            }
        }

        let fs = store();
        let repo = repo_over::<Extensionless>(&fs, "base/deep");
        repo.save(&Extensionless(".".to_owned()))
            .await
            .expect("the guard passes: \".\" holds no separator");

        let keys: Vec<_> = fs
            .list(&Path::new("base"))
            .await
            .unwrap()
            .into_iter()
            .map(|e| e.path.as_str().to_owned())
            .collect();
        assert_eq!(keys, vec!["base/deep/..".to_owned()]);
    }

    #[tokio::test]
    async fn a_value_its_own_codec_accepts_but_its_accessor_cannot_describe_is_refused_on_load() {
        #[derive(Debug, PartialEq)]
        struct Fragile(Option<EntityId>);

        impl Persistable for Fragile {
            fn entity_id(&self) -> &EntityId {
                self.0.as_ref().expect("guarded by integrity_check")
            }
            fn extension() -> &'static str {
                "bin"
            }
            fn integrity_check(&self) -> Result<(), Cow<'static, str>> {
                if self.0.is_none() {
                    return Err(Cow::Borrowed("no id"));
                }
                Ok(())
            }
            fn to_bytes(&self) -> Result<Bytes, Cow<'static, str>> {
                Ok(Bytes::from(self.0.clone().unwrap_or_default()))
            }
            fn from_bytes(bytes: &[u8]) -> Result<Self, Cow<'static, str>> {
                Ok(Self(
                    (!bytes.is_empty()).then(|| String::from_utf8_lossy(bytes).into_owned()),
                ))
            }
        }

        let fs = store();
        fs.write(&Path::new("base/abc.bin"), Bytes::new())
            .await
            .expect("plant id-less bytes under a live key");

        let err = repo_over::<Fragile>(&fs, "base")
            .load(&"abc".to_owned())
            .await
            .expect_err("an id-less decode must be refused, not dereferenced");
        assert!(
            matches!(err, RepositoryErrorKind::IntegrityCheckFailed { ref reason } if reason == "no id"),
            "expected the load-leg integrity guard, got {err:?}"
        );
    }

    #[tokio::test]
    async fn a_slashed_extension_reopens_the_traversal_a_plain_one_closes() {
        macro_rules! ext_dto {
            ($name:ident, $ext:expr) => {
                #[derive(PartialEq)]
                struct $name(EntityId);

                impl Persistable for $name {
                    fn entity_id(&self) -> &EntityId {
                        &self.0
                    }
                    fn extension() -> &'static str {
                        $ext
                    }
                    fn to_bytes(&self) -> Result<Bytes, Cow<'static, str>> {
                        Ok(Bytes::from(self.0.clone()))
                    }
                    fn from_bytes(bytes: &[u8]) -> Result<Self, Cow<'static, str>> {
                        Ok(Self(String::from_utf8_lossy(bytes).into_owned()))
                    }
                }
            };
        }

        ext_dto!(PlainExt, "bin");
        ext_dto!(SlashedExt, "/x");

        async fn key_of<T: Persistable + Send + Sync>(dto: T) -> String {
            let fs = store();
            repo_over::<T>(&fs, "base")
                .save(&dto)
                .await
                .expect("the separator guard passes either way");
            let keys: Vec<_> = fs
                .list(&Path::new("base"))
                .await
                .unwrap()
                .into_iter()
                .map(|e| e.path.as_str().to_owned())
                .collect();
            assert_eq!(keys.len(), 1, "one save, one key: {keys:?}");
            keys.into_iter().next().expect("one key")
        }

        assert_eq!(
            key_of(PlainExt(".".to_owned())).await,
            "base/..bin",
            "a slash-free extension keeps the `.` inside a single segment"
        );
        assert_eq!(
            key_of(SlashedExt(".".to_owned())).await,
            "base/../x",
            "a slashed extension composes a traversal segment regardless"
        );
    }

    #[tokio::test]
    async fn distinct_ids_never_share_one_key() {
        let fs = store();
        let repo = repo_over::<Draft>(&fs, "root");

        for id in ["a/b", "a//b", "/a/b"] {
            repo.save(&Draft {
                id: id.to_owned(),
                payload: 1,
            })
            .await
            .expect_err("each collapses onto `root/a/b.yaml`");
        }

        for id in ["a-b", "a.b"] {
            repo.save(&Draft {
                id: id.to_owned(),
                payload: 1,
            })
            .await
            .expect("a separator-free id is fine");
        }
        assert_eq!(fs.list(&Path::new("root")).await.unwrap().len(), 2);
    }

    #[test]
    fn joining_under_a_base_adds_exactly_one_separator() {
        let cases = [
            ("root", "root/x"),
            ("root/", "root/x"),
            ("s3://bucket", "s3://bucket/x"),
            ("s3://bucket/", "s3://bucket/x"),
            ("file:///", "file:///x"),
            ("file:///var/lib", "file:///var/lib/x"),
            ("/", "/x"),
            ("", "/x"),
        ];
        for (base, want) in cases {
            assert_eq!(join_under(base, "x"), want, "base {base:?}");
        }
    }

    #[tokio::test]
    async fn a_scheme_rooted_base_keys_under_the_scheme() {
        let fs = store();
        repo_over::<Draft>(&fs, "file:///")
            .save(&Draft {
                id: ID_A.to_owned(),
                payload: 1,
            })
            .await
            .expect("save");

        let keys: Vec<String> = fs
            .list(&Path::new(""))
            .await
            .expect("list")
            .into_iter()
            .map(|e| e.path.as_str().to_owned())
            .collect();
        assert_eq!(keys, vec![format!("file:///{ID_A}.yaml")]);
    }

    #[tokio::test]
    async fn exists_finds_a_saved_id_under_any_letter_case() {
        let fs = store();
        let repo = repo_over::<Sealed>(&fs, "mem:");
        repo.save(&sealed(ID_HEX, 1)).await.expect("save lowercase");

        assert!(repo.exists(&ID_HEX.to_uppercase()).await.expect("exists"));

        let keys: Vec<String> = fs
            .list(&Path::new(""))
            .await
            .expect("list")
            .into_iter()
            .map(|e| e.path.as_str().to_owned())
            .collect();
        assert_eq!(keys, vec![format!("mem:/{ID_HEX}.json")]);
    }

    #[tokio::test]
    async fn load_of_a_differently_cased_id_still_returns_the_stored_value() {
        let fs = store();
        let repo = repo_over::<Sealed>(&fs, "mem:");
        let value = sealed(ID_HEX, 9);
        repo.save(&value).await.expect("save lowercase");

        let loaded = repo
            .load(&ID_HEX.to_uppercase())
            .await
            .expect("a case-differing id must round-trip, not IntegrityCheckFailed");

        assert_eq!(loaded, value);
    }

    #[tokio::test]
    async fn a_differing_value_saved_under_an_equal_id_overwrites() {
        let fs = store();
        let repo = repo_over::<Sealed>(&fs, "mem:");

        repo.save(&sealed(ID_A, 1)).await.expect("first save");
        repo.save(&sealed(ID_A, 2)).await.expect("second save");

        assert_eq!(repo.load(&ID_A.to_owned()).await.unwrap(), sealed(ID_A, 2));
    }

    #[tokio::test]
    async fn load_of_an_absent_id_folds_to_io_not_found() {
        let fs = store();
        let repo = repo_over::<Sealed>(&fs, "mem:");

        let err = repo
            .load(&ID_A.to_owned())
            .await
            .expect_err("nothing saved");

        assert!(
            matches!(err, RepositoryErrorKind::Io(IoErrorKind::NotFound { .. })),
            "expected Io(NotFound), got {err:?}"
        );
    }

    #[tokio::test]
    async fn undecodable_bytes_surface_as_encoding_mismatch_carrying_the_dtos_reason() {
        let fs = store();
        let repo = repo_over::<Sealed>(&fs, "mem:");
        fs.write(
            &Path::new(format!("mem:/{ID_A}.json")),
            Bytes::from_static(b"not-a-sealed"),
        )
        .await
        .expect("seed raw bytes");

        let err = repo.load(&ID_A.to_owned()).await.expect_err("undecodable");

        match err {
            RepositoryErrorKind::EncodingMismatch { actual } => {
                assert!(
                    actual.contains("separator"),
                    "lost the DTO's reason: {actual}"
                );
            }
            other => panic!("expected EncodingMismatch, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn an_encoder_failure_on_save_surfaces_as_encoding_mismatch() {
        let fs = store();
        let repo = repo_over::<Unencodable>(&fs, "mem:");

        let err = repo
            .save(&Unencodable(ID_A.to_owned()))
            .await
            .expect_err("encoder always fails");

        match err {
            RepositoryErrorKind::EncodingMismatch { actual } => {
                assert_eq!(actual, "encoder always fails");
            }
            other => panic!("expected EncodingMismatch, got {other:?}"),
        }
        assert!(
            !repo.exists(&ID_A.to_owned()).await.unwrap(),
            "a failed encode must not have written anything"
        );
    }

    #[tokio::test]
    async fn a_misfiled_object_fails_the_id_vs_key_check() {
        let fs = store();
        let repo = repo_over::<Sealed>(&fs, "mem:");
        fs.write(
            &Path::new(format!("mem:/{ID_A}.json")),
            sealed(ID_B, 1).to_bytes().unwrap(),
        )
        .await
        .expect("seed misfiled object");

        let err = repo.load(&ID_A.to_owned()).await.expect_err("misfiled");

        match err {
            RepositoryErrorKind::IntegrityCheckFailed { reason } => {
                assert!(reason.contains(ID_A) && reason.contains(ID_B), "{reason}");
            }
            other => panic!("expected IntegrityCheckFailed, got {other:?}"),
        }
    }

    #[test]
    fn every_filesystem_error_variant_folds_into_io() {
        let cases: Vec<(FileSystemErrorKind, &str)> = vec![
            (
                FileSystemErrorKind::NotFound {
                    path: Path::new("p"),
                },
                "NotFound",
            ),
            (
                FileSystemErrorKind::PermissionDenied {
                    path: Path::new("p"),
                },
                "PermissionDenied",
            ),
            (FileSystemErrorKind::AuthFailed, "PermissionDenied"),
            (
                FileSystemErrorKind::UnsupportedScheme {
                    scheme: "gs".to_owned(),
                },
                "Unsupported",
            ),
            (
                FileSystemErrorKind::ConnectionFailed {
                    msg: "refused".to_owned(),
                },
                "Network",
            ),
            (
                FileSystemErrorKind::InvalidPath {
                    path: Path::new("p"),
                    msg: "m".to_owned(),
                },
                "Malformed",
            ),
            (
                FileSystemErrorKind::Internal {
                    msg: "m".to_owned(),
                },
                "Malformed",
            ),
        ];

        for (fs_err, want) in cases {
            let label = format!("{fs_err:?}");
            let RepositoryErrorKind::Io(io) = RepositoryErrorKind::from(fs_err) else {
                panic!("{label} did not fold into Io");
            };
            let got = match io {
                IoErrorKind::NotFound { .. } => "NotFound",
                IoErrorKind::PermissionDenied { .. } => "PermissionDenied",
                IoErrorKind::Unsupported { .. } => "Unsupported",
                IoErrorKind::Network { .. } => "Network",
                IoErrorKind::Malformed { .. } => "Malformed",
            };
            assert_eq!(got, want, "{label} folded into the wrong family");
        }
    }

    #[test]
    fn every_variant_renders_its_payload_in_its_message() {
        let cases = [
            (
                RepositoryErrorKind::Io(IoErrorKind::NotFound {
                    describe: "zzz-io-subject".to_owned(),
                }),
                "zzz-io-subject",
            ),
            (
                RepositoryErrorKind::EncodingMismatch {
                    actual: Cow::Borrowed("zzz-actual"),
                },
                "zzz-actual",
            ),
            (
                RepositoryErrorKind::IntegrityCheckFailed {
                    reason: "zzz-reason".into(),
                },
                "zzz-reason",
            ),
        ];

        for (err, needle) in cases {
            let message = err.message();
            assert!(
                message.contains(needle),
                "{err:?} message {message:?} omits {needle}"
            );
        }
    }

    #[test]
    fn severity_is_error_for_every_variant() {
        let cases = [
            RepositoryErrorKind::Io(IoErrorKind::NotFound {
                describe: "p".to_owned(),
            }),
            RepositoryErrorKind::EncodingMismatch {
                actual: "yaml".into(),
            },
            RepositoryErrorKind::IntegrityCheckFailed { reason: "r".into() },
        ];
        for case in &cases {
            assert_eq!(case.severity_default(), Severity::Error, "{case:?}");
        }
    }

    #[test]
    fn cause_chains_through_io_and_stops_elsewhere() {
        #[derive(Debug)]
        struct Inner;
        impl std::fmt::Display for Inner {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("inner")
            }
        }
        impl std::error::Error for Inner {}

        let chained = RepositoryErrorKind::Io(IoErrorKind::Network {
            describe: "fs".to_owned(),
            reason: "refused".into(),
            source: Some(Box::new(Inner)),
        });
        assert!(chained.cause().is_some());

        let unchained = RepositoryErrorKind::IntegrityCheckFailed { reason: "r".into() };
        assert!(unchained.cause().is_none());
    }

    #[tokio::test]
    async fn a_caller_over_dyn_repository_accepts_a_second_impl() {
        #[derive(Debug, Default)]
        struct SingleSlotRepo(tokio::sync::RwLock<Option<Sealed>>);

        #[async_trait]
        impl Repository<Sealed> for SingleSlotRepo {
            async fn save(&self, value: &Sealed) -> Result<(), RepositoryErrorKind> {
                *self.0.write().await = Some(value.clone());
                Ok(())
            }
            async fn load(&self, id: &EntityId) -> Result<Sealed, RepositoryErrorKind> {
                self.0.read().await.clone().ok_or_else(|| {
                    RepositoryErrorKind::Io(IoErrorKind::NotFound {
                        describe: id.clone(),
                    })
                })
            }
            async fn exists(&self, _: &EntityId) -> Result<bool, RepositoryErrorKind> {
                Ok(self.0.read().await.is_some())
            }
        }

        async fn round_trip(repo: Arc<dyn Repository<Sealed>>, value: Sealed) {
            repo.save(&value).await.expect("save");
            assert!(repo.exists(&value.id).await.expect("exists"));
            assert_eq!(repo.load(&value.id).await.expect("load"), value);
        }

        let fs = store();
        round_trip(Arc::new(repo_over::<Sealed>(&fs, "mem:")), sealed(ID_A, 1)).await;
        round_trip(Arc::new(SingleSlotRepo::default()), sealed(ID_A, 1)).await;
    }

    struct NotSendNotSync(#[allow(dead_code)] std::rc::Rc<u8>, EntityId);

    impl Persistable for NotSendNotSync {
        fn entity_id(&self) -> &EntityId {
            &self.1
        }
        fn extension() -> &'static str {
            "bin"
        }
        fn to_bytes(&self) -> Result<Bytes, Cow<'static, str>> {
            Err(Cow::Borrowed("unused"))
        }
        fn from_bytes(_: &[u8]) -> Result<Self, Cow<'static, str>> {
            Err(Cow::Borrowed("unused"))
        }
    }

    fn needs_send_sync<T: Send + Sync>() {}

    #[test]
    fn file_system_repository_is_send_and_sync_regardless_of_t() {
        needs_send_sync::<FileSystemRepository<NotSendNotSync>>();
    }

    #[cfg(feature = "io-aws")]
    #[test]
    fn open_s3_and_s3a_compose_identical_keys() {
        let repo_s3 =
            FileSystemRepository::<Sealed>::open(Path::new("s3://bucket/prefix")).unwrap();
        let repo_s3a =
            FileSystemRepository::<Sealed>::open(Path::new("s3a://bucket/prefix")).unwrap();
        let key_s3 = repo_s3.key_for(&"abc".to_owned()).unwrap();
        let key_s3a = repo_s3a.key_for(&"abc".to_owned()).unwrap();
        assert_eq!(key_s3.as_str(), "s3://bucket/prefix/abc.json");
        assert_eq!(key_s3a.as_str(), "s3://bucket/prefix/abc.json");
        assert_eq!(key_s3.as_str(), key_s3a.as_str());
    }

    #[cfg(feature = "io-aws")]
    #[test]
    fn a_hand_built_s3_backend_normalizes_an_s3a_base_through_new() {
        let backend = || Arc::new(crate::io::backends::s3_fs::S3FileSystem::new("bucket"));
        let from_s3a =
            FileSystemRepository::<Sealed>::new(backend(), Path::new("s3a://bucket/prefix"));
        let from_s3 =
            FileSystemRepository::<Sealed>::new(backend(), Path::new("s3://bucket/prefix"));

        let key_s3a = from_s3a.key_for(&"abc".to_owned()).unwrap();
        assert_eq!(key_s3a.as_str(), "s3://bucket/prefix/abc.json");
        assert_eq!(
            key_s3a.as_str(),
            from_s3.key_for(&"abc".to_owned()).unwrap().as_str()
        );
    }

    #[cfg(feature = "io-aws")]
    #[test]
    fn the_bucket_is_the_first_segment_whatever_the_prefix_depth() {
        for (base, want) in [
            ("s3://bucket", "bucket"),
            ("s3://bucket/", "bucket"),
            ("s3://bucket/a/b/c", "bucket"),
            ("s3a://bucket/a/b", "bucket"),
            ("s3://", ""),
        ] {
            assert_eq!(extract_bucket(base), want, "base {base:?}");
        }
    }

    #[cfg(not(feature = "io-aws"))]
    #[test]
    fn open_s3_without_io_aws_is_unsupported() {
        for (base, scheme) in [("s3://bucket/prefix", "s3"), ("s3a://bucket/prefix", "s3a")] {
            let err = FileSystemRepository::<Sealed>::open(Path::new(base)).unwrap_err();
            match err {
                RepositoryErrorKind::Io(IoErrorKind::Unsupported { describe, .. }) => {
                    assert_eq!(describe, scheme, "base {base:?}");
                }
                other => panic!("expected Unsupported for {base:?}, got {other:?}"),
            }
        }
    }

    #[test]
    fn open_local_puts_directory_on_backend_root_not_repository_base() {
        let dir = tempfile::tempdir().unwrap();
        let repo =
            FileSystemRepository::<Draft>::open(Path::new(dir.path().to_str().unwrap())).unwrap();
        assert_eq!(repo.base.as_str(), "");
        let key = repo.key_for(&"abc".to_owned()).unwrap();
        assert_eq!(key.as_str(), "/abc.yaml");
    }

    #[tokio::test]
    async fn open_local_round_trips_over_a_real_backend() {
        let dir = tempfile::tempdir().unwrap();
        let repo =
            FileSystemRepository::<Draft>::open(Path::new(dir.path().to_str().unwrap())).unwrap();
        let value = Draft {
            id: "abc".to_owned(),
            payload: 7,
        };
        repo.save(&value).await.unwrap();
        let loaded = repo.load(&"abc".to_owned()).await.unwrap();
        assert_eq!(loaded, value);
    }

    #[tokio::test]
    async fn open_strips_the_file_scheme_off_the_backend_root() {
        let dir = tempfile::tempdir().unwrap();
        let repo = FileSystemRepository::<Draft>::open(Path::new(format!(
            "file://{}",
            dir.path().to_str().unwrap()
        )))
        .unwrap();
        assert_eq!(repo.base.as_str(), "");

        let value = Draft {
            id: "abc".to_owned(),
            payload: 3,
        };
        repo.save(&value).await.unwrap();
        assert_eq!(repo.load(&"abc".to_owned()).await.unwrap(), value);

        assert!(dir.path().join("abc.yaml").exists());
    }

    #[test]
    fn open_windows_drive_letter_dispatches_local_both_forms() {
        let backslash = FileSystemRepository::<Draft>::open(Path::new("C:\\Users\\x")).unwrap();
        let forwardslash = FileSystemRepository::<Draft>::open(Path::new("C:/Users/x")).unwrap();
        assert_eq!(backslash.base.as_str(), "");
        assert_eq!(forwardslash.base.as_str(), "");
        assert_eq!(
            backslash.key_for(&"abc".to_owned()).unwrap().as_str(),
            "/abc.yaml"
        );
        assert_eq!(
            forwardslash.key_for(&"abc".to_owned()).unwrap().as_str(),
            "/abc.yaml"
        );
    }

    #[test]
    fn open_doubled_separator_drive_letter_dispatches_local() {
        for uri in ["C://Users/x", "c://data"] {
            let repo = FileSystemRepository::<Draft>::open(Path::new(uri)).unwrap();
            assert_eq!(repo.base.as_str(), "");
            assert!(repo
                .key_for(&"abc".to_owned())
                .unwrap()
                .as_str()
                .ends_with("/abc.yaml"));
        }
    }

    #[test]
    fn open_unknown_multi_letter_scheme_still_errors() {
        let err = FileSystemRepository::<Draft>::open(Path::new("gs://bucket")).unwrap_err();
        match err {
            RepositoryErrorKind::Io(IoErrorKind::Unsupported { describe, .. }) => {
                assert_eq!(describe, "gs");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn open_unsupported_scheme_errors_not_panics() {
        for scheme_url in ["gs://bucket/prefix", "http://example.com/x"] {
            let err = FileSystemRepository::<Draft>::open(Path::new(scheme_url)).unwrap_err();
            match err {
                RepositoryErrorKind::Io(IoErrorKind::Unsupported { describe, .. }) => {
                    let expected_scheme = scheme_url.split_once("://").unwrap().0;
                    assert_eq!(describe, expected_scheme);
                }
                other => panic!("expected Unsupported, got {other:?}"),
            }
        }
    }

    #[test]
    fn open_does_not_touch_the_filesystem_on_an_absent_directory() {
        let dir = tempfile::tempdir().unwrap();
        let absent = dir.path().join("not-created-yet");

        FileSystemRepository::<Draft>::open(Path::new(absent.to_str().unwrap()))
            .expect("an absent root is not open's error to raise");

        assert!(!absent.exists(), "open must not create its root");
    }

    #[tokio::test]
    async fn open_absent_directory_fails_on_first_save_not_on_open() {
        let absent = "/definitely/does/not/exist/on/this/machine";
        let repo = FileSystemRepository::<Draft>::open(Path::new(absent)).unwrap();
        let value = Draft {
            id: "abc".to_owned(),
            payload: 1,
        };
        let result = repo.save(&value).await;
        assert!(result.is_err());
    }
}
