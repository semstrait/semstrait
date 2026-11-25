use std::borrow::Cow;
use std::sync::Arc;

use bytes::Bytes;
use semstrait_common::identity::EntityId;
use semstrait_common::io::backends::local_fs::LocalFileSystem;
use semstrait_common::io::{
    FileSystemRepository, InMemoryFileSystem, IoErrorKind, Path, Persistable, Repository,
    RepositoryErrorKind,
};

#[derive(Debug, PartialEq)]
struct Doc(EntityId, u32);

impl Persistable for Doc {
    fn entity_id(&self) -> &EntityId {
        &self.0
    }
    fn extension() -> &'static str {
        "bin"
    }
    fn to_bytes(&self) -> Result<Bytes, Cow<'static, str>> {
        Ok(Bytes::from(format!("{}|{}", self.0, self.1)))
    }
    fn from_bytes(b: &[u8]) -> Result<Self, Cow<'static, str>> {
        let s = std::str::from_utf8(b).map_err(|e| Cow::Owned(e.to_string()))?;
        let (id, n) = s.split_once('|').ok_or(Cow::Borrowed("no sep"))?;
        Ok(Doc(
            id.to_owned(),
            n.parse().map_err(|_| Cow::Borrowed("bad num"))?,
        ))
    }
}

fn repo(dir: &std::path::Path, base: &str) -> FileSystemRepository<Doc> {
    FileSystemRepository::new(Arc::new(LocalFileSystem::new(dir)), Path::new(base))
}

#[tokio::test]
async fn every_base_form_round_trips_over_a_real_backend() {
    for base in ["root", "root/", "", "/", "a/b", "file:///root"] {
        let dir = tempfile::tempdir().expect("tmpdir");
        let r = repo(dir.path(), base);

        r.save(&Doc("id-1".into(), 7))
            .await
            .unwrap_or_else(|e| panic!("save failed for base {base:?}: {e:?}"));
        let loaded = r
            .load(&"id-1".to_owned())
            .await
            .unwrap_or_else(|e| panic!("load failed for base {base:?}: {e:?}"));

        assert_eq!(
            loaded,
            Doc("id-1".into(), 7),
            "base {base:?} round-tripped to a different value"
        );
    }
}

#[tokio::test]
async fn distinct_bases_stay_isolated_on_a_real_backend() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let a = repo(dir.path(), "tenant-a");
    let b = repo(dir.path(), "tenant-b");

    a.save(&Doc("id-1".into(), 1)).await.expect("save a");
    b.save(&Doc("id-1".into(), 2)).await.expect("save b");

    assert_eq!(
        a.load(&"id-1".to_owned()).await.expect("load a").1,
        1,
        "tenant-a read back tenant-b's value"
    );
    assert_eq!(
        b.load(&"id-1".to_owned()).await.expect("load b").1,
        2,
        "tenant-b read back tenant-a's value"
    );
}

#[tokio::test]
async fn an_id_the_os_cannot_name_fails_without_panicking() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let r = repo(dir.path(), "root");
    let too_long: EntityId = "z".repeat(300);

    let saved = r.save(&Doc(too_long.clone(), 1)).await;
    assert!(
        saved.is_err(),
        "a 300-byte id must not report success: {saved:?}"
    );
    assert!(
        !r.exists(&too_long).await.unwrap_or(false),
        "nothing may be stored under an id the OS cannot name"
    );
}

#[tokio::test]
async fn a_scheme_qualified_base_and_its_bare_form_are_one_location_here_and_two_in_memory() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let fs = Arc::new(LocalFileSystem::new(dir.path()));
    let bare: FileSystemRepository<Doc> = FileSystemRepository::new(fs.clone(), Path::new("root"));
    let url: FileSystemRepository<Doc> =
        FileSystemRepository::new(fs.clone(), Path::new("file:///root"));

    bare.save(&Doc("id-1".into(), 1)).await.expect("save bare");
    url.save(&Doc("id-1".into(), 2))
        .await
        .expect("last-writer-wins: the bases collapse, so this overwrites");
    assert_eq!(
        bare.load(&"id-1".to_owned()).await.expect("load bare").1,
        2,
        "expected the url-form write to be visible through the bare form"
    );

    let mem = Arc::new(InMemoryFileSystem::default());
    let m_bare: FileSystemRepository<Doc> =
        FileSystemRepository::new(mem.clone(), Path::new("root"));
    let m_url: FileSystemRepository<Doc> =
        FileSystemRepository::new(mem.clone(), Path::new("file:///root"));
    m_bare
        .save(&Doc("id-1".into(), 1))
        .await
        .expect("save bare");
    m_url
        .save(&Doc("id-1".into(), 2))
        .await
        .expect("the double keys them separately");
    assert_eq!(
        m_bare.load(&"id-1".to_owned()).await.expect("load bare").1,
        1,
        "the double kept two objects where the real backend has one"
    );
}

#[tokio::test]
async fn a_dotdot_key_the_memory_double_accepts_is_refused_by_a_real_backend() {
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
        fn from_bytes(b: &[u8]) -> Result<Self, Cow<'static, str>> {
            Ok(Extensionless(String::from_utf8_lossy(b).into_owned()))
        }
    }

    let dir = tempfile::tempdir().expect("tmpdir");
    let r: FileSystemRepository<Extensionless> = FileSystemRepository::new(
        Arc::new(LocalFileSystem::new(dir.path())),
        Path::new("base"),
    );

    let err = r
        .save(&Extensionless(".".to_owned()))
        .await
        .expect_err("a real backend must refuse a `..` segment");
    let RepositoryErrorKind::Io(IoErrorKind::Malformed { describe, reason }) = &err else {
        panic!("expected the transport's path parse to refuse `..`, got {err:?}");
    };
    assert_eq!(describe, "base/..");
    assert!(
        reason.contains("illegal character sequence \"..\""),
        "the refusal must name the traversal segment: {reason}"
    );

    r.save(&Extensionless("x".to_owned()))
        .await
        .expect("an ordinary id still saves");
}
