use async_trait::async_trait;
use bytes::Bytes;

use crate::diagnostic::{Diagnose, Severity};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Path(String);

impl Path {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn parse(value: impl Into<String>) -> Result<Self, FileSystemErrorKind> {
        let value = value.into();
        if value.is_empty() {
            return Err(FileSystemErrorKind::InvalidPath {
                path: Self(value),
                msg: "empty path".to_owned(),
            });
        }
        if let Some((scheme, _)) = value.split_once("://") {
            if !is_drive_letter(scheme) && !is_known_scheme(scheme) {
                return Err(FileSystemErrorKind::UnsupportedScheme {
                    scheme: scheme.to_owned(),
                });
            }
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for Path {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

fn is_known_scheme(scheme: &str) -> bool {
    matches!(scheme, "file" | "s3" | "s3a")
}

pub(crate) fn is_drive_letter(scheme: &str) -> bool {
    let mut chars = scheme.chars();
    matches!((chars.next(), chars.next()), (Some(c), None) if c.is_ascii_alphabetic())
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileEntry {
    pub path: Path,
    pub size: u64,
    pub modified_at: Option<std::time::SystemTime>,
}

impl FileEntry {
    pub fn new(path: Path, size: u64, modified_at: Option<std::time::SystemTime>) -> Self {
        Self {
            path,
            size,
            modified_at,
        }
    }
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum FileSystemErrorKind {
    NotFound { path: Path },

    UnsupportedScheme { scheme: String },

    InvalidPath { path: Path, msg: String },

    ConnectionFailed { msg: String },

    AuthFailed,

    PermissionDenied { path: Path },

    Internal { msg: String },
}

impl Diagnose for FileSystemErrorKind {
    fn message(&self) -> String {
        match self {
            Self::NotFound { path } => format!("not found: {path}"),
            Self::UnsupportedScheme { scheme } => {
                format!("unsupported scheme: {scheme}")
            }
            Self::InvalidPath { path, msg } => format!("invalid path '{path}': {msg}"),
            Self::ConnectionFailed { msg } => format!("connection failed: {msg}"),
            Self::AuthFailed => "authentication failed".to_owned(),
            Self::PermissionDenied { path } => format!("permission denied: {path}"),
            Self::Internal { msg } => format!("internal error: {msg}"),
        }
    }

    fn severity_default(&self) -> Severity {
        Severity::Error
    }
}

#[allow(clippy::double_must_use)]
#[async_trait]
pub trait FileSystem: Send + Sync + std::fmt::Debug {
    async fn list(&self, prefix: &Path) -> Result<Vec<FileEntry>, FileSystemErrorKind>;

    async fn read(&self, path: &Path) -> Result<Bytes, FileSystemErrorKind>;

    async fn write(&self, path: &Path, data: Bytes) -> Result<(), FileSystemErrorKind>;

    async fn exists(&self, path: &Path) -> Result<bool, FileSystemErrorKind>;
}

#[derive(Clone, Debug, Default)]
pub struct InMemoryFileSystem {
    objects: std::sync::Arc<tokio::sync::RwLock<std::collections::HashMap<String, Bytes>>>,
}

impl InMemoryFileSystem {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl FileSystem for InMemoryFileSystem {
    async fn list(&self, prefix: &Path) -> Result<Vec<FileEntry>, FileSystemErrorKind> {
        let prefix = prefix.as_str();
        let guard = self.objects.read().await;
        let entries = guard
            .iter()
            .filter(|(key, _)| key.starts_with(prefix))
            .map(|(key, bytes)| FileEntry::new(Path::new(key.clone()), bytes.len() as u64, None))
            .collect();
        Ok(entries)
    }

    async fn read(&self, path: &Path) -> Result<Bytes, FileSystemErrorKind> {
        let guard = self.objects.read().await;
        guard
            .get(path.as_str())
            .cloned()
            .ok_or_else(|| FileSystemErrorKind::NotFound { path: path.clone() })
    }

    async fn write(&self, path: &Path, data: Bytes) -> Result<(), FileSystemErrorKind> {
        let mut guard = self.objects.write().await;
        guard.insert(path.as_str().to_owned(), data);
        Ok(())
    }

    async fn exists(&self, path: &Path) -> Result<bool, FileSystemErrorKind> {
        let guard = self.objects.read().await;
        Ok(guard.contains_key(path.as_str()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn _assert_object_safe(_: &dyn FileSystem) {}

    fn boxed() -> Box<dyn FileSystem> {
        Box::new(InMemoryFileSystem::new())
    }

    #[test]
    fn path_parse_accepts_known_schemes() {
        for uri in ["s3://bucket/key", "s3a://bucket/key", "file:///abs/path"] {
            assert!(Path::parse(uri).is_ok(), "expected {uri} to parse");
        }
    }

    #[test]
    fn path_parse_accepts_s3a_but_rejects_gs() {
        assert!(Path::parse("s3a://bucket/key").is_ok());
        let err = Path::parse("gs://bucket/key").unwrap_err();
        match err {
            FileSystemErrorKind::UnsupportedScheme { scheme } => assert_eq!(scheme, "gs"),
            other => panic!("expected UnsupportedScheme, got {other:?}"),
        }
    }

    #[test]
    fn path_parse_rejects_descoped_schemes() {
        for uri in [
            "gs://bucket/key",
            "abfss://c@a.dfs.core.windows.net/key",
            "abfs://c@a/key",
        ] {
            let err = Path::parse(uri).unwrap_err();
            assert!(
                matches!(err, FileSystemErrorKind::UnsupportedScheme { .. }),
                "expected {uri} to be rejected as UnsupportedScheme, got {err:?}"
            );
        }
    }

    #[test]
    fn path_parse_accepts_bare_local_path() {
        let p = Path::parse("/abs/local/path").unwrap();
        assert_eq!(p.as_str(), "/abs/local/path");
        let rel = Path::parse("./relative/path").unwrap();
        assert_eq!(rel.as_str(), "./relative/path");
    }

    #[test]
    fn path_parse_accepts_windows_drive_letter_forms() {
        for uri in [
            "C:\\Users\\x",
            "C:/Users/x",
            "C://Users/x",
            "c://data",
            "D://x/y",
        ] {
            assert!(Path::parse(uri).is_ok(), "expected {uri} to parse");
        }
    }

    #[test]
    fn path_parse_still_rejects_multi_letter_unknown_scheme() {
        let err = Path::parse("gs://bucket/key").unwrap_err();
        match err {
            FileSystemErrorKind::UnsupportedScheme { scheme } => assert_eq!(scheme, "gs"),
            other => panic!("expected UnsupportedScheme, got {other:?}"),
        }
    }

    #[test]
    fn path_parse_rejects_unknown_scheme() {
        let err = Path::parse("ftp://host/file").unwrap_err();
        match err {
            FileSystemErrorKind::UnsupportedScheme { scheme } => assert_eq!(scheme, "ftp"),
            other => panic!("expected UnsupportedScheme, got {other:?}"),
        }
    }

    #[test]
    fn path_parse_rejects_empty() {
        let err = Path::parse("").unwrap_err();
        assert!(matches!(err, FileSystemErrorKind::InvalidPath { .. }));
    }

    #[test]
    fn path_new_skips_validation() {
        assert_eq!(Path::new("ftp://anything").as_str(), "ftp://anything");
    }

    #[test]
    fn file_entry_constructor_sets_fields() {
        let e = FileEntry::new(Path::new("mem/x"), 42, None);
        assert_eq!(e.path.as_str(), "mem/x");
        assert_eq!(e.size, 42);
        assert!(e.modified_at.is_none());
    }

    #[test]
    fn diagnose_message_not_found_carries_path() {
        let e = FileSystemErrorKind::NotFound {
            path: Path::new("s3://bucket/missing"),
        };
        assert_eq!(e.message(), "not found: s3://bucket/missing");
    }

    #[test]
    fn diagnose_message_unsupported_scheme_carries_scheme() {
        let e = FileSystemErrorKind::UnsupportedScheme {
            scheme: "ftp".to_owned(),
        };
        assert_eq!(e.message(), "unsupported scheme: ftp");
    }

    #[test]
    fn diagnose_severity_is_error_for_every_variant() {
        let cases = [
            FileSystemErrorKind::NotFound {
                path: Path::new("p"),
            },
            FileSystemErrorKind::UnsupportedScheme {
                scheme: "x".to_owned(),
            },
            FileSystemErrorKind::InvalidPath {
                path: Path::new("p"),
                msg: "m".to_owned(),
            },
            FileSystemErrorKind::ConnectionFailed {
                msg: "m".to_owned(),
            },
            FileSystemErrorKind::AuthFailed,
            FileSystemErrorKind::PermissionDenied {
                path: Path::new("p"),
            },
            FileSystemErrorKind::Internal {
                msg: "m".to_owned(),
            },
        ];
        for c in &cases {
            assert_eq!(c.severity_default(), Severity::Error);
        }
    }

    #[tokio::test]
    async fn write_then_read_round_trips() {
        let fs = boxed();
        let path = Path::new("mem/round.bin");
        fs.write(&path, Bytes::from_static(b"payload"))
            .await
            .unwrap();
        let got = fs.read(&path).await.unwrap();
        assert_eq!(got.as_ref(), b"payload");
    }

    #[tokio::test]
    async fn write_overwrites_existing_object() {
        let fs = InMemoryFileSystem::new();
        let path = Path::new("mem/k");
        fs.write(&path, Bytes::from_static(b"old")).await.unwrap();
        fs.write(&path, Bytes::from_static(b"new")).await.unwrap();
        assert_eq!(fs.read(&path).await.unwrap().as_ref(), b"new");
    }

    #[tokio::test]
    async fn read_missing_emits_not_found() {
        let fs = InMemoryFileSystem::new();
        let err = fs.read(&Path::new("mem/absent")).await.unwrap_err();
        match err {
            FileSystemErrorKind::NotFound { path } => assert_eq!(path.as_str(), "mem/absent"),
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn exists_reflects_presence() {
        let fs = InMemoryFileSystem::new();
        let path = Path::new("mem/present");
        assert!(!fs.exists(&path).await.unwrap());
        fs.write(&path, Bytes::from_static(b"x")).await.unwrap();
        assert!(fs.exists(&path).await.unwrap());
    }

    #[tokio::test]
    async fn exists_is_exact_not_prefix() {
        let fs = InMemoryFileSystem::new();
        fs.write(&Path::new("dir/file.bin"), Bytes::from_static(b"x"))
            .await
            .unwrap();
        assert!(!fs.exists(&Path::new("dir/")).await.unwrap());
        assert!(!fs.exists(&Path::new("dir")).await.unwrap());
    }

    #[tokio::test]
    async fn list_returns_all_descendants_under_prefix() {
        let fs = InMemoryFileSystem::new();
        for key in [
            "data/a.bin",
            "data/sub/b.bin",
            "data/sub/deep/c.bin",
            "other/d.bin",
        ] {
            fs.write(&Path::new(key), Bytes::from_static(b"x"))
                .await
                .unwrap();
        }
        let mut got: Vec<String> = fs
            .list(&Path::new("data/"))
            .await
            .unwrap()
            .into_iter()
            .map(|e| e.path.as_str().to_owned())
            .collect();
        got.sort();
        assert_eq!(
            got,
            vec![
                "data/a.bin".to_owned(),
                "data/sub/b.bin".to_owned(),
                "data/sub/deep/c.bin".to_owned(),
            ]
        );
    }

    #[tokio::test]
    async fn list_reports_object_size() {
        let fs = InMemoryFileSystem::new();
        fs.write(&Path::new("p/x"), Bytes::from_static(b"12345"))
            .await
            .unwrap();
        let entries = fs.list(&Path::new("p/")).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].size, 5);
    }

    #[tokio::test]
    async fn list_empty_prefix_returns_everything() {
        let fs = InMemoryFileSystem::new();
        fs.write(&Path::new("a"), Bytes::from_static(b"x"))
            .await
            .unwrap();
        fs.write(&Path::new("b"), Bytes::from_static(b"y"))
            .await
            .unwrap();
        let entries = fs.list(&Path::new("")).await.unwrap();
        assert_eq!(entries.len(), 2);
    }
}
