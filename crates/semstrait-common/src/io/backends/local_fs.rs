use async_trait::async_trait;
use bytes::Bytes;

use super::object_store_fs as plumbing;
use crate::io::fs::{FileEntry, FileSystem, FileSystemErrorKind, Path};

#[derive(Debug)]
pub struct LocalFileSystem {
    root: std::path::PathBuf,
}

impl LocalFileSystem {
    pub fn new(root: impl Into<std::path::PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn store(
        &self,
        path: &Path,
    ) -> Result<object_store::local::LocalFileSystem, FileSystemErrorKind> {
        object_store::local::LocalFileSystem::new_with_prefix(&self.root)
            .map_err(|e| plumbing::fold(e, path))
    }

    fn to_key(path: &Path) -> Path {
        match path.as_str().strip_prefix("file://") {
            Some(rest) => Path::new(rest),
            None => path.clone(),
        }
    }

    fn reject_foreign_scheme(path: &Path) -> Result<(), FileSystemErrorKind> {
        match path.as_str().split_once("://") {
            Some(("file", _)) | None => Ok(()),
            Some((scheme, _)) => Err(FileSystemErrorKind::UnsupportedScheme {
                scheme: scheme.to_owned(),
            }),
        }
    }
}

#[async_trait]
impl FileSystem for LocalFileSystem {
    async fn list(&self, prefix: &Path) -> Result<Vec<FileEntry>, FileSystemErrorKind> {
        Self::reject_foreign_scheme(prefix)?;
        let store = self.store(prefix)?;
        let scheme = if prefix.as_str().starts_with("file://") {
            "file:///"
        } else {
            ""
        };
        plumbing::list_literal(&store, &Self::to_key(prefix), |key| {
            format!("{scheme}{key}")
        })
        .await
    }

    async fn read(&self, path: &Path) -> Result<Bytes, FileSystemErrorKind> {
        Self::reject_foreign_scheme(path)?;
        plumbing::read(&self.store(path)?, &Self::to_key(path), path).await
    }

    async fn write(&self, path: &Path, data: Bytes) -> Result<(), FileSystemErrorKind> {
        Self::reject_foreign_scheme(path)?;
        plumbing::write(&self.store(path)?, &Self::to_key(path), path, data).await
    }

    async fn exists(&self, path: &Path) -> Result<bool, FileSystemErrorKind> {
        Self::reject_foreign_scheme(path)?;
        plumbing::exists(&self.store(path)?, &Self::to_key(path), path).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn foreign_scheme_is_rejected_before_conversion() {
        let err = LocalFileSystem::reject_foreign_scheme(&Path::new("s3://bucket/key"))
            .expect_err("a local back-end must reject an s3 path");
        assert!(matches!(
            err,
            FileSystemErrorKind::UnsupportedScheme { ref scheme } if scheme == "s3"
        ));
    }

    #[test]
    fn own_scheme_and_scheme_less_paths_are_accepted() {
        LocalFileSystem::reject_foreign_scheme(&Path::new("file:///tmp/x")).expect("file:// ok");
        LocalFileSystem::reject_foreign_scheme(&Path::new("data/a.bin")).expect("relative ok");
    }

    fn rooted() -> (tempfile::TempDir, LocalFileSystem) {
        let dir = tempfile::tempdir().expect("tempdir");
        let fs = LocalFileSystem::new(dir.path());
        (dir, fs)
    }

    #[tokio::test]
    async fn write_then_read_round_trips_the_bytes() {
        let (_dir, fs) = rooted();
        let path = Path::new("data/x.bin");

        fs.write(&path, Bytes::from_static(b"hello"))
            .await
            .expect("write");

        assert_eq!(
            fs.read(&path).await.expect("read"),
            Bytes::from_static(b"hello")
        );
    }

    #[tokio::test]
    async fn list_matches_a_partial_segment_which_the_store_alone_would_not() {
        let (_dir, fs) = rooted();
        for key in ["data/x.bin", "database/y.bin", "data/sub/z.bin"] {
            fs.write(&Path::new(key), Bytes::from_static(b"b"))
                .await
                .expect("write");
        }

        let mut got: Vec<String> = fs
            .list(&Path::new("dat"))
            .await
            .expect("list")
            .into_iter()
            .map(|e| e.path.as_str().to_owned())
            .collect();
        got.sort();

        assert_eq!(got, ["data/sub/z.bin", "data/x.bin", "database/y.bin"]);
    }

    #[tokio::test]
    async fn a_widened_query_is_narrowed_back_to_the_literal_prefix() {
        let (_dir, fs) = rooted();
        for key in ["data/x.bin", "data/sub/z.bin"] {
            fs.write(&Path::new(key), Bytes::from_static(b"b"))
                .await
                .expect("write");
        }

        let got: Vec<String> = fs
            .list(&Path::new("data/x"))
            .await
            .expect("list")
            .into_iter()
            .map(|e| e.path.as_str().to_owned())
            .collect();

        assert_eq!(got, ["data/x.bin"]);
    }

    #[tokio::test]
    async fn a_trailing_separator_scopes_out_a_similarly_prefixed_sibling() {
        let (_dir, fs) = rooted();
        for key in ["data/x.bin", "database/y.bin", "data/sub/z.bin"] {
            fs.write(&Path::new(key), Bytes::from_static(b"b"))
                .await
                .expect("write");
        }

        let mut got: Vec<String> = fs
            .list(&Path::new("data/"))
            .await
            .expect("list")
            .into_iter()
            .map(|e| e.path.as_str().to_owned())
            .collect();
        got.sort();

        assert_eq!(got, ["data/sub/z.bin", "data/x.bin"]);
    }

    #[tokio::test]
    async fn a_list_entry_carries_the_size_and_a_modified_time() {
        let (_dir, fs) = rooted();
        fs.write(&Path::new("data/x.bin"), Bytes::from_static(b"12345"))
            .await
            .expect("write");

        let entries = fs.list(&Path::new("data/")).await.expect("list");

        let entry = entries.first().expect("one entry");
        assert_eq!(entry.size, 5);
        assert!(entry.modified_at.is_some(), "the store supplies a mtime");
        assert_eq!(
            fs.read(&entry.path).await.expect("read back"),
            Bytes::from_static(b"12345")
        );
    }

    #[tokio::test]
    async fn exists_reports_absence_as_false_not_as_an_error() {
        let (_dir, fs) = rooted();
        let path = Path::new("data/absent.bin");

        assert!(!fs.exists(&path).await.expect("absence is not an error"));

        fs.write(&path, Bytes::from_static(b"b"))
            .await
            .expect("write");
        assert!(fs.exists(&path).await.expect("exists"));
    }

    #[tokio::test]
    async fn reading_an_absent_object_is_not_found_naming_the_requested_path() {
        let (_dir, fs) = rooted();
        let path = Path::new("data/absent.bin");

        let err = fs.read(&path).await.expect_err("absent");

        match err {
            FileSystemErrorKind::NotFound { path: got } => assert_eq!(got, path),
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_second_write_replaces_the_whole_object() {
        let (_dir, fs) = rooted();
        let path = Path::new("data/x.bin");

        fs.write(&path, Bytes::from_static(b"a-longer-first-value"))
            .await
            .expect("first");
        fs.write(&path, Bytes::from_static(b"short"))
            .await
            .expect("second");

        assert_eq!(
            fs.read(&path).await.expect("read"),
            Bytes::from_static(b"short")
        );
    }

    #[tokio::test]
    async fn an_empty_prefix_lists_everything() {
        let (_dir, fs) = rooted();
        for key in ["a.bin", "deep/b.bin"] {
            fs.write(&Path::new(key), Bytes::from_static(b"b"))
                .await
                .expect("write");
        }

        let got = fs.list(&Path::new("")).await.expect("list");

        assert_eq!(got.len(), 2, "{got:?}");
    }

    #[tokio::test]
    async fn a_file_scheme_path_addresses_the_same_object_as_the_bare_key() {
        let (_dir, fs) = rooted();

        fs.write(&Path::new("file:///data/x.bin"), Bytes::from_static(b"v"))
            .await
            .expect("write");

        assert_eq!(
            fs.read(&Path::new("data/x.bin")).await.expect("bare read"),
            Bytes::from_static(b"v"),
            "the scheme must not become part of the key"
        );
        assert!(fs
            .exists(&Path::new("file:///data/x.bin"))
            .await
            .expect("exists"));
        let listed = fs.list(&Path::new("file:///data/")).await.expect("list");
        assert_eq!(listed.len(), 1, "{listed:?}");
    }

    #[tokio::test]
    async fn a_listed_entry_comes_back_in_the_addressing_of_the_prefix() {
        let (_dir, fs) = rooted();
        fs.write(&Path::new("data/x.bin"), Bytes::from_static(b"v"))
            .await
            .expect("write");

        for (prefix, entry) in [
            ("file:///data/", "file:///data/x.bin"),
            ("data/", "data/x.bin"),
        ] {
            let listed = fs.list(&Path::new(prefix)).await.expect("list");
            let paths: Vec<&str> = listed.iter().map(|e| e.path.as_str()).collect();
            assert_eq!(paths, [entry], "{prefix}");
        }
    }

    #[tokio::test]
    async fn a_not_found_names_the_callers_scheme_qualified_path() {
        let (_dir, fs) = rooted();
        let asked = Path::new("file:///data/absent.bin");

        match fs.read(&asked).await.expect_err("absent") {
            FileSystemErrorKind::NotFound { path } => assert_eq!(path, asked),
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_foreign_scheme_is_refused_by_every_method() {
        let (_dir, fs) = rooted();
        let path = Path::new("s3://bucket/key");

        for err in [
            fs.read(&path).await.err(),
            fs.write(&path, Bytes::new()).await.err(),
            fs.exists(&path).await.err(),
            fs.list(&path).await.err(),
        ] {
            match err.expect("every method must refuse") {
                FileSystemErrorKind::UnsupportedScheme { scheme } => assert_eq!(scheme, "s3"),
                other => panic!("expected UnsupportedScheme, got {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn a_listed_entry_re_reads_when_the_key_contains_a_percent() {
        let (_dir, fs) = rooted();
        let path = Path::new("data/50%off.csv");
        fs.write(&path, Bytes::from_static(b"discount"))
            .await
            .expect("write");

        let listed = fs.list(&Path::new("data/")).await.expect("list");
        let entry = listed.first().expect("one entry");

        let reread = fs.read(&entry.path).await.expect("re-read");
        assert_eq!(reread, Bytes::from_static(b"discount"));

        let exists = fs.exists(&entry.path).await.expect("re-exists");
        assert!(exists);

        let relisted = fs.list(&entry.path).await.expect("re-list");
        assert_eq!(relisted.len(), 1, "{relisted:?}");
    }

    #[tokio::test]
    async fn a_traversal_segment_is_rejected_as_an_invalid_path_not_rewritten() {
        let (_dir, fs) = rooted();

        let err = fs
            .read(&Path::new("a/../../b"))
            .await
            .expect_err("traversal must error, not silently rewrite");

        assert!(
            matches!(err, FileSystemErrorKind::InvalidPath { .. }),
            "{err:?}"
        );
    }
}
