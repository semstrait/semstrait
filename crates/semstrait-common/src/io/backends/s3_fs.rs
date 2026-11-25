use async_trait::async_trait;
use bytes::Bytes;

use super::object_store_fs as plumbing;
use crate::io::fs::{FileEntry, FileSystem, FileSystemErrorKind, Path};

#[derive(Debug)]
pub struct S3FileSystem {
    bucket: String,
}

impl S3FileSystem {
    pub fn new(bucket: impl Into<String>) -> Self {
        Self {
            bucket: bucket.into(),
        }
    }

    fn store(&self, path: &Path) -> Result<object_store::aws::AmazonS3, FileSystemErrorKind> {
        object_store::aws::AmazonS3Builder::from_env()
            .with_bucket_name(&self.bucket)
            .build()
            .map_err(|e| plumbing::fold(e, path))
    }

    fn strip_bucket(&self, path: &Path) -> Path {
        let scheme_and_bucket = format!("s3://{}/", self.bucket);
        match path.as_str().strip_prefix(&scheme_and_bucket) {
            Some(key) => Path::new(key),
            None => path.clone(),
        }
    }

    fn reject_foreign_scheme(path: &Path) -> Result<(), FileSystemErrorKind> {
        match path.as_str().split_once("://") {
            Some(("s3", _)) | None => Ok(()),
            Some((scheme, _)) => Err(FileSystemErrorKind::UnsupportedScheme {
                scheme: scheme.to_owned(),
            }),
        }
    }

    async fn list_in<S: object_store::ObjectStore>(
        &self,
        store: &S,
        prefix: &Path,
    ) -> Result<Vec<FileEntry>, FileSystemErrorKind> {
        let key_prefix = self.strip_bucket(prefix);
        let bucket = self.bucket.clone();
        plumbing::list_literal(store, &key_prefix, move |key| {
            format!("s3://{bucket}/{key}")
        })
        .await
    }

    async fn read_in<S: object_store::ObjectStoreExt>(
        &self,
        store: &S,
        path: &Path,
    ) -> Result<Bytes, FileSystemErrorKind> {
        plumbing::read(store, &self.strip_bucket(path), path).await
    }

    async fn write_in<S: object_store::ObjectStoreExt>(
        &self,
        store: &S,
        path: &Path,
        data: Bytes,
    ) -> Result<(), FileSystemErrorKind> {
        plumbing::write(store, &self.strip_bucket(path), path, data).await
    }

    async fn exists_in<S: object_store::ObjectStoreExt>(
        &self,
        store: &S,
        path: &Path,
    ) -> Result<bool, FileSystemErrorKind> {
        plumbing::exists(store, &self.strip_bucket(path), path).await
    }
}

#[async_trait]
impl FileSystem for S3FileSystem {
    async fn list(&self, prefix: &Path) -> Result<Vec<FileEntry>, FileSystemErrorKind> {
        Self::reject_foreign_scheme(prefix)?;
        self.list_in(&self.store(prefix)?, prefix).await
    }

    async fn read(&self, path: &Path) -> Result<Bytes, FileSystemErrorKind> {
        Self::reject_foreign_scheme(path)?;
        self.read_in(&self.store(path)?, path).await
    }

    async fn write(&self, path: &Path, data: Bytes) -> Result<(), FileSystemErrorKind> {
        Self::reject_foreign_scheme(path)?;
        self.write_in(&self.store(path)?, path, data).await
    }

    async fn exists(&self, path: &Path) -> Result<bool, FileSystemErrorKind> {
        Self::reject_foreign_scheme(path)?;
        self.exists_in(&self.store(path)?, path).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn foreign_scheme_is_rejected_before_conversion() {
        let err = S3FileSystem::reject_foreign_scheme(&Path::new("file:///tmp/x"))
            .expect_err("an s3 back-end must reject a file path");
        assert!(matches!(
            err,
            FileSystemErrorKind::UnsupportedScheme { ref scheme } if scheme == "file"
        ));
    }

    #[test]
    fn own_scheme_and_scheme_less_paths_are_accepted() {
        S3FileSystem::reject_foreign_scheme(&Path::new("s3://bucket/key")).expect("s3:// ok");
        S3FileSystem::reject_foreign_scheme(&Path::new("data/a.bin")).expect("relative ok");
    }

    #[test]
    fn the_bucket_prefix_comes_off_and_only_this_buckets_does() {
        let fs = S3FileSystem::new("mine");
        let key = Path::new("models/a.json");

        assert_eq!(fs.strip_bucket(&Path::new("s3://mine/models/a.json")), key);
        assert_eq!(fs.strip_bucket(&key), key, "a bare key is already a key");

        let foreign = Path::new("s3://other/models/a.json");
        assert_eq!(fs.strip_bucket(&foreign), foreign);
    }

    #[test]
    fn a_bucket_name_prefix_is_not_mistaken_for_the_bucket() {
        let fs = S3FileSystem::new("mine");
        let sibling = Path::new("s3://mineral/k");

        assert_eq!(fs.strip_bucket(&sibling), sibling);
    }

    #[tokio::test]
    async fn a_listed_s3_entry_re_addresses_the_same_object_when_the_key_holds_a_percent() {
        let store = object_store::memory::InMemory::new();
        let fs = S3FileSystem::new("mine");
        let asked = Path::new("s3://mine/data/50%off.csv");

        fs.write_in(&store, &asked, Bytes::from_static(b"discount"))
            .await
            .expect("write");

        let listed = fs
            .list_in(&store, &Path::new("s3://mine/data/"))
            .await
            .expect("list");
        let entry = listed.first().expect("one entry");
        assert_eq!(entry.path.as_str(), asked.as_str(), "listed != asked");

        assert_eq!(
            fs.read_in(&store, &entry.path).await.expect("re-read"),
            Bytes::from_static(b"discount")
        );
        assert!(fs.exists_in(&store, &entry.path).await.expect("re-exists"));
        assert_eq!(
            fs.list_in(&store, &entry.path)
                .await
                .expect("re-list")
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn a_foreign_scheme_is_refused_by_every_method_before_any_store_is_built() {
        let fs = S3FileSystem::new("mine");
        let path = Path::new("file:///tmp/x");

        for err in [
            fs.read(&path).await.err(),
            fs.write(&path, Bytes::new()).await.err(),
            fs.exists(&path).await.err(),
            fs.list(&path).await.err(),
        ] {
            match err.expect("every method must refuse") {
                FileSystemErrorKind::UnsupportedScheme { scheme } => assert_eq!(scheme, "file"),
                other => panic!("expected UnsupportedScheme, got {other:?}"),
            }
        }
    }
}
