use bytes::Bytes;
use futures_util::TryStreamExt;
use object_store::path::Path as ObjectPath;
use object_store::{ObjectStore, ObjectStoreExt, PutPayload};

use crate::io::fs::{FileEntry, FileSystemErrorKind, Path};

pub(super) fn fold(err: object_store::Error, path: &Path) -> FileSystemErrorKind {
    use object_store::Error as E;
    match err {
        E::NotFound { .. } => FileSystemErrorKind::NotFound { path: path.clone() },
        E::PermissionDenied { .. } => FileSystemErrorKind::PermissionDenied { path: path.clone() },
        E::Unauthenticated { .. } => FileSystemErrorKind::AuthFailed,
        E::InvalidPath { source } => FileSystemErrorKind::InvalidPath {
            path: path.clone(),
            msg: source.to_string(),
        },
        E::NotSupported { source } => FileSystemErrorKind::Internal {
            msg: format!("operation not supported: {source}"),
        },
        E::NotImplemented {
            operation,
            implementer,
        } => FileSystemErrorKind::Internal {
            msg: format!("{implementer} does not implement {operation}"),
        },
        E::Generic { store, source } => FileSystemErrorKind::ConnectionFailed {
            msg: format!("{store}: {source}"),
        },
        E::AlreadyExists { .. } | E::Precondition { .. } | E::NotModified { .. } => {
            FileSystemErrorKind::Internal {
                msg: format!("unexpected conditional-request outcome for {path}"),
            }
        }
        E::UnknownConfigurationKey { key, store } => FileSystemErrorKind::Internal {
            msg: format!("unknown configuration key {key:?} for store {store}"),
        },
        E::JoinError { source } => FileSystemErrorKind::Internal {
            msg: format!("spawned task failed: {source}"),
        },
        other => FileSystemErrorKind::Internal {
            msg: format!("object_store: {other}"),
        },
    }
}

pub(super) async fn list_literal<S: ObjectStore>(
    store: &S,
    prefix: &Path,
    to_public: impl Fn(&str) -> String,
) -> Result<Vec<FileEntry>, FileSystemErrorKind> {
    let mut literal = ObjectPath::parse(prefix.as_str())
        .map_err(|e| fold(object_store::Error::InvalidPath { source: e }, prefix))?
        .as_ref()
        .to_owned();
    if prefix.as_str().ends_with('/') && !literal.is_empty() {
        literal.push('/');
    }

    let widened = match literal.rfind('/') {
        Some(cut) => ObjectPath::from(&literal[..cut]),
        None => ObjectPath::default(),
    };

    let metas = store
        .list(Some(&widened))
        .try_collect::<Vec<_>>()
        .await
        .map_err(|e| fold(e, prefix))?;

    Ok(metas
        .into_iter()
        .filter(|meta| meta.location.as_ref().starts_with(&literal))
        .map(|meta| {
            FileEntry::new(
                Path::new(to_public(meta.location.as_ref())),
                meta.size,
                Some(meta.last_modified.into()),
            )
        })
        .collect())
}

pub(super) async fn read<S: ObjectStoreExt>(
    store: &S,
    key: &Path,
    report_as: &Path,
) -> Result<Bytes, FileSystemErrorKind> {
    let key = ObjectPath::parse(key.as_str())
        .map_err(|e| fold(object_store::Error::InvalidPath { source: e }, report_as))?;
    let got = store.get(&key).await.map_err(|e| fold(e, report_as))?;
    got.bytes().await.map_err(|e| fold(e, report_as))
}

pub(super) async fn write<S: ObjectStoreExt>(
    store: &S,
    key: &Path,
    report_as: &Path,
    data: Bytes,
) -> Result<(), FileSystemErrorKind> {
    let key = ObjectPath::parse(key.as_str())
        .map_err(|e| fold(object_store::Error::InvalidPath { source: e }, report_as))?;
    store
        .put(&key, PutPayload::from_bytes(data))
        .await
        .map(|_| ())
        .map_err(|e| fold(e, report_as))
}

pub(super) async fn exists<S: ObjectStoreExt>(
    store: &S,
    key: &Path,
    report_as: &Path,
) -> Result<bool, FileSystemErrorKind> {
    let key = ObjectPath::parse(key.as_str())
        .map_err(|e| fold(object_store::Error::InvalidPath { source: e }, report_as))?;
    match store.head(&key).await {
        Ok(_) => Ok(true),
        Err(object_store::Error::NotFound { .. }) => Ok(false),
        Err(e) => Err(fold(e, report_as)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_object_folds_to_not_found_carrying_the_callers_path() {
        let path = Path::new("data/x.bin");
        let err = fold(
            object_store::Error::NotFound {
                path: "whatever-the-store-called-it".to_owned(),
                source: Box::new(std::io::Error::from(std::io::ErrorKind::NotFound)),
            },
            &path,
        );
        match err {
            FileSystemErrorKind::NotFound { path: got } => assert_eq!(got, path),
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[test]
    fn credential_rejection_folds_to_auth_failed_not_permission_denied() {
        let path = Path::new("s3://b/k");
        let auth = fold(
            object_store::Error::Unauthenticated {
                path: "k".to_owned(),
                source: Box::new(std::io::Error::other("bad key")),
            },
            &path,
        );
        assert!(
            matches!(auth, FileSystemErrorKind::AuthFailed),
            "got {auth:?}"
        );

        let denied = fold(
            object_store::Error::PermissionDenied {
                path: "k".to_owned(),
                source: Box::new(std::io::Error::other("no acl")),
            },
            &path,
        );
        match denied {
            FileSystemErrorKind::PermissionDenied { path: got } => assert_eq!(got, path),
            other => panic!("expected PermissionDenied, got {other:?}"),
        }
    }

    #[test]
    fn an_endpoint_failure_folds_to_connection_failed_naming_the_store() {
        let err = fold(
            object_store::Error::Generic {
                store: "S3",
                source: Box::new(std::io::Error::other("dns")),
            },
            &Path::new("s3://b/k"),
        );
        match err {
            FileSystemErrorKind::ConnectionFailed { msg } => {
                assert!(msg.contains("S3") && msg.contains("dns"), "{msg}");
            }
            other => panic!("expected ConnectionFailed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_root_prefix_lists_everything_rather_than_matching_nothing() {
        let store = object_store::memory::InMemory::new();
        for key in ["a.bin", "b/c.bin"] {
            write(
                &store,
                &Path::new(key),
                &Path::new(key),
                Bytes::from_static(b"x"),
            )
            .await
            .expect("write");
        }

        let got = list_literal(&store, &Path::new("/"), str::to_owned)
            .await
            .expect("list");

        assert_eq!(got.len(), 2, "{got:?}");
    }
}
