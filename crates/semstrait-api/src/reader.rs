use std::fmt::Display;

use semstrait_common::io::backends::local_fs::LocalFileSystem;
#[cfg(feature = "aws")]
use semstrait_common::io::backends::s3_fs::S3FileSystem;
use semstrait_common::io::{FileSystem, Path};
use semstrait_common::{Diagnostic, IoErrorKind, Location, SourceId, Span};

#[cfg(not(feature = "aws"))]
use crate::error::InputErrorKind;
use crate::error::{Refusal, SemstraitError};

pub(crate) async fn read(path: &str) -> Result<String, Refusal> {
    let (filesystem, key) = located(path)?;
    let bytes = (filesystem.read(&Path::new(key)).await)
        .map_err(|kind| at(path, SemstraitError::FileSystem(kind)))?;
    String::from_utf8(bytes.into()).map_err(|error| malformed(path, error))
}

#[cfg(feature = "cli")]
pub(crate) fn read_stdin(stdin: impl std::io::Read) -> Result<String, Refusal> {
    std::io::read_to_string(stdin).map_err(|error| malformed("stdin", error))
}

pub(crate) fn malformed(path: &str, error: impl Display) -> Refusal {
    let (describe, reason) = (path.to_owned(), error.to_string().into());
    let malformed = IoErrorKind::Malformed { describe, reason };
    at(path, SemstraitError::Io(malformed))
}

#[cfg(feature = "cli")]
pub(crate) async fn write(path: &str, bytes: Vec<u8>) -> Result<(), Refusal> {
    let (filesystem, key) = located(path)?;
    (filesystem.write(&Path::new(key), bytes.into()).await)
        .map_err(|kind| at(path, SemstraitError::FileSystem(kind)))
}

fn located(path: &str) -> Result<(Box<dyn FileSystem>, &str), Refusal> {
    Ok(match bucket_and_key(path) {
        Some((bucket, key)) => (s3(bucket).map_err(|kind| at(path, kind))?, key),
        None => {
            let (root, name) = path.rsplit_once('/').unwrap_or((".", path));
            (Box::new(LocalFileSystem::new(format!("{root}/"))), name)
        }
    })
}

pub(crate) fn at(path: &str, kind: SemstraitError) -> Refusal {
    let location = Location::new(SourceId::named(path), Span::unknown());
    (Diagnostic::new(kind).with_location(location), Vec::new())
}

fn bucket_and_key(path: &str) -> Option<(&str, &str)> {
    let object = path.strip_prefix("s3://")?;
    Some(object.split_once('/').unwrap_or((object, "")))
}

#[cfg(feature = "aws")]
fn s3(bucket: &str) -> Result<Box<dyn FileSystem>, SemstraitError> {
    Ok(Box::new(S3FileSystem::new(bucket)))
}

#[cfg(not(feature = "aws"))]
fn s3(_: &str) -> Result<Box<dyn FileSystem>, SemstraitError> {
    Err(SemstraitError::Input(InputErrorKind::S3WithoutAws))
}

#[cfg(test)]
mod tests {
    use super::bucket_and_key;

    #[test]
    fn an_s3_uri_splits_into_its_bucket_and_key_and_a_local_path_does_not() {
        let cases = [
            (
                "s3://lake/models/sales.yaml",
                Some(("lake", "models/sales.yaml")),
            ),
            ("s3://lake", Some(("lake", ""))),
            ("/models/sales.yaml", None),
        ];

        for (path, split) in cases {
            assert_eq!(bucket_and_key(path), split, "{path}");
        }
    }
}
