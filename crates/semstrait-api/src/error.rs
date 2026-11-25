use std::borrow::Cow;
use std::convert::identity;
use std::error::Error;

use semstrait_adapter::InvalidRewrite;
use semstrait_common::io::RepositoryErrorKind;
use semstrait_common::{
    split_by_severity, Diagnose, Diagnostic, Diagnostics, FileSystemErrorKind, IoErrorKind,
    Severity,
};
use semstrait_manifest::CompileError;
use semstrait_model::{ModelBuildErrorKind, ParseErrorKind, ValidateErrorKind};
use semstrait_planner::{Advisory, PlanError};
use thiserror::Error;

use crate::catalogs::CatalogConnectionError;
use crate::dialect::NoEngineAdapter;
use crate::wire::WireError;

#[non_exhaustive]
#[derive(Debug)]
pub enum SemstraitError {
    Parse(ParseErrorKind),
    Build(ModelBuildErrorKind),
    Validate(ValidateErrorKind),
    Compile(CompileError),
    Plan(PlanError),
    Advisory(Advisory),
    Adapt(InvalidRewrite),
    Repository(RepositoryErrorKind),
    FileSystem(FileSystemErrorKind),
    Io(IoErrorKind),
    Input(InputErrorKind),
    Catalog(CatalogConnectionError),
    NoEngineAdapter(NoEngineAdapter),
    Wire(WireError),
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Error)]
pub enum InputErrorKind {
    #[error("the manifest does not decode: {reason}")]
    UndecodableManifest { reason: Cow<'static, str> },
    #[error("an s3:// path needs the aws feature")]
    S3WithoutAws,
    #[error(
        "a posted model may not use `${{...}}`, because the server expands no environment \
         variable in HTTP input"
    )]
    EnvVarInPostedModel,
}

impl Diagnose for InputErrorKind {
    fn message(&self) -> String {
        self.to_string()
    }

    fn severity_default(&self) -> Severity {
        Severity::Error
    }
}

pub(crate) type Refusal = (Diagnostic<SemstraitError>, Diagnostics<SemstraitError>);

pub(crate) type Outcome<T> = Result<(T, Diagnostics<SemstraitError>), Refusal>;

pub(crate) type Staged<T, E, W> = Result<(T, Diagnostics<W>), (Diagnostic<E>, Diagnostics<W>)>;

pub(crate) fn refusal(kind: SemstraitError) -> Refusal {
    (Diagnostic::new(kind), Vec::new())
}

#[cfg(feature = "rest")]
pub(crate) fn wired(error: WireError) -> Refusal {
    refusal(SemstraitError::Wire(error))
}

pub(crate) fn lifted<K: Diagnose>(
    diagnostics: Diagnostics<K>,
    lift: fn(K) -> SemstraitError,
) -> impl Iterator<Item = Diagnostic<SemstraitError>> {
    diagnostics
        .into_iter()
        .map(move |diagnostic| diagnostic.map_kind(lift))
}

pub(crate) fn refused<K: Diagnose>(
    diagnostics: Diagnostics<K>,
    lift: fn(K) -> SemstraitError,
) -> Refusal {
    let (errors, warnings) = split_by_severity(diagnostics);
    let mut ordered = lifted(errors, lift).chain(lifted(warnings, lift));
    let no_model = || Diagnostic::new(SemstraitError::Build(ModelBuildErrorKind::NoSource));
    (ordered.next().unwrap_or_else(no_model), ordered.collect())
}

pub(crate) fn joined<T, E: Diagnose, W: Diagnose>(
    warnings: Diagnostics<SemstraitError>,
    staged: Staged<T, E, W>,
    error: fn(E) -> SemstraitError,
    warning: fn(W) -> SemstraitError,
) -> Outcome<T> {
    let all = |later: Diagnostics<W>| warnings.into_iter().chain(lifted(later, warning)).collect();
    match staged {
        Ok((output, later)) => Ok((output, all(later))),
        Err((refusal, later)) => Err((refusal.map_kind(error), all(later))),
    }
}

pub(crate) fn beside<T>(warnings: Diagnostics<SemstraitError>, outcome: Outcome<T>) -> Outcome<T> {
    joined(warnings, outcome, identity, identity)
}

impl SemstraitError {
    fn kind(&self) -> &dyn Diagnose {
        match self {
            Self::Parse(kind) => kind,
            Self::Build(kind) => kind,
            Self::Validate(kind) => kind,
            Self::Compile(kind) => kind,
            Self::Plan(kind) => kind,
            Self::Advisory(kind) => kind,
            Self::Adapt(kind) => kind,
            Self::Repository(kind) => kind,
            Self::FileSystem(kind) => kind,
            Self::Io(kind) => kind,
            Self::Input(kind) => kind,
            Self::Catalog(kind) => kind,
            Self::NoEngineAdapter(kind) => kind,
            Self::Wire(kind) => kind,
        }
    }
}

impl Diagnose for SemstraitError {
    fn message(&self) -> String {
        self.kind().message()
    }

    fn severity_default(&self) -> Severity {
        self.kind().severity_default()
    }

    fn cause(&self) -> Option<&(dyn Error + 'static)> {
        self.kind().cause()
    }
}
