use crate::error::catalogs::CatalogsParseErrorKind;
use crate::error::parse::ParseErrorKind;
use crate::error::validate::ValidateErrorKind;
use semstrait_common::diagnostic::{Diagnose, Severity};
use std::io::ErrorKind as IoErrorKind;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ModelBuildErrorKind {
    NoSource,

    SourceIo { path: String, error: IoErrorKind },

    Parse(ParseErrorKind),
    CatalogsParse(CatalogsParseErrorKind),
    Validate(ValidateErrorKind),
}

impl From<ParseErrorKind> for ModelBuildErrorKind {
    fn from(k: ParseErrorKind) -> Self {
        Self::Parse(k)
    }
}

impl From<ValidateErrorKind> for ModelBuildErrorKind {
    fn from(k: ValidateErrorKind) -> Self {
        Self::Validate(k)
    }
}

impl From<CatalogsParseErrorKind> for ModelBuildErrorKind {
    fn from(k: CatalogsParseErrorKind) -> Self {
        Self::CatalogsParse(k)
    }
}

impl Diagnose for ModelBuildErrorKind {
    fn message(&self) -> String {
        match self {
            ModelBuildErrorKind::Parse(k) => k.message(),
            ModelBuildErrorKind::CatalogsParse(k) => k.message(),
            ModelBuildErrorKind::Validate(k) => k.message(),
            ModelBuildErrorKind::NoSource => {
                "model loader was invoked without a configured source".to_string()
            }
            ModelBuildErrorKind::SourceIo { path, error } => {
                format!("failed to read `{}`: {:?}", path, error)
            }
        }
    }

    fn severity_default(&self) -> Severity {
        match self {
            ModelBuildErrorKind::Parse(k) => k.severity_default(),
            ModelBuildErrorKind::CatalogsParse(k) => k.severity_default(),
            ModelBuildErrorKind::Validate(k) => k.severity_default(),
            ModelBuildErrorKind::NoSource | ModelBuildErrorKind::SourceIo { .. } => Severity::Error,
        }
    }

    fn cause(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ModelBuildErrorKind::Parse(k) => k.cause(),
            ModelBuildErrorKind::CatalogsParse(k) => k.cause(),
            ModelBuildErrorKind::Validate(k) => k.cause(),
            ModelBuildErrorKind::NoSource | ModelBuildErrorKind::SourceIo { .. } => None,
        }
    }
}
