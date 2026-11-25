use std::fmt;
use std::str::FromStr;

use thiserror::Error;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::diagnostic::{Diagnose, Severity};

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(rename_all = "lowercase")
)]
pub enum Dialect {
    DataFusion,
    Substrait,
    Iceberg,
}

impl Dialect {
    pub const fn reference_release(self) -> Option<Release> {
        match self {
            Self::DataFusion => Some(Release::DATAFUSION_REFERENCE),
            Self::Substrait | Self::Iceberg => None,
        }
    }

    pub const fn oldest_release(self) -> Option<Release> {
        match self {
            Self::DataFusion => Some(Release::DATAFUSION_OLDEST),
            Self::Substrait | Self::Iceberg => None,
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::DataFusion => "datafusion",
            Self::Substrait => "substrait",
            Self::Iceberg => "iceberg",
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        match name {
            "datafusion" => Some(Self::DataFusion),
            "substrait" => Some(Self::Substrait),
            "iceberg" => Some(Self::Iceberg),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Release(pub u32);

impl Release {
    pub const DATAFUSION_REFERENCE: Release = Release(55);
    pub const DATAFUSION_OLDEST: Release = Release(53);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DialectRelease {
    pub dialect: Dialect,
    pub release: Option<Release>,
}

impl From<Dialect> for DialectRelease {
    fn from(dialect: Dialect) -> Self {
        Self {
            dialect,
            release: dialect.reference_release(),
        }
    }
}

impl FromStr for DialectRelease {
    type Err = DialectReleaseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (name, release_text) = match s.split_once('=') {
            Some((name, release)) => (name, Some(release)),
            None => (s, None),
        };
        let dialect = Dialect::from_name(name)
            .ok_or_else(|| DialectReleaseError::UnknownDialect(name.to_owned()))?;
        let Some(written) = release_text else {
            return Ok(Self::from(dialect));
        };
        let major: u32 = written
            .parse()
            .map_err(|_| DialectReleaseError::NotAMajor {
                dialect,
                written: written.to_owned(),
            })?;
        let release = Release(major);
        let oldest = dialect
            .oldest_release()
            .ok_or(DialectReleaseError::NoReleases(dialect))?;
        if release < oldest {
            return Err(DialectReleaseError::BelowOldest {
                dialect,
                release,
                oldest,
            });
        }
        Ok(Self {
            dialect,
            release: Some(release),
        })
    }
}

impl fmt::Display for DialectRelease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.dialect.name())?;
        if let Some(release) = self.release {
            write!(f, "={}", release.0)?;
        }
        Ok(())
    }
}

#[non_exhaustive]
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DialectReleaseError {
    #[error("unknown dialect: {0}")]
    UnknownDialect(String),

    #[error("{0:?} has no releases")]
    NoReleases(Dialect),

    #[error("{dialect:?}'s release is not a major version: {written}")]
    NotAMajor { dialect: Dialect, written: String },

    #[error("{dialect:?}'s release {release:?} is below its oldest, {oldest:?}")]
    BelowOldest {
        dialect: Dialect,
        release: Release,
        oldest: Release,
    },
}

impl Diagnose for DialectReleaseError {
    fn message(&self) -> String {
        format!("{self}")
    }

    fn severity_default(&self) -> Severity {
        Severity::Error
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_release_and_oldest_release_for_datafusion() {
        assert_eq!(
            Dialect::DataFusion.reference_release(),
            Some(Release::DATAFUSION_REFERENCE)
        );
        assert_eq!(
            Dialect::DataFusion.oldest_release(),
            Some(Release::DATAFUSION_OLDEST)
        );
        assert_eq!(Release::DATAFUSION_REFERENCE, Release(55));
        assert_eq!(Release::DATAFUSION_OLDEST, Release(53));
    }

    #[test]
    fn reference_release_and_oldest_release_are_none_for_substrait_and_iceberg() {
        assert_eq!(Dialect::Substrait.reference_release(), None);
        assert_eq!(Dialect::Substrait.oldest_release(), None);
        assert_eq!(Dialect::Iceberg.reference_release(), None);
        assert_eq!(Dialect::Iceberg.oldest_release(), None);
    }

    #[test]
    fn from_dialect_uses_the_reference_release() {
        assert_eq!(
            DialectRelease::from(Dialect::DataFusion),
            DialectRelease {
                dialect: Dialect::DataFusion,
                release: Some(Release::DATAFUSION_REFERENCE),
            }
        );
        assert_eq!(
            DialectRelease::from(Dialect::Substrait),
            DialectRelease {
                dialect: Dialect::Substrait,
                release: None,
            }
        );
    }

    #[test]
    fn parse_datafusion_gives_the_reference_release() {
        let parsed: DialectRelease = "datafusion".parse().unwrap();
        assert_eq!(parsed.dialect, Dialect::DataFusion);
        assert_eq!(parsed.release, Some(Release(55)));
    }

    #[test]
    fn parse_datafusion_with_an_explicit_oldest_release() {
        let parsed: DialectRelease = "datafusion=53".parse().unwrap();
        assert_eq!(parsed.release, Some(Release(53)));
    }

    #[test]
    fn parse_datafusion_at_the_reference_release() {
        let parsed: DialectRelease = "datafusion=55".parse().unwrap();
        assert_eq!(parsed.release, Some(Release(55)));
    }

    #[test]
    fn parse_datafusion_above_the_reference_release() {
        let parsed: DialectRelease = "datafusion=56".parse().unwrap();
        assert_eq!(parsed.release, Some(Release(56)));
    }

    #[test]
    fn parse_substrait_has_no_release() {
        let parsed: DialectRelease = "substrait".parse().unwrap();
        assert_eq!(parsed.dialect, Dialect::Substrait);
        assert_eq!(parsed.release, None);
    }

    #[test]
    fn parse_iceberg_has_no_release() {
        let parsed: DialectRelease = "iceberg".parse().unwrap();
        assert_eq!(parsed.dialect, Dialect::Iceberg);
        assert_eq!(parsed.release, None);
    }

    #[test]
    fn parse_refuses_a_release_below_the_oldest() {
        let err: DialectReleaseError = "datafusion=52".parse::<DialectRelease>().unwrap_err();
        assert_eq!(
            err,
            DialectReleaseError::BelowOldest {
                dialect: Dialect::DataFusion,
                release: Release(52),
                oldest: Release::DATAFUSION_OLDEST,
            }
        );
    }

    #[test]
    fn parse_refuses_a_release_on_a_dialect_with_no_releases() {
        let err: DialectReleaseError = "substrait=1".parse::<DialectRelease>().unwrap_err();
        assert_eq!(err, DialectReleaseError::NoReleases(Dialect::Substrait));
    }

    #[test]
    fn parse_refuses_a_release_that_is_not_a_major() {
        let cases = [
            ("datafusion=53.*", "53.*"),
            ("datafusion=53.1", "53.1"),
            ("datafusion=", ""),
        ];
        for (input, expected_written) in cases {
            let err: DialectReleaseError = input.parse::<DialectRelease>().unwrap_err();
            assert_eq!(
                err,
                DialectReleaseError::NotAMajor {
                    dialect: Dialect::DataFusion,
                    written: expected_written.to_owned(),
                }
            );
        }
    }

    #[test]
    fn parse_refuses_an_unknown_dialect() {
        let err: DialectReleaseError = "duckdb".parse::<DialectRelease>().unwrap_err();
        assert_eq!(
            err,
            DialectReleaseError::UnknownDialect("duckdb".to_owned())
        );
    }

    #[test]
    fn display_round_trips_every_accepted_input() {
        let inputs = [
            "datafusion",
            "datafusion=53",
            "datafusion=55",
            "datafusion=56",
            "substrait",
            "iceberg",
        ];
        for input in inputs {
            let parsed: DialectRelease = input.parse().unwrap();
            let displayed = parsed.to_string();
            let reparsed: DialectRelease = displayed.parse().unwrap();
            assert_eq!(parsed, reparsed);
        }
    }

    #[test]
    fn display_writes_name_equals_release_when_a_release_is_set() {
        let parsed: DialectRelease = "datafusion=55".parse().unwrap();
        assert_eq!(parsed.to_string(), "datafusion=55");
    }

    #[test]
    fn display_writes_a_bare_name_when_there_is_no_release() {
        let parsed: DialectRelease = "substrait".parse().unwrap();
        assert_eq!(parsed.to_string(), "substrait");
    }

    #[test]
    fn dialect_release_error_implements_diagnose() {
        let err = DialectReleaseError::UnknownDialect("duckdb".to_owned());
        assert!(!err.message().is_empty());
        assert_eq!(err.severity_default(), Severity::Error);
        assert!(err.cause().is_none());
    }
}
