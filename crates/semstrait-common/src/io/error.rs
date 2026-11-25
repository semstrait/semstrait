use std::borrow::Cow;
use std::error::Error as StdError;
use std::fmt;

use crate::diagnostic::{Diagnose, Severity};

#[non_exhaustive]
#[derive(Debug)]
pub enum IoErrorKind {
    NotFound {
        describe: String,
    },

    PermissionDenied {
        describe: String,
    },

    Network {
        describe: String,
        reason: Cow<'static, str>,
        source: Option<Box<dyn StdError + Send + Sync>>,
    },

    Unsupported {
        describe: String,
        reason: Cow<'static, str>,
    },

    Malformed {
        describe: String,
        reason: Cow<'static, str>,
    },
}

impl Diagnose for IoErrorKind {
    fn message(&self) -> String {
        match self {
            Self::NotFound { describe } => format!("not found: {describe}"),
            Self::PermissionDenied { describe } => format!("permission denied: {describe}"),
            Self::Network {
                describe, reason, ..
            } => format!("network error on {describe}: {reason}"),
            Self::Unsupported { describe, reason } => {
                format!("unsupported: {describe} ({reason})")
            }
            Self::Malformed { describe, reason } => {
                format!("malformed payload from {describe}: {reason}")
            }
        }
    }

    fn severity_default(&self) -> Severity {
        Severity::Error
    }

    fn cause(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Network {
                source: Some(e), ..
            } => Some(e.as_ref()),
            _ => None,
        }
    }
}

impl fmt::Display for IoErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message())
    }
}

impl StdError for IoErrorKind {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Diagnose::cause(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn not_found_message_carries_describe() {
        let e = IoErrorKind::NotFound {
            describe: "/abs/missing.yaml".into(),
        };
        assert_eq!(e.message(), "not found: /abs/missing.yaml");
    }

    #[test]
    fn permission_denied_message_carries_describe() {
        let e = IoErrorKind::PermissionDenied {
            describe: "s3://bucket/secret".into(),
        };
        assert_eq!(e.message(), "permission denied: s3://bucket/secret");
    }

    #[test]
    fn unsupported_message_carries_describe_and_reason() {
        let e = IoErrorKind::Unsupported {
            describe: "s3://bucket/key".into(),
            reason: "io-aws feature disabled".into(),
        };
        assert_eq!(
            e.message(),
            "unsupported: s3://bucket/key (io-aws feature disabled)"
        );
    }

    #[test]
    fn malformed_message_carries_describe_and_reason() {
        let e = IoErrorKind::Malformed {
            describe: "<in-conversion>".into(),
            reason: "invalid UTF-8 at byte 12".into(),
        };
        assert_eq!(
            e.message(),
            "malformed payload from <in-conversion>: invalid UTF-8 at byte 12"
        );
    }

    #[test]
    fn network_message_combines_describe_and_reason() {
        let e = IoErrorKind::Network {
            describe: "s3://bucket/key".into(),
            reason: "connection refused".into(),
            source: None,
        };
        assert_eq!(
            e.message(),
            "network error on s3://bucket/key: connection refused"
        );
    }

    #[test]
    fn severity_default_is_error_for_every_variant() {
        let cases = [
            IoErrorKind::NotFound {
                describe: "x".into(),
            },
            IoErrorKind::PermissionDenied {
                describe: "x".into(),
            },
            IoErrorKind::Network {
                describe: "x".into(),
                reason: "r".into(),
                source: None,
            },
            IoErrorKind::Unsupported {
                describe: "x".into(),
                reason: "r".into(),
            },
            IoErrorKind::Malformed {
                describe: "x".into(),
                reason: "r".into(),
            },
        ];
        for c in &cases {
            assert_eq!(c.severity_default(), Severity::Error);
        }
    }

    #[test]
    fn cause_is_some_for_network_with_source() {
        #[derive(Debug)]
        struct Inner;
        impl fmt::Display for Inner {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("inner")
            }
        }
        impl StdError for Inner {}

        let e = IoErrorKind::Network {
            describe: "x".into(),
            reason: "r".into(),
            source: Some(Box::new(Inner)),
        };
        assert!(Diagnose::cause(&e).is_some());
    }

    #[test]
    fn cause_is_none_for_network_without_source() {
        let e = IoErrorKind::Network {
            describe: "x".into(),
            reason: "r".into(),
            source: None,
        };
        assert!(Diagnose::cause(&e).is_none());
    }

    #[test]
    fn cause_is_none_for_non_network_variants() {
        let cases = [
            IoErrorKind::NotFound {
                describe: "x".into(),
            },
            IoErrorKind::PermissionDenied {
                describe: "x".into(),
            },
            IoErrorKind::Unsupported {
                describe: "x".into(),
                reason: "r".into(),
            },
            IoErrorKind::Malformed {
                describe: "x".into(),
                reason: "r".into(),
            },
        ];
        for c in &cases {
            assert!(Diagnose::cause(c).is_none());
        }
    }

    #[test]
    fn display_delegates_to_message() {
        let e = IoErrorKind::NotFound {
            describe: "x".into(),
        };
        assert_eq!(format!("{e}"), e.message());
    }

    #[test]
    fn std_error_source_chains_to_diagnose_cause() {
        #[derive(Debug)]
        struct Inner;
        impl fmt::Display for Inner {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("inner")
            }
        }
        impl StdError for Inner {}

        let e = IoErrorKind::Network {
            describe: "x".into(),
            reason: "r".into(),
            source: Some(Box::new(Inner)),
        };
        assert!((&e as &dyn StdError).source().is_some());
    }
}
