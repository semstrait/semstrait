use semstrait_common::diagnostic::{Diagnose, Severity};

use crate::types::DriftKind;

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum CatalogProviderErrorKind {
    NotAvailable { msg: String },

    TableNotFound { fqn: String },

    NamespaceNotFound { ns: String },

    SnapshotNotFound { table: String, version: String },

    ConnectionFailed { msg: String },

    AuthFailed,

    AuthDenied { resource: String },

    Timeout { millis: u64 },

    SchemaDrift { kind: DriftKind },

    MalformedPartition { msg: String },

    MalformedResponse { msg: String },

    Internal { msg: String },
}

impl Diagnose for CatalogProviderErrorKind {
    fn message(&self) -> String {
        match self {
            Self::NotAvailable { msg } => format!("catalog not available: {msg}"),
            Self::TableNotFound { fqn } => format!("table not found: {fqn}"),
            Self::NamespaceNotFound { ns } => format!("namespace not found: {ns}"),
            Self::SnapshotNotFound { table, version } => {
                format!("snapshot not found: {table}@{version}")
            }
            Self::ConnectionFailed { msg } => format!("connection failed: {msg}"),
            Self::AuthFailed => "authentication failed".to_owned(),
            Self::AuthDenied { resource } => format!("access denied: {resource}"),
            Self::Timeout { millis } => format!("timed out after {millis}ms"),
            Self::SchemaDrift { kind } => format!("breaking schema drift: {}", kind.describe()),
            Self::MalformedPartition { msg } => format!("malformed partition spec: {msg}"),
            Self::MalformedResponse { msg } => format!("malformed catalog response: {msg}"),
            Self::Internal { msg } => format!("internal error: {msg}"),
        }
    }

    fn severity_default(&self) -> Severity {
        Severity::Error
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_renders_table_not_found() {
        let e = CatalogProviderErrorKind::TableNotFound {
            fqn: "sales.orders".to_owned(),
        };
        assert_eq!(e.message(), "table not found: sales.orders");
    }

    #[test]
    fn message_renders_namespace_not_found() {
        let e = CatalogProviderErrorKind::NamespaceNotFound {
            ns: "sales".to_owned(),
        };
        assert_eq!(e.message(), "namespace not found: sales");
    }

    #[test]
    fn message_renders_snapshot_not_found() {
        let e = CatalogProviderErrorKind::SnapshotNotFound {
            table: "sales.orders".to_owned(),
            version: "42".to_owned(),
        };
        assert_eq!(e.message(), "snapshot not found: sales.orders@42");
    }

    #[test]
    fn message_renders_auth_denied_with_resource() {
        let e = CatalogProviderErrorKind::AuthDenied {
            resource: "sales.orders".to_owned(),
        };
        assert_eq!(e.message(), "access denied: sales.orders");
    }

    #[test]
    fn message_renders_timeout_with_millis() {
        let e = CatalogProviderErrorKind::Timeout { millis: 2500 };
        assert_eq!(e.message(), "timed out after 2500ms");
    }

    #[test]
    fn message_renders_schema_drift_detail() {
        let e = CatalogProviderErrorKind::SchemaDrift {
            kind: DriftKind::ColumnRemoved {
                name: "amount".to_owned(),
            },
        };
        assert_eq!(e.message(), "breaking schema drift: column removed: amount");
    }

    #[test]
    fn every_variant_is_error_severity() {
        let cases = [
            CatalogProviderErrorKind::NotAvailable { msg: "x".into() },
            CatalogProviderErrorKind::TableNotFound { fqn: "x".into() },
            CatalogProviderErrorKind::NamespaceNotFound { ns: "x".into() },
            CatalogProviderErrorKind::SnapshotNotFound {
                table: "x".into(),
                version: "1".into(),
            },
            CatalogProviderErrorKind::ConnectionFailed { msg: "x".into() },
            CatalogProviderErrorKind::AuthFailed,
            CatalogProviderErrorKind::AuthDenied {
                resource: "x".into(),
            },
            CatalogProviderErrorKind::Timeout { millis: 1 },
            CatalogProviderErrorKind::SchemaDrift {
                kind: DriftKind::ColumnAdded { name: "c".into() },
            },
            CatalogProviderErrorKind::MalformedPartition { msg: "x".into() },
            CatalogProviderErrorKind::MalformedResponse { msg: "x".into() },
            CatalogProviderErrorKind::Internal { msg: "x".into() },
        ];
        for c in &cases {
            assert_eq!(c.severity_default(), Severity::Error);
        }
    }
}
