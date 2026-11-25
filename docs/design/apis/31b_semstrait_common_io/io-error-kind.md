---
id: 31b-io-error-kind
status: ratified
version: 2.1
impl: drift
---

# IoErrorKind

This leaf owns the IoErrorKind typed-kind enum, variant taxonomy, Diagnose impl, stage-of-origin identification, and cross-crate composition.

## 7. `IoErrorKind`

```rust
#[non_exhaustive]
#[derive(Debug)]
pub enum IoErrorKind {
    NotFound { describe: String },

    PermissionDenied { describe: String },

    Network {
        describe: String,
        reason: Cow<'static, str>,
        /// Foreign-error chain per `30 §5.4` (variant-side wrapping).
        /// The wrapped error participates in the std::error::Error chain
        /// via `Diagnose::cause()`.
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    Unsupported { describe: String, reason: Cow<'static, str> },

    Malformed { describe: String, reason: Cow<'static, str> },
}
```

**Variant taxonomy.**

<!-- cross-cut: none -->

| Variant            | Trigger                                                                                               | Example                                                                            |
| ------------------ | ----------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `NotFound`         | Target path / key does not exist                                                                      | `LocalFile::read_raw` on a non-existent path; S3 `NoSuchKey`                       |
| `PermissionDenied` | Access denied by the back-end                                                                         | Unix `EACCES`; S3 `AccessDenied`                                                   |
| `Network`          | Transport-level failure (connection refused, DNS, timeout, TLS)                                       | S3 call timed out; endpoint unreachable                                            |
| `Unsupported`      | Scheme or back-end not available in the current feature set, or operation not supported by a back-end | `s3://` URL parsed without `io-aws` enabled; conditional write attempted           |
| `Malformed`        | Payload or URI violates the expected shape                                                            | UTF-8 decode failure via `String::from_io_bytes`; `s3://` with no bucket component |


### 7.1 `Diagnose` impl

```rust
impl Diagnose for IoErrorKind {
    fn message(&self) -> String {
        match self {
            Self::NotFound { describe } =>
                format!("not found: {describe}"),
            Self::PermissionDenied { describe } =>
                format!("permission denied: {describe}"),
            Self::Network { describe, reason, .. } =>
                format!("network error on {describe}: {reason}"),
            Self::Unsupported { describe, reason } =>
                format!("unsupported: {describe} ({reason})"),
            Self::Malformed { describe, reason } =>
                format!("malformed payload from {describe}: {reason}"),
        }
    }

    fn severity_default(&self) -> Severity { Severity::Error }

    fn cause(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Network { source: Some(e), .. } => Some(e.as_ref()),
            _ => None,
        }
    }
}

impl std::fmt::Display for IoErrorKind { /* delegates to Diagnose::message */ }
impl std::error::Error for IoErrorKind {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Diagnose::cause(self)
    }
}
```

### 7.2 Stage-of-origin and identification

Per `30 §5`, identification is by variant identity (renaming a variant is MAJOR; adding one inside `#[non_exhaustive]` is MINOR). There is no central code allocation; matching `IoErrorKind::NotFound { .. }` is the stable contract.

The stage of origin (`fs.read`, `repository.load`, etc.) appears as a `tracing` `stage` field on the surrounding span (`30 §6.5`), not in the kind. Domain wrappers building a `Diagnostic<IoErrorKind>` from a back-end call attach a caller-known location (file path, manifest ID) at construction.

### 7.3 Cross-crate composition

Domain error kinds in upstream crates (`semstrait_model::ModelLoadErrorKind`, `semstrait_manifest::RepositoryErrorKind`) declare `Io(IoErrorKind)` as a variant per D.ii cross-stage nesting (`30 §7.4`). Because `IoErrorKind` is `#[non_exhaustive]`, adding a variant here remains MINOR through the entire dependency chain.

Example (in `33 §11`):

```rust
#[non_exhaustive]
pub enum RepositoryErrorKind {
    Io(IoErrorKind),                    // wraps transport failure
    ManifestIdCollision { id: String }, // domain-level
    EncodingMismatch { expected: String, actual: String },
    IntegrityCheckFailed { reason: String },
}
```

---

## Testing Strategy

- IoErrorKind #[non_exhaustive] (§7)
- variant taxonomy complete for v1 back-ends (§7 table)
- Diagnose impl delegates to variant-specific message (§7.1)
- severity_default → Severity::Error (§7.1)
- cause() returns Some(&inner) for Network variant (§7.1)
- cross-crate composition via Io(IoErrorKind) variant (§7.3)

See the shared behavioral harness: [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md)
