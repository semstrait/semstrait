pub mod constraints;
pub mod diagnostic;
pub mod dialect;
pub mod glob;
pub mod identity;
pub mod tree;

#[cfg(feature = "io")]
pub mod io;

#[cfg(feature = "serde")]
pub mod granularity;
#[cfg(feature = "serde")]
pub mod names;

pub use crate::constraints::{AggregationConstraints, DimensionConstraints, MeasureConstraints};
pub use crate::diagnostic::{
    split_by_severity, Diagnose, Diagnostic, Diagnostics, Location, Severity, SourceId, Span,
};
pub use crate::dialect::{Dialect, DialectRelease, DialectReleaseError, Release};
pub use crate::glob::GlobPattern;
pub use crate::identity::{
    derive_uuid_v5, entity_namespace, mint_entity_id, mint_named_entity_id, EntityId,
    EntityNamespace, ENTITY_ROOT_NAMESPACE, MODEL_ROOT_NAMESPACE,
};
pub use crate::tree::{Rewriter, Tree, ValidateError, Visitor};

#[cfg(feature = "io")]
pub use crate::io::{FileSystem, FileSystemErrorKind, IoErrorKind};

#[cfg(feature = "serde")]
pub use crate::granularity::Granularity;
#[cfg(feature = "serde")]
pub use crate::names::{DataKindName, SemanticsName};
