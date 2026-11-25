#![allow(clippy::result_large_err)]

pub mod catalog;
pub mod error;
pub mod expr;
pub mod loader;
pub mod model;
pub mod parse;
pub mod repository;
#[cfg(feature = "schemars")]
pub mod schema;
pub mod source_fs;
pub mod validation;
pub(crate) mod yaml;

pub use catalog::{CatalogAuthMethod, CatalogEntry, CatalogsConfig, SecretKeyMapping};
pub use error::{CatalogsParseErrorKind, ModelBuildErrorKind, ParseErrorKind, ValidateErrorKind};
pub use expr::{UnresolvedPhysicalExpr, UnresolvedSemanticExpr};
pub use loader::SemanticModelLoader;
pub use model::builder::{
    DatasetBuilder, GrainsetBuilder, JoinsetBuilder, NestedDatasetBuilder, NestedGrainsetBuilder,
    NestedJoinsetBuilder, NestedUnionsetBuilder, SemanticModelBuilder, UnionsetBuilder,
};
pub use model::entity_btree_map::EntityBTreeMap;
pub use model::identity::IdentityProfile;
pub use model::serde::data_type::UnresolvedDataType;
pub use model::serde::ParseError;
pub use model::types::data_kinds::{
    AnyDataKindRef, CatalogRef, ComplexDataKind, ComplexDataKindRef, ComplexExtras, DataKind,
    DataKindBase, DataKindForm, DataKindVariant, Dataset, DatasetBody, ExtrasFlavor, Grainset,
    GrainsetBody, Joinset, JoinsetBody, LeafExtras, NestedDataKind, NestedDataKindRef,
    NestedDataset, NestedGrainset, NestedJoinset, NestedUnionset, PartitionDef, PublicDataKind,
    PublicDataKindRef, SimpleDataKind, SimpleDataKindRef, StorageConfig, StorageFormat, UnionMode,
    Unionset, UnionsetBody,
};
pub use model::types::entities::{
    AdditivityType, Agg, AiContext, AuthoringBody, BucketSpec, BucketedDimensionBody, Cardinality,
    CrossFilter, DataKindFilter, Dimension, DimensionDecl, DimensionRef, DimensionSite,
    DimensionType, EventsBody, Integrity, JoinKeySpec, Keyness, Measure, MeasureRef, MeasureSite,
    MetadataDimensionBody, MetadataDimensionRecipe, MetadataExtraction, MetadataSource, Metric,
    MetricRef, MetricSite, NamedSemanticBody, Optional, PathTokenRef, Reference, Relationship,
    ScdBody, ScdType, SemanticBody, SemanticInterface, SemanticMapping, SemanticMappingValue,
    SemanticSite, SemiAdditivity, SemiAdditivityStrategy, SnapshotBody, TemporalDimensionBody,
    TemporalShape, TemporalShapeKind, TimeseriesBody, DEFAULT_KEYSET,
};
pub use model::SemanticModel;
pub use parse::{parse, parse_catalogs, parse_catalogs_with_source, parse_with_source};
pub use source_fs::{InMemoryFs, LocalFs, SourceFs};
pub use validation::validate;
