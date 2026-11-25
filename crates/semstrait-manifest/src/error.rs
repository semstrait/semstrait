use std::fmt;

use thiserror::Error;

use semstrait_catalog::FileFormat;
use semstrait_common::diagnostic::{Diagnose, Severity};
use semstrait_common::{DataKindName, EntityId, Granularity, SemanticsName};
use semstrait_ir::{Cardinality, CrossFilter, SemanticRole, TemporalType};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{CallError, Literal, NamedReference, TypeError};

use crate::manifest::expression::PhysicalExprId;
use crate::manifest::{KeysetName, SourceAlias};

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ReferenceSite {
    Signature(String),
    SemanticSource(String),
    Keyset(String),
    Semantics(String),
    Source(String),
    Bitmap,
    Bitmask(String),
    ExpressionPool,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReferenceTargetKind {
    Semantics,
    SemanticSources,
    Signatures,
    Sources,
    Expressions,
    Keysets,
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ReferenceScope {
    DataKind(EntityId),
    Appearance(EntityId),
}

impl fmt::Display for ReferenceScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReferenceScope::DataKind(id) => write!(f, "data-kind {id}"),
            ReferenceScope::Appearance(id) => write!(f, "appearance {id}"),
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShapeAxis {
    DataType,
    Category,
    Aggregation,
    Additivity,
}

impl fmt::Display for ShapeAxis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ShapeAxis::DataType => "data type",
            ShapeAxis::Category => "category",
            ShapeAxis::Aggregation => "aggregation",
            ShapeAxis::Additivity => "additivity",
        })
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelationshipSide {
    From,
    To,
}

impl fmt::Display for RelationshipSide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            RelationshipSide::From => "from",
            RelationshipSide::To => "to",
        })
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Error)]
pub enum CompileError {
    #[error("unknown reference `{name}` in {scope}")]
    UnknownReference { name: String, scope: ReferenceScope },

    #[error("cyclic reference among semantics: {}", cycle.join(" -> "))]
    CyclicReference { cycle: Vec<EntityId> },

    #[error("cyclic semantic source composition: {}", cycle.join(" -> "))]
    CyclicSemanticSource { cycle: Vec<EntityId> },

    #[error(
        "semantic source `{semantic_source}` carries the alias `{found}`; its source role \
         and index give `{expected}`"
    )]
    SourceAliasMismatch {
        semantic_source: EntityId,
        expected: SourceAlias,
        found: SourceAlias,
    },

    #[error(
        "physical source `{physical_source}` carries the alias `{found}`; its index gives \
         `{expected}`"
    )]
    PhysicalSourceAliasMismatch {
        physical_source: EntityId,
        expected: SourceAlias,
        found: SourceAlias,
    },

    #[error("physical source {physical_source_id}: column `{column}` has no canonical type")]
    UntypedCatalogColumn {
        physical_source_id: EntityId,
        column: String,
    },

    #[error(
        "the column {:?} at expression {} is not two parts under an alias its place names: \
         `{}`",
        .column.parts(),
        .expression,
        .expected.iter().map(SourceAlias::as_str).collect::<Vec<_>>().join("`, `")
    )]
    ColumnQualifierMismatch {
        expression: PhysicalExprId,
        column: NamedReference,
        expected: Vec<SourceAlias>,
    },

    #[error(
        "the column {:?} at expression {} names no semantics that semantic source {} serves",
        .column.parts(),
        .expression,
        .semantic_source
    )]
    UnservedConditionLeaf {
        expression: PhysicalExprId,
        column: NamedReference,
        semantic_source: EntityId,
    },

    #[error("the column {parts:?} at `{node}` holds an empty part, which names nothing")]
    EmptyColumnPart { node: String, parts: Vec<String> },

    #[error(
        "joinset `{joinset}` admits {units} sub-compositions, over the ceiling of {ceiling}; \
         nothing is truncated — reduce the relationships between its constituents"
    )]
    EnumerationCeilingExceeded {
        joinset: String,
        units: usize,
        ceiling: usize,
    },

    #[error(
        "joinset `{joinset}`: the sub-composition over relationships {second:?} derives the name \
         `{derived_name}`, which `{first}` already holds, so the two would be one entry"
    )]
    EnumeratedUnitNameCollision {
        joinset: String,
        derived_name: String,
        first: String,
        second: Vec<String>,
    },

    #[error("type inference failed at `{node}`: {reason}")]
    TypeInferenceFailure { node: String, reason: String },

    #[error("type error at `{node}`: {error}")]
    TypeError { node: String, error: TypeError },

    #[error("call refused at `{node}`: {error}")]
    CallRefused { node: String, error: CallError },

    #[error(
        "`{node}`: `date_trunc` truncates in {}, and a semantic model's `date_trunc` takes \
         'UTC' only",
        timezone_in_words(.timezone)
    )]
    DateTruncTimezoneNotUtc { node: String, timezone: Literal },

    #[error("column reference in semantic expression under manual mapping (binding {binding})")]
    ColumnInSemanticExprUnderManualMapping { binding: EntityId },

    #[error(
        "semantic_sources {unit}: the `filter:` for the semantics `{semantic}` names a \
         physical column `{column}`, but a composed source's filter reads only what its \
         operands project (ADR 0065)"
    )]
    ColumnInComposedFilter {
        unit: String,
        semantic: String,
        column: String,
    },

    #[error(
        "semantic kind mismatch for `{name}`: authored as {authored_kind:?}, \
         registered as {registered_kind:?}"
    )]
    SemanticKindMismatch {
        authored_kind: SemanticRole,
        registered_kind: SemanticRole,
        name: String,
    },

    #[error(
        "shape mismatch for `{name}` on {axis}: authored as {authored}, \
         already registered as {registered}"
    )]
    SemanticShapeMismatch {
        name: String,
        axis: ShapeAxis,
        authored: String,
        registered: String,
    },

    #[error(
        "semantics `{semantics}` references `{referenced}`, which semantic source \
         `{semantic_source}` does not expose"
    )]
    UnexposedReference {
        semantics: String,
        referenced: String,
        semantic_source: String,
    },

    #[error(
        "metric `{semantics}` on semantic source `{semantic_source}`: neither its site \
         nor its pooled declaration states `expr:`"
    )]
    MetricWithoutExpression {
        semantics: String,
        semantic_source: String,
    },

    #[error("row-level semantics `{semantics}` references `{referenced}`, a Measure or Metric")]
    AggregateReferenceInRowLevelBody {
        semantics: String,
        referenced: String,
    },

    #[error("catalog has no source {source_id} (locator `{locator}`)")]
    CatalogSourceMissing {
        source_id: EntityId,
        locator: String,
    },

    #[error("catalog fetch failed for source {source_id} (locator `{locator}`): {reason}")]
    CatalogFetch {
        source_id: EntityId,
        locator: String,
        reason: String,
    },

    #[error("dataset `{dataset}`: storage pattern `{pattern}` matched no file")]
    SourcePatternMatchedNothing { dataset: String, pattern: String },

    #[error("dataset `{dataset}`: storage entry `{entry}` could not be listed: {reason}")]
    SourceListingFailed {
        dataset: String,
        entry: String,
        reason: String,
    },

    #[error(
        "physical source {physical_source_id}: column `{column}` has native type \
         `{native_type}`, which has no canonical counterpart"
    )]
    UnmappableNativeType {
        column: String,
        expr_id: Option<PhysicalExprId>,
        physical_source_id: EntityId,
        native_type: String,
    },

    #[error(
        "physical source {physical_source_id}: column `{column}` is read by an expression \
         but absent from the observed schema"
    )]
    ColumnNotInSourceSchema {
        column: String,
        physical_source_id: EntityId,
    },

    #[error("physical source {physical_source_id}: column `{column}` is declared more than once")]
    DuplicateSourceColumn {
        physical_source_id: EntityId,
        column: String,
    },

    #[error("dataset `{dataset}` authors both `paths:` and `tables:`: {reason}")]
    StorageMixesPathsAndTables { dataset: String, reason: String },

    #[error(
        "dataset `{dataset}`: storage entry `{entry}` is a relative path; write an absolute \
         path or a URI"
    )]
    RelativeStoragePath { dataset: String, entry: String },

    #[error(
        "dataset `{dataset}`: storage entry `{entry}` has the scheme `{scheme}`, which \
         compile does not read"
    )]
    UnsupportedStorageScheme {
        dataset: String,
        entry: String,
        scheme: String,
    },

    #[error(
        "dataset `{dataset}`: storage entry `{entry}` reads the folder `{base_path}`, which \
         has no normal form: a folder URI ends in exactly one `/`"
    )]
    BasePathNotNormal {
        dataset: String,
        entry: String,
        base_path: String,
    },

    #[error("dataset `{dataset}`: `paths:` names `{uri}` twice")]
    DuplicateStorageUri { dataset: String, uri: String },

    #[error("dataset `{dataset}`: storage entry `{entry}` names `{uri}`, which holds no object")]
    StorageObjectMissing {
        dataset: String,
        entry: String,
        uri: String,
    },

    #[error("dataset `{dataset}`: storage entry `{entry}` names `{uri}`, an object of 0 bytes")]
    StorageObjectEmpty {
        dataset: String,
        entry: String,
        uri: String,
    },

    #[error(
        "dataset `{dataset}`: storage entries `{first}` and `{second}` read one base path or \
         one object, so the Dataset would read an object twice"
    )]
    StorageReadTwice {
        dataset: String,
        first: String,
        second: String,
    },

    #[error(
        "dataset `{dataset}`: storage entry `{entry}` holds the folder `{folder}`, which is \
         no `key=value` folder; a folder holds its objects directly or in Hive `key=value` \
         folders"
    )]
    PartitionFolderWithoutKey {
        dataset: String,
        entry: String,
        folder: String,
    },

    #[error(
        "dataset `{dataset}`: storage entry `{entry}` holds the object `{object}` beside \
         partition folders"
    )]
    ObjectBesidePartitions {
        dataset: String,
        entry: String,
        object: String,
    },

    #[error(
        "dataset `{dataset}`: storage entry `{entry}` holds the partition `{folder}` with the \
         keys {keys:?}, and its first partition has the keys {first:?}"
    )]
    PartitionKeySequencesDiffer {
        dataset: String,
        entry: String,
        folder: String,
        keys: Vec<String>,
        first: Vec<String>,
    },

    #[error(
        "dataset `{dataset}`: storage entry `{entry}` holds the folder `{folder}`, whose key \
         or value is empty"
    )]
    PartitionKeyValueEmpty {
        dataset: String,
        entry: String,
        folder: String,
    },

    #[error(
        "dataset `{dataset}`: storage entry `{entry}` spells one partition key as `{first}` \
         and as `{second}`"
    )]
    PartitionKeySpelledTwice {
        dataset: String,
        entry: String,
        first: String,
        second: String,
    },

    #[error(
        "dataset `{dataset}`: storage entry `{entry}` holds the folder `{folder}`, whose path \
         names the key `{key}` twice"
    )]
    PartitionKeyRepeated {
        dataset: String,
        entry: String,
        folder: String,
        key: String,
    },

    #[error(
        "dataset `{dataset}`: a `partition_def` entry names `{dimension}`, which is no \
         Dimension of the Dataset"
    )]
    PartitionEntryUnknownDimension { dataset: String, dimension: String },

    #[error("dataset `{dataset}`: two `partition_def` entries name the Dimension `{dimension}`")]
    PartitionEntryRepeated { dataset: String, dimension: String },

    #[error(
        "dataset `{dataset}`: the `partition_def` entry for `{dimension}` names a Dimension \
         that maps no column directly"
    )]
    PartitionEntryNotColumn { dataset: String, dimension: String },

    #[error(
        "dataset `{dataset}`: the `partition_def` entry for `{dimension}` names the key \
         `{key}`, and storage entry `{entry}` has the keys {keys:?}"
    )]
    PartitionEntryInapplicable {
        dataset: String,
        entry: String,
        dimension: String,
        key: String,
        keys: Vec<String>,
    },

    #[error(
        "dataset `{dataset}`: storage entry `{entry}` holds the value `{value}` of the key \
         `{key}`, which the `partition_def` entry for `{dimension}` does not list"
    )]
    PartitionValueNotListed {
        dataset: String,
        entry: String,
        dimension: String,
        key: String,
        value: String,
    },

    #[error(
        "dataset `{dataset}`: storage entry `{entry}` holds the value `{value}` of the key \
         that `{dimension}` maps, which does not read as `{data_type}`, its type"
    )]
    PartitionValueNotCast {
        dataset: String,
        entry: String,
        dimension: String,
        value: String,
        data_type: DataType,
    },

    #[error(
        "dataset `{dataset}`: storage entry `{entry}` is read as {}; a path source is read as \
         `parquet`, and a table source is an `iceberg` table",
        .format.map_or_else(|| "no format".to_owned(), |format| format!("`{format}`"))
    )]
    UnsupportedSourceFormat {
        dataset: String,
        entry: String,
        format: Option<FileFormat>,
    },

    #[error(
        "dataset `{dataset}`: `tables:` entry `{entry}` is a glob; a table entry names one table"
    )]
    TableGlob { dataset: String, entry: String },

    #[error("dataset `{dataset_id}` declares no physical source to bind against")]
    DatasetWithoutSource { dataset_id: EntityId },

    #[error("orphan expression in pool: {expr_id} is named by no signature and no Joinset source")]
    OrphanExpression { expr_id: PhysicalExprId },

    #[error(
        "metadata dimension `{semantic}`: physical source `{physical_source}` (`{locator}`) has \
         no location segment at position {token}"
    )]
    MetadataSegmentUnavailable {
        semantic: EntityId,
        physical_source: EntityId,
        locator: String,
        token: u32,
    },

    #[error("metadata dimension `{semantic}`: extraction form `{form}` is not ratified")]
    MetadataExtractionUnsupported {
        semantic: EntityId,
        form: &'static str,
    },

    #[error(
        "semantic registry holds {registry_len} entries, which exceeds the {bit_width}-bit \
         bit_position width"
    )]
    BitmaskValueOverflow { registry_len: usize, bit_width: u32 },

    #[error("relationship `{relationship}`: join key `{key}` is not exposed by its `{side}` side")]
    RelationshipJoinKeyUnknownOnSide {
        relationship: String,
        key: String,
        side: RelationshipSide,
    },

    #[error(
        "relationship `{relationship}`: join key `{key}` resolves to a {role:?}, not a Dimension"
    )]
    RelationshipJoinKeyNotADimension {
        relationship: String,
        key: String,
        role: SemanticRole,
    },

    #[error(
        "relationship `{relationship}`: join key `{key}` resolves to {from_id} on `from` but \
         {to_id} on `to`"
    )]
    RelationshipJoinKeyIdMismatch {
        relationship: String,
        key: String,
        from_id: EntityId,
        to_id: EntityId,
    },

    #[error("relationship `{relationship}`: join key `{key}` is declared more than once")]
    RelationshipJoinKeyDuplicated { relationship: String, key: String },

    #[error("relationship `{relationship}`: {reason}")]
    RelationshipResolution {
        relationship: String,
        reason: String,
    },

    #[error("cross-grain unit `{unit}`: {reason}")]
    CrossGrainResolution { unit: String, reason: String },

    #[error(
        "relationship `{relationship}`: `cardinality:` is required unless the basis is `cross`"
    )]
    RelationshipMissingCardinality { relationship: String },

    #[error(
        "relationship `{relationship}`: resolved cardinality {cardinality:?} may not carry a \
         directional cross_filter {cross_filter:?}"
    )]
    RelationshipManyToManyCrossFilterDirectional {
        relationship: String,
        cardinality: Cardinality,
        cross_filter: CrossFilter,
    },

    #[error(
        "data kind `{data_kind}`: keyset `{keyset}` names `{site}`, which is not a pooled \
         Semantics and so has no registry entry to reference"
    )]
    KeysetMemberNotPooled {
        data_kind: EntityId,
        keyset: KeysetName,
        site: String,
    },

    #[error(
        "data kind `{data_kind}`: keyset `{keyset}` names `{site}`, which resolves to a \
         {role:?}, not a Dimension"
    )]
    KeysetMemberNotADimension {
        data_kind: EntityId,
        keyset: KeysetName,
        site: String,
        role: SemanticRole,
    },

    #[error(
        "composite `{composite}`: keyset `{keyset}` claims {group:?}, which falls outside every \
         boundary element its children admit; `{dimension}` is the member they contradict"
    )]
    KeysetOutsideBoundary {
        composite: EntityId,
        keyset: KeysetName,
        group: Vec<EntityId>,
        dimension: EntityId,
    },

    #[error(
        "composite `{composite}`: the derived keyset name `{keyset}` is already held by an \
         authored keyset on the same composite, and one name is one entity"
    )]
    DerivedKeysetNameCollision {
        composite: EntityId,
        keyset: KeysetName,
    },

    #[error(
        "composite `{composite}`: the claims {shapes:?} are different time axes, so the composed \
         surface has no one shape to publish and compile will not select one"
    )]
    ComposedTemporalDisagreement {
        composite: EntityId,
        shapes: Vec<(EntityId, TemporalType)>,
    },

    #[error(
        "grainset `{grainset}`: children {} have the same grain ({}) and \
         serve the same names",
        quoted_list(.children),
        time_axis_phrase(.time_axis)
    )]
    GrainsetGrainRepeated {
        grainset: DataKindName,
        children: Vec<DataKindName>,
        time_axis: Option<(SemanticsName, Granularity)>,
    },

    #[error(
        "composite `{composite}`: its constituent `{constituent}` carries no granularity, \
         declared or derived, so a request has no granularity to route it by; declare one \
         as `extras.temporal.granularity:`"
    )]
    ComposedConstituentMissingGranularity {
        composite: EntityId,
        constituent: EntityId,
    },

    #[error(
        "composite `{composite}`: its constituent `{constituent}` carries a time axis but \
         publishes no temporal shape (for example, the shapes below it tie at one rank on \
         different time axes), so a request has no granularity to route it by; \
         it may state one of the shapes below it, with a granularity"
    )]
    ComposedConstituentMissingTemporalShape {
        composite: EntityId,
        constituent: EntityId,
    },

    #[error(
        "composite `{composite}`: the declared granularity {stated} is not served by the \
         constituents {constituents:?} below it"
    )]
    StatedGranularityNotServed {
        composite: EntityId,
        stated: Granularity,
        constituents: Vec<(EntityId, Granularity)>,
    },

    #[error(
        "`{node}`: the tagged literal {literal:?} narrows to the declared {declared}, and a \
         tagged literal's narrowing cast is refused"
    )]
    NarrowingLiteral {
        node: String,
        literal: Literal,
        declared: DataType,
    },

    #[error("model overrode catalog for source {source_id}: `{model_value}` (catalog: `{catalog_value}`)")]
    CatalogOverride {
        source_id: EntityId,
        catalog_value: String,
        model_value: String,
    },

    #[error("expression {expr_id}: {error}")]
    UndeclaredOption {
        expr_id: PhysicalExprId,
        error: CallError,
    },

    #[error(
        "the manifest's `dialect` is `{dialect}`, and a manifest is written and read only in \
         `canonical`"
    )]
    ManifestDialect { dialect: String },

    #[error("dangling reference from {from:?} to {to_kind:?}: id {target_id} not found")]
    DanglingReference {
        from: ReferenceSite,
        to_kind: ReferenceTargetKind,
        target_id: String,
    },

    #[error("structural violation at {site:?}: {rule}")]
    StructuralViolation {
        site: ReferenceSite,
        rule: &'static str,
    },

    #[error(
        "definition names `{first}` and `{second}` are equal without regard to case, which \
         an engine that folds identifiers would collide"
    )]
    DuplicateDefinitionName { first: String, second: String },

    #[error("definition name `{name}` starts with the reserved prefix `{prefix}`")]
    ReservedDefinitionName { name: String, prefix: &'static str },

    #[error(
        "relationship `{relationship}`: its `filter:` names `{semantics}`, a {role:?}, but a \
         relationship filter reads only row-level semantics"
    )]
    MeasureOrMetricInRelationshipFilter {
        relationship: String,
        semantics: String,
        role: SemanticRole,
    },

    #[error(
        "semantics `{semantics}` reads the named Filter `{filter}` from its `expr:` or `filter:`"
    )]
    NamedFilterReference { semantics: String, filter: String },

    #[error(
        "semantic source `{semantic_source}` exposes `{semantics}`, but no operand or child \
         maps it"
    )]
    UnservedComposedSemantics {
        semantics: String,
        semantic_source: String,
    },

    #[error(
        "Grainset `{grainset}` cannot compute `{metric}`: no child maps `{name}`, which it reads"
    )]
    GrainsetMetricReadsUnmapped {
        grainset: String,
        metric: String,
        name: String,
    },

    #[error(
        "Grainset `{grainset}` cannot compute `{metric}`: no child maps `{measure}` together \
         with `{dimension}`, which its filter reads"
    )]
    GrainsetMetricFilterUnmapped {
        grainset: String,
        metric: String,
        measure: String,
        dimension: String,
    },

    #[error(
        "Grainset `{grainset}` cannot compute `{metric}`: its part key reads `{dimension}`, and \
         `{child}`, which gives `{measure}`, does not map `{dimension}` at the grain the key reads"
    )]
    GrainsetMetricPartKeyUnformed {
        grainset: String,
        metric: String,
        measure: String,
        dimension: String,
        child: String,
    },

    #[error(
        "Grainset `{grainset}` cannot compute `{metric}`: either the children that serve the \
         names it reads share no keyset that maps them together (a Metric without `agg:` needs \
         one, and so does the formula grain of a Metric with `agg:`), or a child that gives a \
         value maps a Dimension of the formula grain coarser than the finest granularity at \
         which such a child maps it"
    )]
    GrainsetMetricWithoutKeyset { grainset: String, metric: String },

    #[error(
        "relationship `{relationship}`: its join basis names `{key}`, which semantic source \
         `{semantic_source}` does not map"
    )]
    JoinKeyNotMapped {
        relationship: String,
        key: String,
        semantic_source: String,
    },

    #[error("dataset `{dataset}`: `auto` on a constituent needs an observed schema, and none was observed")]
    AutoMappingWithoutSchema { dataset: String },

    #[error(
        "semantic source `{semantic_source}`: reduction leaf `{leaf}` has no signature for \
         `{semantics}`, which a sibling reduction leaf has"
    )]
    ReductionLeafMissingSignature {
        semantic_source: EntityId,
        leaf: EntityId,
        semantics: String,
    },

    #[error(
        "semantic source `{semantic_source}`: `{semantics}` writes no `by`, and its Datasets \
         give no single time axis to default to"
    )]
    SemiAdditiveByNoDefault {
        semantics: String,
        semantic_source: String,
    },

    #[error(
        "semantic source `{semantic_source}`: `{semantics}`'s `by` (`{by}`) is not a temporal \
         Dimension, or is one of its axes"
    )]
    SemiAdditiveByInvalid {
        semantics: String,
        by: String,
        semantic_source: String,
    },

    #[error(
        "semantic source `{semantic_source}`: `{semantics}`'s `{dimension}` is not a Dimension \
         of the Datasets it reads"
    )]
    SemiAdditiveDimensionNotRead {
        semantics: String,
        dimension: String,
        semantic_source: String,
    },

    #[error(
        "semantic source `{semantic_source}`: `{semantics}`'s `{dimension}` is mapped by two \
         or more of the Datasets it reads, and is not the time axis of exactly one of them"
    )]
    SemiAdditiveDimensionAmbiguous {
        semantics: String,
        dimension: String,
        semantic_source: String,
    },

    #[error(
        "semantic source `{semantic_source}` exposes `{semantics}`, but does not serve its \
         `by` (`{by}`)"
    )]
    SemiAdditiveByNotServed {
        semantics: String,
        by: String,
        semantic_source: String,
    },

    #[error("`{semantics}`'s axis `{axis}` names no definition")]
    SemiAdditiveAxisUnknown { semantics: String, axis: String },

    #[error("`{semantics}`'s `by` (`{by}`) names no definition")]
    SemiAdditiveByUnknown { semantics: String, by: String },

    #[error("`{semantics}`'s axis `{axis}` resolves to a {role:?}, not a Dimension")]
    SemiAdditiveAxisNotADimension {
        semantics: String,
        axis: String,
        role: SemanticRole,
    },

    #[error(
        "semantic source `{semantic_source}` exposes `{semantics}`, but does not serve its \
         axis `{axis}`"
    )]
    SemiAdditiveAxisNotServed {
        semantics: String,
        axis: String,
        semantic_source: String,
    },
}

fn timezone_in_words(timezone: &Literal) -> String {
    match timezone {
        Literal::String(zone) => format!("the timezone '{zone}'"),
        Literal::Null(_) => "a NULL timezone".to_owned(),
        other => format!("the timezone {other:?}"),
    }
}

impl Diagnose for CompileError {
    fn message(&self) -> String {
        format!("{self}")
    }

    fn severity_default(&self) -> Severity {
        match self {
            CompileError::CatalogOverride { .. } => Severity::Warning,
            _ => Severity::Error,
        }
    }
}

fn time_axis_phrase(time_axis: &Option<(SemanticsName, Granularity)>) -> String {
    match time_axis {
        Some((axis, granularity)) => format!("`{axis}` at {granularity}"),
        None => "no time axis".to_owned(),
    }
}

fn quoted_list(names: &[DataKindName]) -> String {
    let quoted: Vec<String> = names.iter().map(|name| format!("`{name}`")).collect();
    match quoted.split_last() {
        Some((last, [])) => last.clone(),
        Some((last, rest)) => format!("{} and {last}", rest.join(", ")),
        None => String::new(),
    }
}

#[cfg(test)]
mod compile_error_tests {
    use super::*;
    use semstrait_common::diagnostic::{split_by_severity, Diagnostic, Diagnostics};

    #[test]
    fn warning_variants_default_to_warning_severity() {
        let override_ = CompileError::CatalogOverride {
            source_id: "s1".into(),
            catalog_value: "v1".into(),
            model_value: "v2".into(),
        };
        assert_eq!(override_.severity_default(), Severity::Warning);

        let unmappable = CompileError::UnmappableNativeType {
            column: "region".into(),
            expr_id: None,
            physical_source_id: "s1".into(),
            native_type: "variant".into(),
        };
        let hard = CompileError::UnknownReference {
            name: "foo".into(),
            scope: ReferenceScope::Appearance("b1".into()),
        };
        assert_eq!(unmappable.severity_default(), Severity::Error);
        assert_eq!(hard.severity_default(), Severity::Error);
    }

    #[test]
    fn a_type_error_is_a_hard_error_that_names_its_semantics() {
        let refused = CompileError::TypeError {
            node: "is_big".into(),
            error: TypeError::Cast {
                from: DataType::Boolean.into(),
                to: DataType::Date,
            },
        };
        assert_eq!(refused.severity_default(), Severity::Error);
        let msg = refused.message();
        assert!(msg.contains("is_big"), "names the semantics: {msg}");
        assert!(msg.contains("boolean"), "renders typing's refusal: {msg}");
    }

    #[test]
    fn single_channel_partitions_by_severity() {
        let diags: Diagnostics<CompileError> = vec![
            Diagnostic::new(CompileError::CyclicReference {
                cycle: vec!["a".into(), "b".into()],
            }),
            Diagnostic::new(CompileError::CatalogOverride {
                source_id: "s1".into(),
                catalog_value: "v1".into(),
                model_value: "v2".into(),
            }),
            Diagnostic::new(CompileError::CatalogSourceMissing {
                source_id: "s9".into(),
                locator: "lake.orders".into(),
            }),
        ];
        let (errors, warnings) = split_by_severity(diags);
        assert_eq!(errors.len(), 2);
        assert_eq!(warnings.len(), 1);
        assert!(matches!(
            warnings[0].kind,
            CompileError::CatalogOverride { .. }
        ));
    }

    #[test]
    fn display_renders_pointer_fields() {
        let cyclic = CompileError::CyclicReference {
            cycle: vec!["m_a".into(), "m_b".into()],
        };
        assert_eq!(
            cyclic.message(),
            "cyclic reference among semantics: m_a -> m_b"
        );

        let mismatch = CompileError::SemanticKindMismatch {
            authored_kind: SemanticRole::Measure,
            registered_kind: SemanticRole::Dimension,
            name: "revenue".into(),
        };
        assert!(mismatch.message().contains("revenue"));
        assert!(mismatch.message().contains("Measure"));
        assert!(mismatch.message().contains("Dimension"));
    }

    #[test]
    fn both_enumeration_refusals_are_hard_errors_naming_the_joinset() {
        let ceiling = CompileError::EnumerationCeilingExceeded {
            joinset: "sales".to_owned(),
            units: 26_704,
            ceiling: 2000,
        };
        assert_eq!(ceiling.severity_default(), Severity::Error);
        let msg = ceiling.message();
        assert!(msg.contains("sales"), "the Joinset is named: {msg}");
        assert!(
            msg.contains("26704"),
            "the count reached is rendered: {msg}"
        );
        assert!(
            msg.contains("nothing is truncated"),
            "the message states that nothing is dropped: {msg}"
        );

        let collision = CompileError::EnumeratedUnitNameCollision {
            joinset: "sales".to_owned(),
            derived_name: "sales.r1.r2".to_owned(),
            first: "sales.r1.r2".to_owned(),
            second: vec!["r1".to_owned(), "r2".to_owned(), "r3".to_owned()],
        };
        assert_eq!(collision.severity_default(), Severity::Error);
        let msg = collision.message();
        assert!(msg.contains("sales.r1.r2"), "the derived name: {msg}");
        assert!(msg.contains("r3"), "the colliding relationship set: {msg}");
    }

    #[test]
    fn both_keyset_member_refusals_are_hard_errors_naming_their_own_cause() {
        let unpooled = CompileError::KeysetMemberNotPooled {
            data_kind: "dk.orders".into(),
            keyset: KeysetName::from("natural"),
            site: "order_ts".to_owned(),
        };
        assert_eq!(unpooled.severity_default(), Severity::Error);
        let msg = unpooled.message();
        assert!(msg.contains("dk.orders"), "names the DataKind: {msg}");
        assert!(msg.contains("natural"), "names the keyset: {msg}");
        assert!(msg.contains("order_ts"), "names the site: {msg}");
        assert!(
            !msg.contains("Dimension"),
            "no entry resolved, so no role may be claimed: {msg}"
        );

        let wrong_role = CompileError::KeysetMemberNotADimension {
            data_kind: "dk.orders".into(),
            keyset: KeysetName::from("natural"),
            site: "revenue".to_owned(),
            role: SemanticRole::Measure,
        };
        assert_eq!(wrong_role.severity_default(), Severity::Error);
        let msg = wrong_role.message();
        assert!(msg.contains("dk.orders"), "names the DataKind: {msg}");
        assert!(msg.contains("natural"), "names the keyset: {msg}");
        assert!(msg.contains("revenue"), "names the site: {msg}");
        assert!(msg.contains("Measure"), "names the resolved role: {msg}");
        assert!(
            msg.contains("not a Dimension"),
            "names the role required: {msg}"
        );
    }

    #[test]
    fn reference_scope_display() {
        assert_eq!(
            ReferenceScope::DataKind("k1".into()).to_string(),
            "data-kind k1"
        );
        assert_eq!(
            ReferenceScope::Appearance("b1".into()).to_string(),
            "appearance b1"
        );
    }

    #[test]
    fn catalog_fetch_is_hard_error_and_renders_reason() {
        let fetch = CompileError::CatalogFetch {
            source_id: "s1".into(),
            locator: "lake.orders".into(),
            reason: "authentication failed".into(),
        };
        assert_eq!(fetch.severity_default(), Severity::Error);
        let msg = fetch.message();
        assert!(msg.contains("lake.orders"), "locator rendered: {msg}");
        assert!(
            msg.contains("authentication failed"),
            "reason rendered: {msg}"
        );
        assert!(!matches!(fetch, CompileError::CatalogSourceMissing { .. }));
    }

    #[test]
    fn dataset_without_source_is_hard_error_and_names_the_dataset() {
        let without = CompileError::DatasetWithoutSource {
            dataset_id: "orders".into(),
        };
        assert_eq!(without.severity_default(), Severity::Error);
        let msg = without.message();
        assert!(msg.contains("orders"), "names the Dataset: {msg}");
        assert!(!matches!(
            without,
            CompileError::CatalogSourceMissing { .. }
        ));
        assert!(
            !msg.contains("catalog"),
            "does not blame the catalog: {msg}"
        );
    }

    const REVENUE_ENTRY: PhysicalExprId = PhysicalExprId::new(uuid::Uuid::from_u128(
        0x253641c7_819a_5b60_a3dc_c276bfceb36d,
    ));

    #[test]
    fn orphan_expression_is_hard_error_and_renders_handle() {
        let orphan = CompileError::OrphanExpression {
            expr_id: REVENUE_ENTRY,
        };
        assert_eq!(orphan.severity_default(), Severity::Error);
        assert_eq!(
            orphan.message(),
            "orphan expression in pool: 253641c7-819a-5b60-a3dc-c276bfceb36d is named by no \
             signature and no Joinset source"
        );
    }

    #[test]
    fn an_undeclared_option_renders_the_entry_s_id() {
        let fault = CompileError::UndeclaredOption {
            expr_id: REVENUE_ENTRY,
            error: CallError::UnknownOption {
                function: "concat",
                option: "null_policy".to_owned(),
            },
        };
        assert_eq!(
            fault.message(),
            "expression 253641c7-819a-5b60-a3dc-c276bfceb36d: `concat` has no option \
             `null_policy`"
        );
    }
}
