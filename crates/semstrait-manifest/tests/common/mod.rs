#![allow(dead_code)]

pub mod golden;

use std::time::{Duration, UNIX_EPOCH};

use async_trait::async_trait;
use bytes::Bytes;
use semstrait_catalog::{
    CatalogId, CatalogProvider, CatalogProviderErrorKind, DriftReport, FileFormat, NamespaceRef,
    Partition, ResolvedTable, Schema as NativeSchema, SchemaColumn as NativeColumn,
    SnapshotMetadata, SnapshotVersion, TableRef,
};
use semstrait_common::diagnostic::Diagnostic;
use semstrait_common::io::{FileSystem, InMemoryFileSystem, Path};
use semstrait_common::EntityId;
use semstrait_common::Granularity;
use semstrait_ir::PhysicalExpr;
use semstrait_manifest::{
    compile, CompileError, Link, ManifestProvenance, PhysicalExprId, RoleDefinition,
    SemanticManifest, SemanticSignature, SemanticSource,
};
use semstrait_model::expr::{dim, lit};
use semstrait_model::{
    parse, Agg, Dataset, DatasetBuilder, Dimension, DimensionSite, Measure, MeasureSite,
    MetadataSource, Metric, MetricSite, NestedDataset, PathTokenRef, SemanticMapping,
    SemanticModel, StorageFormat, TemporalShape, UnresolvedSemanticExpr,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{
    AggregateAllowed, AggregateCall, Aggregation, Call, CastFailure, Expr, ExprOperationExt,
    FunctionRef, Leaf, Literal, NamedReference, NullOrdering, Over, SemanticLeaf, SortDir, SortKey,
    UntypedLiteral, WindowCall,
};
use semstrait_sql::functions::registry;

use golden::Golden;

pub fn test_provenance() -> ManifestProvenance {
    ManifestProvenance::new(
        UNIX_EPOCH + Duration::from_secs(1_700_000_000),
        UNIX_EPOCH + Duration::from_secs(1_800_000_000),
    )
}

pub fn lake_folder(name: &str) -> String {
    format!("s3://lake/{name}/")
}

pub async fn seeded_fs(keys: impl IntoIterator<Item = impl Into<String>>) -> InMemoryFileSystem {
    let fs = InMemoryFileSystem::default();
    for key in keys {
        fs.write(&Path::new(key), Bytes::from_static(b"PAR1"))
            .await
            .expect("seeding the in-memory filesystem");
    }
    fs
}

pub async fn fixture_fs(names: &[&str]) -> InMemoryFileSystem {
    seeded_fs((names.iter()).map(|name| format!("{}part-0.parquet", lake_folder(name)))).await
}

pub fn constituent(name: &str, names: &[&str]) -> NestedDataset {
    NestedDataset::builder(name)
        .path(lake_folder(name))
        .format(StorageFormat::Parquet)
        .semantic_mapping(identity_mapping(names))
        .build()
}

pub fn dated(name: &str, names: &[&str], granularity: Granularity) -> NestedDataset {
    dated_on(name, names, "date", granularity)
}

pub fn dated_on(name: &str, names: &[&str], axis: &str, granularity: Granularity) -> NestedDataset {
    NestedDataset::builder(name)
        .path(lake_folder(name))
        .format(StorageFormat::Parquet)
        .semantic_mapping(identity_mapping(names))
        .temporal(TemporalShape::events(axis, Some(granularity)))
        .build()
}

pub fn identity_mapping(names: &[&str]) -> SemanticMapping {
    let mut mapping = SemanticMapping::builder();
    for mapped in names {
        mapping = mapping.column(*mapped, *mapped);
    }
    mapping.build()
}

pub fn agg(name: &str) -> Agg {
    Agg::of_name(name).expect("a catalog aggregate")
}

pub fn path_token(token: u32) -> MetadataSource {
    MetadataSource::Path(
        serde_json::from_str::<PathTokenRef>(&format!("{{\"token\":{token}}}"))
            .expect("a path-token reference decodes from its own wire form"),
    )
}

pub fn dimension(name: &str, data_type: DataType) -> Dimension {
    Dimension::builder(name)
        .data_type(data_type)
        .categorical()
        .build()
}

pub fn bodied(name: &str, data_type: DataType, expr: UnresolvedSemanticExpr) -> Dimension {
    Dimension::builder(name)
        .data_type(data_type)
        .categorical()
        .expr(expr)
        .build()
}

pub fn measure(name: &str, data_type: DataType, aggregate: &str) -> Measure {
    Measure::builder(name)
        .data_type(data_type)
        .agg(agg(aggregate))
        .build()
}

pub fn orders_model(
    dimensions: Vec<Dimension>,
    measures: Vec<Measure>,
    dataset: impl FnOnce(DatasetBuilder) -> DatasetBuilder,
) -> SemanticModel {
    let mut model = SemanticModel::default();
    model.name = "sales".to_owned();
    let mut orders = Dataset::builder("orders").table("warehouse.orders");
    for dimension in dimensions {
        orders = orders.dimension(DimensionSite::r#ref(dimension.name().as_str()));
        insert_dimension(&mut model, dimension);
    }
    for measure in measures {
        orders = orders.measure(MeasureSite::r#ref(measure.name().as_str()));
        insert_measure(&mut model, measure);
    }
    insert_dataset(&mut model, dataset(orders).build());
    model
}

pub fn orders_model_with_metrics(
    dimensions: Vec<Dimension>,
    measures: Vec<Measure>,
    metrics: Vec<Metric>,
) -> SemanticModel {
    let mut model = orders_model(dimensions, measures, |orders| {
        (metrics.iter()).fold(orders, |orders, metric| {
            orders.metric(MetricSite::r#ref(metric.name().as_str()))
        })
    });
    for metric in metrics {
        insert_metric(&mut model, metric);
    }
    model
}

pub fn decimal_18_2() -> DataType {
    DataType::Decimal {
        precision: 18,
        scale: 2,
    }
}

pub fn insert_dimension(model: &mut SemanticModel, dimension: Dimension) {
    let id = dimension.entity_id().expect("the mint is eager").to_owned();
    model.dimensions.insert(id, dimension);
}

pub fn insert_measure(model: &mut SemanticModel, measure: Measure) {
    let id = measure.entity_id().expect("the mint is eager").to_owned();
    model.measures.insert(id, measure);
}

pub fn insert_metric(model: &mut SemanticModel, metric: Metric) {
    let id = metric.entity_id().expect("the mint is eager").to_owned();
    model.metrics.insert(id, metric);
}

pub fn insert_dataset(model: &mut SemanticModel, dataset: Dataset) {
    let id = dataset.body.base.id.clone().expect("the mint is eager");
    model.datasets.insert(id, dataset);
}

pub fn model_from_yaml(yaml: &str) -> SemanticModel {
    let (model, warnings) = parse(yaml)
        .unwrap_or_else(|d| panic!("parse must succeed for:\n{yaml}\ngot {d:?}"))
        .build()
        .unwrap_or_else(|d| panic!("build must succeed for:\n{yaml}\ngot {d:?}"));
    assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
    model
}

pub async fn compile_model(model: SemanticModel) -> SemanticManifest {
    compile_model_with(model, None).await
}

pub async fn compile_model_with(
    model: SemanticModel,
    catalog: Option<&dyn CatalogProvider>,
) -> SemanticManifest {
    compile_with_on(model, catalog, &InMemoryFileSystem::new()).await
}

pub fn names(manifest: &SemanticManifest) -> Vec<String> {
    let mut out: Vec<String> = manifest
        .semantic_sources
        .values()
        .map(|entry| entry.name.as_str().to_owned())
        .collect();
    out.sort();
    out
}

pub async fn compile_model_on(model: SemanticModel, fs: &dyn FileSystem) -> SemanticManifest {
    compile_with_on(model, None, fs).await
}

async fn compile_with_on(
    model: SemanticModel,
    catalog: Option<&dyn CatalogProvider>,
    fs: &dyn FileSystem,
) -> SemanticManifest {
    let (manifest, warnings) = compile(model, catalog, fs, test_provenance())
        .await
        .expect("the fixture model must compile clean");
    assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
    assert!(
        manifest.validate().is_ok(),
        "a compiled artifact must satisfy its own validator: {:?}",
        manifest.validate().err()
    );
    manifest
}

pub async fn compile_error(model: SemanticModel) -> CompileError {
    compile_error_with(model, None).await
}

pub async fn compile_error_with(
    model: SemanticModel,
    catalog: Option<&dyn CatalogProvider>,
) -> CompileError {
    compile_error_with_on(model, catalog, &InMemoryFileSystem::new()).await
}

pub async fn compile_error_on(model: SemanticModel, fs: &dyn FileSystem) -> CompileError {
    compile_error_with_on(model, None, fs).await
}

pub async fn compile_error_with_on(
    model: SemanticModel,
    catalog: Option<&dyn CatalogProvider>,
    fs: &dyn FileSystem,
) -> CompileError {
    compile_diagnostic_with_on(model, catalog, fs).await.kind
}

pub async fn compile_diagnostic(model: SemanticModel) -> Diagnostic<CompileError> {
    compile_diagnostic_with(model, None).await
}

pub async fn compile_diagnostic_with(
    model: SemanticModel,
    catalog: Option<&dyn CatalogProvider>,
) -> Diagnostic<CompileError> {
    compile_diagnostic_with_on(model, catalog, &InMemoryFileSystem::new()).await
}

pub async fn compile_diagnostic_with_on(
    model: SemanticModel,
    catalog: Option<&dyn CatalogProvider>,
    fs: &dyn FileSystem,
) -> Diagnostic<CompileError> {
    compile(model, catalog, fs, test_provenance())
        .await
        .expect_err("the model must be refused")
        .0
}

pub async fn refusal_golden(case: &str, model: SemanticModel, expected: CompileError) -> Golden {
    refusal_golden_with_on(case, model, None, &InMemoryFileSystem::new(), expected).await
}

pub async fn refusal_golden_on(
    case: &str,
    model: SemanticModel,
    fs: &dyn FileSystem,
    expected: CompileError,
) -> Golden {
    refusal_golden_with_on(case, model, None, fs, expected).await
}

pub async fn refusal_golden_with_on(
    case: &str,
    model: SemanticModel,
    catalog: Option<&dyn CatalogProvider>,
    fs: &dyn FileSystem,
    expected: CompileError,
) -> Golden {
    let diagnostic = compile_diagnostic_with_on(model, catalog, fs).await;
    assert_eq!(diagnostic.kind, expected, "{case}");
    Golden::message(case, diagnostic.to_string())
}

#[derive(Debug)]
pub struct MapCatalog {
    tables: std::collections::BTreeMap<String, (TableRef, NativeSchema)>,
    partitions: Vec<Partition>,
    format: FileFormat,
    dialect: Option<semstrait_common::Dialect>,
}

impl Default for MapCatalog {
    fn default() -> Self {
        Self {
            tables: std::collections::BTreeMap::new(),
            partitions: Vec::new(),
            format: FileFormat::Iceberg,
            dialect: None,
        }
    }
}

impl MapCatalog {
    pub fn new(known: TableRef, schema: NativeSchema) -> Self {
        Self::default().with_table(known, schema)
    }

    pub fn with_table(mut self, table: TableRef, schema: NativeSchema) -> Self {
        self.tables.insert(table.fully_qualified(), (table, schema));
        self
    }

    pub fn with_partitions(mut self, partitions: Vec<Partition>) -> Self {
        self.partitions = partitions;
        self
    }

    pub fn with_format(mut self, format: FileFormat) -> Self {
        self.format = format;
        self
    }

    pub fn with_dialect(mut self, dialect: semstrait_common::Dialect) -> Self {
        self.dialect = Some(dialect);
        self
    }

    fn known(
        &self,
        table: &TableRef,
    ) -> Result<&(TableRef, NativeSchema), CatalogProviderErrorKind> {
        self.tables.get(&table.fully_qualified()).ok_or_else(|| {
            CatalogProviderErrorKind::TableNotFound {
                fqn: table.fully_qualified(),
            }
        })
    }
}

#[async_trait]
impl CatalogProvider for MapCatalog {
    fn id(&self) -> CatalogId {
        CatalogId::new("fake")
    }

    fn dialect(&self) -> Option<semstrait_common::Dialect> {
        self.dialect
    }

    async fn list_tables(
        &self,
        _namespace: &NamespaceRef,
    ) -> Result<Vec<TableRef>, CatalogProviderErrorKind> {
        Ok(self
            .tables
            .values()
            .map(|(table, _)| table.clone())
            .collect())
    }

    async fn resolve_table(
        &self,
        table: &TableRef,
    ) -> Result<ResolvedTable, CatalogProviderErrorKind> {
        let (table, schema) = self.known(table)?;
        Ok(ResolvedTable::new(
            table.clone(),
            Path::new(format!("s3://wh/db/{}", table.fully_qualified())),
            self.format,
            schema.clone(),
            self.partitions.clone(),
            SnapshotMetadata::new(SnapshotVersion::Id(7), Some(7)),
        ))
    }

    async fn get_schema(&self, table: &TableRef) -> Result<NativeSchema, CatalogProviderErrorKind> {
        Ok(self.known(table)?.1.clone())
    }

    async fn get_partitions(
        &self,
        _table: &TableRef,
    ) -> Result<Vec<Partition>, CatalogProviderErrorKind> {
        Ok(self.partitions.clone())
    }

    async fn get_snapshot(
        &self,
        _table: &TableRef,
        _version: SnapshotVersion,
    ) -> Result<SnapshotMetadata, CatalogProviderErrorKind> {
        Ok(SnapshotMetadata::current_unpinned())
    }

    async fn check_schema_drift(
        &self,
        table: &TableRef,
        _expected: &NativeSchema,
    ) -> Result<DriftReport, CatalogProviderErrorKind> {
        Err(CatalogProviderErrorKind::TableNotFound {
            fqn: table.fully_qualified(),
        })
    }
}

pub fn warehouse_orders() -> TableRef {
    TableRef::new(NamespaceRef::new(["warehouse"]), "orders")
}

pub fn warehouse_catalog(tables: &[(&str, &[(&str, &str)])]) -> MapCatalog {
    (tables.iter()).fold(MapCatalog::default(), |catalog, &(name, columns)| {
        let columns = (columns.iter())
            .map(|&(column, native)| NativeColumn::new(column, native, true))
            .collect();
        catalog.with_table(
            TableRef::new(NamespaceRef::new(["warehouse"]), name),
            NativeSchema::new(columns),
        )
    })
}

pub fn catalog(columns: &[(&str, &str)]) -> MapCatalog {
    warehouse_catalog(&[("orders", columns)])
}

pub fn definition_id(manifest: &SemanticManifest, name: &str) -> EntityId {
    manifest
        .semantic_definitions
        .by_name(name)
        .unwrap_or_else(|| panic!("no definition named `{name}`"))
        .id
        .clone()
}

pub fn find_signature<'m>(
    manifest: &'m SemanticManifest,
    source_name: &str,
    semantic: &str,
) -> Option<&'m SemanticSignature> {
    let definition_id = definition_id(manifest, semantic);
    listed(manifest, by_name(manifest, source_name))
        .find(|signature| signature.definition_id == definition_id)
}

pub fn signature<'m>(
    manifest: &'m SemanticManifest,
    source_name: &str,
    semantic: &str,
) -> &'m SemanticSignature {
    find_signature(manifest, source_name, semantic)
        .unwrap_or_else(|| panic!("`{source_name}` realises no `{semantic}`"))
}

pub fn has_signature(manifest: &SemanticManifest, source_name: &str, semantic: &str) -> bool {
    find_signature(manifest, source_name, semantic).is_some()
}

pub fn listed<'m>(
    manifest: &'m SemanticManifest,
    source: &'m SemanticSource,
) -> impl Iterator<Item = &'m SemanticSignature> {
    manifest.signatures_of(source)
}

pub fn source_signatures(
    manifest: &SemanticManifest,
    root: PhysicalExprId,
) -> Vec<&SemanticSignature> {
    let holds = |source: &SemanticSource| {
        listed(manifest, source).any(|signature| {
            signature.expression_id == root || signature.filter_expr_id == Some(root)
        }) || matches!(
            source.role_definition,
            RoleDefinition::Joinset { on_expr_id, filter_expr_id, .. }
                if on_expr_id == Some(root) || filter_expr_id == Some(root)
        )
    };
    let source = (manifest.semantic_sources.values())
        .find(|source| holds(source))
        .unwrap_or_else(|| panic!("no semantic source holds the entry {root}"));
    listed(manifest, source).collect()
}

pub fn stored_tree(manifest: &SemanticManifest, root: PhysicalExprId) -> PhysicalExpr {
    manifest
        .expressions
        .reassemble(&root, &source_signatures(manifest, root), &[])
        .expect("the entry's tree reads whole")
}

pub fn link_free_tree(manifest: &SemanticManifest, root: PhysicalExprId) -> PhysicalExpr {
    let entry =
        (manifest.expressions.get(&root)).unwrap_or_else(|| panic!("the entry {root} is stored"));
    assert_eq!(
        entry.expr.links().count(),
        0,
        "the entry {root} holds no link (ADR 0088, rules 5 and 14)"
    );
    stored_tree(manifest, root)
}

pub fn tree(manifest: &SemanticManifest, signature: &SemanticSignature) -> PhysicalExpr {
    stored_tree(manifest, signature.expression_id)
}

pub async fn stored_with<const N: usize>(
    model: SemanticModel,
    catalog: Option<&dyn CatalogProvider>,
    source: &str,
    names: [&str; N],
) -> [PhysicalExpr; N] {
    let manifest = compile_model_with(model, catalog).await;
    names.map(|name| tree(&manifest, signature(&manifest, source, name)))
}

pub fn filter_tree(manifest: &SemanticManifest, signature: &SemanticSignature) -> PhysicalExpr {
    let root = signature
        .filter_expr_id
        .expect("the signature carries a filter root");
    stored_tree(manifest, root)
}

pub fn stored_expr(manifest: &SemanticManifest, root: PhysicalExprId) -> &Expr<Link> {
    let entry =
        (manifest.expressions.get(&root)).unwrap_or_else(|| panic!("the entry {root} is stored"));
    entry.expr.expr()
}

pub fn link(id: PhysicalExprId) -> Expr<Link> {
    Expr::Leaf(Link(id))
}

pub fn string<L: Leaf, A: Aggregation>(value: &str) -> Expr<L, A> {
    Expr::Literal(Literal::String(value.to_owned()))
}

pub fn zone<L: Leaf, A: Aggregation>(region: Expr<L, A>) -> Expr<L, A> {
    Expr::Case {
        whens: vec![(
            Expr::InList {
                value: Box::new(region),
                list: vec![string("eu"), string("uk")],
                negated: false,
            },
            string("emea"),
        )],
        else_: Some(Box::new(string("other"))),
    }
}

pub fn revenue_and_refunds(metrics: Vec<Metric>) -> SemanticModel {
    let sum = |name: &str| {
        Measure::builder(name)
            .data_type(DataType::Integer)
            .agg(agg("sum"))
    };
    orders_model_with_metrics(
        vec![
            dimension("status", DataType::String),
            dimension("channel", DataType::String),
        ],
        vec![
            sum("revenue")
                .filter(dim("status").expect("a name").eq(lit("paid")))
                .build(),
            sum("refunds").build(),
        ],
        metrics,
    )
}

pub fn is(name: &str, value: &str) -> PhysicalExpr {
    cast(orders_column(name), DataType::String).eq(string(value))
}

pub fn zone_model() -> SemanticModel {
    orders_model(
        vec![
            dimension("region", DataType::String),
            bodied(
                "zone",
                DataType::String,
                zone(dim("region").expect("a name")),
            ),
        ],
        Vec::new(),
        |orders| orders,
    )
}

pub fn by_name<'m>(manifest: &'m SemanticManifest, name: &str) -> &'m SemanticSource {
    manifest
        .semantic_sources
        .values()
        .find(|entry| entry.name.as_str() == name)
        .unwrap_or_else(|| {
            let present: Vec<&str> = manifest
                .semantic_sources
                .values()
                .map(|entry| entry.name.as_str())
                .collect();
            panic!("no semantic source named `{name}`; present: {present:?}")
        })
}

pub fn semantic_alias<'m>(manifest: &'m SemanticManifest, name: &str) -> &'m str {
    by_name(manifest, name).alias.as_str()
}

pub fn source_alias(manifest: &SemanticManifest, dataset: &str) -> String {
    let entry = by_name(manifest, dataset);
    let RoleDefinition::Dataset { source_id, .. } = &entry.role_definition else {
        panic!("`{dataset}` is not a Dataset");
    };
    manifest.sources[source_id].alias.as_str().to_owned()
}

pub fn function(name: &str) -> FunctionRef<AggregateAllowed> {
    registry()
        .get(name)
        .unwrap_or_else(|| panic!("the catalog declares `{name}`"))
}

pub fn call(name: &str, args: Vec<UnresolvedSemanticExpr>) -> UnresolvedSemanticExpr {
    let function = function(name)
        .try_into()
        .expect("a function that a row-level tree holds");
    Expr::Function(Call::new(function, args).expect("a call that fits its declaration"))
}

pub fn lag(value: UnresolvedSemanticExpr, order: UnresolvedSemanticExpr) -> UnresolvedSemanticExpr {
    let FunctionRef::Window(lag) = function("lag") else {
        panic!("`lag` is a window function");
    };
    let over = Over {
        partition_by: Vec::new(),
        order_by: vec![SortKey {
            expr: order,
            dir: SortDir::Asc {
                nulls: NullOrdering::Unspecified,
            },
        }],
        window_frame: None,
    };
    let call = WindowCall::builder(lag, vec![value])
        .over(over)
        .build()
        .expect("a lag call that fits its declaration");
    Expr::Function(Call::Window(call))
}

pub fn untyped(literal: Literal) -> UnresolvedSemanticExpr {
    Expr::Leaf(SemanticLeaf::Untyped(UntypedLiteral(literal)))
}

pub fn physical_call<L: Leaf>(name: &str, args: Vec<Expr<L>>) -> Expr<L> {
    Expr::Function(Call::new(function(name), args).expect("a call that fits its declaration"))
}

pub fn column(alias: &str, name: &str) -> PhysicalExpr {
    Expr::Column(NamedReference::qualified(alias, name).expect("an alias and a name"))
}

pub fn orders_column(name: &str) -> PhysicalExpr {
    column("ps0", name)
}

pub fn aggregate<L: Leaf>(
    name: &str,
    distinct: bool,
    arg: Expr<L>,
    filter: Option<Expr<L>>,
) -> Expr<L> {
    let FunctionRef::Aggregate(function, capability) = function(name) else {
        panic!("`{name}` is no aggregate function");
    };
    let call = AggregateCall::builder(function, vec![arg])
        .distinct(distinct)
        .maybe_filter(filter)
        .build()
        .expect("an aggregate call that fits its declaration");
    Expr::Function(Call::Aggregate(call, capability))
}

pub fn integer_sum(alias: &str, name: &str) -> PhysicalExpr {
    integer_sum_filtered(alias, name, None)
}

pub fn integer_sum_filtered(alias: &str, name: &str, filter: Option<PhysicalExpr>) -> PhysicalExpr {
    let column = cast(column(alias, name), DataType::Integer);
    cast(aggregate("sum", false, column, filter), DataType::Integer)
}

pub fn cast<L: Leaf>(input: Expr<L>, target: DataType) -> Expr<L> {
    Expr::Cast {
        input: Box::new(input),
        target,
        on_failure: CastFailure::Error,
    }
}
