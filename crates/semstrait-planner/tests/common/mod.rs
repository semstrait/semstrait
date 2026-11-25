#![allow(dead_code)]

pub mod golden;

use std::time::UNIX_EPOCH;

use async_trait::async_trait;
use semstrait_catalog::{
    CatalogId, CatalogProvider, CatalogProviderErrorKind, DriftReport, FileFormat, NamespaceRef,
    Partition, ResolvedTable, Schema, SchemaColumn, SnapshotMetadata, SnapshotVersion, TableRef,
};
use semstrait_common::io::{FileSystem, InMemoryFileSystem, Path};
use semstrait_common::{DataKindName, Diagnostic, Diagnostics, Dialect, Granularity, Tree};
use semstrait_ir::{
    field_name, AggregateNode, CrossNode, FilterNode, JoinNode, MultisetNode, Name, Parameter,
    PhysicalExpr, PlanNode, ProjectNode, Reader, ScanNode, SemAnnotation, SemanticPlan, SourceRole,
    TableScan,
};
use semstrait_manifest::{
    compile, ManifestProvenance, SemanticManifest, SemanticSignature, SemanticSource,
};
use semstrait_model::parse;
use semstrait_planner::{
    Advisory, Decision, DimensionModifier, FramePath, PlanError, Planner, ReadModel, Request,
    RequestContext, Selection, Strategy, StrategyInput,
};
use semstrait_sql::expr::{
    AggregateAllowed, AggregateCall, Call, Expr, ExprOperationExt, FunctionRef, Literal,
    NamedReference, NullOrdering, Over, SortDir, SortKey, WindowCall,
};
use semstrait_sql::functions::registry;

#[derive(Debug)]
pub struct Replaced(pub Decision);

impl<Arm> Strategy<Arm> for Replaced {
    fn decide(&self, _: &dyn ReadModel, _: &Arm, _: &StrategyInput) -> Result<Decision, PlanError> {
        Ok(self.0.clone())
    }
}

#[derive(Debug, Default)]
pub struct Catalog {
    tables: Vec<ResolvedTable>,
    dialect: Option<Dialect>,
}

impl Catalog {
    pub fn table(mut self, locator: &str, format: FileFormat, columns: &[(&str, &str)]) -> Self {
        let (namespace, name) = locator
            .rsplit_once('.')
            .expect("a `namespace.table` locator");
        let columns = (columns.iter())
            .map(|(column, native)| SchemaColumn::new(*column, *native, true))
            .collect();
        self.tables.push(ResolvedTable::new(
            TableRef::new(NamespaceRef::new(namespace.split('.')), name),
            Path::new(format!("s3://warehouse/{namespace}/{name}")),
            format,
            Schema::new(columns),
            Vec::new(),
            SnapshotMetadata::new(SnapshotVersion::Id(7), Some(7)),
        ));
        self
    }

    pub fn with_dialect(self, dialect: Dialect) -> Self {
        Self {
            dialect: Some(dialect),
            ..self
        }
    }

    fn resolved(&self, table: &TableRef) -> Result<&ResolvedTable, CatalogProviderErrorKind> {
        (self.tables.iter())
            .find(|resolved| &resolved.table == table)
            .ok_or_else(|| CatalogProviderErrorKind::TableNotFound {
                fqn: table.fully_qualified(),
            })
    }
}

#[async_trait]
impl CatalogProvider for Catalog {
    fn id(&self) -> CatalogId {
        CatalogId::new("fake")
    }

    fn dialect(&self) -> Option<Dialect> {
        self.dialect
    }

    async fn list_tables(
        &self,
        namespace: &NamespaceRef,
    ) -> Result<Vec<TableRef>, CatalogProviderErrorKind> {
        let tables = self.tables.iter().map(|resolved| &resolved.table);
        Ok(tables
            .filter(|table| &table.namespace == namespace)
            .cloned()
            .collect())
    }

    async fn resolve_table(
        &self,
        table: &TableRef,
    ) -> Result<ResolvedTable, CatalogProviderErrorKind> {
        self.resolved(table).cloned()
    }

    async fn get_schema(&self, table: &TableRef) -> Result<Schema, CatalogProviderErrorKind> {
        Ok(self.resolved(table)?.schema.clone())
    }

    async fn get_partitions(
        &self,
        table: &TableRef,
    ) -> Result<Vec<Partition>, CatalogProviderErrorKind> {
        Ok(self.resolved(table)?.partitions.clone())
    }

    async fn get_snapshot(
        &self,
        table: &TableRef,
        _version: SnapshotVersion,
    ) -> Result<SnapshotMetadata, CatalogProviderErrorKind> {
        Ok(self.resolved(table)?.snapshot.clone())
    }

    async fn check_schema_drift(
        &self,
        table: &TableRef,
        _expected: &Schema,
    ) -> Result<DriftReport, CatalogProviderErrorKind> {
        Err(CatalogProviderErrorKind::TableNotFound {
            fqn: table.fully_qualified(),
        })
    }
}

pub async fn compiled(yaml: &str, catalog: &Catalog, fs: &InMemoryFileSystem) -> SemanticManifest {
    let (model, _) = parse(yaml)
        .unwrap_or_else(|diagnostics| panic!("parse refused:\n{yaml}\n{diagnostics:?}"))
        .build()
        .unwrap_or_else(|diagnostics| panic!("build refused:\n{yaml}\n{diagnostics:?}"));
    let provenance = ManifestProvenance::new(UNIX_EPOCH, UNIX_EPOCH);
    let (manifest, _) = compile(model, Some(catalog), fs, provenance)
        .await
        .unwrap_or_else(|diagnostic| panic!("compile refused:\n{yaml}\n{diagnostic:?}"));
    manifest
}

pub async fn seeded(objects: &[&str]) -> InMemoryFileSystem {
    let fs = InMemoryFileSystem::new();
    for object in objects {
        fs.write(&Path::new(*object), b"PAR1".as_slice().into())
            .await
            .expect("seeding the in-memory file system");
    }
    fs
}

pub fn model(semantics: &str, datasets: &str) -> String {
    format!("semantic_model:\n  name: skeleton\n{semantics}  datasets:\n{datasets}")
}

pub async fn over_objects(semantics: &str, datasets: &str, objects: &[&str]) -> SemanticManifest {
    let fs = seeded(objects).await;
    compiled(&model(semantics, datasets), &Catalog::default(), &fs).await
}

pub const ORDERS: &str = r#"
    - name: orders
      extras:
        storage:
          tables: [warehouse.orders]
        semantic_mapping:
          region: region
          channel: channel
          customer_id: customer_id
          ordered_at: ordered_at
          revenue: amount
          order_count: order_id
          web_revenue: amount
      dimensions:
        - ref: region
        - ref: channel
        - ref: customer_id
        - ref: ordered_at
      measures:
        - ref: revenue
        - ref: order_count
        - ref: web_revenue
      metrics:
        - ref: aov
        - ref: non_web
      filters:
        - name: is_eu
          expr:
            eq: [region, {lit: eu}]
        - name: eu_web
          expr:
            and: [is_eu, {eq: [channel, {lit: web}]}]
"#;

pub const TWINS: &str = r#"
    - name: orders_a
      extras:
        storage:
          tables: [warehouse.orders]
        semantic_mapping:
          region: region
          revenue: amount
      dimensions:
        - ref: region
      measures:
        - ref: revenue
    - name: orders_b
      extras:
        storage:
          tables: [warehouse.orders]
        semantic_mapping:
          region: region
          revenue: amount
      dimensions:
        - ref: region
      measures:
        - ref: revenue
"#;

pub const SEMANTICS: &str = r#"
  dimensions:
    - name: region
      data_type: string
      type: categorical
    - name: channel
      data_type: string
      type: categorical
    - name: customer_id
      data_type: string
      type: categorical
    - name: ordered_at
      data_type: date
      type:
        temporal:
          granularities: [day]
  measures:
    - name: revenue
      data_type: double
      agg: sum
    - name: order_count
      data_type: bigint
      agg: count
    - name: web_revenue
      data_type: double
      agg: sum
      filter:
        eq: [channel, {lit: web}]
  metrics:
    - name: aov
      data_type: double
      expr:
        divide: [revenue, order_count]
    - name: non_web
      data_type: double
      expr:
        subtract: [revenue, web_revenue]
"#;

pub fn orders_catalog() -> Catalog {
    Catalog::default().table(
        "warehouse.orders",
        FileFormat::Iceberg,
        &[
            ("order_id", "long"),
            ("customer_id", "string"),
            ("region", "string"),
            ("channel", "string"),
            ("amount", "double"),
            ("ordered_at", "date"),
        ],
    )
}

pub async fn skeleton() -> SemanticManifest {
    with_datasets(ORDERS, &orders_catalog()).await
}

pub async fn with_datasets(datasets: &str, catalog: &Catalog) -> SemanticManifest {
    with_semantics(SEMANTICS, datasets, catalog).await
}

pub async fn with_semantics(
    semantics: &str,
    datasets: &str,
    catalog: &Catalog,
) -> SemanticManifest {
    compiled(
        &model(semantics, datasets),
        catalog,
        &InMemoryFileSystem::new(),
    )
    .await
}

const HARNESS: &str = r#"
semantic_model:
  name: harness
  dimensions:
    - name: region
      data_type: string
      type: categorical
    - name: channel
      data_type: string
      type: categorical
  measures:
    - name: revenue
      data_type: double
      agg: sum
    - name: clicks
      data_type: bigint
      agg: sum
  datasets:
    - name: orders
      extras:
        storage:
          tables: [warehouse.orders]
        semantic_mapping:
          region: region
          revenue: amount
      dimensions:
        - ref: region
      measures:
        - ref: revenue
    - name: events
      extras:
        storage:
          format: parquet
          paths: [s3://semstrait-e2e/events/]
        semantic_mapping:
          channel: channel
          clicks: clicks
      dimensions:
        - ref: channel
      measures:
        - ref: clicks
"#;

pub const EVENTS_OBJECT: &str = "s3://semstrait-e2e/events/part-0.parquet";

pub async fn harness() -> SemanticManifest {
    let catalog = Catalog::default().table(
        "warehouse.orders",
        FileFormat::Iceberg,
        &[("region", "string"), ("amount", "double")],
    );
    let fs = seeded(&[EVENTS_OBJECT]).await;
    compiled(HARNESS, &catalog, &fs).await
}

pub const SPEND_BOOKINGS: &str = r#"
semantic_model:
  name: spend_bookings
  dimensions:
    - name: campaign_id
      data_type: string
      type: categorical
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day]
  measures:
    - name: cost
      data_type: bigint
      agg: sum
    - name: revenue
      data_type: bigint
      agg: sum
  joinsets:
    - name: spend_bookings
      dimensions:
        - ref: campaign_id
        - ref: date
      measures:
        - ref: cost
        - ref: revenue
      datasets:
        - name: spend
          extras:
            storage:
              tables: [warehouse.spend]
        - name: bookings
          extras:
            storage:
              tables: [warehouse.bookings]
      relationships:
        - name: spend_to_bookings
          from: spend
          to: bookings
          on: [campaign_id, date]
          cardinality: one_to_one
          optional: none
          cross_filter: both
          filter:
            gt_eq: [date, {lit: {date: "2026-03-15"}}]
"#;

pub async fn spend_bookings(yaml: &str) -> SemanticManifest {
    spend_bookings_with(yaml, &[]).await
}

pub async fn spend_bookings_with(yaml: &str, more: &[(&str, &str)]) -> SemanticManifest {
    let bookings = [
        ("campaign_id", "string"),
        ("date", "date"),
        ("revenue", "long"),
    ];
    let catalog = Catalog::default()
        .table(
            "warehouse.spend",
            FileFormat::Iceberg,
            &[
                ("campaign_id", "string"),
                ("date", "date"),
                ("cost", "long"),
            ],
        )
        .table(
            "warehouse.bookings",
            FileFormat::Iceberg,
            &[&bookings[..], more].concat(),
        );
    compiled(yaml, &catalog, &InMemoryFileSystem::new()).await
}

const TXN_BY_GRAIN: &str = r#"
semantic_model:
  name: txn_by_grain
  dimensions:
    - name: event_date
      data_type: date
      type:
        temporal:
          granularities: [day]
    - name: country
      data_type: string
      type: categorical
    - name: txn_type
      data_type: string
      type: categorical
  measures:
    - name: revenue
      data_type: double
      agg: sum
    - name: quantity
      data_type: bigint
      agg: sum
    - name: revenue_target
      data_type: double
      agg: sum
    - name: avg_qty
      data_type: double
      agg: avg
  grainsets:
    - name: txn_by_grain
      dimensions:
        - ref: event_date
          is_key: true
        - ref: country
          is_key: true
        - ref: txn_type
      measures:
        - ref: revenue
        - ref: quantity
        - ref: revenue_target
        - ref: avg_qty
      datasets:
"#;

pub const TRANSACTIONS: &str = r#"
        - name: transactions
          extras:
            storage:
              tables: [warehouse.transactions]
            semantic_mapping:
              event_date: created_at
              country: country_code
              txn_type: transaction_type
              revenue: amount
              quantity: qty
              avg_qty: qty
            temporal:
              timeseries:
                occurred_at: event_date
              granularity: day
"#;

pub const TXN_MONTHLY_AGG: &str = r#"
        - name: txn_monthly_agg
          extras:
            storage:
              tables: [warehouse.txn_monthly_agg]
            semantic_mapping:
              event_date: month_start
              country: country_code
              revenue: total_revenue
              revenue_target: target
            temporal:
              timeseries:
                occurred_at: event_date
              granularity: month
"#;

pub const TXN_WEEKLY_AGG: &str = r#"
        - name: txn_weekly_agg
          extras:
            storage:
              tables: [warehouse.txn_weekly_agg]
            semantic_mapping:
              event_date: week_start
              country: country_code
              revenue: total_revenue
              quantity: total_quantity
            temporal:
              timeseries:
                occurred_at: event_date
              granularity: week
"#;

pub async fn txn_by_grain(children: &[&str]) -> SemanticManifest {
    let catalog = Catalog::default()
        .table(
            "warehouse.transactions",
            FileFormat::Iceberg,
            &[
                ("created_at", "date"),
                ("country_code", "string"),
                ("transaction_type", "string"),
                ("amount", "double"),
                ("qty", "long"),
            ],
        )
        .table(
            "warehouse.txn_monthly_agg",
            FileFormat::Iceberg,
            &[
                ("month_start", "date"),
                ("country_code", "string"),
                ("total_revenue", "double"),
                ("target", "double"),
            ],
        )
        .table(
            "warehouse.txn_weekly_agg",
            FileFormat::Iceberg,
            &[
                ("week_start", "date"),
                ("country_code", "string"),
                ("total_revenue", "double"),
                ("total_quantity", "long"),
            ],
        );
    let yaml = format!("{TXN_BY_GRAIN}{}", children.concat());
    compiled(&yaml, &catalog, &InMemoryFileSystem::new()).await
}

pub fn alias(manifest: &SemanticManifest, name: &str) -> String {
    semantic_source(manifest, name).alias.as_str().to_owned()
}

fn semantic_source<'m>(manifest: &'m SemanticManifest, name: &str) -> &'m SemanticSource {
    (manifest.semantic_sources.values())
        .find(|source| source.name.as_str() == name)
        .expect("the semantic source")
}

pub fn signature<'m>(
    manifest: &'m SemanticManifest,
    source: &str,
    semantics: &str,
) -> &'m SemanticSignature {
    signature_of(manifest, source, semantics).expect("its signature")
}

pub fn signature_of<'m>(
    manifest: &'m SemanticManifest,
    source: &str,
    semantics: &str,
) -> Option<&'m SemanticSignature> {
    let definition = (manifest.semantic_definitions.by_name(semantics)).expect("the semantics");
    (manifest.signatures_of(semantic_source(manifest, source)))
        .find(|signature| signature.definition_id == definition.id)
}

pub fn without_signature(manifest: &mut SemanticManifest, source: &str, semantics: &str) {
    let id = signature(manifest, source, semantics).id.clone();
    manifest.semantic_signatures.remove(&id);
    assert!(signature_of(manifest, source, semantics).is_none());
}

pub fn qualified(qualifier: &str, name: &str) -> PhysicalExpr {
    Expr::Column(NamedReference::qualified(qualifier, name).expect("a name"))
}

pub fn integer(value: i32) -> PhysicalExpr {
    Expr::Literal(Literal::Integer(value))
}

pub fn select(names: &[&str]) -> Request {
    Request::new(
        (names.iter())
            .map(|name| Selection::Field((*name).into()))
            .collect(),
    )
}

pub fn at_granularity(name: &str, granularity: Granularity, rest: &[&str]) -> Request {
    let bucketed = Selection::Dimension {
        name: name.into(),
        modifier: Some(DimensionModifier::Granularity(granularity)),
    };
    Request::new(
        std::iter::once(bucketed)
            .chain(rest.iter().map(|name| Selection::Field((*name).into())))
            .collect(),
    )
}

pub fn refusal(
    manifest: &SemanticManifest,
    request: &Request,
) -> (Diagnostic<PlanError>, Diagnostics<Advisory>) {
    refusal_by(&Planner::new(), manifest, request)
}

pub fn refusal_by(
    planner: &Planner,
    manifest: &SemanticManifest,
    request: &Request,
) -> (Diagnostic<PlanError>, Diagnostics<Advisory>) {
    match planner.plan(manifest, request, &RequestContext::default()) {
        Ok(_) => panic!("{request:?} is planned"),
        Err(refused) => refused,
    }
}

pub fn grain_not_servable(
    manifest: &SemanticManifest,
    request: &Request,
    path: &[&str],
    (requested, held): (Option<Granularity>, Option<Granularity>),
    nearest: &[(&str, &[&str])],
) -> Diagnostic<PlanError> {
    let (error, advisories) = refusal(manifest, request);
    let PlanError::GrainNotServable {
        path: at,
        source,
        requested: asked,
        source_granularity,
        nearest: listed,
    } = &error.kind
    else {
        panic!("GrainNotServable, got {error:?}");
    };
    let joined = path.join(" / ");
    assert_eq!(at.to_string(), joined);
    assert_eq!(Some(source.as_str()), path.last().copied());
    assert_eq!((*asked, *source_granularity), (requested, held));
    let listed: Vec<(&str, Vec<&str>)> = (listed.iter())
        .map(|(child, lacks)| {
            (
                child.as_str(),
                lacks.iter().map(|name| name.as_str()).collect(),
            )
        })
        .collect();
    let expected: Vec<(&str, Vec<&str>)> = (nearest.iter())
        .map(|(child, lacks)| (*child, lacks.to_vec()))
        .collect();
    assert_eq!(listed, expected);
    assert_eq!(
        (error.kind.path()).map(ToString::to_string),
        Some(joined),
        "the refusal names its frame"
    );
    assert!(advisories.is_empty());
    error
}

pub fn planned(
    manifest: &SemanticManifest,
    request: &Request,
) -> (SemanticPlan, Diagnostics<Advisory>) {
    planned_in(manifest, request, &RequestContext::default())
}

pub fn planned_in(
    manifest: &SemanticManifest,
    request: &Request,
    context: &RequestContext,
) -> (SemanticPlan, Diagnostics<Advisory>) {
    let plan = || match Planner::new().plan(manifest, request, context) {
        Ok(planned) => planned,
        Err((error, _)) => panic!("{request:?} is refused: {error:?}"),
    };
    let (first, advisories) = plan();
    assert_eq!(first, plan().0, "two calls give equal plans");
    (first, advisories)
}

fn first_inputs(node: &PlanNode) -> impl Iterator<Item = &PlanNode> {
    std::iter::successors(Some(node), |node| node.children().first().copied())
}

pub fn scan_below(node: &PlanNode) -> &ScanNode {
    match first_inputs(node).last() {
        Some(PlanNode::Scan(scan)) => scan,
        other => panic!("the scan, got {other:?}"),
    }
}

pub fn scan_of(plan: &SemanticPlan) -> &ScanNode {
    scan_below(&plan.root)
}

pub fn table_of(scan: &ScanNode) -> &str {
    match &scan.reader {
        Reader::TableScan(TableScan::NamedTable { names, .. }) => &names[2],
        other => panic!("a table's reader, got {other:?}"),
    }
}

pub fn read_table(plan: &SemanticPlan) -> &str {
    table_of(scan_of(plan))
}

pub fn table_below(node: &PlanNode) -> &str {
    table_of(scan_below(node))
}

pub fn tables_read(plan: &SemanticPlan) -> Vec<&str> {
    (plan.root.pre_order())
        .filter_map(|node| match node {
            PlanNode::Scan(scan) => Some(table_of(scan)),
            _ => None,
        })
        .collect()
}

pub fn column(name: &str) -> PhysicalExpr {
    Expr::Column(NamedReference::unqualified(name).expect("a name"))
}

pub fn columns_of<'a>(
    manifest: &'a SemanticManifest,
    table: &str,
) -> impl Fn(&str) -> PhysicalExpr + 'a {
    let source = (manifest.sources.values())
        .find(|source| source.name == table)
        .expect("the physical source");
    |name| Expr::Column(NamedReference::qualified(source.alias.as_str(), name).expect("a name"))
}

pub fn text(value: &str) -> PhysicalExpr {
    Expr::Literal(Literal::String(value.into()))
}

pub fn bigint(value: i64) -> PhysicalExpr {
    Expr::Literal(Literal::BigInt(value))
}

pub fn call(name: &str, args: Vec<PhysicalExpr>) -> PhysicalExpr {
    let function = registry().get(name).expect("a catalog function");
    Expr::Function(Call::new(function, args).expect("a call that fits its declaration"))
}

pub fn markers(project: &ProjectNode) -> Vec<(&str, PhysicalExpr)> {
    (fields(project).into_iter())
        .filter(|(_, tree)| matches!(tree, Expr::Function(Call::Window(_))))
        .collect()
}

pub fn once(marker: &str) -> PhysicalExpr {
    column(marker).eq(bigint(1))
}

pub const LATEST: SortDir = SortDir::Desc {
    nulls: NullOrdering::Last,
};

fn window(name: &str, over: Over<Parameter, AggregateAllowed>) -> PhysicalExpr {
    let Some(FunctionRef::Window(function)) = registry().get(name) else {
        panic!("the catalog declares `{name}`");
    };
    let call = WindowCall::builder(function, Vec::new()).over(over).build();
    Expr::Function(Call::Window(call.expect("a window call")))
}

pub fn rank(partition: Vec<PhysicalExpr>, by: PhysicalExpr, dir: SortDir) -> PhysicalExpr {
    let over = Over {
        partition_by: partition,
        order_by: vec![SortKey { expr: by, dir }],
        window_frame: None,
    };
    window("rank", over)
}

pub fn row_number(partition: Vec<PhysicalExpr>) -> PhysicalExpr {
    let over = Over {
        partition_by: partition,
        order_by: Vec::new(),
        window_frame: None,
    };
    window("row_number", over)
}

pub fn every_marker(node: &PlanNode) -> Vec<(&str, PhysicalExpr)> {
    let own = match node {
        PlanNode::Project(project) => markers(project),
        _ => Vec::new(),
    };
    (own.into_iter())
        .chain(node.children().into_iter().flat_map(every_marker))
        .collect()
}

pub const UNBUCKETED_DATE: &str = "__plan_unbucketed_date";

pub fn latest_by_date(
    marking: &ProjectNode,
    (marker, window): &(&str, PhysicalExpr),
    partition: &[&str],
) {
    assert!(marker.starts_with("__plan_"), "{marker}");
    let partition = partition.iter().map(|name| column(name)).collect();
    assert_eq!(window, &rank(partition, column(UNBUCKETED_DATE), LATEST));
    let bucketing = projection(&marking.input);
    assert_eq!(written(bucketing, UNBUCKETED_DATE), Some(column("date")));
}

pub fn balance_marker<'a>(aggregate: &'a AggregateNode, partition: &[&str]) -> &'a str {
    let marking = projection(&aggregate.input);
    let [marker] = &markers(marking)[..] else {
        panic!("one marker, got {marking:?}");
    };
    latest_by_date(marking, marker, partition);
    let (_, balance) = (aggregate.aggregates.iter())
        .find(|(entry, _)| entry.as_str() == "balance")
        .expect("the entry of `balance`");
    assert_eq!(balance.filter(), Some(&once(marker.0)));
    marker.0
}

pub fn summed<'w>(
    row_level: &ProjectNode,
    windows: &'w [(&'w str, PhysicalExpr)],
    sum: &AggregateCall<Parameter, AggregateAllowed>,
) -> (PhysicalExpr, Option<&'w (&'w str, PhysicalExpr)>) {
    assert_eq!(sum.function().declarations.canonical.name, "sum");
    let [argument] = sum.args() else {
        panic!("one argument, got {:?}", sum.args());
    };
    let marker = sum.filter().map(|condition| {
        (windows.iter())
            .find(|(marker, _)| condition == &once(marker))
            .unwrap_or_else(|| panic!("a marker's condition, got {condition:?}"))
    });
    (computed(row_level, argument), marker)
}

pub fn entry<'a>(
    aggregate: &'a AggregateNode,
    name: &str,
) -> &'a AggregateCall<Parameter, AggregateAllowed> {
    (aggregate.aggregates.iter())
        .find_map(|(entry, call)| (entry.as_str() == name).then_some(call))
        .unwrap_or_else(|| panic!("the entry `{name}`, got {:?}", aggregate.aggregates))
}

pub type Formula = fn(PhysicalExpr, PhysicalExpr) -> PhysicalExpr;

pub fn measure_markers<'a>(
    source: impl Fn(&str) -> PhysicalExpr,
    derivation: &'a ProjectNode,
    (metric, formula): (&str, Formula),
    partition: &[&str],
) -> &'a AggregateNode {
    let aggregate = aggregate_step(&derivation.input);
    let marking = projection(&aggregate.input);
    let windows = markers(marking);
    let [first, second] = &windows[..] else {
        panic!("two markers, got {marking:?}");
    };
    assert_ne!(first.0, second.0);
    let row_level = projection(&projection(&marking.input).input);
    let [a, b] = [("a", first), ("b", second)].map(|(measure, marker)| {
        latest_by_date(marking, marker, partition);
        let (entry, sum) = (aggregate.aggregates.iter())
            .find(|(_, call)| call.filter() == Some(&once(marker.0)))
            .unwrap_or_else(|| panic!("the entry that reads `{}`", marker.0));
        assert_eq!(
            summed(row_level, &windows, sum),
            (source(measure), Some(marker))
        );
        column(entry.as_str())
    });
    assert_eq!(written(derivation, metric), Some(formula(a, b)));
    aggregate
}

pub fn written(project: &ProjectNode, name: &str) -> Option<PhysicalExpr> {
    (fields(project).into_iter()).find_map(|(field, tree)| (field == name).then_some(tree))
}

pub fn names(schema: &semstrait_ir::Schema) -> Vec<&str> {
    (schema.columns.iter())
        .map(|field| field.name.as_str())
        .collect()
}

pub fn output_names(plan: &SemanticPlan) -> Vec<&str> {
    plan.output_names.iter().map(Name::as_str).collect()
}

pub fn datasets(plan: &SemanticPlan) -> Vec<&str> {
    (plan.root.pre_order())
        .flat_map(|node| &node.common().semantic_meta)
        .filter_map(|subject| match subject {
            SemAnnotation::SemanticSource {
                semantic_source,
                source_role: SourceRole::Dataset,
                ..
            } => Some(semantic_source.as_str()),
            _ => None,
        })
        .collect()
}

pub fn answered_by_source(plan: &SemanticPlan, child: &str) {
    let answered = DataKindName::new(child);
    let trace = &plan.root.common().semantic_meta;
    assert!(
        (trace.iter()).any(|subject| {
            matches!(subject, SemAnnotation::SemanticSource { answered_by: Some(name), .. } if *name == answered)
        }),
        "{answered:?} in {trace:?}"
    );
}

pub fn aggregate_step(node: &PlanNode) -> &AggregateNode {
    match node {
        PlanNode::Aggregate(aggregate) => aggregate,
        other => panic!("an aggregate step, got {other:?}"),
    }
}

pub fn aggregate_steps(node: &PlanNode) -> usize {
    (node.pre_order())
        .filter(|node| matches!(node, PlanNode::Aggregate(_)))
        .count()
}

pub fn aggregate_filters(node: &PlanNode) -> Vec<(&str, Option<&PhysicalExpr>)> {
    (node.pre_order())
        .filter_map(|node| match node {
            PlanNode::Aggregate(aggregate) => Some(&aggregate.aggregates),
            _ => None,
        })
        .flatten()
        .map(|(entry, call)| (entry.as_str(), call.filter()))
        .collect()
}

pub fn first_aggregate(node: &PlanNode) -> &AggregateNode {
    (node.pre_order())
        .find_map(|node| match node {
            PlanNode::Aggregate(aggregate) => Some(aggregate),
            _ => None,
        })
        .unwrap_or_else(|| panic!("an aggregate step in {node:?}"))
}

pub fn union_of(plan: &SemanticPlan) -> (&PlanNode, &MultisetNode) {
    (first_inputs(&plan.root))
        .find_map(|above| match above.children().first().copied() {
            Some(PlanNode::Multiset(union)) => Some((above, union)),
            _ => None,
        })
        .unwrap_or_else(|| panic!("a union below the root of {plan:?}"))
}

pub fn projection(node: &PlanNode) -> &ProjectNode {
    match node {
        PlanNode::Project(project) => project,
        other => panic!("a projection, got {other:?}"),
    }
}

pub fn filter(node: &PlanNode) -> &FilterNode {
    match node {
        PlanNode::Filter(filter) => filter,
        other => panic!("a filter, got {other:?}"),
    }
}

pub fn join(node: &PlanNode) -> &JoinNode {
    match node {
        PlanNode::Join(join) => join,
        other => panic!("a join, got {other:?}"),
    }
}

pub fn cross(node: &PlanNode) -> &CrossNode {
    match node {
        PlanNode::Cross(cross) => cross,
        other => panic!("a cross, got {other:?}"),
    }
}

pub fn below_aggregate(plan: &SemanticPlan) -> &ProjectNode {
    projection(&aggregate_step(&plan.root).input)
}

pub fn fields(project: &ProjectNode) -> Vec<(&str, PhysicalExpr)> {
    let names = names(&project.common.output_schema);
    names.into_iter().zip(project.expressions.clone()).collect()
}

pub fn computed(row_level: &ProjectNode, leaf: &PhysicalExpr) -> PhysicalExpr {
    let Expr::Column(field) = leaf else {
        panic!("a field of stage 2, got {leaf:?}");
    };
    written(row_level, &field_name(field)).unwrap_or_else(|| panic!("{field:?} in {row_level:?}"))
}

pub fn parts(node: &PlanNode) -> (&AggregateNode, PhysicalExpr, &AggregateNode) {
    two_steps(node, "sum")
}

pub fn two_steps<'p>(
    node: &'p PlanNode,
    function: &str,
) -> (&'p AggregateNode, PhysicalExpr, &'p AggregateNode) {
    let step_2 = aggregate_step(node);
    let [(_, call)] = step_2.aggregates.as_slice() else {
        panic!("one entry, got {:?}", step_2.aggregates);
    };
    assert_eq!(call.function().declarations.canonical.name, function);
    let formula = projection(&step_2.input);
    let per_group = computed(formula, &call.args()[0]);
    (step_2, per_group, aggregate_step(&formula.input))
}

pub fn filtered(plan: &SemanticPlan) -> (&FilterNode, &ProjectNode) {
    let filter = filter(&aggregate_step(&plan.root).input);
    (filter, projection(&filter.input))
}

pub fn stored(manifest: &SemanticManifest, dataset: &str, semantics: &str) -> PhysicalExpr {
    let signatures: Vec<_> = (manifest.signatures_of(semantic_source(manifest, dataset))).collect();
    let root = signature(manifest, dataset, semantics).expression_id;
    (manifest.expressions)
        .reassemble(&root, &signatures, &[])
        .expect("its tree")
}

pub fn lossy(advisories: &Diagnostics<Advisory>) -> Vec<(String, &str)> {
    (advisories.iter())
        .map(|advisory| match &advisory.kind {
            Advisory::LossyReaggregation { path, semantics } => {
                (path.to_string(), semantics.as_str())
            }
            other => panic!("a LossyReaggregation, got {other:?}"),
        })
        .collect()
}

pub fn advisory_kinds(advisories: Diagnostics<Advisory>) -> Vec<Advisory> {
    (advisories.into_iter())
        .map(|advisory| advisory.kind)
        .collect()
}

fn advice_rows<N: ToString>(
    advisories: Diagnostics<Advisory>,
    row: impl Fn(Advisory) -> Option<(FramePath, DataKindName, Vec<N>)>,
) -> Vec<(String, String, Vec<String>)> {
    (advisory_kinds(advisories).into_iter())
        .filter_map(row)
        .map(|(path, name, others)| {
            (
                path.to_string(),
                name.to_string(),
                others.iter().map(ToString::to_string).collect(),
            )
        })
        .collect()
}

pub fn ties(advisories: Diagnostics<Advisory>) -> Vec<(String, String, Vec<String>)> {
    advice_rows(advisories, |advisory| match advisory {
        Advisory::AmbiguousCandidates { path, chosen, tied } => Some((path, chosen, tied)),
        other => panic!("an AmbiguousCandidates, got {other:?}"),
    })
}

pub fn not_servable_branches(
    advisories: Diagnostics<Advisory>,
) -> Vec<(String, String, Vec<String>)> {
    advice_rows(advisories, |advisory| match advisory {
        Advisory::BranchNotServable {
            path,
            branch,
            lacks,
        } => Some((path, branch, lacks)),
        _ => None,
    })
}
