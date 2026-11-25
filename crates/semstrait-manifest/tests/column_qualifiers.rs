use semstrait_catalog::FileFormat;
use semstrait_common::Granularity;
use semstrait_manifest::{
    compile, mint_path_source_id, CompileError, PhysicalExprId, RoleDefinition, SemanticManifest,
    SourceAlias, StoredExpr,
};
use semstrait_model::expr::{dim, lit, measure as measure_leaf};
use semstrait_model::{
    DatasetBuilder, Dimension, DimensionSite, Joinset, Measure, MeasureSite, Metric, MetricSite,
    NestedDataset, NestedGrainset, Relationship, SemanticMapping, SemanticModel,
    SemanticModelBuilder, StorageFormat, TemporalShape, Unionset, UnresolvedSemanticExpr,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{BinaryOpKind, Expr, ExprOperationExt, NamedReference};

mod common;
use common::{
    agg, by_name, cast, column, compile_error, constituent, dated, dimension, fixture_fs,
    identity_mapping, lake_folder, measure, orders_model, semantic_alias, signature, stored_tree,
    test_provenance,
};

const PROBE: [&str; 3] = ["eu-orders", "eu_orders", "eu_orders_fb2f09"];

const SALES_NAMES: [&str; 3] = ["customer_id", "region", "revenue"];

fn related(name: &str, from: &str, to: &str) -> Relationship {
    Relationship::builder()
        .name(name)
        .from(from)
        .to(to)
        .on(["customer_id"])
        .many_to_one()
        .build()
        .expect("the fixture relationship must build")
}

fn eu_only() -> UnresolvedSemanticExpr {
    ExprOperationExt::eq(dim("region").expect("a name"), lit("eu"))
}

fn categorical(name: &str) -> Dimension {
    Dimension::builder(name)
        .data_type(DataType::String)
        .categorical()
        .build()
}

fn vocabulary(measure_filter: Option<UnresolvedSemanticExpr>) -> SemanticModelBuilder {
    let revenue = Measure::builder("revenue")
        .data_type(DataType::Double)
        .agg(agg("sum"));
    let revenue = match measure_filter {
        Some(predicate) => revenue.filter(predicate).build(),
        None => revenue.build(),
    };
    SemanticModel::builder()
        .name("sales")
        .dimension(categorical("customer_id"))
        .measure(revenue)
}

async fn compiled(model: SemanticModelBuilder, names: &[&str]) -> SemanticManifest {
    let fs = fixture_fs(names).await;
    let model = model.build().expect("the fixture model must build").0;
    let (manifest, warnings) = compile(model, None, &fs, test_provenance())
        .await
        .expect("the fixture compiles");
    assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
    manifest
}

async fn probe() -> SemanticManifest {
    let mut sales = Joinset::builder("sales")
        .dimension(DimensionSite::r#ref("customer_id"))
        .dimension(DimensionSite::r#ref("region"))
        .measure(MeasureSite::r#ref("revenue"))
        .relationship(related("r1", "eu-orders", "eu_orders_fb2f09"))
        .relationship(related("r2", "eu_orders", "eu-orders"));
    for name in PROBE {
        sales = sales.dataset(constituent(name, &SALES_NAMES));
    }
    compiled(
        vocabulary(Some(eu_only()))
            .dimension(categorical("region"))
            .joinset(sales.build()),
        &PROBE,
    )
    .await
}

async fn filtered() -> SemanticManifest {
    let relationship = Relationship::builder()
        .name("orders_to_customers")
        .from("orders")
        .to("customers")
        .on(["customer_id"])
        .many_to_one()
        .filter(eu_only())
        .build()
        .expect("the fixture relationship must build");
    let sales = Joinset::builder("sales")
        .dataset(constituent("orders", &SALES_NAMES))
        .dataset(constituent("customers", &SALES_NAMES))
        .dimension(DimensionSite::r#ref("customer_id"))
        .dimension(DimensionSite::r#ref("region"))
        .measure(MeasureSite::r#ref("revenue"))
        .relationship(relationship);
    let stacked = Unionset::builder("stacked")
        .dataset(constituent("east", &SALES_NAMES))
        .dataset(constituent("west", &SALES_NAMES))
        .dimension(DimensionSite::r#ref("customer_id"))
        .dimension(DimensionSite::r#ref("region"))
        .measure(MeasureSite::r#ref("revenue"));
    compiled(
        vocabulary(Some(eu_only()))
            .dimension(categorical("region"))
            .joinset(sales.build())
            .unionset(stacked.union_all().build()),
        &["orders", "customers", "east", "west"],
    )
    .await
}

async fn implicit_unionset() -> SemanticManifest {
    let orders = NestedDataset::builder("orders")
        .tables(["warehouse.orders_eu", "warehouse.orders_us"])
        .semantic_mapping(identity_mapping(&SALES_NAMES))
        .build();
    let sales = Joinset::builder("sales")
        .dataset(orders)
        .dataset(constituent("customers", &SALES_NAMES))
        .dimension(DimensionSite::r#ref("customer_id"))
        .dimension(DimensionSite::r#ref("region"))
        .measure(MeasureSite::r#ref("revenue"))
        .relationship(related("orders_to_customers", "orders", "customers"));
    compiled(
        vocabulary(None)
            .dimension(categorical("region"))
            .joinset(sales.build()),
        &["customers"],
    )
    .await
}

async fn grainset_constituent() -> SemanticManifest {
    let daily = ["customer_id", "date", "revenue", "quantity"];
    let monthly = ["customer_id", "date", "revenue", "target"];
    let orders = NestedGrainset::builder("orders")
        .temporal(TemporalShape::events("date", Some(Granularity::Month)))
        .dataset(dated("orders_daily", &daily, Granularity::Day))
        .dataset(dated("orders_monthly", &monthly, Granularity::Month))
        .build();
    let sales = Joinset::builder("sales")
        .grainset(orders)
        .dataset(constituent("customers", &SALES_NAMES))
        .dimension(DimensionSite::r#ref("customer_id").with_key())
        .dimension(DimensionSite::r#ref("region"))
        .dimension(DimensionSite::r#ref("date"))
        .measure(MeasureSite::r#ref("revenue"))
        .measure(MeasureSite::r#ref("quantity"))
        .measure(MeasureSite::r#ref("target"))
        .metric(MetricSite::r#ref("quantity_per_target"))
        .relationship(related("orders_to_customers", "orders", "customers"));
    let quantity_per_target = Metric::builder("quantity_per_target")
        .data_type(DataType::Double)
        .expr(measure_leaf("quantity").expect("a name") / measure_leaf("target").expect("a name"))
        .filter(ExprOperationExt::eq(
            dim("customer_id").expect("a name"),
            lit("c1"),
        ))
        .build();
    compiled(
        vocabulary(None)
            .dimension(categorical("region"))
            .dimension(dimension("date", DataType::Date))
            .measure(measure("quantity", DataType::Double, "sum"))
            .measure(measure("target", DataType::Double, "sum"))
            .metric(quantity_per_target)
            .joinset(sales.build()),
        &["orders_daily", "orders_monthly", "customers"],
    )
    .await
}

fn sales_constituents(manifest: &SemanticManifest) -> Vec<&str> {
    let mut aliases = ["sales.orders", "sales.customers"]
        .map(|name| semantic_alias(manifest, name))
        .to_vec();
    aliases.sort_unstable();
    aliases
}

fn join_roots(manifest: &SemanticManifest, name: &str) -> (PhysicalExprId, Option<PhysicalExprId>) {
    match &by_name(manifest, name).role_definition {
        RoleDefinition::Joinset {
            on_expr_id,
            filter_expr_id,
            ..
        } => (
            on_expr_id.expect("a keyed unit has a condition"),
            *filter_expr_id,
        ),
        other => panic!("`{name}` must be a `Joinset`, got {other:?}"),
    }
}

#[tokio::test]
async fn a_datasets_tree_names_its_physical_source_and_a_join_condition_its_constituents() {
    let manifest = probe().await;
    let read = mint_path_source_id(&lake_folder("eu-orders"), Some(FileFormat::Parquet));
    assert_eq!(manifest.sources[&read].alias.as_str(), "ps1");

    assert_eq!(
        stored_tree(
            &manifest,
            signature(&manifest, "sales.eu-orders", "customer_id").expression_id
        ),
        cast(column("ps1", "customer_id"), DataType::String),
        "a Dataset's tree: `Column([\"ps1\", …])`, cast to its Dimension's declared type \
         (ADR 0089, rule 16)"
    );
    assert_eq!(
        stored_tree(&manifest, join_roots(&manifest, "sales.r1").0),
        Expr::BinaryOp {
            op: BinaryOpKind::Eq,
            left: Box::new(column("ssd4", "customer_id")),
            right: Box::new(column("ssd0", "customer_id")),
        },
        "a join condition: `Column([\"ssd4\", \"customer_id\"])`, each constituent's \
         semantics (ADR 0075)"
    );
    assert!(manifest.validate().is_ok());
}

#[tokio::test]
async fn a_join_condition_names_an_implicit_unionset_by_its_own_alias() {
    let manifest = implicit_unionset().await;
    assert!(
        matches!(
            by_name(&manifest, "sales.orders").role_definition,
            RoleDefinition::Unionset { .. }
        ),
        "a Dataset of two tables reduces to a Unionset (ADR 0049)"
    );
    assert_eq!(
        stored_tree(&manifest, join_roots(&manifest, "sales").0),
        Expr::BinaryOp {
            op: BinaryOpKind::Eq,
            left: Box::new(column(
                semantic_alias(&manifest, "sales.orders"),
                "customer_id"
            )),
            right: Box::new(column(
                semantic_alias(&manifest, "sales.customers"),
                "customer_id"
            )),
        },
        "each side is `Column(<constituent alias>.customer_id)`, the semantic form"
    );
    let warnings = manifest.validate().expect("the compiled artifact is valid");
    assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
}

#[tokio::test]
async fn validate_refuses_a_condition_leaf_its_implicit_unionset_does_not_serve() {
    let mut manifest = implicit_unionset().await;
    let condition = join_roots(&manifest, "sales").0;
    let orders = by_name(&manifest, "sales.orders").id.clone();
    let leaf = NamedReference::qualified(semantic_alias(&manifest, "sales.orders"), "bogus")
        .expect("an alias and a name");
    let other =
        NamedReference::qualified(semantic_alias(&manifest, "sales.customers"), "customer_id")
            .expect("an alias and a name");
    let entry =
        (manifest.expressions.physical.get_mut(&condition)).expect("the condition is stored");
    entry.expr = StoredExpr::from(Expr::BinaryOp {
        op: BinaryOpKind::Eq,
        left: Box::new(Expr::Column(leaf.clone())),
        right: Box::new(Expr::Column(other)),
    });
    let faults: Vec<CompileError> = (manifest.validate().expect_err("`bogus` is not served"))
        .into_iter()
        .map(|fault| fault.kind)
        .collect();
    assert_eq!(
        faults,
        vec![CompileError::UnservedConditionLeaf {
            expression: condition,
            column: leaf,
            semantic_source: orders,
        }]
    );
}

fn revenue_filter(manifest: &SemanticManifest, source: &str) -> PhysicalExprId {
    signature(manifest, source, "revenue")
        .filter_expr_id
        .expect("`revenue` carries a filter")
}

#[tokio::test]
async fn a_joinsets_composed_filter_names_only_its_operands() {
    let manifest = probe().await;
    let columns = manifest
        .expressions
        .columns(&revenue_filter(&manifest, "sales"));
    assert!(!columns.is_empty(), "the filter reads `region`");
    for (_, column) in columns {
        assert!(
            matches!(column.parts(), [qualifier, _] if ["ssd0", "ssj3"].contains(&qualifier.as_str())),
            "{column:?} names a source that is not an operand of `sales`"
        );
    }
    for operand in ["sales.r2", "sales.eu_orders_fb2f09"] {
        assert!(
            signature(&manifest, operand, "revenue")
                .filter_expr_id
                .is_some(),
            "`{operand}` carries its own filter on its own signature"
        );
    }
}

#[tokio::test]
async fn validate_refuses_a_column_that_is_not_two_parts_under_the_alias_its_place_names() {
    let (probe, filtered, implicit) = (probe().await, filtered().await, implicit_unionset().await);
    let grainset = grainset_constituent().await;
    for manifest in [&probe, &filtered, &implicit, &grainset] {
        assert!(manifest.validate().is_ok(), "the unedited manifest passes");
    }

    let customer_id = |manifest, source| signature(manifest, source, "customer_id").expression_id;
    let dataset_tree = customer_id(&probe, "sales.eu-orders");
    let condition = join_roots(&probe, "sales").0;
    let projection = customer_id(&probe, "sales");
    let probe_filter = revenue_filter(&probe, "sales");
    let relationship_filter = join_roots(&filtered, "sales")
        .1
        .expect("the relationship's `filter:` is stored");
    let composed_filter = revenue_filter(&filtered, "sales");
    let stack_tree = customer_id(&filtered, "stacked");
    let stack_filter = revenue_filter(&filtered, "stacked");

    let named = |name| semantic_alias(&filtered, name);
    let constituents = sales_constituents(&filtered);
    let implicit_condition = join_roots(&implicit, "sales").0;
    let RoleDefinition::Unionset { branches, .. } =
        &by_name(&implicit, "sales.orders").role_definition
    else {
        panic!("`sales.orders` reduces to a Unionset");
    };
    let reduction_leaf = implicit.semantic_sources[&branches[0]].alias.as_str();
    let grainset_condition = join_roots(&grainset, "sales").0;
    let computed_filter = signature(&grainset, "sales.orders", "quantity_per_target")
        .filter_expr_id
        .expect("the Metric's `filter:` is stored");
    let own = semantic_alias(&grainset, "sales.orders");
    for (manifest, entry_id, column, expected, case) in [
        (
            &probe,
            dataset_tree,
            NamedReference::unqualified("customer_id").expect("a name"),
            vec!["ps1"],
            "a one-part column in a Dataset's tree",
        ),
        (
            &probe,
            dataset_tree,
            NamedReference::qualified("ps2", "customer_id").expect("an alias and a name"),
            vec!["ps1"],
            "another physical source's alias in a Dataset's tree",
        ),
        (
            &probe,
            condition,
            NamedReference::qualified("ssj3", "customer_id").expect("an alias and a name"),
            vec!["ssd0", "ssd1", "ssd4"],
            "an operand that is a composition, in a join condition",
        ),
        (
            &probe,
            condition,
            NamedReference::new(["ssd4", "eu", "customer_id"]).expect("three parts"),
            vec!["ssd0", "ssd1", "ssd4"],
            "a column of three parts, in a join condition",
        ),
        (
            &implicit,
            implicit_condition,
            NamedReference::qualified(reduction_leaf, "customer_id").expect("an alias and a name"),
            sales_constituents(&implicit),
            "a reduction leaf inside a Unionset constituent, in a join condition",
        ),
        (
            &grainset,
            grainset_condition,
            NamedReference::qualified(
                semantic_alias(&grainset, "sales.orders.orders_daily"),
                "customer_id",
            )
            .expect("an alias and a name"),
            sales_constituents(&grainset),
            "a child inside a Grainset constituent, in a join condition",
        ),
        (
            &probe,
            projection,
            NamedReference::qualified("ssd4", "customer_id").expect("an alias and a name"),
            vec!["ssd0", "ssj3"],
            "a constituent below an operand, in a Joinset's projection",
        ),
        (
            &probe,
            probe_filter,
            NamedReference::qualified("ssd4", "region").expect("an alias and a name"),
            vec!["ssd0", "ssj3"],
            "a constituent below an operand, in a Joinset's composed filter",
        ),
        (
            &filtered,
            relationship_filter,
            NamedReference::qualified(named("stacked.east"), "region")
                .expect("an alias and a name"),
            constituents.clone(),
            "a Dataset that is not a constituent, in a relationship filter",
        ),
        (
            &filtered,
            composed_filter,
            NamedReference::qualified(named("stacked.east"), "region")
                .expect("an alias and a name"),
            constituents,
            "a source that is not an operand, in a Joinset's composed filter",
        ),
        (
            &filtered,
            stack_tree,
            NamedReference::qualified(named("stacked.east"), "customer_id")
                .expect("an alias and a name"),
            vec![named("stacked")],
            "a branch, in a Unionset's projection",
        ),
        (
            &filtered,
            stack_filter,
            NamedReference::qualified(named("stacked.west"), "region")
                .expect("an alias and a name"),
            vec![named("stacked")],
            "a branch, in a Unionset's filter",
        ),
        (
            &grainset,
            computed_filter,
            NamedReference::qualified(
                semantic_alias(&grainset, "sales.orders.orders_daily"),
                "customer_id",
            )
            .expect("an alias and a name"),
            vec![own],
            "a child, in the filter of a Metric that a Grainset computes",
        ),
        (
            &grainset,
            computed_filter,
            NamedReference::unqualified("customer_id").expect("a name"),
            vec![own],
            "a one-part column in the filter of a Metric that a Grainset computes",
        ),
    ] {
        let mut decoded = manifest.clone();
        if let Some(entry) = decoded.expressions.physical.get_mut(&entry_id) {
            entry.expr = StoredExpr::from(Expr::Column(column.clone()));
        }
        let faults: Vec<CompileError> = decoded
            .validate()
            .expect_err(case)
            .into_iter()
            .map(|fault| fault.kind)
            .collect();
        assert_eq!(
            faults,
            vec![CompileError::ColumnQualifierMismatch {
                expression: entry_id,
                column,
                expected: expected.into_iter().map(SourceAlias::from).collect(),
            }],
            "{case}"
        );
    }
}

#[tokio::test]
async fn compile_refuses_a_mapped_column_with_an_empty_name() {
    let explicit = |orders: DatasetBuilder| {
        orders.semantic_mapping(SemanticMapping::builder().column("plain", "").build())
    };
    for (model, semantics) in [
        (
            orders_model(
                vec![dimension("plain", DataType::String)],
                Vec::new(),
                explicit,
            ),
            "plain",
        ),
        (
            orders_model(
                vec![dimension("", DataType::String)],
                Vec::new(),
                |orders| orders,
            ),
            "",
        ),
    ] {
        assert_eq!(
            compile_error(model).await,
            CompileError::EmptyColumnPart {
                node: format!("semantic_mapping {semantics}"),
                parts: vec![String::new()],
            },
            "{semantics:?}"
        );
    }
}

#[tokio::test]
async fn compile_refuses_a_composed_column_with_an_empty_name() {
    let mapped = |name: &str| {
        NestedDataset::builder(name)
            .path(lake_folder(name))
            .format(StorageFormat::Parquet)
            .semantic_mapping(
                SemanticMapping::builder()
                    .column("customer_id", "customer_id")
                    .column("", "region")
                    .build(),
            )
            .build()
    };
    let sales = Joinset::builder("sales")
        .dataset(mapped("orders"))
        .dataset(mapped("customers"))
        .dimension(DimensionSite::r#ref("customer_id"))
        .dimension(DimensionSite::r#ref(""))
        .relationship(related("orders_to_customers", "orders", "customers"));
    let stacked = Unionset::builder("stacked")
        .dataset(mapped("east"))
        .dataset(mapped("west"))
        .dimension(DimensionSite::r#ref("customer_id"))
        .dimension(DimensionSite::r#ref(""));
    let dimensions = || {
        SemanticModel::builder()
            .name("sales")
            .dimension(categorical("customer_id"))
            .dimension(categorical(""))
    };
    for (model, files, node, alias) in [
        (
            dimensions().joinset(sales.build()),
            ["orders", "customers"],
            "sales",
            "ssd1",
        ),
        (
            dimensions().unionset(stacked.union_all().build()),
            ["east", "west"],
            "stacked",
            "ssu1",
        ),
    ] {
        let model = model.build().expect("the fixture model must build").0;
        let fs = fixture_fs(&files).await;
        let (diagnostic, _) = compile(model, None, &fs, test_provenance())
            .await
            .expect_err(node);
        assert_eq!(
            diagnostic.kind,
            CompileError::EmptyColumnPart {
                node: node.to_owned(),
                parts: vec![alias.to_owned(), String::new()],
            },
            "{node}"
        );
    }
}

#[test]
fn a_column_with_an_empty_part_is_named_with_its_parts() {
    let fault = CompileError::EmptyColumnPart {
        node: "semantic_mapping plain".to_owned(),
        parts: vec![String::new()],
    };
    assert_eq!(
        fault.to_string(),
        "the column [\"\"] at `semantic_mapping plain` holds an empty part, which names \
         nothing"
    );
}

#[test]
fn a_misplaced_column_is_named_with_the_aliases_its_place_admits() {
    let fault = CompileError::ColumnQualifierMismatch {
        expression: PhysicalExprId::new(uuid::Uuid::from_u128(7)),
        column: NamedReference::unqualified("amount").expect("a name"),
        expected: vec![SourceAlias::from("ssd0"), SourceAlias::from("ssj3")],
    };
    assert_eq!(
        fault.to_string(),
        "the column [\"amount\"] at expression 00000000-0000-0000-0000-000000000007 is not \
         two parts under an alias its place names: `ssd0`, `ssj3`"
    );
}
