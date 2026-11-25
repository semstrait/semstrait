use std::collections::BTreeSet;

use semstrait_ir::PhysicalExpr;
use semstrait_ir::SemanticRole;
use semstrait_manifest::{
    compile, CompileError, ExprLayer, PhysicalExprId, RoleDefinition, SemanticManifest,
    SemanticSignature, SemanticSource,
};
use semstrait_model::expr::{col, dim, lit, measure, metric};
use semstrait_model::{
    DataKindFilter, Dataset, Dimension, DimensionSite, Joinset, JoinsetBuilder, Measure,
    MeasureRef, MeasureSite, Metric, MetricRef, MetricSite, NestedDataset, Optional, Relationship,
    SemanticMapping, SemanticModel, SemanticSite, StorageFormat, Unionset, UnionsetBuilder,
    UnresolvedSemanticExpr,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{BinaryOpKind, Expr, ExprOperationExt, Leaf, Literal};

trait RefSiteFilter {
    fn set_filter(&mut self, predicate: UnresolvedSemanticExpr);
}

impl RefSiteFilter for MeasureRef {
    fn set_filter(&mut self, predicate: UnresolvedSemanticExpr) {
        self.filter = Some(predicate);
    }
}

impl RefSiteFilter for MetricRef {
    fn set_filter(&mut self, predicate: UnresolvedSemanticExpr) {
        self.filter = Some(predicate);
    }
}

fn set_site_filter<I: std::fmt::Debug, R: RefSiteFilter + std::fmt::Debug>(
    site: &mut SemanticSite<I, R>,
    predicate: UnresolvedSemanticExpr,
    label: &str,
) {
    match site {
        SemanticSite::Reference(reference) => reference.set_filter(predicate),
        other => panic!("the fixture exposes `{label}` at a ref site, got {other:?}"),
    }
}

mod common;
use common::{
    agg, aggregate, by_name, cast, catalog, column, compile_error, compile_error_with,
    compile_model_with, constituent, decimal_18_2, fixture_fs, lake_folder, link, link_free_tree,
    listed, physical_call, source_alias, stored_expr, stored_tree, string, test_provenance,
    warehouse_catalog, MapCatalog,
};

fn mapped_constituent(name: &str) -> NestedDataset {
    NestedDataset::builder(name)
        .path(lake_folder(name))
        .format(StorageFormat::Parquet)
        .semantic_mapping(
            SemanticMapping::builder()
                .column("customer_id", "customer_id")
                .column("region", "region_code")
                .column("revenue", "revenue")
                .build(),
        )
        .build()
}

fn model_builder() -> semstrait_model::SemanticModelBuilder {
    pooled_vocabulary(None)
}

fn pooled_vocabulary(
    measure_filter: Option<UnresolvedSemanticExpr>,
) -> semstrait_model::SemanticModelBuilder {
    pooled_vocabulary_with(measure_filter, None)
}

fn pooled_vocabulary_with(
    measure_filter: Option<UnresolvedSemanticExpr>,
    body: Option<UnresolvedSemanticExpr>,
) -> semstrait_model::SemanticModelBuilder {
    let revenue = Measure::builder("revenue")
        .data_type(DataType::Double)
        .agg(agg("sum"));
    let revenue = match (body, measure_filter) {
        (Some(body), Some(predicate)) => revenue.expr(body).filter(predicate).build(),
        (Some(body), None) => revenue.expr(body).build(),
        (None, Some(predicate)) => revenue.filter(predicate).build(),
        (None, None) => revenue.build(),
    };
    SemanticModel::builder()
        .name("sales")
        .dimension(
            Dimension::builder("customer_id")
                .data_type(DataType::String)
                .categorical()
                .build(),
        )
        .dimension(
            Dimension::builder("region")
                .data_type(DataType::String)
                .categorical()
                .build(),
        )
        .measure(revenue)
}

fn sales_joinset(optional: Optional) -> JoinsetBuilder {
    sales_joinset_over(
        optional,
        constituent("orders", &SALES_NAMES),
        constituent("customers", &SALES_NAMES),
    )
}

const SALES_NAMES: [&str; 3] = ["customer_id", "region", "revenue"];

fn sales_joinset_over(
    optional: Optional,
    from: NestedDataset,
    to: NestedDataset,
) -> JoinsetBuilder {
    let relationship = Relationship::builder()
        .name("orders_to_customers")
        .from("orders")
        .to("customers")
        .on(["customer_id"])
        .many_to_one()
        .optional(optional)
        .build()
        .expect("the fixture relationship must build");
    Joinset::builder("sales")
        .dataset(from)
        .dataset(to)
        .dimension(DimensionSite::r#ref("customer_id"))
        .dimension(DimensionSite::r#ref("region"))
        .measure(MeasureSite::r#ref("revenue"))
        .relationship(relationship)
}

fn joined_model(optional: Optional) -> SemanticModel {
    model_builder()
        .joinset(sales_joinset(optional).build())
        .build()
        .expect("the fixture model must build")
        .0
}

fn stacked_unionset_over(east: NestedDataset, west: NestedDataset) -> UnionsetBuilder {
    Unionset::builder("stacked")
        .dataset(east)
        .dataset(west)
        .dimension(DimensionSite::r#ref("customer_id"))
        .dimension(DimensionSite::r#ref("region"))
        .measure(MeasureSite::r#ref("revenue"))
}

fn stacked_unionset() -> UnionsetBuilder {
    stacked_unionset_over(
        constituent("east", &SALES_NAMES),
        constituent("west", &SALES_NAMES),
    )
}

fn stacked_model() -> SemanticModel {
    model_builder()
        .unionset(stacked_unionset().union_all().build())
        .build()
        .expect("the fixture model must build")
        .0
}

fn doubled() -> Metric {
    Metric::builder("doubled")
        .data_type(DataType::Double)
        .expr(measure("revenue").expect("a name") + measure("revenue").expect("a name"))
        .build()
}

fn joined_filtered_measure_inlined_in_metric_model() -> SemanticModel {
    pooled_vocabulary(Some(ExprOperationExt::eq(
        dim("region").expect("a name"),
        lit("eu"),
    )))
    .metric(doubled())
    .joinset(
        sales_joinset(Optional::None)
            .metric(MetricSite::r#ref("doubled"))
            .build(),
    )
    .build()
    .expect("the fixture model must build")
    .0
}

fn bodied_pooled_vocabulary(
    measure_filter: Option<UnresolvedSemanticExpr>,
) -> semstrait_model::SemanticModelBuilder {
    pooled_vocabulary_with(measure_filter, col("amount"))
}

fn joined_cast_wrapped_filtered_measure_inlined_in_metric_model() -> SemanticModel {
    bodied_pooled_vocabulary(Some(ExprOperationExt::eq(
        dim("region").expect("a name"),
        lit("eu"),
    )))
    .metric(doubled())
    .joinset(
        column_filtered_joinset(Optional::None)
            .metric(MetricSite::r#ref("doubled"))
            .build(),
    )
    .build()
    .expect("the fixture model must build")
    .0
}

fn bodied_revenue_catalog() -> MapCatalog {
    catalog(&[
        ("customer_id", "string"),
        ("region", "string"),
        ("amount", "decimal(18,2)"),
    ])
}

fn eu_only() -> DataKindFilter {
    DataKindFilter::builder("eu_only")
        .expr(ExprOperationExt::eq(
            dim("region").expect("a name"),
            lit("eu"),
        ))
        .build()
}

fn is_eu_metric() -> Metric {
    Metric::builder("is_eu")
        .data_type(DataType::Boolean)
        .expr(ExprOperationExt::eq(
            dim("region").expect("a name"),
            lit("eu"),
        ))
        .build()
}

fn joined_measure_filter_referencing_metric_model() -> SemanticModel {
    let mut model = model_builder()
        .metric(is_eu_metric())
        .joinset(
            sales_joinset(Optional::None)
                .metric(MetricSite::r#ref("is_eu"))
                .build(),
        )
        .build()
        .expect("the fixture model must build")
        .0;
    let joinset = model
        .joinsets
        .values_mut()
        .next()
        .expect("the fixture holds one joinset");
    let revenue = joinset
        .semantic_interface
        .measures
        .values_mut()
        .next()
        .expect("the joinset exposes one measure");
    set_site_filter(revenue, metric("is_eu").expect("a name"), "revenue");
    model
}

fn joined_bodied_model() -> SemanticModel {
    model_builder()
        .metric(doubled())
        .joinset(
            sales_joinset(Optional::None)
                .metric(MetricSite::r#ref("doubled"))
                .filter(eu_only())
                .build(),
        )
        .build()
        .expect("the fixture model must build")
        .0
}

fn stacked_bodied_model() -> SemanticModel {
    model_builder()
        .metric(doubled())
        .unionset(
            stacked_unionset()
                .metric(MetricSite::r#ref("doubled"))
                .filter(eu_only())
                .union_all()
                .build(),
        )
        .build()
        .expect("the fixture model must build")
        .0
}

fn joined_filtered_model() -> SemanticModel {
    pooled_vocabulary(Some(ExprOperationExt::eq(
        dim("region").expect("a name"),
        lit("eu"),
    )))
    .joinset(sales_joinset(Optional::None).build())
    .build()
    .expect("the fixture model must build")
    .0
}

fn stacked_filtered_model() -> SemanticModel {
    pooled_vocabulary(Some(ExprOperationExt::eq(
        dim("region").expect("a name"),
        lit("eu"),
    )))
    .unionset(stacked_unionset().union_all().build())
    .build()
    .expect("the fixture model must build")
    .0
}

fn column_filter_catalog() -> MapCatalog {
    catalog(&[
        ("customer_id", "string"),
        ("region", "string"),
        ("revenue", "double"),
        ("is_active", "boolean"),
    ])
}

fn table_constituent(name: &str) -> NestedDataset {
    NestedDataset::builder(name)
        .table("warehouse.orders")
        .build()
}

fn column_filtered_joinset(optional: Optional) -> JoinsetBuilder {
    sales_joinset_over(
        optional,
        table_constituent("orders"),
        table_constituent("customers"),
    )
}

fn column_filtered_unionset() -> UnionsetBuilder {
    stacked_unionset_over(table_constituent("east"), table_constituent("west"))
}

fn joined_column_filtered_model() -> SemanticModel {
    pooled_vocabulary(Some(ExprOperationExt::eq(
        col("is_active").expect("a name"),
        lit(true),
    )))
    .joinset(column_filtered_joinset(Optional::None).build())
    .build()
    .expect("the fixture model must build")
    .0
}

fn stacked_column_filtered_model() -> SemanticModel {
    pooled_vocabulary(Some(ExprOperationExt::eq(
        col("is_active").expect("a name"),
        lit(true),
    )))
    .unionset(column_filtered_unionset().union_all().build())
    .build()
    .expect("the fixture model must build")
    .0
}

fn joined_filtered_model_with(
    optional: Optional,
    pooled: Option<UnresolvedSemanticExpr>,
    site: Option<UnresolvedSemanticExpr>,
) -> SemanticModel {
    let model = pooled_vocabulary(pooled)
        .joinset(sales_joinset(optional).build())
        .build()
        .expect("the fixture model must build")
        .0;
    match site {
        Some(predicate) => with_site_filter(model, predicate),
        None => model,
    }
}

fn with_site_filter(mut model: SemanticModel, predicate: UnresolvedSemanticExpr) -> SemanticModel {
    let joinset = (model.joinsets.values_mut())
        .next()
        .expect("the fixture holds one joinset");
    let measure = (joinset.semantic_interface.measures.values_mut())
        .next()
        .expect("the joinset exposes one measure");
    set_site_filter(measure, predicate, "revenue");
    model
}

fn joined_mapped_model(pooled: UnresolvedSemanticExpr) -> SemanticModel {
    pooled_vocabulary(Some(pooled))
        .joinset(
            sales_joinset_over(
                Optional::None,
                mapped_constituent("orders"),
                mapped_constituent("customers"),
            )
            .build(),
        )
        .build()
        .expect("the fixture model must build")
        .0
}

fn filtered_doubled() -> Metric {
    Metric::builder("doubled")
        .data_type(DataType::Double)
        .expr(measure("revenue").expect("a name") + measure("revenue").expect("a name"))
        .filter(ExprOperationExt::eq(
            dim("region").expect("a name"),
            lit("eu"),
        ))
        .build()
}

fn joined_filtered_metric_model(site: Option<UnresolvedSemanticExpr>) -> SemanticModel {
    let mut model = model_builder()
        .metric(filtered_doubled())
        .joinset(
            sales_joinset(Optional::None)
                .metric(MetricSite::r#ref("doubled"))
                .build(),
        )
        .build()
        .expect("the fixture model must build")
        .0;
    if let Some(predicate) = site {
        let joinset = model
            .joinsets
            .values_mut()
            .next()
            .expect("the fixture holds one joinset");
        let metric = joinset
            .semantic_interface
            .metrics
            .values_mut()
            .next()
            .expect("the joinset exposes one metric");
        set_site_filter(metric, predicate, "doubled");
    }
    model
}

fn stacked_filtered_metric_model() -> SemanticModel {
    model_builder()
        .metric(filtered_doubled())
        .unionset(
            stacked_unionset()
                .metric(MetricSite::r#ref("doubled"))
                .union_all()
                .build(),
        )
        .build()
        .expect("the fixture model must build")
        .0
}

fn filtered_is_eu() -> Metric {
    Metric::builder("is_eu")
        .data_type(DataType::Boolean)
        .expr(ExprOperationExt::eq(
            dim("region").expect("a name"),
            lit("eu"),
        ))
        .filter(ExprOperationExt::neq(
            dim("customer_id").expect("a name"),
            lit("x"),
        ))
        .build()
}

fn joined_row_level_metric_model() -> SemanticModel {
    model_builder()
        .metric(filtered_is_eu())
        .joinset(
            sales_joinset(Optional::None)
                .metric(MetricSite::r#ref("is_eu"))
                .build(),
        )
        .build()
        .expect("the fixture model must build")
        .0
}

fn stacked_row_level_metric_model() -> SemanticModel {
    model_builder()
        .metric(filtered_is_eu())
        .unionset(
            stacked_unionset()
                .metric(MetricSite::r#ref("is_eu"))
                .union_all()
                .build(),
        )
        .build()
        .expect("the fixture model must build")
        .0
}

fn column_filtered_doubled() -> Metric {
    Metric::builder("doubled")
        .data_type(DataType::Double)
        .expr(measure("revenue").expect("a name") + measure("revenue").expect("a name"))
        .filter(ExprOperationExt::eq(
            col("is_active").expect("a name"),
            lit(true),
        ))
        .build()
}

fn joined_column_filtered_metric_model() -> SemanticModel {
    model_builder()
        .metric(column_filtered_doubled())
        .joinset(
            column_filtered_joinset(Optional::None)
                .metric(MetricSite::r#ref("doubled"))
                .build(),
        )
        .build()
        .expect("the fixture model must build")
        .0
}

fn stacked_column_filtered_metric_model() -> SemanticModel {
    model_builder()
        .metric(column_filtered_doubled())
        .unionset(
            column_filtered_unionset()
                .metric(MetricSite::r#ref("doubled"))
                .union_all()
                .build(),
        )
        .build()
        .expect("the fixture model must build")
        .0
}

fn chain_model() -> SemanticModel {
    let joinset = Joinset::builder("sales")
        .dataset(constituent("orders", &SALES_NAMES))
        .dataset(constituent("customers", &SALES_NAMES))
        .dataset(constituent("regions", &SALES_NAMES))
        .dimension(DimensionSite::r#ref("customer_id"))
        .dimension(DimensionSite::r#ref("region"))
        .measure(MeasureSite::r#ref("revenue"))
        .relationship(
            Relationship::builder()
                .name("r1")
                .from("orders")
                .to("customers")
                .on(["customer_id"])
                .many_to_one()
                .build()
                .expect("the fixture relationship must build"),
        )
        .relationship(
            Relationship::builder()
                .name("r2")
                .from("orders")
                .to("regions")
                .on(["region"])
                .many_to_one()
                .build()
                .expect("the fixture relationship must build"),
        )
        .build();
    model_builder()
        .joinset(joinset)
        .build()
        .expect("the fixture model must build")
        .0
}

fn out_of_order_chain_model(optional: Optional) -> SemanticModel {
    out_of_order_chain_model_with(optional, None)
}

fn out_of_order_chain_model_with(
    optional: Optional,
    measure_filter: Option<UnresolvedSemanticExpr>,
) -> SemanticModel {
    let edge = |name: &str, from: &str, to: &str, on: &str| {
        Relationship::builder()
            .name(name)
            .from(from)
            .to(to)
            .on([on])
            .many_to_one()
            .optional(optional)
            .build()
            .expect("the fixture relationship must build")
    };
    let joinset = Joinset::builder("sales")
        .dataset(constituent("zeta", &SALES_NAMES))
        .dataset(constituent("alpha", &SALES_NAMES))
        .dataset(constituent("beta", &SALES_NAMES))
        .dimension(DimensionSite::r#ref("customer_id"))
        .dimension(DimensionSite::r#ref("region"))
        .measure(MeasureSite::r#ref("revenue"))
        .relationship(edge("r1", "alpha", "beta", "customer_id"))
        .relationship(edge("r2", "beta", "zeta", "region"))
        .build();
    pooled_vocabulary(measure_filter)
        .joinset(joinset)
        .build()
        .expect("the fixture model must build")
        .0
}

async fn compiled(model: SemanticModel) -> SemanticManifest {
    let fs = fixture_fs(&[
        "orders",
        "customers",
        "regions",
        "east",
        "west",
        "zeta",
        "alpha",
        "beta",
    ])
    .await;
    let (manifest, warnings) = compile(model, None, &fs, test_provenance())
        .await
        .expect("the fixture model must compile");
    assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
    assert!(
        manifest.validate().is_ok(),
        "a compiled artifact must satisfy its own validator: {:?}",
        manifest.validate().err()
    );
    manifest
}

fn definition_id(manifest: &SemanticManifest, name: &str) -> String {
    manifest
        .semantic_definitions
        .entries
        .values()
        .find(|definition| definition.name.as_str() == name)
        .unwrap_or_else(|| panic!("no semantics named `{name}`"))
        .id
        .clone()
}

fn signature_of<'m>(
    manifest: &'m SemanticManifest,
    source: &str,
    semantics: &str,
) -> &'m SemanticSignature {
    let wanted = definition_id(manifest, semantics);
    by_name(manifest, source)
        .semantic_signature_ids
        .iter()
        .filter_map(|id| manifest.semantic_signatures.get(id))
        .find(|signature| signature.definition_id == wanted)
        .unwrap_or_else(|| panic!("`{source}` owns no signature for `{semantics}`"))
}

fn projection(manifest: &SemanticManifest, source: &str, semantics: &str) -> PhysicalExpr {
    let signature = signature_of(manifest, source, semantics);
    let role = manifest.semantic_definitions.entries[&signature.definition_id]
        .role_definition
        .role();
    let holds_no_link = matches!(
        (&by_name(manifest, source).role_definition, role),
        (
            RoleDefinition::Joinset { .. } | RoleDefinition::Unionset { .. },
            SemanticRole::Dimension | SemanticRole::Measure
        ) | (RoleDefinition::Joinset { .. }, SemanticRole::Filter)
    );
    if holds_no_link {
        link_free_tree(manifest, signature.expression_id)
    } else {
        stored_tree(manifest, signature.expression_id)
    }
}

fn side(manifest: &SemanticManifest, operand: &str, name: &str) -> PhysicalExpr {
    column(by_name(manifest, operand).alias.as_str(), name)
}

fn filter_tree(manifest: &SemanticManifest, source: &str, semantics: &str) -> PhysicalExpr {
    let root = signature_of(manifest, source, semantics)
        .filter_expr_id
        .unwrap_or_else(|| panic!("`{source}`'s signature for `{semantics}` carries no filter"));
    match by_name(manifest, source).role_definition {
        RoleDefinition::Joinset { .. } => link_free_tree(manifest, root),
        _ => stored_tree(manifest, root),
    }
}

fn signed(manifest: &SemanticManifest, source: &str) -> Vec<String> {
    let mut names: Vec<String> = by_name(manifest, source)
        .semantic_signature_ids
        .iter()
        .filter_map(|id| manifest.semantic_signatures.get(id))
        .map(|signature| {
            manifest.semantic_definitions.entries[&signature.definition_id]
                .name
                .as_str()
                .to_owned()
        })
        .collect();
    names.sort();
    names
}

fn eq_eu<L: Leaf>(left: Expr<L>) -> Expr<L> {
    left.eq(Expr::Literal(Literal::String("eu".to_owned())))
}

fn mapped(alias: &str, name: &str, declared: DataType) -> PhysicalExpr {
    cast(column(alias, name), declared)
}

fn coalesce(from: PhysicalExpr, to: PhysicalExpr) -> PhysicalExpr {
    physical_call("coalesce", vec![from, to])
}

fn joined_revenue_sum(manifest: &SemanticManifest, filter: Option<PhysicalExpr>) -> PhysicalExpr {
    let read = |operand| side(manifest, operand, "revenue");
    aggregate(
        "sum",
        false,
        coalesce(read("sales.orders"), read("sales.customers")),
        filter,
    )
}

fn bits_match_mask(manifest: &SemanticManifest, entry: &SemanticSource) -> bool {
    let listed: BTreeSet<u32> = entry
        .semantic_signature_ids
        .iter()
        .filter_map(|id| manifest.semantic_signatures.get(id))
        .map(|signature| {
            manifest.semantic_definitions.entries[&signature.definition_id].bit_position
        })
        .collect();
    let set: BTreeSet<u32> = manifest
        .semantic_definitions
        .entries
        .values()
        .filter(|definition| entry.semantic_bitmask.contains(definition.bit_position))
        .map(|definition| definition.bit_position)
        .collect();
    listed == set
}

#[tokio::test]
async fn a_name_both_operands_map_reads_coalesce_in_its_join_types_order() {
    for (optional, first, second) in [
        (Optional::None, "sales.orders", "sales.customers"),
        (Optional::Left, "sales.orders", "sales.customers"),
        (Optional::Right, "sales.customers", "sales.orders"),
        (Optional::Both, "sales.orders", "sales.customers"),
    ] {
        let manifest = compiled(joined_model(optional)).await;
        let read = |name| coalesce(side(&manifest, first, name), side(&manifest, second, name));
        for name in ["customer_id", "region"] {
            assert_eq!(
                projection(&manifest, "sales", name),
                read(name),
                "{optional:?} reads `{name}` from `{first}`, then `{second}`"
            );
        }
        assert_eq!(
            projection(&manifest, "sales", "revenue"),
            aggregate("sum", false, read("revenue"), None),
            "{optional:?} aggregates the `Coalesce`"
        );
        assert_eq!(
            signed(&manifest, "sales"),
            vec!["customer_id", "region", "revenue"],
            "one signature per exposed name that has a body-free realisation"
        );
        assert!(
            bits_match_mask(&manifest, by_name(&manifest, "sales")),
            "the composition's own bits equal its mask"
        );
    }
}

#[tokio::test]
async fn a_composed_measure_whose_type_is_not_its_declared_type_is_cast_to_it() {
    let declared = [
        ("quantity", DataType::Integer, "sum"),
        ("cost", decimal_18_2(), "sum"),
        ("orders", DataType::Integer, "count"),
    ];
    let measures = || {
        declared
            .iter()
            .map(|(name, _, _)| MeasureSite::r#ref(*name))
            .collect::<Vec<_>>()
    };
    let names: Vec<&str> = SALES_NAMES
        .iter()
        .copied()
        .chain(declared.iter().map(|(name, _, _)| *name))
        .collect();
    let stack = stacked_unionset_over(constituent("east", &names), constituent("west", &names));
    let join = sales_joinset_over(
        Optional::Left,
        constituent("orders", &names),
        constituent("customers", &names),
    );
    let model = declared
        .iter()
        .fold(model_builder(), |model, (name, data_type, aggregate)| {
            model.measure(
                Measure::builder(*name)
                    .data_type(data_type.clone())
                    .agg(agg(aggregate))
                    .build(),
            )
        })
        .unionset(
            measures()
                .into_iter()
                .fold(stack, |stack, site| stack.measure(site))
                .union_all()
                .build(),
        )
        .joinset(
            measures()
                .into_iter()
                .fold(join, |join, site| join.measure(site))
                .build(),
        )
        .build()
        .expect("the fixture model must build")
        .0;
    let manifest = compiled(model).await;
    let stack_alias = by_name(&manifest, "stacked").alias.clone();
    for (name, data_type, aggregate_name) in declared {
        let over = |input| {
            cast(
                aggregate(aggregate_name, false, input, None),
                data_type.clone(),
            )
        };
        assert_eq!(
            projection(&manifest, "stacked", name),
            over(column(stack_alias.as_str(), name)),
            "`{name}` on the Unionset"
        );
        assert_eq!(
            projection(&manifest, "sales", name),
            over(coalesce(
                side(&manifest, "sales.orders", name),
                side(&manifest, "sales.customers", name),
            )),
            "`{name}` on the Joinset"
        );
    }
    assert_eq!(
        projection(&manifest, "stacked", "revenue"),
        aggregate("sum", false, column(stack_alias.as_str(), "revenue"), None)
    );
}

#[tokio::test]
async fn a_unionsets_signatures_read_its_own_alias() {
    let manifest = compiled(stacked_model()).await;
    let stacked = by_name(&manifest, "stacked");

    assert_eq!(
        signed(&manifest, "stacked"),
        vec!["customer_id", "region", "revenue"]
    );
    for name in ["customer_id", "region"] {
        assert_eq!(
            projection(&manifest, "stacked", name),
            column(stacked.alias.as_str(), name),
            "`{name}` reads the stack's own column"
        );
    }
    assert_eq!(
        projection(&manifest, "stacked", "revenue"),
        aggregate(
            "sum",
            false,
            column(stacked.alias.as_str(), "revenue"),
            None
        )
    );
    assert!(bits_match_mask(&manifest, stacked));
}

#[tokio::test]
async fn every_implicit_joinset_unit_owns_signatures_matching_its_mask() {
    let manifest = compiled(chain_model()).await;

    let units: Vec<&SemanticSource> = manifest
        .semantic_sources
        .values()
        .filter(|entry| matches!(entry.role_definition, RoleDefinition::Joinset { .. }))
        .collect();
    assert_eq!(units.len(), 3, "three constituents, two edges, three units");
    for unit in units {
        assert!(
            !unit.semantic_signature_ids.is_empty(),
            "`{}` owns no signature",
            unit.name.as_str()
        );
        assert!(
            bits_match_mask(&manifest, unit),
            "`{}`'s bits and its mask disagree",
            unit.name.as_str()
        );
    }
}

fn operand_names<'m>(manifest: &'m SemanticManifest, unit: &str) -> (&'m str, &'m str) {
    match &by_name(manifest, unit).role_definition {
        RoleDefinition::Joinset { from, to, .. } => (
            manifest.semantic_sources[from].name.as_str(),
            manifest.semantic_sources[to].name.as_str(),
        ),
        other => panic!("`{unit}` must be a `Joinset`, got {other:?}"),
    }
}

#[tokio::test]
async fn a_cumulative_unit_reads_its_composite_operand_under_any_authored_order() {
    for optional in [Optional::None, Optional::Left] {
        let manifest = compiled(out_of_order_chain_model(optional)).await;
        let (from, to) = operand_names(&manifest, "sales");
        assert_eq!(
            (from, to),
            ("sales.r1", "sales.zeta"),
            "the fixture's whole unit must join the composite operand to the leaf"
        );
        for name in ["customer_id", "region"] {
            assert_eq!(
                projection(&manifest, "sales", name),
                coalesce(side(&manifest, from, name), side(&manifest, to, name)),
                "{optional:?} reads `{from}`, then `{to}`, for `{name}`"
            );
        }
        assert!(
            bits_match_mask(&manifest, by_name(&manifest, "sales")),
            "the cumulative unit's own bits equal its mask"
        );
    }
}

#[tokio::test]
async fn a_full_join_over_a_composite_operand_coalesces_both_aliases() {
    let manifest = compiled(out_of_order_chain_model(Optional::Both)).await;
    let (from, to) = operand_names(&manifest, "sales");
    assert_eq!(
        projection(&manifest, "sales", "region"),
        coalesce(
            side(&manifest, from, "region"),
            side(&manifest, to, "region"),
        ),
    );
}

#[tokio::test]
async fn a_joinset_lowers_a_bodied_metric_and_a_named_filter_on_itself() {
    let manifest = compiled(joined_bodied_model()).await;
    let sum = joined_revenue_sum(&manifest, None);

    assert_eq!(
        signed(&manifest, "sales"),
        vec!["customer_id", "doubled", "eu_only", "region", "revenue"],
        "one signature per exposed name, the two bodied roles included"
    );
    assert_eq!(
        projection(&manifest, "sales", "doubled"),
        sum.clone() + sum,
        "read through its links, `doubled` sums the Joinset's own `revenue` twice"
    );
    assert_eq!(
        projection(&manifest, "sales", "eu_only"),
        eu_copies(&manifest, "sales.orders", "sales.customers"),
        "`eu_only` is its body, once for each operand"
    );
    for unit in ["sales", "sales.orders", "sales.customers"] {
        assert!(
            bits_match_mask(&manifest, by_name(&manifest, unit)),
            "`{unit}`'s own bits and its mask disagree"
        );
    }
}

#[tokio::test]
async fn a_unionset_lowers_a_bodied_metric_and_a_named_filter_over_its_own_alias() {
    let manifest = compiled(stacked_bodied_model()).await;
    let stacked = by_name(&manifest, "stacked");
    let region = signature_of(&manifest, "stacked", "region").expression_id;
    let eu_only = signature_of(&manifest, "stacked", "eu_only").expression_id;
    let sum = aggregate(
        "sum",
        false,
        column(stacked.alias.as_str(), "revenue"),
        None,
    );

    assert_eq!(
        signed(&manifest, "stacked"),
        vec!["customer_id", "doubled", "eu_only", "region", "revenue"]
    );
    assert_eq!(
        projection(&manifest, "stacked", "doubled"),
        sum.clone() + sum,
        "read through its links, `doubled` sums the stack's own `revenue` twice"
    );
    assert_eq!(
        projection(&manifest, "stacked", "eu_only"),
        eq_eu(column(stacked.alias.as_str(), "region")),
        "`eu_only` reads the stack's own `region`"
    );
    assert_eq!(
        stored_expr(&manifest, eu_only),
        &eq_eu(link(region)),
        "a Unionset's named filter links the stack's own `region`"
    );
    for unit in ["stacked", "stacked.east", "stacked.west"] {
        assert!(
            bits_match_mask(&manifest, by_name(&manifest, unit)),
            "`{unit}`'s own bits and its mask disagree"
        );
    }
}

#[tokio::test]
async fn a_composed_measure_carries_its_own_view_only() {
    let manifest = compiled(joined_filtered_model()).await;
    let signature = signature_of(&manifest, "sales", "revenue");
    let filter_root = signature
        .filter_expr_id
        .expect("the composition carries a predicate");

    assert_eq!(
        filter_tree(&manifest, "sales", "revenue"),
        eu_copies(&manifest, "sales.orders", "sales.customers"),
        "the composition's own view, with no copy of an operand's own filter"
    );
    assert!(
        signature_of(&manifest, "sales.orders", "revenue")
            .filter_expr_id
            .is_some(),
        "the preserved side carries its own filter on its own signature"
    );
    assert_eq!(
        manifest
            .expressions
            .get(&filter_root)
            .expect("the filter root is stored")
            .layer,
        ExprLayer::Scalar,
        "the filter is row-level"
    );

    assert_ne!(
        filter_root, signature.expression_id,
        "the filter entry is not the expression's entry"
    );
}

#[tokio::test]
async fn a_stacked_measure_carries_its_own_filter_over_its_projections() {
    let manifest = compiled(stacked_filtered_model()).await;

    assert_eq!(
        filter_tree(&manifest, "stacked", "revenue"),
        Expr::BinaryOp {
            op: BinaryOpKind::Eq,
            left: Box::new(projection(&manifest, "stacked", "region")),
            right: Box::new(Expr::Literal(Literal::String("eu".to_owned()))),
        },
        "the stack's predicate reads its own projected column"
    );
    for branch in ["stacked.east", "stacked.west"] {
        assert!(
            signature_of(&manifest, branch, "revenue")
                .filter_expr_id
                .is_some(),
            "`{branch}` lowers the pooled predicate over its own columns too"
        );
    }
}

#[tokio::test]
async fn a_unionset_measures_filter_links_the_unionsets_own_dimension() {
    let manifest = compiled(stacked_filtered_model()).await;
    let region = signature_of(&manifest, "stacked", "region").expression_id;
    let filter = (signature_of(&manifest, "stacked", "revenue").filter_expr_id)
        .expect("the stack carries a predicate");
    assert_eq!(stored_expr(&manifest, filter), &eq_eu(link(region)));
    assert_eq!(
        filter_tree(&manifest, "stacked", "revenue"),
        eq_eu(column(
            by_name(&manifest, "stacked").alias.as_str(),
            "region"
        ))
    );
}

#[tokio::test]
async fn a_joinsets_measures_filter_holds_no_link() {
    let manifest = compiled(joined_filtered_model()).await;
    let filter = (signature_of(&manifest, "sales", "revenue").filter_expr_id)
        .expect("the Joinset carries a predicate");
    assert_eq!(
        link_free_tree(&manifest, filter),
        eu_copies(&manifest, "sales.orders", "sales.customers")
    );
}

fn sources_whose_filter_links_their_own_region(manifest: &SemanticManifest) -> Vec<&str> {
    let mut linking = Vec::new();
    for source in manifest.semantic_sources.values() {
        let signatures: Vec<_> = listed(manifest, source).collect();
        let own: BTreeSet<PhysicalExprId> = signatures
            .iter()
            .map(|signature| signature.expression_id)
            .collect();
        for signature in &signatures {
            for root in [Some(signature.expression_id), signature.filter_expr_id]
                .into_iter()
                .flatten()
            {
                let links: Vec<_> = manifest.expressions.physical[&root].expr.links().collect();
                let name = source.name.as_str();
                assert!(
                    links.iter().all(|link| own.contains(link)),
                    "`{name}`'s entry {root} links its own signatures only"
                );
                if Some(root) != signature.filter_expr_id || links.is_empty() {
                    continue;
                }
                assert!(
                    !matches!(source.role_definition, RoleDefinition::Joinset { .. }),
                    "`{name}`'s filter holds no link"
                );
                let region = signature_of(manifest, name, "region").expression_id;
                assert_eq!(stored_expr(manifest, root), &eq_eu(link(region)), "{name}");
                linking.push(name);
            }
        }
    }
    linking.sort_unstable();
    linking
}

#[tokio::test]
async fn nested_compositions_link_their_own_signatures() {
    let eu = || {
        Some(ExprOperationExt::eq(
            dim("region").expect("a name"),
            lit("eu"),
        ))
    };
    let manifest = compiled(out_of_order_chain_model_with(Optional::None, eu())).await;
    assert_eq!(
        sources_whose_filter_links_their_own_region(&manifest),
        ["sales.alpha", "sales.beta", "sales.zeta"]
    );
    for (joinset, from, to) in [
        ("sales", "sales.r1", "sales.zeta"),
        ("sales.r1", "sales.alpha", "sales.beta"),
    ] {
        assert_eq!(
            filter_tree(&manifest, joinset, "revenue"),
            eu_copies(&manifest, from, to)
        );
    }

    let two_tables = |name: &str| {
        NestedDataset::builder(name)
            .tables([format!("warehouse.{name}_a"), format!("warehouse.{name}_b")])
            .build()
    };
    let columns: &[(&str, &str)] = &[
        ("customer_id", "string"),
        ("region", "string"),
        ("revenue", "double"),
    ];
    let catalog =
        warehouse_catalog(&["east_a", "east_b", "west_a", "west_b"].map(|table| (table, columns)));
    let model = pooled_vocabulary(eu())
        .unionset(
            Unionset::builder("stacked")
                .dataset(two_tables("east"))
                .dataset(two_tables("west"))
                .dimension(DimensionSite::r#ref("customer_id"))
                .dimension(DimensionSite::r#ref("region"))
                .measure(MeasureSite::r#ref("revenue"))
                .union_all()
                .build(),
        )
        .build()
        .expect("the fixture model must build")
        .0;
    let manifest = compile_model_with(model, Some(&catalog)).await;
    let linking = sources_whose_filter_links_their_own_region(&manifest);
    for unionset in ["stacked", "stacked.east", "stacked.west"] {
        assert!(linking.contains(&unionset), "{unionset} in {linking:?}");
        assert_eq!(
            filter_tree(&manifest, unionset, "revenue"),
            eq_eu(side(&manifest, unionset, "region"))
        );
    }
}

#[tokio::test]
async fn a_site_filter_replaces_the_pooled_one_on_the_composition_and_its_operands() {
    let manifest = compiled(joined_filtered_model_with(
        Optional::None,
        Some(ExprOperationExt::eq(
            dim("region").expect("a name"),
            lit("eu"),
        )),
        Some(ExprOperationExt::neq(
            dim("customer_id").expect("a name"),
            lit("x"),
        )),
    ))
    .await;

    assert_eq!(
        filter_tree(&manifest, "sales", "revenue"),
        neq_x_copies(&manifest, "sales.orders", "sales.customers"),
    );
    for operand in ["sales.orders", "sales.customers"] {
        assert!(
            !format!("{:?}", filter_tree(&manifest, operand, "revenue")).contains("String(\"eu\")"),
            "the site predicate replaces the pooled one on `{operand}` too"
        );
    }
}

fn eu_copies(manifest: &SemanticManifest, from: &str, to: &str) -> PhysicalExpr {
    eq_eu(side(manifest, from, "region")) & eq_eu(side(manifest, to, "region"))
}

fn neq_x_copies(manifest: &SemanticManifest, from: &str, to: &str) -> PhysicalExpr {
    let neq_x = |operand| side(manifest, operand, "customer_id").neq(string("x"));
    neq_x(from) & neq_x(to)
}

#[tokio::test]
async fn a_filter_on_a_joinset_holds_one_copy_for_each_operand() {
    let eu = || {
        Some(ExprOperationExt::eq(
            dim("region").expect("a name"),
            lit("eu"),
        ))
    };
    for optional in [
        Optional::None,
        Optional::Left,
        Optional::Right,
        Optional::Both,
    ] {
        let manifest = compiled(joined_filtered_model_with(optional, eu(), None)).await;
        assert_eq!(
            filter_tree(&manifest, "sales", "revenue"),
            eu_copies(&manifest, "sales.orders", "sales.customers"),
            "{optional:?}"
        );
    }

    let model = pooled_vocabulary(eu())
        .joinset(
            sales_joinset_over(
                Optional::None,
                constituent("orders", &SALES_NAMES),
                constituent("customers", &["customer_id", "revenue"]),
            )
            .build(),
        )
        .build()
        .expect("the fixture model must build")
        .0;
    let manifest = compiled(model).await;
    assert_eq!(
        filter_tree(&manifest, "sales", "revenue"),
        eq_eu(side(&manifest, "sales.orders", "region")),
        "a filter that reads no name both operands map has one copy"
    );

    let vip_eu = DataKindFilter::builder("vip_eu")
        .expr(
            ExprOperationExt::eq(dim("region").expect("a name"), lit("eu"))
                & ExprOperationExt::eq(dim("segment").expect("a name"), lit("vip")),
        )
        .build();
    let segment = Dimension::builder("segment")
        .data_type(DataType::String)
        .categorical()
        .build();
    let model = model_builder()
        .dimension(segment)
        .joinset(
            sales_joinset_over(
                Optional::None,
                constituent("orders", &SALES_NAMES),
                constituent("customers", &["customer_id", "segment"]),
            )
            .dimension(DimensionSite::r#ref("segment"))
            .filter(vip_eu)
            .build(),
        )
        .build()
        .expect("the fixture model must build")
        .0;
    let manifest = compiled(model).await;
    let vip = side(&manifest, "sales.customers", "segment")
        .eq(Expr::Literal(Literal::String("vip".to_owned())));
    assert_eq!(
        projection(&manifest, "sales", "vip_eu"),
        eq_eu(side(&manifest, "sales.orders", "region")) & vip,
        "one copy, each name read from the operand that maps it"
    );
}

#[tokio::test]
async fn the_composed_filter_names_the_semantics_and_the_operands_own_filter_its_column() {
    let manifest = compiled(joined_mapped_model(ExprOperationExt::eq(
        dim("region").expect("a name"),
        lit("eu"),
    )))
    .await;

    assert_eq!(
        filter_tree(&manifest, "sales", "revenue"),
        eu_copies(&manifest, "sales.orders", "sales.customers"),
        "the composition's own view names the semantics it projects"
    );
    assert_eq!(
        filter_tree(&manifest, "sales.orders", "revenue"),
        eq_eu(cast(
            column(&source_alias(&manifest, "sales.orders"), "region_code"),
            DataType::String,
        )),
        "the operand's own filter names its physical column, on its own signature"
    );
}

#[tokio::test]
async fn a_composite_operands_own_filter_stays_on_its_signature() {
    let manifest = compiled(out_of_order_chain_model_with(
        Optional::None,
        Some(ExprOperationExt::eq(
            dim("region").expect("a name"),
            lit("eu"),
        )),
    ))
    .await;
    assert_eq!(
        operand_names(&manifest, "sales"),
        ("sales.r1", "sales.zeta"),
        "the cumulative unit joins the composite operand to the leaf"
    );
    assert_eq!(
        projection(&manifest, "sales.r1", "region"),
        coalesce(
            side(&manifest, "sales.alpha", "region"),
            side(&manifest, "sales.beta", "region"),
        ),
        "the immediate operand's own value"
    );

    assert_eq!(
        filter_tree(&manifest, "sales.r1", "revenue"),
        eu_copies(&manifest, "sales.alpha", "sales.beta"),
        "the sub-composition's own view, over its operands"
    );
    assert_eq!(
        filter_tree(&manifest, "sales", "revenue"),
        eu_copies(&manifest, "sales.r1", "sales.zeta"),
        "the cumulative unit's own view reads its immediate operands"
    );
}

#[tokio::test]
async fn an_unfiltered_composed_measure_carries_no_predicate() {
    let manifest = compiled(joined_model(Optional::None)).await;
    assert!(
        signature_of(&manifest, "sales", "revenue")
            .filter_expr_id
            .is_none(),
        "no `filter:` anywhere authors no predicate here"
    );
}

#[tokio::test]
async fn a_composed_measure_filter_referencing_a_scalar_bodied_metric_is_refused() {
    match compile_error(joined_measure_filter_referencing_metric_model()).await {
        CompileError::AggregateReferenceInRowLevelBody {
            semantics,
            referenced,
        } => {
            assert_eq!(semantics, "revenue");
            assert_eq!(referenced, "is_eu");
        }
        other => panic!("expected AggregateReferenceInRowLevelBody, got {other:?}"),
    }
}

#[tokio::test]
async fn a_composed_measures_filter_omits_a_pooled_physical_column_leaf() {
    let catalog = column_filter_catalog();
    for (manifest, composition, constituents) in [
        (
            compile_model_with(joined_column_filtered_model(), Some(&catalog)).await,
            "sales",
            ["sales.orders", "sales.customers"],
        ),
        (
            compile_model_with(stacked_column_filtered_model(), Some(&catalog)).await,
            "stacked",
            ["stacked.east", "stacked.west"],
        ),
    ] {
        assert!(
            signature_of(&manifest, composition, "revenue")
                .filter_expr_id
                .is_none(),
            "`{composition}`'s own filter is omitted, and nothing is copied in its place"
        );
        for constituent in constituents {
            assert!(
                signature_of(&manifest, constituent, "revenue")
                    .filter_expr_id
                    .is_some(),
                "`{constituent}` still lowers the pooled predicate over its own physical column"
            );
        }
    }
}

#[tokio::test]
async fn a_composed_measures_own_site_filter_refuses_a_physical_column_leaf() {
    let model = pooled_vocabulary(None)
        .joinset(column_filtered_joinset(Optional::None).build())
        .build()
        .expect("the fixture model must build")
        .0;
    let model = with_site_filter(
        model,
        ExprOperationExt::eq(col("is_active").expect("a name"), lit(true)),
    );
    match compile_error_with(model, Some(&column_filter_catalog())).await {
        CompileError::ColumnInComposedFilter {
            unit,
            semantic,
            column,
        } => {
            assert_eq!(unit, "sales");
            assert_eq!(semantic, "revenue");
            assert_eq!(column, "is_active");
        }
        other => panic!("expected ColumnInComposedFilter, got {other:?}"),
    }
}

#[tokio::test]
async fn a_composed_metric_lowers_its_body_over_the_compositions_own_measures() {
    let manifest = compiled(joined_filtered_metric_model(None)).await;
    let revenue = signature_of(&manifest, "sales", "revenue").expression_id;
    let doubled = signature_of(&manifest, "sales", "doubled").expression_id;
    let sum = joined_revenue_sum(&manifest, None);

    assert_eq!(
        stored_expr(&manifest, doubled),
        &(link(revenue) + link(revenue)),
        "`doubled` links the Joinset's own `revenue`"
    );
    assert_eq!(
        projection(&manifest, "sales", "doubled"),
        sum.clone() + sum,
        "read through its links, `doubled` sums the Joinset's own `revenue` twice"
    );
    assert_eq!(
        filter_tree(&manifest, "sales", "doubled"),
        eu_copies(&manifest, "sales.orders", "sales.customers"),
        "the Joinset's own filter for `doubled`, a copy for each operand"
    );
    for operand in ["sales.orders", "sales.customers"] {
        assert!(
            signature_of(&manifest, operand, "doubled")
                .filter_expr_id
                .is_some(),
            "`{operand}`'s own `doubled` keeps its own filter"
        );
    }
}

#[tokio::test]
async fn a_composed_metric_on_a_unionset_links_the_stacks_own_signatures() {
    let manifest = compiled(stacked_filtered_metric_model()).await;
    let alias = by_name(&manifest, "stacked").alias.as_str();
    let region = signature_of(&manifest, "stacked", "region").expression_id;
    let filter = signature_of(&manifest, "stacked", "doubled")
        .filter_expr_id
        .expect("the stack carries its own filter for `doubled`");
    let sum = aggregate("sum", false, column(alias, "revenue"), None);

    assert_eq!(
        projection(&manifest, "stacked", "doubled"),
        sum.clone() + sum,
        "read through its links, `doubled` sums the stack's own `revenue` twice"
    );
    assert_eq!(
        stored_expr(&manifest, filter),
        &eq_eu(link(region)),
        "the filter links the stack's own `region`"
    );
    assert_eq!(
        filter_tree(&manifest, "stacked", "doubled"),
        eq_eu(column(alias, "region")),
        "the filter reads the stack's own `region`"
    );
}

#[tokio::test]
async fn a_composed_metrics_site_filter_holds_a_copy_for_each_operand() {
    let manifest = compiled(joined_filtered_metric_model(Some(ExprOperationExt::neq(
        dim("customer_id").expect("a name"),
        lit("x"),
    ))))
    .await;

    assert_eq!(
        filter_tree(&manifest, "sales", "doubled"),
        neq_x_copies(&manifest, "sales.orders", "sales.customers"),
        "the site's filter, once for each operand, joined with `And`"
    );
}

#[tokio::test]
async fn a_composed_row_level_metric_holds_its_filter_in_its_own_tree() {
    let case = |filter, formula| Expr::Case {
        whens: vec![(filter, formula)],
        else_: None,
    };

    let manifest = compiled(joined_row_level_metric_model()).await;
    let region = coalesce(
        side(&manifest, "sales.orders", "region"),
        side(&manifest, "sales.customers", "region"),
    );
    assert_eq!(
        projection(&manifest, "sales", "is_eu"),
        case(
            neq_x_copies(&manifest, "sales.orders", "sales.customers"),
            eq_eu(region),
        ),
        "the Joinset's `is_eu` holds a copy of its filter for each operand"
    );
    assert!(
        signature_of(&manifest, "sales", "is_eu")
            .filter_expr_id
            .is_none(),
        "the Joinset's `is_eu` has no filter entry"
    );

    let manifest = compiled(stacked_row_level_metric_model()).await;
    let alias = by_name(&manifest, "stacked").alias.as_str();
    assert_eq!(
        projection(&manifest, "stacked", "is_eu"),
        case(
            column(alias, "customer_id").neq(string("x")),
            eq_eu(column(alias, "region")),
        ),
        "the stack's `is_eu` holds its filter over the stack's own columns"
    );
    assert!(
        signature_of(&manifest, "stacked", "is_eu")
            .filter_expr_id
            .is_none(),
        "the stack's `is_eu` has no filter entry"
    );
}

#[tokio::test]
async fn a_composed_metrics_pooled_physical_column_filter_is_refused() {
    let catalog = column_filter_catalog();
    for (model, composition) in [
        (joined_column_filtered_metric_model(), "sales"),
        (stacked_column_filtered_metric_model(), "stacked"),
    ] {
        match compile_error_with(model, Some(&catalog)).await {
            CompileError::ColumnInComposedFilter {
                unit,
                semantic,
                column,
            } => {
                assert_eq!(unit, composition);
                assert_eq!(semantic, "doubled");
                assert_eq!(column, "is_active");
            }
            other => panic!("expected ColumnInComposedFilter, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn a_composed_metric_reads_the_compositions_own_filtered_measure() {
    let manifest = compiled(joined_filtered_measure_inlined_in_metric_model()).await;

    let alias = source_alias(&manifest, "sales.orders");
    let filtered_fold = aggregate(
        "sum",
        false,
        mapped(&alias, "revenue", DataType::Double),
        Some(eq_eu(mapped(&alias, "region", DataType::String))),
    );
    assert_eq!(
        projection(&manifest, "sales.orders", "doubled"),
        Expr::BinaryOp {
            op: BinaryOpKind::Add,
            left: Box::new(filtered_fold.clone()),
            right: Box::new(filtered_fold),
        },
        "the constituent's own `doubled` signature reads with `revenue`'s predicate"
    );

    let composed_fold = joined_revenue_sum(
        &manifest,
        Some(eu_copies(&manifest, "sales.orders", "sales.customers")),
    );
    assert_eq!(
        projection(&manifest, "sales", "doubled"),
        composed_fold.clone() + composed_fold,
        "the composed Metric reads the Joinset's own filtered `revenue`"
    );
    assert!(
        signature_of(&manifest, "sales", "doubled")
            .filter_expr_id
            .is_none(),
        "`doubled` states no filter of its own"
    );
}

#[tokio::test]
async fn a_composed_metric_reads_the_compositions_own_cast_wrapped_measure() {
    let manifest = compile_model_with(
        joined_cast_wrapped_filtered_measure_inlined_in_metric_model(),
        Some(&bodied_revenue_catalog()),
    )
    .await;

    let alias = source_alias(&manifest, "sales.orders");
    let filtered_cast_fold = cast(
        aggregate(
            "sum",
            false,
            column(&alias, "amount"),
            Some(eq_eu(column(&alias, "region"))),
        ),
        DataType::Double,
    );
    assert_eq!(
        projection(&manifest, "sales.orders", "doubled"),
        Expr::BinaryOp {
            op: BinaryOpKind::Add,
            left: Box::new(filtered_cast_fold.clone()),
            right: Box::new(filtered_cast_fold),
        },
        "the constituent's own `doubled` signature keeps the observed-column Cast and \
         reads with `revenue`'s predicate inside it"
    );

    let composed_cast_fold = cast(
        joined_revenue_sum(
            &manifest,
            Some(eu_copies(&manifest, "sales.orders", "sales.customers")),
        ),
        DataType::Double,
    );
    assert_eq!(
        projection(&manifest, "sales", "doubled"),
        composed_cast_fold.clone() + composed_cast_fold,
        "the composed Metric reads the Joinset's own cast-wrapped, filtered `revenue`"
    );
    assert!(
        signature_of(&manifest, "sales", "doubled")
            .filter_expr_id
            .is_none(),
        "`doubled` states no filter of its own"
    );
}

#[tokio::test]
async fn composed_identity_is_identical_across_two_compiles() {
    let first = compiled(chain_model()).await;
    let second = compiled(chain_model()).await;
    assert_eq!(first.semantic_signatures, second.semantic_signatures);
    assert_eq!(first.expressions, second.expressions);
}

#[tokio::test]
async fn a_measure_gives_one_result_on_a_dataset_and_on_each_composition_over_it() {
    for (aggregate_name, native, declared, casts_its_column, casts) in [
        ("sum", "int", DataType::Integer, false, true),
        ("avg", "decimal(18,2)", DataType::Double, true, false),
        ("median", "double", DataType::String, false, true),
        ("sum", "int", DataType::BigInt, true, false),
    ] {
        let catalog = catalog(&[("customer_id", "string"), ("value", native)]);
        let model = |model: semstrait_model::SemanticModelBuilder| {
            model
                .name("sales")
                .dimension(
                    Dimension::builder("customer_id")
                        .data_type(DataType::String)
                        .categorical()
                        .build(),
                )
                .measure(
                    Measure::builder("value")
                        .data_type(declared.clone())
                        .agg(agg(aggregate_name))
                        .build(),
                )
                .build()
                .expect("the fixture model must build")
                .0
        };
        let over = |input| {
            let call = aggregate(aggregate_name, false, input, None);
            if casts {
                cast(call, declared.clone())
            } else {
                call
            }
        };

        let dataset = Dataset::builder("orders")
            .table("warehouse.orders")
            .dimension(DimensionSite::r#ref("customer_id"))
            .measure(MeasureSite::r#ref("value"))
            .build();
        let manifest = compile_model_with(
            model(SemanticModel::builder().dataset(dataset)),
            Some(&catalog),
        )
        .await;
        let value = column("ps0", "value");
        assert_eq!(
            projection(&manifest, "orders", "value"),
            over(if casts_its_column {
                cast(value, declared.clone())
            } else {
                value
            }),
            "`{aggregate_name}` over `{native}` on the Dataset"
        );

        let over_the_table = |name: &str| {
            NestedDataset::builder(name)
                .table("warehouse.orders")
                .build()
        };
        let to = |constituent: &str| {
            Relationship::builder()
                .name(format!("to_{constituent}"))
                .from("orders")
                .to(constituent)
                .on(["customer_id"])
                .many_to_one()
                .build()
                .expect("the fixture relationship must build")
        };
        let joinset = Joinset::builder("sales")
            .dataset(over_the_table("orders"))
            .dataset(over_the_table("customers"))
            .dataset(over_the_table("regions"))
            .dimension(DimensionSite::r#ref("customer_id"))
            .measure(MeasureSite::r#ref("value"))
            .relationship(to("customers"))
            .relationship(to("regions"))
            .build();
        let unionset = Unionset::builder("stacked")
            .dataset(over_the_table("east"))
            .dataset(over_the_table("west"))
            .dimension(DimensionSite::r#ref("customer_id"))
            .measure(MeasureSite::r#ref("value"))
            .union_all()
            .build();
        let composed = SemanticModel::builder().joinset(joinset).unionset(unionset);
        let manifest = compile_model_with(model(composed), Some(&catalog)).await;
        let read =
            |from, to| coalesce(side(&manifest, from, "value"), side(&manifest, to, "value"));
        assert_eq!(
            projection(&manifest, "sales.to_customers", "value"),
            over(read("sales.orders", "sales.customers")),
            "`{aggregate_name}` over `{native}` on the Joinset unit over two Datasets"
        );
        assert_eq!(
            projection(&manifest, "sales", "value"),
            over(read("sales.to_customers", "sales.regions")),
            "`{aggregate_name}` over `{native}` on the Joinset unit over a Joinset unit"
        );
        assert_eq!(
            projection(&manifest, "stacked", "value"),
            over(side(&manifest, "stacked", "value")),
            "`{aggregate_name}` over `{native}` on the Unionset"
        );
    }
}
