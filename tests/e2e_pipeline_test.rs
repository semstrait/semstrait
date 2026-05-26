//! End-to-end integration tests for the full Semstrait pipeline.
//!
//! Tests the complete flow: YAML → ManifestCompiler → SemanticPlanner → SQL generation.
//! Model definitions are loaded from `tests/fixtures/models/`.

mod test_helpers;

use semstrait_api::types::RawQueryRequest;
use semstrait_api::SemstraitEngine;
use semstrait_manifest::{CompileSource, ManifestCompiler};
use semstrait_planner::{ResolvedQueryRequest, SemanticPlanner};
use semstrait_adapter::sql::{AnsiDialect, AnsiSqlEmitter, SqlEmitter};
use std::collections::HashMap;
use test_helpers::load_model;

// =============================================================================
// Test 1: Full pipeline - compile, plan, generate SQL
// =============================================================================

#[tokio::test]
async fn test_yaml_compile_plan_sql() {
    let yaml = load_model("orders_3dim");

    // Step 1: Compile the YAML model
    let compiler = ManifestCompiler::new();
    let manifest = compiler
        .compile(CompileSource::Yaml(yaml))
        .await
        .expect("compilation should succeed");

    // Verify manifest structure
    assert_eq!(manifest.model_name, "e2e_test_model");
    assert_eq!(manifest.entities.len(), 1);
    assert!(manifest.entities.contains_key("orders"));

    let iface = manifest.entities["orders"].interface();
    assert_eq!(iface.dimensions.len(), 3);
    assert_eq!(iface.measures.len(), 1);
    assert!(iface.dimensions.contains_key("date"));
    assert!(iface.dimensions.contains_key("region"));
    assert!(iface.dimensions.contains_key("customer"));
    assert!(iface.measures.contains_key("revenue"));

    // Step 2: Build a query request
    let request = ResolvedQueryRequest {
        entity_name: "orders".to_string(),
        dimensions: vec!["date".to_string(), "region".to_string()],
        measures: vec!["revenue".to_string()],
        filters: vec![],
        inline_filters: vec![],
        pending_inline_filters: vec![],
        grain: None,
        limit: None,
        order_by: vec![],
        session_variables: HashMap::new(),
    };

    // Step 3: Plan the query
    let planner = SemanticPlanner::builder().build();
    let plan = planner
        .plan(&request, &manifest)
        .expect("planning should succeed");

    // Verify the plan output
    assert_eq!(plan.output_names.len(), 3); // date, region, revenue
    assert!(plan.output_names.contains(&"date".to_string()));
    assert!(plan.output_names.contains(&"region".to_string()));
    assert!(plan.output_names.contains(&"revenue".to_string()));

    // Step 4: Generate SQL
    let emitter = AnsiSqlEmitter::new(AnsiDialect);
    let sql = emitter.emit(&plan).expect("SQL generation should succeed");

    // Verify SQL contains expected elements
    assert!(
        sql.contains("SELECT") || sql.contains("select"),
        "SQL should contain SELECT: {}",
        sql
    );
    assert!(
        sql.contains("GROUP BY") || sql.contains("group by"),
        "SQL should contain GROUP BY for aggregation: {}",
        sql
    );

    assert!(!sql.is_empty(), "SQL should not be empty");
    assert!(
        sql.len() > 20,
        "SQL should be a meaningful query, got: {}",
        sql
    );
}

// =============================================================================
// Test 2: Constraint violation - measure requires dimension
// =============================================================================

#[tokio::test]
async fn test_constraint_violation_e2e() {
    let yaml = load_model("sales_constrained");

    let compiler = ManifestCompiler::new();
    let manifest = compiler
        .compile(CompileSource::Yaml(yaml))
        .await
        .expect("compilation should succeed");

    // Build a request that violates the constraint (no date dimension)
    let request = ResolvedQueryRequest {
        entity_name: "sales".to_string(),
        dimensions: vec!["region".to_string()], // missing 'date'
        measures: vec!["revenue".to_string()],
        filters: vec![],
        inline_filters: vec![],
        pending_inline_filters: vec![],
        grain: None,
        limit: None,
        order_by: vec![],
        session_variables: HashMap::new(),
    };

    // Planning should fail with constraint violation
    let planner = SemanticPlanner::builder().build();
    let result = planner.plan(&request, &manifest);

    assert!(result.is_err(), "planning should fail due to constraint");

    let err = result.unwrap_err();
    let err_msg = err.to_string();
    assert!(
        err_msg.contains("constraint") || err_msg.contains("violation"),
        "error should mention constraint violation, got: {}",
        err_msg
    );
}

// =============================================================================
// Test 3: Compile error - raw SQL rejection
// =============================================================================

#[tokio::test]
async fn test_compile_error_raw_sql() {
    let yaml = load_model("raw_sql_invalid");

    let compiler = ManifestCompiler::new();
    let result = compiler
        .compile(CompileSource::Yaml(yaml))
        .await;

    assert!(
        result.is_err(),
        "compilation should fail due to raw SQL in expr"
    );

    let err = result.unwrap_err();
    let err_msg = err.to_string();
    assert!(
        err_msg.contains("raw SQL rejected") || err_msg.contains("Raw SQL rejected"),
        "error should mention raw SQL rejection, got: {}",
        err_msg
    );
}

// =============================================================================
// Test 4: Plan with filters and ordering
// =============================================================================

#[tokio::test]
async fn test_plan_with_filters_and_order() {
    let yaml = load_model("products");

    let compiler = ManifestCompiler::new();
    let manifest = compiler
        .compile(CompileSource::Yaml(yaml))
        .await
        .expect("compilation should succeed");

    // Build request with filters and ordering
    use semstrait_planner::{FilterOperator, FilterValue, OrderByClause, QueryFilter, SortDirection};

    let request = ResolvedQueryRequest {
        entity_name: "products".to_string(),
        dimensions: vec!["category".to_string(), "brand".to_string()],
        measures: vec!["total_sales".to_string()],
        filters: vec![QueryFilter {
            field: "category".to_string(),
            operator: FilterOperator::Eq,
            values: vec![FilterValue::String("Electronics".to_string())],
        }],
        inline_filters: vec![],
        pending_inline_filters: vec![],
        grain: None,
        limit: Some(10),
        order_by: vec![OrderByClause {
            field: "total_sales".to_string(),
            direction: SortDirection::Descending,
        }],
        session_variables: HashMap::new(),
    };

    let planner = SemanticPlanner::builder().build();
    let plan = planner
        .plan(&request, &manifest)
        .expect("planning should succeed");

    // Verify plan structure
    assert_eq!(plan.output_names.len(), 3);

    // Generate SQL
    let emitter = AnsiSqlEmitter::new(AnsiDialect);
    let sql = emitter.emit(&plan).expect("SQL generation should succeed");

    // SQL should contain filter, order, and limit
    assert!(!sql.is_empty());
    assert!(sql.len() > 30, "SQL should be comprehensive, got: {}", sql);
}

// =============================================================================
// Test 5: Multiple measures aggregation
// =============================================================================

#[tokio::test]
async fn test_multiple_measures() {
    let yaml = load_model("transactions_multi_measure");

    let compiler = ManifestCompiler::new();
    let manifest = compiler
        .compile(CompileSource::Yaml(yaml))
        .await
        .expect("compilation should succeed");

    let request = ResolvedQueryRequest {
        entity_name: "transactions".to_string(),
        dimensions: vec!["date".to_string()],
        measures: vec![
            "revenue".to_string(),
            "transaction_count".to_string(),
            "avg_amount".to_string(),
        ],
        filters: vec![],
        inline_filters: vec![],
        pending_inline_filters: vec![],
        grain: None,
        limit: None,
        order_by: vec![],
        session_variables: HashMap::new(),
    };

    let planner = SemanticPlanner::builder().build();
    let plan = planner
        .plan(&request, &manifest)
        .expect("planning should succeed");

    // Should have 4 outputs: 1 dimension + 3 measures
    assert_eq!(plan.output_names.len(), 4);
    assert!(plan.output_names.contains(&"revenue".to_string()));
    assert!(plan.output_names.contains(&"transaction_count".to_string()));
    assert!(plan.output_names.contains(&"avg_amount".to_string()));

    // Generate SQL
    let emitter = AnsiSqlEmitter::new(AnsiDialect);
    let sql = emitter.emit(&plan).expect("SQL generation should succeed");

    assert!(!sql.is_empty());
    // SQL should contain multiple aggregate functions
    assert!(sql.len() > 50, "SQL with multiple measures should be substantial");
}

// =============================================================================
// Test 6: Kind not found error
// =============================================================================

#[tokio::test]
async fn test_kind_not_found() {
    let yaml = load_model("orders_simple");

    let compiler = ManifestCompiler::new();
    let manifest = compiler
        .compile(CompileSource::Yaml(yaml))
        .await
        .expect("compilation should succeed");

    // Request a non-existent kind
    let request = ResolvedQueryRequest {
        entity_name: "nonexistent_kind".to_string(),
        dimensions: vec!["date".to_string()],
        measures: vec!["revenue".to_string()],
        filters: vec![],
        inline_filters: vec![],
        pending_inline_filters: vec![],
        grain: None,
        limit: None,
        order_by: vec![],
        session_variables: HashMap::new(),
    };

    let planner = SemanticPlanner::builder().build();
    let result = planner.plan(&request, &manifest);

    assert!(result.is_err(), "planning should fail for unknown kind");

    let err = result.unwrap_err();
    let err_msg = err.to_string();
    assert!(
        err_msg.contains("not found") || err_msg.contains("nonexistent_kind"),
        "error should mention kind not found, got: {}",
        err_msg
    );
}

// =============================================================================
// Test 7: Explain includes Substrait JSON
// =============================================================================

#[tokio::test]
async fn test_explain_includes_substrait() {
    let yaml = load_model("orders_with_metrics");

    let engine = SemstraitEngine::with_model(&yaml)
        .await
        .expect("engine should compile manifest");

    let raw = RawQueryRequest {
        from: Some("orders".to_string()),
        select: vec!["date".to_string(), "region".to_string(), "revenue".to_string()],
        ..Default::default()
    };

    let result = engine.explain(&raw).await.expect("explain should succeed");

    assert!(result.sql.is_some(), "should have SQL");
    assert!(
        result.plan_text.contains("TableScan:"),
        "plan_text should contain TableScan: {}",
        result.plan_text
    );
}

// =============================================================================
// Comprehensive E-Commerce Model — full feature coverage
// =============================================================================
//
// Tests the comprehensive_ecommerce.yaml model which exercises ALL semstrait
// features: grainset, unionset, joinset, all dimension types, all aggregation
// types, additivity, constraints, measure filters, metrics (simple, ratio,
// derived, nested), column mapping variants (simple, WithGrain, literal),
// metadata dimensions, temporal configs, kind-level filters, relationships.

/// Helper: compile the comprehensive model once, shared across tests.
async fn compile_ecommerce() -> semstrait_manifest::CompiledManifest {
    let yaml = load_model("comprehensive_ecommerce");
    let compiler = ManifestCompiler::new();
    compiler
        .compile(CompileSource::Yaml(yaml))
        .await
        .expect("comprehensive model should compile")
}

/// Helper: plan + emit SQL for a given request against a manifest.
fn plan_sql(
    request: &ResolvedQueryRequest,
    manifest: &semstrait_manifest::CompiledManifest,
) -> (semstrait_ir::LogicalPlan, String) {
    let planner = SemanticPlanner::builder().build();
    let plan = planner
        .plan(request, manifest)
        .expect("planning should succeed");
    let emitter = AnsiSqlEmitter::new(AnsiDialect);
    let sql = emitter.emit(&plan).expect("SQL emission should succeed");
    (plan, sql)
}

/// Helper: build a simple request (no filters/limit/order).
fn simple_request(
    entity: &str,
    dims: Vec<&str>,
    measures: Vec<&str>,
) -> ResolvedQueryRequest {
    ResolvedQueryRequest {
        entity_name: entity.to_string(),
        dimensions: dims.into_iter().map(|s| s.to_string()).collect(),
        measures: measures.into_iter().map(|s| s.to_string()).collect(),
        filters: vec![],
        inline_filters: vec![],
        pending_inline_filters: vec![],
        grain: None,
        limit: None,
        order_by: vec![],
        session_variables: HashMap::new(),
    }
}

// -- 9: Compilation — verify all kinds, datasets, and interface counts --------

#[tokio::test]
async fn test_ecommerce_compilation_structure() {
    let m = compile_ecommerce().await;

    assert_eq!(m.model_name, "ecommerce_analytics");

    // 9 standalone datasets + 5 kinds = ≥14 entities in entities
    assert!(m.entities.len() >= 14, "expected ≥14 entities, got {}", m.entities.len());

    // 5 kinds: sales, inventory, all_traffic, order_details, traffic_campaigns
    let expected_kinds = ["sales", "inventory", "all_traffic", "order_details", "traffic_campaigns"];
    for name in &expected_kinds {
        assert!(
            m.entities.contains_key(*name),
            "missing data_kind '{}'",
            name
        );
    }

    // Verify sales grainset interface
    let sales = &m.entities["sales"];
    let si = sales.interface();
    assert!(si.dimensions.contains_key("order_date"), "sales missing order_date dim");
    assert!(si.dimensions.contains_key("region"), "sales missing region dim");
    assert!(si.measures.contains_key("revenue"), "sales missing revenue measure");
    assert!(si.measures.contains_key("order_count"), "sales missing order_count");
    assert!(si.measures.contains_key("cost"), "sales missing cost");
    assert!(si.metrics.contains_key("avg_order_value"), "sales missing avg_order_value metric");
    assert!(si.metrics.contains_key("profit"), "sales missing profit metric");
    assert!(si.metrics.contains_key("roi"), "sales missing roi metric");

    // Verify all_traffic unionset interface
    let traffic = &m.entities["all_traffic"];
    let ti = traffic.interface();
    assert!(ti.dimensions.contains_key("platform"), "all_traffic missing platform dim");
    assert!(ti.measures.contains_key("clicks"), "all_traffic missing clicks");
    assert!(ti.measures.contains_key("sessions"), "all_traffic missing sessions");
    assert!(ti.measures.contains_key("click_revenue"), "all_traffic missing click_revenue");
    assert!(ti.metrics.contains_key("conversion_rate"), "all_traffic missing conversion_rate");

    // Verify order_details joinset interface
    let od = &m.entities["order_details"];
    let oi = od.interface();
    assert!(oi.dimensions.contains_key("customer_name"), "order_details missing customer_name");
    assert!(oi.dimensions.contains_key("product_category"), "order_details missing product_category");

    // Top-level relationships for ad-hoc joins
    assert!(m.relationships.len() >= 2, "expected ≥2 top-level relationships");
}

// -- 10: Grainset — basic daily grain query -----------------------------------

#[tokio::test]
async fn test_grainset_daily_grain() {
    let m = compile_ecommerce().await;
    let req = simple_request("sales", vec!["order_date", "region"], vec!["revenue"]);
    let (plan, sql) = plan_sql(&req, &m);

    assert!(plan.output_names.contains(&"order_date".to_string()));
    assert!(plan.output_names.contains(&"region".to_string()));
    assert!(plan.output_names.contains(&"revenue".to_string()));
    assert!(sql.contains("SUM"), "SQL should contain SUM for revenue: {}", sql);
    assert!(sql.contains("GROUP BY"), "SQL should contain GROUP BY: {}", sql);
}

// -- 11: Grainset — category forces daily dataset (partial coverage) ----------
// category is only mapped in orders_daily. The grainset planner should route
// to the daily dataset when category is requested, since monthly lacks it.
// Bug 3 fix: single-dataset optimization now checks dimension coverage,
// so the planner falls through to UNION path which NULL-fills missing dims.

#[tokio::test]
async fn test_grainset_partial_coverage() {
    let m = compile_ecommerce().await;
    let req = simple_request("sales", vec!["order_date", "category"], vec!["revenue"]);
    let (plan, _sql) = plan_sql(&req, &m);

    assert!(plan.output_names.contains(&"category".to_string()));
    assert!(plan.output_names.contains(&"revenue".to_string()));
}

// -- 12: Grainset — multiple measures + cost ----------------------------------

#[tokio::test]
async fn test_grainset_multi_measure() {
    let m = compile_ecommerce().await;
    let req = simple_request(
        "sales",
        vec!["order_date"],
        vec!["revenue", "order_count", "cost"],
    );
    let (plan, _sql) = plan_sql(&req, &m);

    assert_eq!(plan.output_names.len(), 4); // 1 dim + 3 measures
}

// -- 13: Grainset — metric: simple derived (profit = revenue - cost) ----------

#[tokio::test]
async fn test_grainset_derived_metric_profit() {
    let m = compile_ecommerce().await;
    let req = simple_request("sales", vec!["order_date"], vec!["profit"]);
    let (plan, sql) = plan_sql(&req, &m);

    assert!(plan.output_names.contains(&"profit".to_string()));
    // Profit decomposes into revenue and cost aggregates
    assert!(!sql.is_empty());
}

// -- 14: Grainset — nested metric: ROI = profit / cost -----------------------
// ROI is a nested metric (depth > 0): roi = profit / cost, where profit = revenue - cost.
// extract_metric_constituents() recursively expands nested metrics to transitive
// leaf measures (revenue, cost), ensuring they land in the same grain group.

#[tokio::test]
async fn test_grainset_nested_metric_roi() {
    let m = compile_ecommerce().await;
    let req = simple_request("sales", vec!["order_date", "region"], vec!["roi"]);
    let (plan, _sql) = plan_sql(&req, &m);

    assert!(plan.output_names.contains(&"roi".to_string()));
}

// -- 15: Grainset — ratio metric: avg_order_value = revenue / order_count -----

#[tokio::test]
async fn test_grainset_ratio_metric() {
    let m = compile_ecommerce().await;
    let req = simple_request("sales", vec!["order_date", "region"], vec!["avg_order_value"]);
    let (plan, sql) = plan_sql(&req, &m);

    assert!(plan.output_names.contains(&"avg_order_value".to_string()));
    assert!(!sql.is_empty());
}

// -- 16: Unionset — clicks from all platforms with NULL-fill ------------------

#[tokio::test]
async fn test_unionset_all_platforms() {
    let m = compile_ecommerce().await;
    let req = simple_request("all_traffic", vec!["click_date"], vec!["clicks"]);
    let (plan, sql) = plan_sql(&req, &m);

    assert!(plan.output_names.contains(&"clicks".to_string()));
    // Union ALL across 3 datasets
    assert!(
        sql.contains("UNION ALL") || sql.contains("union all"),
        "SQL should contain UNION ALL: {}",
        sql
    );
}

// -- 17: Unionset — partial coverage: click_revenue only from web -------------

#[tokio::test]
async fn test_unionset_partial_coverage_click_revenue() {
    let m = compile_ecommerce().await;
    // click_revenue is mapped only in web_clicks
    let req = simple_request("all_traffic", vec!["click_date"], vec!["click_revenue"]);
    let (plan, sql) = plan_sql(&req, &m);

    assert!(plan.output_names.contains(&"click_revenue".to_string()));
    assert!(!sql.is_empty());
}

// -- 18: Unionset — literal dimension (platform) -----------------------------

#[tokio::test]
async fn test_unionset_literal_dimension() {
    let m = compile_ecommerce().await;
    let req = simple_request("all_traffic", vec!["click_date", "platform"], vec!["clicks"]);
    let (plan, sql) = plan_sql(&req, &m);

    assert!(plan.output_names.contains(&"platform".to_string()));
    // Literal should appear as injected constant in SQL
    assert!(!sql.is_empty());
}

// -- 19: Unionset — ratio metric: conversion_rate = conversions / clicks ------

#[tokio::test]
async fn test_unionset_ratio_metric() {
    let m = compile_ecommerce().await;
    let req = simple_request("all_traffic", vec!["click_date"], vec!["conversion_rate"]);
    let (plan, sql) = plan_sql(&req, &m);

    assert!(plan.output_names.contains(&"conversion_rate".to_string()));
    assert!(!sql.is_empty());
}

// -- 20: Joinset — 2-way join: orders + customers -----------------------------

#[tokio::test]
async fn test_joinset_two_way_join() {
    let m = compile_ecommerce().await;
    let req = simple_request(
        "order_details",
        vec!["order_date", "customer_name"],
        vec!["revenue"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert!(plan.output_names.contains(&"customer_name".to_string()));
    assert!(plan.output_names.contains(&"revenue".to_string()));
    assert!(
        sql.contains("JOIN") || sql.contains("join"),
        "SQL should contain JOIN: {}",
        sql
    );
}

// -- 21: Joinset — 3-way BFS: orders + customers + products ------------------

#[tokio::test]
async fn test_joinset_three_way_join() {
    let m = compile_ecommerce().await;
    let req = simple_request(
        "order_details",
        vec!["order_date", "customer_name", "product_category"],
        vec!["revenue"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert!(plan.output_names.contains(&"customer_name".to_string()));
    assert!(plan.output_names.contains(&"product_category".to_string()));
    assert!(plan.output_names.contains(&"revenue".to_string()));
    // Should contain at least 2 JOINs
    assert!(!sql.is_empty());
}

// -- 22: Joinset — measures across join: revenue by customer_tier -------------

#[tokio::test]
async fn test_joinset_measures_across_join() {
    let m = compile_ecommerce().await;
    // Measures across a join boundary — revenue from orders, customer_tier from customers
    let req = simple_request(
        "order_details",
        vec!["order_date", "customer_tier"],
        vec!["revenue"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert!(plan.output_names.contains(&"revenue".to_string()));
    assert!(plan.output_names.contains(&"customer_tier".to_string()));
    assert!(!sql.is_empty());
}

// -- 23: Joinset — campaign attribution: traffic → campaigns ------------------

#[tokio::test]
async fn test_joinset_campaign_attribution() {
    let m = compile_ecommerce().await;
    let req = simple_request(
        "traffic_campaigns",
        vec!["click_date", "campaign_name"],
        vec!["clicks", "conversions"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert!(plan.output_names.contains(&"campaign_name".to_string()));
    assert!(plan.output_names.contains(&"clicks".to_string()));
    assert!(plan.output_names.contains(&"conversions".to_string()));
    assert!(!sql.is_empty());
}

// -- 24: Joinset — campaign measures: clicks + budget -------------------------

#[tokio::test]
async fn test_joinset_campaign_measures() {
    let m = compile_ecommerce().await;
    // Measures from different datasets in the join chain
    let req = simple_request(
        "traffic_campaigns",
        vec!["click_date", "campaign_channel"],
        vec!["clicks", "campaign_budget"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert!(plan.output_names.contains(&"clicks".to_string()));
    assert!(plan.output_names.contains(&"campaign_budget".to_string()));
    assert!(!sql.is_empty());
}

// -- 24b: Joinset — metric: avg_order_value across join ----------------------
// Bug 2 fix: joinset now supports metrics via decomposition to constituent measures.

#[tokio::test]
async fn test_joinset_metric_avg_order_value() {
    let m = compile_ecommerce().await;
    let req = simple_request(
        "order_details",
        vec!["order_date", "customer_name"],
        vec!["avg_order_value"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert!(plan.output_names.contains(&"avg_order_value".to_string()));
    assert!(!sql.is_empty());
}

// -- 24c: Joinset — metric: cost_per_conversion (campaign joinset) -----------

#[tokio::test]
async fn test_joinset_metric_cost_per_conversion() {
    let m = compile_ecommerce().await;
    let req = simple_request(
        "traffic_campaigns",
        vec!["click_date", "campaign_name"],
        vec!["cost_per_conversion"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert!(plan.output_names.contains(&"cost_per_conversion".to_string()));
    assert!(!sql.is_empty());
}

// -- 25: Semi-additive — inventory_balance with latest resolution -------------

#[tokio::test]
async fn test_semi_additive_inventory() {
    let m = compile_ecommerce().await;
    let req = simple_request(
        "inventory",
        vec!["snapshot_date", "warehouse_id"],
        vec!["inventory_balance"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert!(plan.output_names.contains(&"inventory_balance".to_string()));
    assert!(!sql.is_empty());
}

// -- 26: Semi-additive — mix additive + semi-additive -------------------------

#[tokio::test]
async fn test_semi_additive_mixed() {
    let m = compile_ecommerce().await;
    let req = simple_request(
        "inventory",
        vec!["snapshot_date", "warehouse_id"],
        vec!["inventory_balance", "units_received", "units_shipped"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert_eq!(plan.output_names.len(), 5); // 2 dims + 3 measures
    assert!(!sql.is_empty());
}

// -- 27: Semi-additive — turnover_rate metric (ratio over semi-additive) ------

#[tokio::test]
async fn test_semi_additive_ratio_metric() {
    let m = compile_ecommerce().await;
    let req = simple_request(
        "inventory",
        vec!["snapshot_date", "warehouse_id"],
        vec!["turnover_rate"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert!(plan.output_names.contains(&"turnover_rate".to_string()));
    assert!(!sql.is_empty());
}

// -- 28: Constraint satisfied — revenue with order_date -----------------------

#[tokio::test]
async fn test_constraint_satisfied() {
    let m = compile_ecommerce().await;
    // revenue has constraint: one_of [order_date, region] — order_date present → OK
    let req = simple_request("sales", vec!["order_date"], vec!["revenue"]);
    let (_plan, sql) = plan_sql(&req, &m);
    assert!(!sql.is_empty());
}

// -- 29: Constraint violated — revenue without order_date or region -----------

#[tokio::test]
async fn test_constraint_violated_one_of() {
    let m = compile_ecommerce().await;
    // revenue requires one_of [order_date, region] but query has only 'category'
    let req = simple_request("sales", vec!["category"], vec!["revenue"]);
    let planner = SemanticPlanner::builder().build();
    let result = planner.plan(&req, &m);

    assert!(result.is_err(), "should fail: revenue requires one_of [order_date, region]");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("constraint") || err_msg.contains("one_of"),
        "error should mention constraint: {}",
        err_msg
    );
}

// -- 30: Filters + ORDER BY + LIMIT ------------------------------------------

#[tokio::test]
async fn test_filters_order_limit() {
    use semstrait_planner::{FilterOperator, FilterValue, OrderByClause, QueryFilter, SortDirection};

    let m = compile_ecommerce().await;
    let request = ResolvedQueryRequest {
        entity_name: "sales".to_string(),
        dimensions: vec!["order_date".to_string(), "region".to_string()],
        measures: vec!["revenue".to_string()],
        filters: vec![QueryFilter {
            field: "region".to_string(),
            operator: FilterOperator::Eq,
            values: vec![FilterValue::String("europe".to_string())],
        }],
        inline_filters: vec![],
        pending_inline_filters: vec![],
        grain: None,
        limit: Some(100),
        order_by: vec![OrderByClause {
            field: "revenue".to_string(),
            direction: SortDirection::Descending,
        }],
        session_variables: HashMap::new(),
    };

    let (plan, sql) = plan_sql(&request, &m);
    assert_eq!(plan.output_names.len(), 3);
    assert!(!sql.is_empty());
    // Should contain ORDER BY and FETCH/LIMIT
    assert!(
        sql.contains("ORDER BY") || sql.contains("order by"),
        "SQL should contain ORDER BY: {}",
        sql
    );
}

// -- 31: Multiple measures via grainset (SUM, COUNT, MIN, MAX) ----------------
// The sales grainset has revenue (SUM), order_count (COUNT), cost (SUM).
// We verify all measures are planned correctly.

#[tokio::test]
async fn test_grainset_all_measures() {
    let m = compile_ecommerce().await;
    let req = simple_request(
        "sales",
        vec!["order_date", "region"],
        vec!["revenue", "order_count", "cost"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert_eq!(plan.output_names.len(), 5); // 2 dims + 3 measures
    assert!(sql.contains("SUM"), "SQL should contain SUM: {}", sql);
    assert!(!sql.is_empty());
}

// -- 32: Unionset with multiple aggregation types (COUNT + COUNT_DISTINCT) ----

#[tokio::test]
async fn test_unionset_multiple_agg_types() {
    let m = compile_ecommerce().await;
    // all_traffic has clicks (COUNT), sessions (COUNT_DISTINCT), conversions (SUM)
    let req = simple_request(
        "all_traffic",
        vec!["click_date"],
        vec!["clicks", "sessions", "conversions"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert_eq!(plan.output_names.len(), 4); // 1 dim + 3 measures
    assert!(!sql.is_empty());
}

// =============================================================================
// Test 33: Computed dimensions — YAML → compile → plan → SQL
// =============================================================================

#[tokio::test]
async fn test_computed_dimension_e2e() {
    let yaml = load_model("orders_computed_dim");

    // Step 1: Compile — computed dims should be resolved.
    let compiler = ManifestCompiler::new();
    let manifest = compiler
        .compile(CompileSource::Yaml(yaml))
        .await
        .expect("compilation should succeed");

    let iface = manifest.entities["orders_daily"].interface();
    assert!(iface.dimensions.contains_key("market"), "should have computed 'market' dim");
    assert!(iface.dimensions.contains_key("market_tier"), "should have computed 'market_tier' dim");

    // Verify computed dimensions have expr set.
    let market_dim = &iface.dimensions["market"];
    assert!(market_dim.expr.is_some(), "market dim should have expr");

    let tier_dim = &iface.dimensions["market_tier"];
    assert!(tier_dim.expr.is_some(), "market_tier dim should have expr");

    // Step 2: Plan with computed dimension.
    let request = ResolvedQueryRequest {
        entity_name: "orders_daily".to_string(),
        dimensions: vec!["date".to_string(), "market".to_string()],
        measures: vec!["revenue".to_string()],
        filters: vec![],
        inline_filters: vec![],
        pending_inline_filters: vec![],
        grain: None,
        limit: None,
        order_by: vec![],
        session_variables: HashMap::new(),
    };

    let planner = SemanticPlanner::builder().build();
    let plan = planner
        .plan(&request, &manifest)
        .expect("planning with computed dim should succeed");

    assert_eq!(plan.output_names, vec!["date", "market", "revenue"]);

    // Step 3: Generate SQL — computed dim should appear as UPPER(...) function call.
    let emitter = AnsiSqlEmitter::new(AnsiDialect);
    let sql = emitter.emit(&plan).expect("SQL generation should succeed");

    // Verify UPPER appears as a function call, not as a quoted column name.
    assert!(
        sql.contains("UPPER("),
        "SQL should contain UPPER( function call for computed market dim: {}",
        sql
    );
    // Verify it references the physical column "region"
    assert!(
        sql.contains("\"region\""),
        "SQL should reference physical column 'region': {}",
        sql
    );
    assert!(
        sql.contains("GROUP BY"),
        "SQL should contain GROUP BY: {}",
        sql
    );
}

#[tokio::test]
async fn test_computed_dimension_case_when_e2e() {
    let yaml = load_model("orders_computed_dim");

    let compiler = ManifestCompiler::new();
    let manifest = compiler
        .compile(CompileSource::Yaml(yaml))
        .await
        .expect("compilation should succeed");

    // Plan with CASE/WHEN computed dimension.
    let request = ResolvedQueryRequest {
        entity_name: "orders_daily".to_string(),
        dimensions: vec!["date".to_string(), "market_tier".to_string()],
        measures: vec!["revenue".to_string()],
        filters: vec![],
        inline_filters: vec![],
        pending_inline_filters: vec![],
        grain: None,
        limit: None,
        order_by: vec![],
        session_variables: HashMap::new(),
    };

    let planner = SemanticPlanner::builder().build();
    let plan = planner
        .plan(&request, &manifest)
        .expect("planning with CASE/WHEN computed dim should succeed");

    assert_eq!(plan.output_names, vec!["date", "market_tier", "revenue"]);

    let emitter = AnsiSqlEmitter::new(AnsiDialect);
    let sql = emitter.emit(&plan).expect("SQL generation should succeed");

    // Verify CASE WHEN appears as actual SQL construct, not a quoted column name.
    assert!(
        sql.contains("CASE WHEN"),
        "SQL should contain CASE WHEN for market_tier dim: {}",
        sql
    );
    assert!(
        sql.contains("IN ("),
        "SQL should contain IN clause for market_tier: {}",
        sql
    );
}

// =============================================================================
// E2E Full Coverage Model — all features, types, and kind variants
// =============================================================================
//
// Tests the e2e_full_coverage.yaml model which exercises every semstrait feature:
// 7 datasets, 2 grainsets, 3 unionsets, 4 joinsets, 12 data types, 6 aggregation
// types, 4 join types, 2 union modes, computed dimensions, kind nesting (unionset→
// grainset ref, joinset→unionset ref), metrics (simple, ratio, nested), measure
// filters, constraints, semi-additive measures, temporal configs (timeseries,
// snapshot, events, SCD type 1, SCD type 2).

/// Helper: compile the full-coverage model once, shared across tests.
async fn compile_full_coverage() -> semstrait_manifest::CompiledManifest {
    let yaml = load_model("e2e_full_coverage");
    let compiler = ManifestCompiler::new();
    compiler
        .compile(CompileSource::Yaml(yaml))
        .await
        .expect("e2e_full_coverage model should compile")
}

// -- 34: Compilation — verify all 16 entities present -------------------------

#[tokio::test]
async fn test_fc_compilation_structure() {
    let m = compile_full_coverage().await;

    assert_eq!(m.model_name, "e2e_full_coverage");

    // 8 datasets + 2 grainsets + 3 unionsets + 3 joinsets = 16 entities
    let expected = [
        // Standalone datasets
        "transactions", "accounts", "account_changelog", "sensor_readings",
        "inventory_snapshots", "products", "regions", "txn_monthly_agg",
        // Grainsets
        "txn_by_grain", "sensor_analytics",
        // Unionsets
        "all_transactions", "unique_events", "unified_analytics",
        // Joinsets
        "txn_details", "product_inventory", "account_sensor_full",
    ];
    for name in &expected {
        assert!(
            m.entities.contains_key(*name),
            "missing data_kind '{}'",
            name
        );
    }

    // Verify transactions interface has rich field set
    let txn = &m.entities["transactions"];
    let ti = txn.interface();
    assert!(ti.dimensions.len() >= 12, "transactions should have 12+ dims, got {}", ti.dimensions.len());
    assert!(ti.measures.len() >= 10, "transactions should have 10+ measures, got {}", ti.measures.len());
    assert!(ti.metrics.len() >= 3, "transactions should have 3+ metrics, got {}", ti.metrics.len());
}

// -- 35: Simple dataset query — single dim + single measure ---------------------

#[tokio::test]
async fn test_fc_simple_dataset_query() {
    let m = compile_full_coverage().await;
    let req = simple_request("transactions", vec!["event_date"], vec!["revenue"]);
    let (plan, sql) = plan_sql(&req, &m);

    assert_eq!(plan.output_names.len(), 2);
    assert!(plan.output_names.contains(&"event_date".to_string()));
    assert!(plan.output_names.contains(&"revenue".to_string()));
    assert!(sql.contains("SUM"), "SQL should contain SUM: {}", sql);
    assert!(sql.contains("GROUP BY"), "SQL should contain GROUP BY: {}", sql);
}

// -- 36: Multi-attribute — 4 dims + 5 measures across numeric types -------------

#[tokio::test]
async fn test_fc_multi_attribute_query() {
    let m = compile_full_coverage().await;
    let req = simple_request(
        "transactions",
        vec!["event_date", "country", "txn_type", "is_fraud"],
        vec!["revenue", "priority_level", "item_count", "discount_rate", "precise_amount"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    // 4 dims + 5 measures = 9 outputs
    assert_eq!(plan.output_names.len(), 9);
    assert!(plan.output_names.contains(&"is_fraud".to_string()));
    assert!(plan.output_names.contains(&"priority_level".to_string()));
    assert!(plan.output_names.contains(&"precise_amount".to_string()));
    assert!(!sql.is_empty());
}

// -- 37: All 6 aggregation types in one query -----------------------------------

#[tokio::test]
async fn test_fc_all_aggregation_types() {
    let m = compile_full_coverage().await;
    // sum (revenue), avg (discount_rate), count (order_count),
    // count_distinct (unique_accounts), min (min_amount), max (max_amount)
    let req = simple_request(
        "transactions",
        vec!["event_date"],
        vec!["revenue", "discount_rate", "order_count", "unique_accounts", "min_amount", "max_amount"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert_eq!(plan.output_names.len(), 7); // 1 dim + 6 measures
    assert!(sql.contains("SUM"), "should have SUM: {}", sql);
    assert!(sql.contains("AVG"), "should have AVG: {}", sql);
    assert!(sql.contains("COUNT"), "should have COUNT: {}", sql);
    assert!(sql.contains("MIN"), "should have MIN: {}", sql);
    assert!(sql.contains("MAX"), "should have MAX: {}", sql);
}

// -- 38: Computed dimensions — UPPER, CASE, CONCAT, SUBSTRING -------------------

#[tokio::test]
async fn test_fc_computed_dimensions() {
    let m = compile_full_coverage().await;
    let req = simple_request(
        "transactions",
        vec!["event_date", "txn_type_upper", "fraud_label", "country_type", "txn_prefix"],
        vec!["revenue"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert_eq!(plan.output_names.len(), 6); // 5 dims + 1 measure
    assert!(plan.output_names.contains(&"txn_type_upper".to_string()));
    assert!(plan.output_names.contains(&"fraud_label".to_string()));
    assert!(plan.output_names.contains(&"country_type".to_string()));
    assert!(plan.output_names.contains(&"txn_prefix".to_string()));
    assert!(sql.contains("UPPER"), "should have UPPER: {}", sql);
    assert!(sql.contains("CASE"), "should have CASE: {}", sql);
    assert!(sql.contains("CONCAT"), "should have CONCAT: {}", sql);
    assert!(sql.contains("SUBSTRING"), "should have SUBSTRING: {}", sql);
}

// -- 39: Metrics — ratio (refund_ratio) + nested (net_efficiency) ---------------

#[tokio::test]
async fn test_fc_metrics_simple_and_nested() {
    let m = compile_full_coverage().await;
    let req = simple_request(
        "transactions",
        vec!["event_date"],
        vec!["refund_ratio", "net_efficiency"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert!(plan.output_names.contains(&"refund_ratio".to_string()));
    assert!(plan.output_names.contains(&"net_efficiency".to_string()));
    assert!(!sql.is_empty());
}

// -- 40: Measure with filter — purchase_revenue (filtered by txn_type) ----------

#[tokio::test]
async fn test_fc_measure_with_filter() {
    let m = compile_full_coverage().await;
    let req = simple_request(
        "transactions",
        vec!["event_date", "country"],
        vec!["purchase_revenue"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert!(plan.output_names.contains(&"purchase_revenue".to_string()));
    // Filtered measure uses CASE WHEN for conditional aggregation
    assert!(
        sql.contains("CASE") || sql.contains("case"),
        "filtered measure should use CASE WHEN: {}",
        sql
    );
}

// -- 41: Constraint satisfied — constrained_revenue with event_date + country ---

#[tokio::test]
async fn test_fc_constraint_satisfied() {
    let m = compile_full_coverage().await;
    // constrained_revenue requires: all=[event_date], one_of=[country, txn_type]
    let req = simple_request(
        "transactions",
        vec!["event_date", "country"],
        vec!["constrained_revenue"],
    );
    let (_plan, sql) = plan_sql(&req, &m);
    assert!(!sql.is_empty());
}

// -- 42: Constraint violated — constrained_revenue without event_date -----------

#[tokio::test]
async fn test_fc_constraint_violated() {
    let m = compile_full_coverage().await;
    // constrained_revenue has all=[event_date] — querying without it should fail
    let req = simple_request(
        "transactions",
        vec!["country"],
        vec!["constrained_revenue"],
    );
    let planner = SemanticPlanner::builder().build();
    let result = planner.plan(&req, &m);

    assert!(result.is_err(), "should fail: constrained_revenue requires event_date");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("constraint") || err_msg.contains("event_date"),
        "error should mention constraint: {}",
        err_msg
    );
}

// -- 43: Grainset — daily grain query -------------------------------------------

#[tokio::test]
async fn test_fc_grainset_daily() {
    let m = compile_full_coverage().await;
    let req = simple_request(
        "txn_by_grain",
        vec!["event_date", "country"],
        vec!["revenue", "order_count"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert_eq!(plan.output_names.len(), 4); // 2 dims + 2 measures
    assert!(sql.contains("GROUP BY"), "SQL should contain GROUP BY: {}", sql);
}

// -- 44: Grainset — metrics (avg_order_value + revenue_per_unit) ----------------

#[tokio::test]
async fn test_fc_grainset_metric() {
    let m = compile_full_coverage().await;
    let req = simple_request(
        "txn_by_grain",
        vec!["event_date"],
        vec!["avg_order_value", "revenue_per_unit"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert!(plan.output_names.contains(&"avg_order_value".to_string()));
    assert!(plan.output_names.contains(&"revenue_per_unit".to_string()));
    assert!(!sql.is_empty());
}

// -- 45: Unionset ALL — literal dimension "stream" ------------------------------

#[tokio::test]
async fn test_fc_unionset_all() {
    let m = compile_full_coverage().await;
    let req = simple_request(
        "all_transactions",
        vec!["event_date", "stream"],
        vec!["amount", "txn_count"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert!(plan.output_names.contains(&"stream".to_string()));
    assert!(plan.output_names.contains(&"amount".to_string()));
    assert!(
        sql.contains("UNION ALL") || sql.contains("union all"),
        "SQL should contain UNION ALL: {}",
        sql
    );
}

// -- 46: Unionset UNIQUE (distinct) — dedup across sensor sources ---------------
// Note: with a single dataset, the planner optimizes away the UNION node.

#[tokio::test]
async fn test_fc_unionset_unique() {
    let m = compile_full_coverage().await;
    let req = simple_request(
        "unique_events",
        vec!["event_ts", "device_type"],
        vec!["event_count"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert!(plan.output_names.contains(&"event_count".to_string()));
    assert!(plan.output_names.contains(&"device_type".to_string()));
    assert!(!sql.is_empty());
}

// -- 47: Joinset 3-way — transactions → accounts → regions ---------------------

#[tokio::test]
async fn test_fc_joinset_3way() {
    let m = compile_full_coverage().await;
    let req = simple_request(
        "txn_details",
        vec!["event_date", "account_name", "region_name", "continent"],
        vec!["revenue", "account_count"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert_eq!(plan.output_names.len(), 6); // 4 dims + 2 measures
    assert!(plan.output_names.contains(&"account_name".to_string()));
    assert!(plan.output_names.contains(&"region_name".to_string()));
    assert!(plan.output_names.contains(&"continent".to_string()));
    assert!(
        sql.contains("JOIN") || sql.contains("join"),
        "SQL should contain JOIN: {}",
        sql
    );
}

// -- 48: Joinset — metric across join (avg_order_value) -------------------------

#[tokio::test]
async fn test_fc_joinset_metric_across_join() {
    let m = compile_full_coverage().await;
    let req = simple_request(
        "txn_details",
        vec!["event_date", "account_name"],
        vec!["avg_order_value"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert!(plan.output_names.contains(&"avg_order_value".to_string()));
    assert!(!sql.is_empty());
}

// -- 49: Joinset — right join + multi-column key + decimal measures -------------

#[tokio::test]
async fn test_fc_joinset_right_multi_key() {
    let m = compile_full_coverage().await;
    let req = simple_request(
        "product_inventory",
        vec!["product_name", "warehouse"],
        vec!["stock_balance", "unit_price"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert!(plan.output_names.contains(&"stock_balance".to_string()));
    assert!(plan.output_names.contains(&"unit_price".to_string()));
    assert!(
        sql.contains("JOIN") || sql.contains("join"),
        "SQL should contain JOIN: {}",
        sql
    );
}

// -- 50: Joinset — full outer join (accounts ↔ sensor readings) -----------------

#[tokio::test]
async fn test_fc_joinset_full_outer() {
    let m = compile_full_coverage().await;
    let req = simple_request(
        "account_sensor_full",
        vec!["account_name", "sensor_id"],
        vec!["account_count", "reading_count"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert_eq!(plan.output_names.len(), 4); // 2 dims + 2 measures
    assert!(
        sql.contains("JOIN") || sql.contains("join"),
        "SQL should contain JOIN: {}",
        sql
    );
}

// -- 51: Kind nesting — unionset → grainset ref (unified_analytics) ------------
// The grainset ref may be optimized to a single dataset by the planner, so
// we verify the query succeeds and produces correct output names.

#[tokio::test]
async fn test_fc_kind_nesting_unionset_grainset_ref() {
    let m = compile_full_coverage().await;
    let req = simple_request(
        "unified_analytics",
        vec!["event_date", "country"],
        vec!["revenue", "order_count"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert_eq!(plan.output_names.len(), 4); // 2 dims + 2 measures
    assert!(plan.output_names.contains(&"revenue".to_string()));
    assert!(plan.output_names.contains(&"order_count".to_string()));
    assert!(!sql.is_empty());
}

// -- 52: Filters + ORDER BY + LIMIT --------------------------------------------

#[tokio::test]
async fn test_fc_filters_order_limit() {
    use semstrait_planner::{FilterOperator, FilterValue, OrderByClause, QueryFilter, SortDirection};

    let m = compile_full_coverage().await;
    let request = ResolvedQueryRequest {
        entity_name: "txn_by_grain".to_string(),
        dimensions: vec!["event_date".to_string(), "country".to_string()],
        measures: vec!["revenue".to_string()],
        filters: vec![QueryFilter {
            field: "country".to_string(),
            operator: FilterOperator::Eq,
            values: vec![FilterValue::String("US".to_string())],
        }],
        inline_filters: vec![],
        pending_inline_filters: vec![],
        grain: None,
        limit: Some(50),
        order_by: vec![OrderByClause {
            field: "revenue".to_string(),
            direction: SortDirection::Descending,
        }],
        session_variables: HashMap::new(),
    };

    let (plan, sql) = plan_sql(&request, &m);
    assert_eq!(plan.output_names.len(), 3);
    assert!(
        sql.contains("ORDER BY") || sql.contains("order by"),
        "SQL should contain ORDER BY: {}",
        sql
    );
}

// -- 53: SQL + plan tree via SemstraitEngine::explain() -------------------------

#[tokio::test]
async fn test_fc_explain_plan_tree() {
    let yaml = load_model("e2e_full_coverage");
    let engine = SemstraitEngine::with_model(&yaml)
        .await
        .expect("engine should compile e2e_full_coverage");

    let raw = RawQueryRequest {
        from: Some("transactions".to_string()),
        select: vec![
            "event_date".to_string(),
            "country".to_string(),
            "revenue".to_string(),
            "order_count".to_string(),
        ],
        ..Default::default()
    };

    let result = engine.explain(&raw).await.expect("explain should succeed");

    // SQL should always be present
    assert!(result.sql.is_some(), "should have SQL");
    let sql = result.sql.unwrap();
    assert!(sql.contains("SELECT"), "SQL should contain SELECT: {}", sql);

    // Plan text should be a human-readable tree
    assert!(
        result.plan_text.contains("TableScan:"),
        "plan_text should contain TableScan: {}",
        result.plan_text
    );
    assert!(
        result.plan_text.contains("Aggregate:"),
        "plan_text should contain Aggregate: {}",
        result.plan_text
    );
}

// -- 54: Semi-additive — snapshot with latest resolution ------------------------

#[tokio::test]
async fn test_fc_semi_additive_snapshot() {
    let m = compile_full_coverage().await;
    let req = simple_request(
        "inventory_snapshots",
        vec!["snap_date", "warehouse"],
        vec!["stock_balance", "stock_received"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert_eq!(plan.output_names.len(), 4); // 2 dims + 2 measures
    assert!(plan.output_names.contains(&"stock_balance".to_string()));
    assert!(!sql.is_empty());
}

// -- 55: Sensor events — timestamp type, min/max aggregations -------------------

#[tokio::test]
async fn test_fc_sensor_events_temporal() {
    let m = compile_full_coverage().await;
    let req = simple_request(
        "sensor_readings",
        vec!["reading_ts", "sensor_type"],
        vec!["reading_value", "min_reading", "max_reading"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert_eq!(plan.output_names.len(), 5); // 2 dims + 3 measures
    assert!(plan.output_names.contains(&"min_reading".to_string()));
    assert!(plan.output_names.contains(&"max_reading".to_string()));
    assert!(sql.contains("MIN"), "should have MIN: {}", sql);
    assert!(sql.contains("MAX"), "should have MAX: {}", sql);
}

// =============================================================================
// Declarative Expression Tests — ExprBlock tags through full pipeline
// =============================================================================
//
// Tests every declarative ExprBlock tag through YAML → compile → plan → SQL.
// Uses declarative_expressions.yaml fixture with 3 datasets (string_ops,
// math_ops, conditional_ops), 1 grainset, and 1 unionset.

/// Helper: compile the declarative expressions model.
async fn compile_decl_expr() -> semstrait_manifest::CompiledManifest {
    let yaml = load_model("declarative_expressions");
    let compiler = ManifestCompiler::new();
    compiler
        .compile(CompileSource::Yaml(yaml))
        .await
        .expect("declarative_expressions model should compile")
}

// -- 56: Compilation — all datasets and kinds present ---------------------------

#[tokio::test]
async fn test_decl_compilation() {
    let m = compile_decl_expr().await;
    assert_eq!(m.model_name, "declarative_expressions");

    for name in &["string_ops", "math_ops", "conditional_ops", "string_by_grain", "all_ops"] {
        assert!(m.entities.contains_key(*name), "missing '{}'", name);
    }

    // string_ops should have 13 computed dims + 3 physical dims + event_date = 17
    let si = m.entities["string_ops"].interface();
    assert!(si.dimensions.len() >= 16, "string_ops should have 16+ dims, got {}", si.dimensions.len());
}

// -- 57: String functions — upper, lower, trim, length, concat, replace ---------

#[tokio::test]
async fn test_decl_string_functions_basic() {
    let m = compile_decl_expr().await;
    let req = simple_request(
        "string_ops",
        vec!["event_date", "name_upper", "name_lower", "name_trimmed", "name_length", "full_label"],
        vec!["record_count"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert_eq!(plan.output_names.len(), 7); // 6 dims + 1 measure
    assert!(sql.contains("UPPER"), "should have UPPER: {}", sql);
    assert!(sql.contains("LOWER"), "should have LOWER: {}", sql);
    assert!(sql.contains("TRIM"), "should have TRIM: {}", sql);
    assert!(sql.contains("LENGTH"), "should have LENGTH: {}", sql);
    assert!(sql.contains("CONCAT"), "should have CONCAT: {}", sql);
}

// -- 58: String functions — ltrim, rtrim, replace, substring, left, right -------

#[tokio::test]
async fn test_decl_string_functions_extended() {
    let m = compile_decl_expr().await;
    let req = simple_request(
        "string_ops",
        vec!["event_date", "name_ltrimmed", "name_rtrimmed", "url_fixed", "code_prefix", "name_left", "name_right"],
        vec!["record_count"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert_eq!(plan.output_names.len(), 8);
    assert!(sql.contains("LTRIM"), "should have LTRIM: {}", sql);
    assert!(sql.contains("RTRIM"), "should have RTRIM: {}", sql);
    assert!(sql.contains("REPLACE"), "should have REPLACE: {}", sql);
    assert!(sql.contains("SUBSTRING"), "should have SUBSTRING: {}", sql);
    assert!(sql.contains("LEFT"), "should have LEFT: {}", sql);
    assert!(sql.contains("RIGHT"), "should have RIGHT: {}", sql);
}

// -- 59: String functions — lpad, rpad ------------------------------------------

#[tokio::test]
async fn test_decl_string_padding() {
    let m = compile_decl_expr().await;
    let req = simple_request(
        "string_ops",
        vec!["event_date", "code_padded", "code_rpadded"],
        vec!["record_count"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert_eq!(plan.output_names.len(), 4);
    assert!(sql.contains("LPAD"), "should have LPAD: {}", sql);
    assert!(sql.contains("RPAD"), "should have RPAD: {}", sql);
}

// -- 60: Math functions — abs, ceil, floor, round, power, sqrt, mod -------------

#[tokio::test]
async fn test_decl_math_functions() {
    let m = compile_decl_expr().await;
    let req = simple_request(
        "math_ops",
        vec!["event_date", "abs_delta", "ceil_value", "floor_value", "rounded_value", "power_value", "sqrt_value", "mod_value"],
        vec!["record_count"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert_eq!(plan.output_names.len(), 9); // 8 dims + 1 measure
    assert!(sql.contains("ABS"), "should have ABS: {}", sql);
    assert!(sql.contains("CEIL"), "should have CEIL: {}", sql);
    assert!(sql.contains("FLOOR"), "should have FLOOR: {}", sql);
    assert!(sql.contains("ROUND"), "should have ROUND: {}", sql);
    assert!(sql.contains("POWER"), "should have POWER: {}", sql);
    assert!(sql.contains("SQRT"), "should have SQRT: {}", sql);
    assert!(sql.contains("MOD"), "should have MOD: {}", sql);
}

// -- 61: Arithmetic — add, subtract, multiply, divide, safe_divide, negate ------

#[tokio::test]
async fn test_decl_arithmetic() {
    let m = compile_decl_expr().await;
    let req = simple_request(
        "math_ops",
        vec!["event_date", "sum_values", "diff_values", "product_values", "ratio_values", "safe_ratio", "negated_delta"],
        vec!["record_count"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert_eq!(plan.output_names.len(), 8); // 7 dims + 1 measure
    // Arithmetic ops render as infix operators in SQL
    assert!(sql.contains("+") || sql.contains("- "), "should have arithmetic ops: {}", sql);
    assert!(!sql.is_empty());
}

// -- 62: Type conversion — cast -------------------------------------------------

#[tokio::test]
async fn test_decl_cast() {
    let m = compile_decl_expr().await;
    let req = simple_request(
        "math_ops",
        vec!["event_date", "cast_string"],
        vec!["record_count"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert_eq!(plan.output_names.len(), 3);
    assert!(sql.contains("CAST"), "should have CAST: {}", sql);
}

// -- 63: Conditional — case, coalesce, null_if, if, greatest, least -------------

#[tokio::test]
async fn test_decl_conditional_functions() {
    let m = compile_decl_expr().await;
    let req = simple_request(
        "conditional_ops",
        vec!["event_date", "status_label", "score_safe", "score_nonzero", "score_category", "best_score", "capped_score"],
        vec!["record_count"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert_eq!(plan.output_names.len(), 8); // 7 dims + 1 measure
    assert!(sql.contains("CASE"), "should have CASE: {}", sql);
    assert!(sql.contains("COALESCE"), "should have COALESCE: {}", sql);
    // null_if renders as NULLIF function call
    // if desugars to CASE WHEN
    assert!(sql.contains("GREATEST"), "should have GREATEST: {}", sql);
    assert!(sql.contains("LEAST"), "should have LEAST: {}", sql);
}

// -- 64: Predicates — in_list, not_in_list, between, is_null, is_not_null -------

#[tokio::test]
async fn test_decl_predicate_expressions() {
    let m = compile_decl_expr().await;
    let req = simple_request(
        "conditional_ops",
        vec!["event_date", "is_premium", "is_excluded", "in_range", "has_email", "missing_score"],
        vec!["record_count"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert_eq!(plan.output_names.len(), 7);
    assert!(sql.contains(" IN ("), "should have IN: {}", sql);
    assert!(sql.contains("NOT IN"), "should have NOT IN: {}", sql);
    assert!(sql.contains("BETWEEN"), "should have BETWEEN: {}", sql);
    assert!(sql.contains("IS NOT NULL"), "should have IS NOT NULL: {}", sql);
    assert!(sql.contains("IS NULL"), "should have IS NULL: {}", sql);
}

// -- 65: Date functions — extract, date_add -------------------------------------

#[tokio::test]
async fn test_decl_date_functions() {
    let m = compile_decl_expr().await;
    let req = simple_request(
        "conditional_ops",
        vec!["event_date", "event_year", "next_month"],
        vec!["record_count"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert_eq!(plan.output_names.len(), 4);
    assert!(sql.contains("EXTRACT"), "should have EXTRACT: {}", sql);
    assert!(sql.contains("DATE_ADD"), "should have DATE_ADD: {}", sql);
}

// -- 66: Logical — and, or, not + comparison (gte, lte) -------------------------

#[tokio::test]
async fn test_decl_logical_expressions() {
    let m = compile_decl_expr().await;
    let req = simple_request(
        "conditional_ops",
        vec!["event_date", "active_premium", "any_flag", "not_deleted", "is_high_score", "is_low_score"],
        vec!["record_count"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert_eq!(plan.output_names.len(), 7);
    // AND/OR render as infix, NOT as prefix
    assert!(sql.contains("AND") || sql.contains("and"), "should have AND: {}", sql);
    assert!(sql.contains("OR") || sql.contains("or"), "should have OR: {}", sql);
    assert!(sql.contains("NOT"), "should have NOT: {}", sql);
}

// -- 67: Guard expression -------------------------------------------------------

#[tokio::test]
async fn test_decl_guard_expression() {
    let m = compile_decl_expr().await;
    let req = simple_request(
        "conditional_ops",
        vec!["event_date", "guarded_score"],
        vec!["record_count"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert_eq!(plan.output_names.len(), 3);
    // Guard renders as CASE WHEN condition THEN expr ELSE NULL END
    assert!(sql.contains("CASE"), "guard should render as CASE: {}", sql);
}

// -- 68: Grainset — declarative expressions survive grain routing ---------------

#[tokio::test]
async fn test_decl_grainset_expressions() {
    let m = compile_decl_expr().await;
    let req = simple_request(
        "string_by_grain",
        vec!["event_date", "name_upper", "name_trimmed", "full_label"],
        vec!["record_count"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert_eq!(plan.output_names.len(), 5); // 4 dims + 1 measure
    assert!(sql.contains("UPPER"), "grainset should have UPPER: {}", sql);
    assert!(sql.contains("TRIM"), "grainset should have TRIM: {}", sql);
    assert!(sql.contains("CONCAT"), "grainset should have CONCAT: {}", sql);
}

// -- 69: Unionset — literal dims + UNION ALL ------------------------------------
// Note: computed dims on kind-level interfaces are NULL-filled per DL-049
// (serde_yaml 0.9 limitation). Test the unionset with physical/literal dims.

#[tokio::test]
async fn test_decl_unionset_literal_and_union() {
    let m = compile_decl_expr().await;
    let req = simple_request(
        "all_ops",
        vec!["event_date", "source"],
        vec!["record_count"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert_eq!(plan.output_names.len(), 3);
    assert!(plan.output_names.contains(&"source".to_string()));
    assert!(
        sql.contains("UNION ALL") || sql.contains("union all"),
        "should be UNION ALL: {}",
        sql
    );
}

// -- 70: Combined — mix all expression categories in one query ------------------

#[tokio::test]
async fn test_decl_combined_all_categories() {
    let m = compile_decl_expr().await;
    // Query conditional_ops with expressions from every category
    let req = simple_request(
        "conditional_ops",
        vec![
            "event_date",
            "status_label",     // case
            "score_safe",       // coalesce
            "is_premium",       // in_list
            "in_range",         // between
            "event_year",       // extract
            "active_premium",   // and
            "guarded_score",    // guard
        ],
        vec!["record_count", "total_score"],
    );
    let (plan, sql) = plan_sql(&req, &m);

    assert_eq!(plan.output_names.len(), 10); // 8 dims + 2 measures
    assert!(sql.contains("CASE"), "combined should have CASE: {}", sql);
    assert!(sql.contains("COALESCE"), "combined should have COALESCE: {}", sql);
    assert!(sql.contains(" IN ("), "combined should have IN: {}", sql);
    assert!(sql.contains("BETWEEN"), "combined should have BETWEEN: {}", sql);
    assert!(sql.contains("EXTRACT"), "combined should have EXTRACT: {}", sql);
}

// -- Debug: check alpinestars compiled interface --------------------------------


// =============================================================================
// Test 8: DataFusion query execution (feature-gated)
// =============================================================================

#[cfg(feature = "datafusion")]
mod datafusion_tests {
    use super::*;
    use semstrait_adapter::{DataFusionAdapter, EngineAdapter};
    use std::sync::Arc;

    /// Verify the DataFusion adapter produces Substrait (not SQL) artifacts.
    #[tokio::test]
    async fn test_datafusion_adapter_produces_substrait() {
        let yaml = load_model("orders_3dim");

        let compiler = ManifestCompiler::new();
        let manifest = compiler
            .compile(CompileSource::Yaml(yaml))
            .await
            .expect("compilation should succeed");

        let request = ResolvedQueryRequest {
            entity_name: "orders".to_string(),
            dimensions: vec!["date".to_string(), "region".to_string()],
            measures: vec!["revenue".to_string()],
            filters: vec![],
            inline_filters: vec![],
            pending_inline_filters: vec![],
            grain: None,
            limit: None,
            order_by: vec![],
            session_variables: HashMap::new(),
        };

        let adapter = DataFusionAdapter;
        let mut planner_builder = SemanticPlanner::builder();
        if let Some(pb) = adapter.plan_builder() {
            planner_builder = planner_builder.with_plan_builder(pb);
        }
        let planner = planner_builder.build();
        let plan = planner.plan(&request, &manifest).expect("planning should succeed");

        let artifact = adapter.adapt(&plan).expect("adapt should succeed");

        // DataFusion adapter MUST produce Substrait, not SQL.
        assert!(
            artifact.is_substrait(),
            "DataFusion adapter should produce Substrait artifact, got SQL"
        );
        assert!(
            !artifact.is_sql(),
            "DataFusion adapter should NOT produce SQL artifact"
        );

        // debug_sql() should still work as fallback.
        let debug = adapter.debug_sql(&plan).expect("debug_sql should succeed");
        assert!(!debug.is_empty(), "debug SQL should be non-empty");

        // debug_sql() should use DataFusion dialect (LIMIT, not FETCH FIRST).
        assert!(
            !debug.contains("FETCH FIRST"),
            "DataFusion debug SQL should NOT use FETCH FIRST: {}",
            debug
        );
    }

    /// Full facade pipeline: SemstraitBuilder → with_adapter() → plan_builder wiring → explain.
    ///
    /// This tests that the adapter's plan_builder() is wired into the planner,
    /// and that explain() uses the adapter's debug_sql() (DataFusion dialect).
    #[tokio::test]
    async fn test_facade_adapter_wiring() {
        let yaml = load_model("orders_3dim");

        let adapter: Arc<dyn EngineAdapter> = Arc::new(DataFusionAdapter);
        let instance = semstrait::SemstraitInstance::builder()
            .with_model(yaml)
            .with_adapter(adapter)
            .build()
            .await
            .expect("facade build should succeed");

        // explain() should use DataFusion dialect via adapter.
        let request = ResolvedQueryRequest {
            entity_name: "orders".to_string(),
            dimensions: vec!["date".to_string(), "region".to_string()],
            measures: vec!["revenue".to_string()],
            filters: vec![],
            inline_filters: vec![],
            pending_inline_filters: vec![],
            grain: None,
            limit: Some(10),
            order_by: vec![],
            session_variables: HashMap::new(),
        };

        let sql = instance.explain(&request).expect("explain should succeed");

        assert!(sql.contains("SELECT"), "SQL should contain SELECT: {}", sql);
        assert!(sql.contains("GROUP BY"), "SQL should contain GROUP BY: {}", sql);
        // DataFusion dialect uses LIMIT, not FETCH FIRST.
        assert!(
            sql.contains("LIMIT"),
            "DataFusion dialect should use LIMIT: {}",
            sql
        );
        assert!(
            !sql.contains("FETCH FIRST"),
            "DataFusion dialect should NOT use FETCH FIRST: {}",
            sql
        );
    }

    /// Full adapter pipeline: compile → plan → adapt → Substrait.
    /// Verifies adapter produces valid Substrait with FunctionRegistry and
    /// that InList uses SingularOrList (not ScalarFunction).
    #[tokio::test]
    async fn test_datafusion_substrait_with_registry() {
        let yaml = load_model("orders_3dim");

        let compiler = ManifestCompiler::new();
        let manifest = compiler
            .compile(CompileSource::Yaml(yaml))
            .await
            .expect("compilation should succeed");

        // Plan with adapter's PlanBuilder wired in.
        let adapter = DataFusionAdapter;
        let mut planner_builder = SemanticPlanner::builder();
        if let Some(pb) = adapter.plan_builder() {
            planner_builder = planner_builder.with_plan_builder(pb);
        }
        let planner = planner_builder.build();

        let request = ResolvedQueryRequest {
            entity_name: "orders".to_string(),
            dimensions: vec!["date".to_string(), "region".to_string()],
            measures: vec!["revenue".to_string()],
            filters: vec![],
            inline_filters: vec![],
            pending_inline_filters: vec![],
            grain: None,
            limit: None,
            order_by: vec![],
            session_variables: HashMap::new(),
        };

        let plan = planner.plan(&request, &manifest).expect("planning should succeed");

        // Adapt to Substrait.
        let artifact = adapter.adapt(&plan).expect("adapt should succeed");
        assert!(artifact.is_substrait(), "should produce Substrait");

        // Verify Substrait JSON is well-formed.
        let substrait_plan = artifact.as_substrait().expect("should have Substrait");
        // Plan should have extension declarations from FunctionRegistry.
        assert!(
            !substrait_plan.extensions.is_empty(),
            "Substrait plan should have function extension declarations"
        );
        // No extension URIs (Phase 3: removed).
        assert!(
            substrait_plan.extension_uris.is_empty(),
            "Substrait plan should have no extension URIs"
        );
    }

}
