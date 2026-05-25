//! End-to-end pipeline tests: YAML → ManifestCompiler → SemanticPlanner → SQL.
//!
//! Exercises the full stack through the facade crate's dependencies.
//! Model definitions are loaded from `tests/fixtures/models/` at the workspace root.

use std::collections::HashMap;

use semstrait_manifest::{CompileSource, ManifestCompiler};
use semstrait_planner::{
    FilterOperator, FilterValue, OrderByClause, QueryFilter, ResolvedQueryRequest,
    SemanticPlanner, SortDirection,
};
use semstrait_adapter::sql::{AnsiDialect, AnsiSqlEmitter, SqlEmitter};

/// Load a test fixture YAML model by name (without extension).
fn load_model(name: &str) -> String {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let path = format!("{}/../../tests/fixtures/models/{}.yaml", manifest_dir, name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to load fixture '{}': {}", path, e))
}

fn make_request(
    kind: &str,
    dims: &[&str],
    measures: &[&str],
) -> ResolvedQueryRequest {
    ResolvedQueryRequest {
        entity_name: kind.to_string(),
        dimensions: dims.iter().map(|s| s.to_string()).collect(),
        measures: measures.iter().map(|s| s.to_string()).collect(),
        filters: vec![],
        inline_filters: vec![],
        grain: None,
        limit: None,
        order_by: vec![],
        session_variables: HashMap::new(),
    }
}

// ============================================================================
// Full pipeline: YAML → compile → plan → SQL
// ============================================================================

#[tokio::test]
async fn e2e_compile_plan_sql() {
    let yaml = load_model("orders_basic");

    // Step 1: Compile YAML → CompiledManifest
    let compiler = ManifestCompiler::new();
    let manifest = compiler
        .compile(CompileSource::Yaml(yaml))
        .await
        .expect("compilation should succeed");

    assert_eq!(manifest.model_name, "e2e_test");
    assert_eq!(manifest.entities.len(), 1);
    assert!(manifest.entities.contains_key("orders"));

    // Step 2: Plan a query
    let planner = SemanticPlanner::builder().build();
    let request = make_request("orders", &["order_date", "region"], &["revenue"]);

    let plan = planner
        .plan(&request, &manifest)
        .expect("planning should succeed");

    assert_eq!(plan.output_names, vec!["order_date", "region", "revenue"]);

    // Step 3: Emit SQL
    let emitter = AnsiSqlEmitter::new(AnsiDialect);
    let sql = emitter.emit(&plan).expect("SQL emission should succeed");

    // Verify the SQL contains expected elements
    let sql_upper = sql.to_uppercase();
    assert!(sql_upper.contains("SELECT"), "SQL should contain SELECT: {}", sql);
    assert!(sql_upper.contains("FROM"), "SQL should contain FROM: {}", sql);
    assert!(
        sql_upper.contains("ORDERS_DAILY"),
        "SQL should reference the table: {}",
        sql
    );
}

#[tokio::test]
async fn e2e_compile_plan_sql_with_filters() {
    let yaml = load_model("orders_basic");

    let compiler = ManifestCompiler::new();
    let manifest = compiler
        .compile(CompileSource::Yaml(yaml))
        .await
        .expect("compilation should succeed");

    let planner = SemanticPlanner::builder().build();
    let mut request = make_request("orders", &["order_date"], &["revenue"]);
    request.filters.push(QueryFilter {
        field: "region".to_string(),
        operator: FilterOperator::Eq,
        values: vec![FilterValue::String("US".to_string())],
    });

    let plan = planner
        .plan(&request, &manifest)
        .expect("planning with filter should succeed");

    let emitter = AnsiSqlEmitter::new(AnsiDialect);
    let sql = emitter.emit(&plan).expect("SQL emission should succeed");

    let sql_upper = sql.to_uppercase();
    assert!(sql_upper.contains("WHERE"), "Filtered SQL should contain WHERE: {}", sql);
}

#[tokio::test]
async fn e2e_compile_plan_sql_with_inline_filter() {
    // Inline request-scope filter (CompiledFilter on `inline_filters`) rides
    // the same scan-layer engine as a named DataKind filter — see
    // `docs/design/foundations/11_names_and_scopes.md §6.4.2`.
    let yaml = load_model("orders_basic");

    let compiler = ManifestCompiler::new();
    let manifest = compiler
        .compile(CompileSource::Yaml(yaml))
        .await
        .expect("compilation should succeed");

    let planner = SemanticPlanner::builder().build();
    let mut request = make_request("orders", &["order_date"], &["revenue"]);
    request.inline_filters = vec![semstrait_manifest::CompiledFilter {
        name: "__inline_filter_0".to_string(),
        expr: semstrait_core::Expr::eq(
            semstrait_core::Expr::entity_ref("region"),
            semstrait_core::Expr::string("US"),
        ),
        expr_source: "region = 'US'".to_string(),
    }];

    let plan = planner
        .plan(&request, &manifest)
        .expect("planning with inline filter should succeed");

    let emitter = AnsiSqlEmitter::new(AnsiDialect);
    let sql = emitter.emit(&plan).expect("SQL emission should succeed");

    let sql_upper = sql.to_uppercase();
    assert!(
        sql_upper.contains("WHERE"),
        "inline-filtered SQL should contain WHERE: {}",
        sql
    );
    assert!(
        sql.contains("'US'"),
        "inline filter literal should appear in SQL: {}",
        sql
    );
}

#[tokio::test]
async fn e2e_compile_plan_sql_with_order_and_limit() {
    let yaml = load_model("orders_basic");

    let compiler = ManifestCompiler::new();
    let manifest = compiler
        .compile(CompileSource::Yaml(yaml))
        .await
        .expect("compilation should succeed");

    let planner = SemanticPlanner::builder().build();
    let mut request = make_request("orders", &["region"], &["revenue"]);
    request.order_by.push(OrderByClause {
        field: "revenue".to_string(),
        direction: SortDirection::Descending,
    });
    request.limit = Some(10);

    let plan = planner
        .plan(&request, &manifest)
        .expect("planning with order+limit should succeed");

    let emitter = AnsiSqlEmitter::new(AnsiDialect);
    let sql = emitter.emit(&plan).expect("SQL emission should succeed");

    let sql_upper = sql.to_uppercase();
    assert!(
        sql_upper.contains("ORDER BY"),
        "SQL should contain ORDER BY: {}",
        sql
    );
    assert!(
        sql_upper.contains("LIMIT") || sql_upper.contains("FETCH"),
        "SQL should contain LIMIT or FETCH: {}",
        sql
    );
}

// ============================================================================
// Constraint violation through full pipeline
// ============================================================================

#[tokio::test]
async fn e2e_constraint_violation() {
    let yaml = load_model("orders_constrained");

    let compiler = ManifestCompiler::new();
    let manifest = compiler
        .compile(CompileSource::Yaml(yaml))
        .await
        .expect("compilation should succeed");

    let planner = SemanticPlanner::builder().build();
    // Request without order_date — violates one_of constraint
    let request = make_request("orders", &["region"], &["revenue"]);

    let result = planner.plan(&request, &manifest);
    assert!(result.is_err(), "should fail with constraint violation");

    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("one_of") || msg.contains("constraint"),
        "error should mention constraint: {}",
        msg
    );
}

#[tokio::test]
async fn e2e_constraint_satisfied() {
    let yaml = load_model("orders_constrained");

    let compiler = ManifestCompiler::new();
    let manifest = compiler
        .compile(CompileSource::Yaml(yaml))
        .await
        .expect("compilation should succeed");

    let planner = SemanticPlanner::builder().build();
    // Request with order_date — satisfies one_of constraint
    let request = make_request("orders", &["order_date", "region"], &["revenue"]);

    let plan = planner
        .plan(&request, &manifest)
        .expect("should succeed when constraint is satisfied");

    assert_eq!(
        plan.output_names,
        vec!["order_date", "region", "revenue"]
    );
}

// ============================================================================
// Error cases through full pipeline
// ============================================================================

#[tokio::test]
async fn e2e_kind_not_found() {
    let yaml = load_model("orders_basic");

    let compiler = ManifestCompiler::new();
    let manifest = compiler
        .compile(CompileSource::Yaml(yaml))
        .await
        .expect("compilation should succeed");

    let planner = SemanticPlanner::builder().build();
    let request = make_request("nonexistent_kind", &["date"], &["revenue"]);

    let result = planner.plan(&request, &manifest);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not found"));
}

#[tokio::test]
async fn e2e_raw_sql_rejected_at_compile() {
    let yaml = load_model("raw_sql_invalid");

    let compiler = ManifestCompiler::new();
    let result = compiler
        .compile(CompileSource::Yaml(yaml))
        .await;

    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("raw SQL rejected"),
        "should reject raw SQL: {}",
        msg
    );
}

#[tokio::test]
async fn e2e_multiple_measures() {
    let yaml = load_model("orders_basic");

    let compiler = ManifestCompiler::new();
    let manifest = compiler
        .compile(CompileSource::Yaml(yaml))
        .await
        .expect("compilation should succeed");

    let planner = SemanticPlanner::builder().build();
    let request = make_request("orders", &["order_date"], &["revenue", "order_count"]);

    let plan = planner
        .plan(&request, &manifest)
        .expect("multi-measure query should succeed");

    assert_eq!(
        plan.output_names,
        vec!["order_date", "revenue", "order_count"]
    );

    let emitter = AnsiSqlEmitter::new(AnsiDialect);
    let sql = emitter.emit(&plan).expect("SQL emission should succeed");
    assert!(!sql.is_empty(), "SQL should not be empty");
}
