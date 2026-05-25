//! Temporary diagnostic: dumps generated SQL for all declarative expression
//! and e2e_full_coverage queries to verify functional correctness.

mod test_helpers;

use semstrait_manifest::{CompileSource, ManifestCompiler};
use semstrait_planner::{ResolvedQueryRequest, SemanticPlanner};
use semstrait_adapter::sql::{AnsiDialect, AnsiSqlEmitter, SqlEmitter};
use std::collections::HashMap;
use test_helpers::load_model;

async fn compile(name: &str) -> semstrait_manifest::CompiledManifest {
    let yaml = load_model(name);
    ManifestCompiler::new()
        .compile(CompileSource::Yaml(yaml))
        .await
        .unwrap()
}

fn sql(
    entity: &str,
    dims: &[&str],
    measures: &[&str],
    m: &semstrait_manifest::CompiledManifest,
) -> String {
    let req = ResolvedQueryRequest {
        entity_name: entity.to_string(),
        dimensions: dims.iter().map(|s| s.to_string()).collect(),
        measures: measures.iter().map(|s| s.to_string()).collect(),
        filters: vec![],
        inline_filters: vec![],
        grain: None,
        limit: None,
        order_by: vec![],
        session_variables: HashMap::new(),
    };
    let planner = SemanticPlanner::builder().build();
    let plan = planner.plan(&req, m).unwrap();
    let emitter = AnsiSqlEmitter::new(AnsiDialect);
    emitter.emit(&plan).unwrap()
}

#[tokio::test]
async fn dump_all_sql() {
    let m = compile("declarative_expressions").await;

    println!("\n=== STRING: upper, lower, trim, length, concat ===");
    println!("{}", sql("string_ops", &["event_date", "name_upper", "name_lower", "name_trimmed", "name_length", "full_label"], &["record_count"], &m));

    println!("\n=== STRING: ltrim, rtrim, replace, substring, left, right ===");
    println!("{}", sql("string_ops", &["event_date", "name_ltrimmed", "name_rtrimmed", "url_fixed", "code_prefix", "name_left", "name_right"], &["record_count"], &m));

    println!("\n=== STRING: lpad, rpad ===");
    println!("{}", sql("string_ops", &["event_date", "code_padded", "code_rpadded"], &["record_count"], &m));

    println!("\n=== MATH: abs, ceil, floor, round, power, sqrt, mod ===");
    println!("{}", sql("math_ops", &["event_date", "abs_delta", "ceil_value", "floor_value", "rounded_value", "power_value", "sqrt_value", "mod_value"], &["record_count"], &m));

    println!("\n=== ARITHMETIC: add, subtract, multiply, divide, safe_divide, negate ===");
    println!("{}", sql("math_ops", &["event_date", "sum_values", "diff_values", "product_values", "ratio_values", "safe_ratio", "negated_delta"], &["record_count"], &m));

    println!("\n=== TYPE: cast ===");
    println!("{}", sql("math_ops", &["event_date", "cast_string"], &["record_count"], &m));

    println!("\n=== CONDITIONAL: case, coalesce, null_if, if, greatest, least ===");
    println!("{}", sql("conditional_ops", &["event_date", "status_label", "score_safe", "score_nonzero", "score_category", "best_score", "capped_score"], &["record_count"], &m));

    println!("\n=== PREDICATE: in_list, not_in_list, between, is_null, is_not_null ===");
    println!("{}", sql("conditional_ops", &["event_date", "is_premium", "is_excluded", "in_range", "has_email", "missing_score"], &["record_count"], &m));

    println!("\n=== DATE: extract, date_add ===");
    println!("{}", sql("conditional_ops", &["event_date", "event_year", "next_month"], &["record_count"], &m));

    println!("\n=== LOGICAL: and, or, not, gte, lte ===");
    println!("{}", sql("conditional_ops", &["event_date", "active_premium", "any_flag", "not_deleted", "is_high_score", "is_low_score"], &["record_count"], &m));

    println!("\n=== GUARD ===");
    println!("{}", sql("conditional_ops", &["event_date", "guarded_score"], &["record_count"], &m));

    println!("\n=== GRAINSET with expressions ===");
    println!("{}", sql("string_by_grain", &["event_date", "name_upper", "name_trimmed", "full_label"], &["record_count"], &m));

    println!("\n=== UNIONSET ===");
    println!("{}", sql("all_ops", &["event_date", "source"], &["record_count"], &m));

    // E2E full coverage
    let fc = compile("e2e_full_coverage").await;

    println!("\n=== FC: simple dataset ===");
    println!("{}", sql("transactions", &["event_date"], &["revenue"], &fc));

    println!("\n=== FC: multi-attribute ===");
    println!("{}", sql("transactions", &["event_date", "country", "txn_type"], &["revenue", "precise_amount", "discount_rate"], &fc));

    println!("\n=== FC: all agg types ===");
    println!("{}", sql("transactions", &["event_date"], &["revenue", "discount_rate", "order_count", "unique_accounts", "min_amount", "max_amount"], &fc));

    println!("\n=== FC: computed dims (inline) ===");
    println!("{}", sql("transactions", &["event_date", "txn_type_upper", "fraud_label", "country_type", "txn_prefix"], &["revenue"], &fc));

    println!("\n=== FC: metrics (ratio + nested) ===");
    println!("{}", sql("transactions", &["event_date"], &["refund_ratio", "net_efficiency"], &fc));

    println!("\n=== FC: measure filter ===");
    println!("{}", sql("transactions", &["event_date", "country"], &["purchase_revenue"], &fc));

    println!("\n=== FC: grainset ===");
    println!("{}", sql("txn_by_grain", &["event_date", "country"], &["revenue", "order_count"], &fc));

    println!("\n=== FC: unionset ALL ===");
    println!("{}", sql("all_transactions", &["event_date", "stream"], &["amount", "txn_count"], &fc));

    println!("\n=== FC: joinset 3-way ===");
    println!("{}", sql("txn_details", &["event_date", "account_name", "region_name"], &["revenue"], &fc));

    println!("\n=== FC: joinset right+multikey ===");
    println!("{}", sql("product_inventory", &["product_name", "warehouse"], &["stock_balance", "unit_price"], &fc));

    println!("\n=== FC: joinset full outer ===");
    println!("{}", sql("account_sensor_full", &["account_name", "sensor_id"], &["account_count", "reading_count"], &fc));

    println!("\n=== FC: semi-additive ===");
    println!("{}", sql("inventory_snapshots", &["snap_date", "warehouse"], &["stock_balance", "stock_received"], &fc));

    println!("\n=== FC: sensor events ===");
    println!("{}", sql("sensor_readings", &["reading_ts", "sensor_type"], &["reading_value", "min_reading", "max_reading"], &fc));
}
