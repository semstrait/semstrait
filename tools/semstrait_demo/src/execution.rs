// DataFusion execution
use datafusion::arrow::record_batch::RecordBatch;
use semstrait::{Schema, QueryRequest};
use substrait::proto::Plan;
use blake3::Hasher;
use serde_json;
use super::ReproducibilityParams;

/// Execute a Substrait plan using DataFusion (now implemented via datafusion_execution)
/// This function is kept for backwards compatibility but delegates to datafusion_execution
pub async fn execute_substrait_plan_via_df_exec(
    ctx: &datafusion::prelude::SessionContext,
    plan_node: &semstrait::plan::PlanNode,
    table_paths: &std::collections::HashMap<String, String>,
) -> anyhow::Result<Vec<RecordBatch>> {
    use crate::datafusion_execution;
    let df = datafusion_execution::execute_plan_node(ctx, plan_node, table_paths).await?;
    let batches = df.collect().await?;
    Ok(batches)
}

/// Print execution results from DataFusion RecordBatches
pub fn print_execution_results(results: &[RecordBatch]) -> anyhow::Result<()> {
    println!("📊 EXECUTION RESULTS");
    println!("===================");

    if results.is_empty() {
        println!("  (No results returned)");
        return Ok(());
    }

    let total_rows: usize = results.iter().map(|batch| batch.num_rows()).sum();
    println!("  Rows: {}", total_rows);
    println!("  Batches: {}", results.len());

    // Print column names from first batch
    if let Some(first_batch) = results.first() {
        println!("  Columns: {}", first_batch.num_columns());
        if !first_batch.schema().fields().is_empty() {
            println!("  Schema: {}",
                first_batch.schema().fields().iter()
                    .map(|f| format!("{}:{}", f.name(), f.data_type()))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }

        // Print first few rows for preview
        let preview_rows = std::cmp::min(10, first_batch.num_rows());
        if preview_rows > 0 {
            println!("  Preview (first {} rows):", preview_rows);

            // Print header
            print!("    ");
            for field in first_batch.schema().fields() {
                print!("| {:<15} ", field.name());
            }
            println!("|");

            // Print separator
            print!("    ");
            for _ in first_batch.schema().fields() {
                print!("+-----------------");
            }
            println!("+");

            // Print data rows
            for row_idx in 0..preview_rows {
                print!("    ");
                for col_idx in 0..first_batch.num_columns() {
                    let col = first_batch.column(col_idx);
                    let value_str = format!("{}", col.as_any().downcast_ref::<arrow::array::StringArray>()
                        .and_then(|arr| arr.value(row_idx).to_string().into())
                        .or_else(|| col.as_any().downcast_ref::<arrow::array::Float64Array>()
                            .and_then(|arr| arr.value(row_idx).to_string().into()))
                        .or_else(|| col.as_any().downcast_ref::<arrow::array::Int64Array>()
                            .and_then(|arr| arr.value(row_idx).to_string().into()))
                        .unwrap_or_else(|| "<unknown>".to_string()));
                    print!("| {:<15} ", &value_str[..std::cmp::min(15, value_str.len())]);
                }
                println!("|");
            }
        }
    }

    Ok(())
}

/// Compute a stable snapshot ID for reproducibility
pub async fn compute_snapshot_id(
    schema: &Schema,
    request: &QueryRequest,
    repro_params: &ReproducibilityParams,
    substrait_plan: &Plan,
    table_paths: &std::collections::HashMap<String, String>,
) -> anyhow::Result<String> {
    let mut hasher = Hasher::new();

    // Include the embedded model YAML (deterministic)
    let schema_yaml = include_str!("../model.yaml");
    hasher.update(schema_yaml.as_bytes());

    // Include canonical request JSON
    let request_json = serde_json::to_string(request)?;
    hasher.update(request_json.as_bytes());

    // Include reproducibility parameters
    let repro_json = serde_json::to_string(repro_params)?;
    hasher.update(repro_json.as_bytes());

    // Include Substrait plan (deterministic serialization)
    let plan_json = serde_json::to_string(substrait_plan)?;
    hasher.update(plan_json.as_bytes());

    // Include data-state fingerprint (rowcount + max_event_time + schema hash per table)
    let snapshot_store = crate::snapshot_store::SnapshotStore::new();
    let data_fingerprint = snapshot_store.compute_data_fingerprint(table_paths).await?;
    hasher.update(data_fingerprint.as_bytes());

    // Generate hash
    let hash = hasher.finalize();
    Ok(hash.to_hex().to_string())
}

/// Print reconciliation analysis between semstrait results and platform SQL
pub fn print_reconciliation(
    semstrait_results: &[RecordBatch],
    adwords_path: &str,
    facebook_path: &str,
) -> anyhow::Result<()> {
    println!("🔍 Platform Baseline vs Semstrait Results");
    println!("=========================================");

    // For now, show semstrait results and note that platform baseline would be computed similarly
    // In a full implementation, we'd run separate DataFusion queries against raw tables
    println!("📊 Semstrait Results:");
    if let Some(first_batch) = semstrait_results.first() {
        if first_batch.num_rows() > 0 {
            // Print results in a table format
            print!("  |");
            for field in first_batch.schema().fields() {
                print!(" {:<15} |", field.name());
            }
            println!();
            print!("  +");
            for _ in first_batch.schema().fields() {
                print!("-----------------+");
            }
            println!("+");

            for row_idx in 0..first_batch.num_rows().min(10) {
                print!("  |");
                for col_idx in 0..first_batch.num_columns() {
                    let col = first_batch.column(col_idx);
                    let value_str = format!("{}", col.as_any().downcast_ref::<arrow::array::StringArray>()
                        .and_then(|arr| arr.value(row_idx).to_string().into())
                        .or_else(|| col.as_any().downcast_ref::<arrow::array::Float64Array>()
                            .and_then(|arr| arr.value(row_idx).to_string().into()))
                        .or_else(|| col.as_any().downcast_ref::<arrow::array::Int64Array>()
                            .and_then(|arr| arr.value(row_idx).to_string().into()))
                        .unwrap_or_else(|| "<unknown>".to_string()));
                    print!(" {:<15} |", &value_str[..std::cmp::min(15, value_str.len())]);
                }
                println!();
            }
        }
    }

    println!();
    println!("📊 Platform Baseline Notes:");
    println!("  - AdWords table: {} (cost + impressions)", adwords_path);
    println!("  - Facebook table: {} (spend + impressions)", facebook_path);
    println!("  - Platform SQL would compute: SUM(cost) as total_cost, SUM(impressions) GROUP BY tableGroup");
    println!();
    println!("💡 Reconciliation Status:");
    println!("  - Semstrait: UNION + CASE WHEN logic for cross-platform metrics");
    println!("  - Platform: Direct aggregation from raw tables");
    println!("  - Expected: Exact match for deterministic queries");
    println!("  ⚠️ Platform baseline computation deferred (would require additional DataFusion queries)");

    Ok(())
}

/// Print drilldown analysis (mock implementation)
pub fn print_drilldown(
    table_group: &str,
    _adwords_path: &str,
    _facebook_path: &str,
) -> anyhow::Result<()> {
    println!("🔬 Drilldown: {} (showing contributing rows)", table_group);
    println!("===============================================");

    match table_group {
        "adwords" => {
            println!("  Row contributions (mock data):");
            println!("  | row_id | cost_contribution | impressions_contribution | event_time_utc |");
            println!("  |--------|------------------|--------------------------|----------------|");
            println!("  | 1      | 150.25           | 15000                    | 2024-01-01T00:00:00Z |");
            println!("  | 2      | 275.50           | 25000                    | 2024-01-01T00:00:06Z |");
            println!("  | 3      | 200.75           | 20000                    | 2024-01-01T00:00:12Z |");
            println!("  📊 AdWords Totals: cost=626.50, impressions=60000");
        }
        "facebook" => {
            println!("  Row contributions (mock data):");
            println!("  | row_id | spend_contribution | impressions_contribution | event_time_utc |");
            println!("  |--------|-------------------|--------------------------|----------------|");
            println!("  | 4      | 125.00            | 30000                    | 2024-01-01T00:03:00Z |");
            println!("  | 5      | 125.50            | 15000                    | 2024-01-01T00:03:06Z |");
            println!("  📊 Facebook Totals: spend=250.50, impressions=45000");
        }
        _ => return Err(anyhow::anyhow!("Unknown table group: {}", table_group)),
    }

    println!("  💡 (Actual drilldown temporarily disabled due to version conflicts)");
    println!("  🔗 Shows exact rows contributing to aggregate metrics");

    Ok(())
}