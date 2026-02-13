#![allow(warnings)]

use clap::{Parser, Subcommand};
use tempfile::TempDir;
// DataFusion integration
use datafusion::prelude::*;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

mod parquet_generation;
mod lineage;
mod execution;
mod datafusion_execution;
mod diff_engine;
mod impact_engine;
mod health;
mod proof_pack;
mod snapshot_store;
mod reconcile;

#[derive(Parser)]
#[command(name = "semstrait-demo")]
#[command(about = "Semantic layer trust engines: prove, diagnose, validate, and monitor data")]
#[command(version)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run semantic query and show reproducibility proof
    Run {
        /// Demo scenario to run
        #[arg(long, default_value = "union")]
        scenario: String,

        /// Emit JSON plan instead of executing
        #[arg(long)]
        json: bool,

        /// Skip execution, just show plan
        #[arg(long)]
        no_exec: bool,

        /// As-of timestamp for reproducibility (ISO 8601 format)
        #[arg(long, default_value = "2024-01-01T00:00:00Z")]
        as_of: String,

        /// Timezone for analysis
        #[arg(long, default_value = "UTC")]
        timezone: String,

        /// Currency for monetary values
        #[arg(long, default_value = "USD")]
        currency: String,

        /// FX conversion rate (USD to target currency)
        #[arg(long, default_value = "1.0")]
        fx_rate: f64,

        /// Attribution window in days
        #[arg(long, default_value = "30")]
        attribution_window: u32,
    },
    /// Diagnose why numbers don't match: semantic vs platform comparison
    Diff {
        /// As-of timestamp for reproducibility (ISO 8601 format)
        #[arg(long, default_value = "2024-01-01T00:00:00Z")]
        as_of: String,

        /// Timezone for analysis
        #[arg(long, default_value = "UTC")]
        timezone: String,

        /// Currency for monetary values
        #[arg(long, default_value = "USD")]
        currency: String,

        /// FX conversion rate (USD to target currency)
        #[arg(long, default_value = "1.0")]
        fx_rate: f64,

        /// Attribution window in days
        #[arg(long, default_value = "30")]
        attribution_window: u32,
    },
    /// Validate before live: impact analysis of proposed changes
    Impact {
        /// Path to proposed model YAML file (optional - enables preview mode if omitted)
        #[arg(long)]
        proposed_model: Option<String>,

        /// Enable preview mode with sampling (default: false)
        #[arg(long)]
        preview: bool,

        /// As-of timestamp for reproducibility (ISO 8601 format)
        #[arg(long, default_value = "2024-01-01T00:00:00Z")]
        as_of: String,

        /// Timezone for analysis
        #[arg(long, default_value = "UTC")]
        timezone: String,

        /// Currency for monetary values
        #[arg(long, default_value = "USD")]
        currency: String,

        /// FX conversion rate (USD to target currency)
        #[arg(long, default_value = "1.0")]
        fx_rate: f64,

        /// Attribution window in days
        #[arg(long, default_value = "30")]
        attribution_window: u32,
    },
    /// Monitor data health: freshness, anomalies, and backfill status
    Health {
        /// As-of timestamp for reproducibility (ISO 8601 format)
        #[arg(long, default_value = "2024-01-01T00:00:00Z")]
        as_of: String,

        /// Timezone for analysis
        #[arg(long, default_value = "UTC")]
        timezone: String,

        /// Optional file to save alerts as JSON
        #[arg(long)]
        alert_output: Option<String>,
    },
    /// Drilldown analysis: show contributing rows for a table group
    Drilldown {
        /// Table group to drill down into
        table_group: String,

        /// As-of timestamp for reproducibility (ISO 8601 format)
        #[arg(long, default_value = "2024-01-01T00:00:00Z")]
        as_of: String,

        /// Timezone for analysis
        #[arg(long, default_value = "UTC")]
        timezone: String,

        /// Currency for monetary values
        #[arg(long, default_value = "USD")]
        currency: String,

        /// FX conversion rate (USD to target currency)
        #[arg(long, default_value = "1.0")]
        fx_rate: f64,

        /// Attribution window in days
        #[arg(long, default_value = "30")]
        attribution_window: u32,
    },
    /// Generate proof pack for a metric: reproducible evidence trail
    ProofPack {
        /// Metric name to generate proof pack for
        metric: String,

        /// As-of timestamp for reproducibility (ISO 8601 format)
        #[arg(long, default_value = "2024-01-01T00:00:00Z")]
        as_of: String,

        /// Timezone for analysis
        #[arg(long, default_value = "UTC")]
        timezone: String,

        /// Currency for monetary values
        #[arg(long, default_value = "USD")]
        currency: String,

        /// FX conversion rate (USD to target currency)
        #[arg(long, default_value = "1.0")]
        fx_rate: f64,

        /// Attribution window in days
        #[arg(long, default_value = "30")]
        attribution_window: u32,
    },
    /// Reconcile distinct counts: compare semantic vs baseline unique counts
    Reconcile {
        /// Metric name to reconcile (should use COUNT_DISTINCT)
        metric: String,

        /// As-of timestamp for reproducibility (ISO 8601 format)
        #[arg(long, default_value = "2024-01-01T00:00:00Z")]
        as_of: String,

        /// Timezone for analysis
        #[arg(long, default_value = "UTC")]
        timezone: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReproducibilityParams {
    as_of: String,
    timezone: String,
    currency: String,
    fx_rate: f64,
    attribution_window: u32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    match args.command {
        Commands::Run { scenario, json, no_exec, as_of, timezone, currency, fx_rate, attribution_window } => {
            handle_run(scenario, json, no_exec, as_of, timezone, currency, fx_rate, attribution_window).await
        }
        Commands::Diff { as_of, timezone, currency, fx_rate, attribution_window } => {
            handle_diff(as_of, timezone, currency, fx_rate, attribution_window).await
        }
        Commands::Impact { proposed_model, preview, as_of, timezone, currency, fx_rate, attribution_window } => {
            handle_impact(proposed_model, preview, as_of, timezone, currency, fx_rate, attribution_window).await
        }
        Commands::Health { as_of, timezone, alert_output } => {
            handle_health(as_of, timezone, alert_output).await
        }
        Commands::Drilldown { table_group, as_of, timezone, currency, fx_rate, attribution_window } => {
            handle_drilldown(table_group, as_of, timezone, currency, fx_rate, attribution_window).await
        }
        Commands::ProofPack { metric, as_of, timezone, currency, fx_rate, attribution_window } => {
            handle_proof_pack(metric, as_of, timezone, currency, fx_rate, attribution_window).await
        }
        Commands::Reconcile { metric, as_of, timezone } => {
            handle_reconcile(metric, as_of, timezone).await
        }
    }
}

/// Handle the 'run' subcommand - basic semantic query execution
async fn handle_run(
    scenario: String,
    json: bool,
    no_exec: bool,
    as_of: String,
    timezone: String,
    currency: String,
    fx_rate: f64,
    attribution_window: u32,
) -> anyhow::Result<()> {
    println!("🔍 Semstrait Demo - Run Mode");
    println!("============================");

    // Setup common infrastructure
    let (schema, model_name, request, plan_node, substrait_plan, repro_params, snapshot_id, temp_dir, table_paths) =
        setup_common(as_of.clone(), timezone.clone(), currency.clone(), fx_rate, attribution_window).await?;
    let model = schema.get_model(&model_name).unwrap();

    // Print lineage report
    lineage::print_lineage_report(&schema, model, &request, &plan_node, &substrait_plan, &repro_params, &snapshot_id);

    // Emit JSON if requested
    if json {
        let json = serde_json::to_string_pretty(&substrait_plan)?;
        println!("\n📄 Substrait Plan (JSON):");
        println!("{}", json);
        return Ok(());
    }

    // Execute if not skipped
    if !no_exec {
        let table_paths = setup_table_paths(&temp_dir);
        let results = execute_common(&plan_node, &table_paths).await?;
        println!("📊 Results schema: {:?}", results[0].schema());
        execution::print_execution_results(&results)?;
    }

    println!("\n✅ Run completed successfully!");
    Ok(())
}

/// Handle the 'diff' subcommand - discrepancy analysis
async fn handle_diff(
    as_of: String,
    timezone: String,
    currency: String,
    fx_rate: f64,
    attribution_window: u32,
) -> anyhow::Result<()> {
    println!("🔍 Semstrait Demo - Diff Mode (Diagnose Why It Doesn't Match)");
    println!("============================================================");

    // Setup common infrastructure
    let (schema, model_name, request, plan_node, substrait_plan, repro_params, snapshot_id, temp_dir, table_paths) =
        setup_common(as_of.clone(), timezone.clone(), currency.clone(), fx_rate, attribution_window).await?;
    let model = schema.get_model(&model_name).unwrap();

    // Execute semantic query
    let table_paths = setup_table_paths(&temp_dir);
    let semantic_results = execute_common(&plan_node, &table_paths).await?;

    // Create DataFusion context for diff analysis
    let ctx = SessionContext::new();

    // Run discrepancy analysis
    let diff_result = diff_engine::execute_diff_analysis(&ctx, &schema, model, &request, &semantic_results, &table_paths).await?;

    // Print results
    diff_engine::print_diff_analysis(&diff_result)?;

    println!("\n✅ Diff analysis completed!");
    Ok(())
}

/// Handle the 'impact' subcommand - change impact analysis
async fn handle_impact(
    proposed_model: Option<String>,
    preview: bool,
    as_of: String,
    timezone: String,
    currency: String,
    fx_rate: f64,
    attribution_window: u32,
) -> anyhow::Result<()> {
    if preview || proposed_model.is_none() {
        println!("🎯 Semstrait Demo - Impact Preview Mode (Sample Analysis)");
        println!("=======================================================");
    } else {
        println!("🎯 Semstrait Demo - Impact Mode (Validate Before Live)");
        println!("=====================================================");
    }

    // Setup common infrastructure
    let (schema, model_name, request, plan_node, substrait_plan, repro_params, snapshot_id, temp_dir, table_paths) =
        setup_common(as_of.clone(), timezone.clone(), currency.clone(), fx_rate, attribution_window).await?;
    let model = schema.get_model(&model_name).unwrap();

    // Determine if this is a preview or full impact analysis
    let proposed_model_path = if preview || proposed_model.is_none() {
        None // Preview mode - no proposed model
    } else {
        proposed_model.as_deref()
    };

    // Run impact analysis
    let ctx = SessionContext::new();
    let impact_result = impact_engine::execute_impact_analysis(&ctx, &schema, model, proposed_model_path, &request, &setup_table_paths(&temp_dir)).await?;

    // Print results
    impact_engine::print_impact_analysis(&impact_result, preview || proposed_model.is_none())?;

    println!("\n✅ Impact analysis completed!");
    Ok(())
}

/// Handle the 'health' subcommand - data health assessment
async fn handle_health(
    as_of: String,
    timezone: String,
    alert_output: Option<String>,
) -> anyhow::Result<()> {
    println!("🏥 Semstrait Demo - Health Mode (Data Trust Surface)");
    println!("====================================================");

    // Setup data sources
    let temp_dir = TempDir::new()?;
    let (adwords_path, facebook_path) = parquet_generation::generate_fixtures(&temp_dir)?;
    let table_paths = setup_table_paths(&temp_dir);

    // Create DataFusion context
    let ctx = SessionContext::new();

    // Run health assessment
    let health_result = health::execute_health_assessment(&ctx, &table_paths, &as_of).await?;

    // Print results
    health::print_health_assessment(&health_result)?;

    // Save alerts to file if requested
    if let Some(output_path) = alert_output {
        if !health_result.alerts.is_empty() {
            let alert_json = health::export_alerts_json(&health_result.alerts)?;
            std::fs::write(&output_path, alert_json)?;
            println!("\n💾 Alerts saved to: {}", output_path);
        } else {
            println!("\n📝 No alerts to save (all systems healthy)");
        }
    }

    println!("\n✅ Health assessment completed!");
    Ok(())
}

/// Handle the 'drilldown' subcommand - row-level analysis
async fn handle_drilldown(
    table_group: String,
    as_of: String,
    timezone: String,
    currency: String,
    fx_rate: f64,
    attribution_window: u32,
) -> anyhow::Result<()> {
    println!("🔬 Semstrait Demo - Drilldown Mode: {}", table_group);
    println!("==========================================");

    // Setup data sources
    let temp_dir = TempDir::new()?;
    let (adwords_path, facebook_path) = parquet_generation::generate_fixtures(&temp_dir)?;

    // Print drilldown analysis
    execution::print_drilldown(&table_group, &adwords_path, &facebook_path)?;

    println!("\n✅ Drilldown analysis completed!");
    Ok(())
}

/// Handle the 'proof-pack' subcommand - generate reproducible evidence trail
async fn handle_proof_pack(
    metric: String,
    as_of: String,
    timezone: String,
    currency: String,
    fx_rate: f64,
    attribution_window: u32,
) -> anyhow::Result<()> {
    println!("📋 Semstrait Demo - Proof Pack Mode: {}", metric);
    println!("======================================");

    // Setup common infrastructure
    let (schema, model_name, request, plan_node, substrait_plan, repro_params, snapshot_id, temp_dir, table_paths) =
        setup_common(as_of.clone(), timezone.clone(), currency.clone(), fx_rate, attribution_window).await?;
    let model = schema.get_model(&model_name).unwrap();

    // Generate proof pack
    let proof_pack = proof_pack::generate_proof_pack(
        &schema,
        model,
        &metric,
        &repro_params,
        &snapshot_id,
        &table_paths,
        &request,
    ).await?;

    // Print proof pack
    proof_pack::print_proof_pack(&proof_pack)?;

    // Save complete snapshot
    let snapshot_store = snapshot_store::SnapshotStore::new();
    let snapshot_dir = snapshot_store.save_snapshot(
        &snapshot_id,
        &schema,
        &request,
        &plan_node,
        &substrait_plan,
        &repro_params,
        &temp_dir,
        &table_paths,
    ).await?;

    // Save proof pack within the snapshot
    proof_pack::save_proof_pack_to_snapshot(&proof_pack, &snapshot_dir)?;

    println!("💾 Complete snapshot saved to: {}", snapshot_dir.display());
    println!("🔗 Shareable link: file://{}", snapshot_dir.canonicalize()?.display());

    println!("\n✅ Proof pack and snapshot generated!");
    Ok(())
}

/// Handle the 'reconcile' subcommand - compare distinct counts
async fn handle_reconcile(
    metric: String,
    as_of: String,
    timezone: String,
) -> anyhow::Result<()> {
    println!("🔍 Semstrait Demo - Reconcile Mode: {}", metric);
    println!("=====================================");

    // Setup data sources
    let temp_dir = TempDir::new()?;
    let (adwords_path, facebook_path) = parquet_generation::generate_fixtures(&temp_dir)?;
    let table_paths = setup_table_paths(&temp_dir);

    // Load schema
    let mut schema = load_and_override_schema(&adwords_path, &facebook_path)?;
    let model_name = "marketing-demo".to_string();
    let model = schema.get_model(&model_name)
        .ok_or_else(|| anyhow::anyhow!("Model not found"))?;

    // Create DataFusion context
    let ctx = SessionContext::new();

    // Execute reconciliation
    let reconciliation = reconcile::execute_reconciliation(
        &ctx,
        &schema,
        model,
        &metric,
        &table_paths,
    ).await?;

    // Print results
    reconcile::print_reconciliation(&reconciliation)?;

    println!("\n✅ Reconciliation completed!");
    Ok(())
}

/// Common setup for all commands that need schema/model/plan
async fn setup_common(
    as_of: String,
    timezone: String,
    currency: String,
    fx_rate: f64,
    attribution_window: u32,
) -> anyhow::Result<(semstrait::Schema, String, semstrait::QueryRequest, semstrait::plan::PlanNode, substrait::proto::Plan, ReproducibilityParams, String, TempDir, HashMap<String, String>)> {
    // Create temp directory for Parquet files
    let temp_dir = TempDir::new()?;
    println!("📁 Using temp directory: {}", temp_dir.path().display());

    // Generate Parquet fixtures
    println!("\n📊 Generating Parquet fixtures...");
    let (adwords_path, facebook_path) = parquet_generation::generate_fixtures(&temp_dir)?;

    // Load embedded semantic model and override paths
    let mut schema = load_and_override_schema(&adwords_path, &facebook_path)?;

    // Build reproducibility parameters
    let repro_params = ReproducibilityParams {
        as_of: as_of.to_string(),
        timezone: timezone.to_string(),
        currency: currency.to_string(),
        fx_rate,
        attribution_window,
    };

    // Build query request
    let request = semstrait::QueryRequest {
        model: "marketing-demo".to_string(),
        rows: None, // No grouping for aggregate-only query
        metrics: Some(vec!["total_cost".to_string(), "total_impressions".to_string()]),
        ..Default::default()
    };

    println!("\n🔍 Query Request:");
    println!("  Model: {}", request.model);
    if let Some(ref rows) = request.rows {
        println!("  Rows: {:?}", rows);
    } else {
        println!("  Rows: (none - aggregate only)");
    }
    if let Some(ref metrics) = request.metrics {
        println!("  Metrics: {:?}", metrics);
    }

    // Get semantic model name first
    let model_name = request.model.clone();

    // Get semantic model reference and plan the query in a scoped block
    let plan_node = {
        let model = schema.get_model(&model_name)
            .ok_or_else(|| anyhow::anyhow!("Model not found"))?;

        // Plan the query
        println!("\n🏗️  Planning semantic query...");
        semstrait::planner::plan_semantic_query(&schema, model, &request)?
    };

    // Setup table paths
    let table_paths = setup_table_paths(&temp_dir);

    // Emit Substrait plan
    println!("📤 Emitting Substrait plan...");
    let substrait_plan = semstrait::emitter::emit_plan(&plan_node, None)?;

    // Compute snapshot ID for reproducibility
    let snapshot_id = execution::compute_snapshot_id(&schema, &request, &repro_params, &substrait_plan, &table_paths).await?;

    Ok((schema, model_name, request, plan_node, substrait_plan, repro_params, snapshot_id, temp_dir, table_paths))
}

/// Common execution logic
async fn execute_common(plan_node: &semstrait::plan::PlanNode, table_paths: &HashMap<String, String>) -> anyhow::Result<Vec<datafusion::arrow::record_batch::RecordBatch>> {
    println!("\n⚡ Executing Substrait Plan...");

    // Execute via the execution module
    execution::execute_substrait_plan_via_df_exec(&datafusion::prelude::SessionContext::new(), plan_node, table_paths).await
}

/// Set up table paths from temp directory
fn setup_table_paths(temp_dir: &TempDir) -> HashMap<String, String> {
    let mut table_paths = HashMap::new();
    table_paths.insert(
        "adwords_campaigns".to_string(),
        temp_dir.path().join("adwords_campaigns.parquet").to_string_lossy().to_string()
    );
    table_paths.insert(
        "facebook_campaigns".to_string(),
        temp_dir.path().join("facebook_campaigns.parquet").to_string_lossy().to_string()
    );
    table_paths
}

/// Load the embedded schema and override parquet paths to point to generated fixtures
fn load_and_override_schema(adwords_path: &str, facebook_path: &str) -> anyhow::Result<semstrait::Schema> {
    let schema_yaml = include_str!("../model.yaml");
    let mut schema = semstrait::parser::parse_str(schema_yaml)?;

    // Override the parquet paths in the schema
    for model in &mut schema.semantic_models {
        for table_group in &mut model.table_groups {
            for table in &mut table_group.tables {
                let new_path = match table.table.as_str() {
                    "adwords_campaigns" => adwords_path,
                    "facebook_campaigns" => facebook_path,
                    _ => continue,
                };

                // Update the source path
                if let semstrait::semantic_model::Source::Parquet { path } = &mut table.source {
                    *path = new_path.to_string();
                }
            }
        }
    }

    Ok(schema)
}