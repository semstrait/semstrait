use semstrait::{Schema, SemanticModel, QueryRequest};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use datafusion::prelude::*;
use super::ReproducibilityParams;

/// A complete proof pack containing all evidence for a metric's calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofPack {
    /// The metric this proof pack is for
    pub metric_name: String,
    /// The snapshot ID that generated this proof pack
    pub snapshot_id: String,
    /// Reproducibility parameters used
    pub reproducibility_params: ReproducibilityParams,
    /// The metric formula/expression
    pub metric_formula: MetricFormula,
    /// Mapping from metric to measures by table group
    pub metric_to_measures: HashMap<String, Vec<MeasureMapping>>,
    /// Per-source metadata and column mappings
    pub source_metadata: HashMap<String, SourceMetadata>,
}

/// The formula/expression for a metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricFormula {
    /// The raw expression string
    pub expression: String,
    /// Human-readable description
    pub description: Option<String>,
    /// Data type of the result
    pub data_type: String,
    /// Whether this metric is additive across dimensions
    pub is_additive: bool,
}

/// Mapping from a measure to its source columns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasureMapping {
    /// The measure name
    pub measure_name: String,
    /// The aggregation function applied
    pub aggregation: String,
    /// The expression that defines this measure
    pub expression: String,
    /// The table group this measure comes from
    pub table_group: String,
    /// The physical table this measure is defined on
    pub table: String,
    /// Column mappings for this measure
    pub column_mappings: HashMap<String, String>,
}

/// Metadata about a data source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceMetadata {
    /// Table name
    pub table_name: String,
    /// Table group this belongs to
    pub table_group: String,
    /// When this data was last ingested
    pub last_ingested_at: DateTime<Utc>,
    /// Row count in this source
    pub row_count: usize,
    /// Maximum event time in this source
    pub max_event_time: Option<DateTime<Utc>>,
    /// Completeness watermark (max_event_time - lag)
    pub completeness_up_to: Option<DateTime<Utc>>,
    /// Schema hash for change detection
    pub schema_hash: String,
    /// Source type (parquet, etc.)
    pub source_type: String,
    /// Source path/location
    pub source_path: String,
}

/// Generate a complete proof pack for a metric
pub async fn generate_proof_pack(
    schema: &Schema,
    model: &SemanticModel,
    metric_name: &str,
    repro_params: &ReproducibilityParams,
    snapshot_id: &str,
    table_paths: &HashMap<String, String>,
    request: &QueryRequest,
) -> anyhow::Result<ProofPack> {
    // Get the metric definition
    let metric = model.get_metric(metric_name)
        .ok_or_else(|| anyhow::anyhow!("Metric '{}' not found", metric_name))?;

    // Build metric formula
    let metric_formula = build_metric_formula(metric)?;

    // Build measure mappings
    let metric_to_measures = build_measure_mappings(schema, model, metric)?;

    // Build source metadata
    let source_metadata = build_source_metadata(table_paths).await?;

    Ok(ProofPack {
        metric_name: metric_name.to_string(),
        snapshot_id: snapshot_id.to_string(),
        reproducibility_params: repro_params.clone(),
        metric_formula,
        metric_to_measures,
        source_metadata,
    })
}

/// Build the metric formula structure
fn build_metric_formula(metric: &semstrait::semantic_model::Metric) -> anyhow::Result<MetricFormula> {
    let expression = match &metric.expr {
        semstrait::semantic_model::MetricExpr::MeasureRef(name) => format!("Measure reference: {}", name),
        semstrait::semantic_model::MetricExpr::Structured(node) => format!("{:?}", node), // Simplified for now
    };

    // Determine if additive (simplified heuristic)
    let is_additive = match &metric.expr {
        semstrait::semantic_model::MetricExpr::MeasureRef(name) => {
            // Check if the referenced measure uses sum aggregation
            // This is a simplification - in practice we'd need to trace through the expression
            name.contains("cost") || name.contains("spend") || name.contains("impressions")
        }
        semstrait::semantic_model::MetricExpr::Structured(_) => false, // CASE expressions are typically not additive
    };

    Ok(MetricFormula {
        expression,
        description: metric.description.clone(),
        data_type: metric.data_type.as_ref()
            .map(|dt| format!("{:?}", dt))
            .unwrap_or_else(|| "unknown".to_string()),
        is_additive,
    })
}

/// Build mappings from metric to measures by table group
fn build_measure_mappings(
    schema: &Schema,
    model: &SemanticModel,
    metric: &semstrait::semantic_model::Metric,
) -> anyhow::Result<HashMap<String, Vec<MeasureMapping>>> {
    let mut mappings = HashMap::new();

    // For cross-tableGroup metrics (like our total_cost), extract the tableGroup-to-measure mappings
    if let semstrait::semantic_model::MetricExpr::Structured(expr) = &metric.expr {
        // This is a simplified implementation - we'd need to parse the CASE expression
        // For the demo, we'll hardcode the mappings based on our model
        let adwords_measures = vec![MeasureMapping {
            measure_name: "cost".to_string(),
            aggregation: "sum".to_string(),
            expression: "cost".to_string(),
            table_group: "adwords".to_string(),
            table: "adwords_campaigns".to_string(),
            column_mappings: HashMap::from([
                ("cost".to_string(), "cost".to_string()),
            ]),
        }];

        let facebook_measures = vec![MeasureMapping {
            measure_name: "spend".to_string(),
            aggregation: "sum".to_string(),
            expression: "spend".to_string(),
            table_group: "facebook".to_string(),
            table: "facebook_campaigns".to_string(),
            column_mappings: HashMap::from([
                ("spend".to_string(), "spend".to_string()),
            ]),
        }];

        mappings.insert("adwords".to_string(), adwords_measures);
        mappings.insert("facebook".to_string(), facebook_measures);
    }

    Ok(mappings)
}

/// Build source metadata for all tables
async fn build_source_metadata(table_paths: &HashMap<String, String>) -> anyhow::Result<HashMap<String, SourceMetadata>> {
    let mut metadata = HashMap::new();

    for (table_name, path) in table_paths {
        let ctx = SessionContext::new();
        let df = ctx.read_parquet(path, Default::default()).await?;
        let batches = df.clone().collect().await?;

        let row_count = batches.iter().map(|b| b.num_rows()).sum();

        // Find max event_time from data
        let max_event_time = find_max_event_time(&df).await?;

        // Calculate schema hash
        let schema_hash = compute_schema_hash(&batches);

        // Determine table group from table name
        let table_group = if table_name.contains("adwords") {
            "adwords"
        } else if table_name.contains("facebook") {
            "facebook"
        } else {
            "unknown"
        };

        metadata.insert(table_name.clone(), SourceMetadata {
            table_name: table_name.clone(),
            table_group: table_group.to_string(),
            last_ingested_at: Utc::now(), // In demo, use current time
            row_count,
            max_event_time,
            completeness_up_to: max_event_time.map(|t| t - chrono::Duration::hours(1)),
            schema_hash,
            source_type: "parquet".to_string(),
            source_path: path.clone(),
        });
    }

    Ok(metadata)
}

/// Find maximum event_time from a DataFrame
async fn find_max_event_time(df: &DataFrame) -> anyhow::Result<Option<DateTime<Utc>>> {
    // Look for event_time_utc column and find max
    if let Ok(max_time_col) = df.clone().select_columns(&["event_time_utc"]) {
        if let Ok(max_time_df) = max_time_col.aggregate(vec![], vec![
            datafusion::functions_aggregate::expr_fn::max(datafusion::logical_expr::col("event_time_utc"))
        ]) {
            let batches = max_time_df.collect().await?;
            if let Some(batch) = batches.first() {
                if let Some(col) = batch.column_by_name("max(event_time_utc)") {
                    if let Some(timestamp_array) = col.as_any().downcast_ref::<datafusion::arrow::array::TimestampMicrosecondArray>() {
                        if let Some(max_ts) = timestamp_array.value(0).into() {
                            return Ok(Some(DateTime::from_timestamp_micros(max_ts).unwrap_or(Utc::now())));
                        }
                    }
                }
            }
        }
    }
    Ok(None)
}

/// Compute a hash of the schema for change detection
fn compute_schema_hash(batches: &[datafusion::arrow::record_batch::RecordBatch]) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();

    if let Some(first_batch) = batches.first() {
        let schema = first_batch.schema();
        for field in schema.fields() {
            field.name().hash(&mut hasher);
            format!("{:?}", field.data_type()).hash(&mut hasher);
        }
    }

    format!("{:x}", hasher.finish())
}

/// Print a proof pack in human-readable format
pub fn print_proof_pack(proof_pack: &ProofPack) -> anyhow::Result<()> {
    println!("📋 PROOF PACK");
    println!("============");
    println!("Metric: {}", proof_pack.metric_name);
    println!("Snapshot ID: {}", proof_pack.snapshot_id);
    println!("As-of: {}", proof_pack.reproducibility_params.as_of);
    println!("Timezone: {}", proof_pack.reproducibility_params.timezone);
    println!("Currency: {} (FX: {:.4})", proof_pack.reproducibility_params.currency, proof_pack.reproducibility_params.fx_rate);
    println!("Attribution Window: {} days", proof_pack.reproducibility_params.attribution_window);
    println!();

    println!("📐 METRIC FORMULA");
    println!("  Expression: {}", proof_pack.metric_formula.expression);
    if let Some(desc) = &proof_pack.metric_formula.description {
        println!("  Description: {}", desc);
    }
    println!("  Data Type: {}", proof_pack.metric_formula.data_type);
    println!("  Additive: {}", proof_pack.metric_formula.is_additive);
    println!();

    println!("🔗 MEASURE MAPPINGS");
    for (table_group, measures) in &proof_pack.metric_to_measures {
        println!("  Table Group: {}", table_group);
        for measure in measures {
            println!("    Measure: {} ({})", measure.measure_name, measure.aggregation);
            println!("    Expression: {}", measure.expression);
            println!("    Table: {}", measure.table);
            println!("    Column Mappings:");
            for (logical, physical) in &measure.column_mappings {
                println!("      {} → {}", logical, physical);
            }
            println!();
        }
    }

    println!("📊 SOURCE METADATA");
    for (table_name, metadata) in &proof_pack.source_metadata {
        println!("  Table: {} ({})", table_name, metadata.table_group);
        println!("    Rows: {}", metadata.row_count);
        println!("    Last Ingested: {}", metadata.last_ingested_at);
        if let Some(max_time) = metadata.max_event_time {
            println!("    Max Event Time: {}", max_time);
        }
        if let Some(completeness) = metadata.completeness_up_to {
            println!("    Complete Up To: {}", completeness);
        }
        println!("    Schema Hash: {}", metadata.schema_hash);
        println!("    Source: {} ({})", metadata.source_type, metadata.source_path);
        println!();
    }

    Ok(())
}

/// Save proof pack to a specific snapshot directory
pub fn save_proof_pack_to_snapshot(proof_pack: &ProofPack, snapshot_dir: &std::path::Path) -> anyhow::Result<()> {
    // Save proof pack as JSON
    let proof_pack_path = snapshot_dir.join("proof_pack.json");
    let json = serde_json::to_string_pretty(proof_pack)?;
    std::fs::write(proof_pack_path, json)?;

    Ok(())
}

/// Save proof pack to disk (legacy function for backwards compatibility)
pub fn save_proof_pack(proof_pack: &ProofPack, snapshot_id: &str) -> anyhow::Result<()> {
    // Create snapshot directory
    let snapshot_dir = std::path::Path::new(".semstrait_demo").join("snapshots").join(snapshot_id);
    std::fs::create_dir_all(&snapshot_dir)?;

    save_proof_pack_to_snapshot(proof_pack, &snapshot_dir)?;

    println!("💾 Proof pack saved to: {}", snapshot_dir.display());
    println!("🔗 Shareable link: file://{}", snapshot_dir.canonicalize()?.display());

    Ok(())
}