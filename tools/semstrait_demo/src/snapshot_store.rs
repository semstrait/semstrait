use semstrait::{Schema, QueryRequest, plan::PlanNode};
use substrait::proto::Plan;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::TempDir;
use chrono::{DateTime, Utc};
use super::ReproducibilityParams;

/// A snapshot store that persists all artifacts needed to reproduce a calculation
pub struct SnapshotStore {
    base_dir: PathBuf,
}

impl SnapshotStore {
    /// Create a new snapshot store
    pub fn new() -> Self {
        let base_dir = PathBuf::from(".semstrait_demo").join("snapshots");
        Self { base_dir }
    }

    /// Save a complete snapshot with all artifacts
    pub async fn save_snapshot(
        &self,
        snapshot_id: &str,
        schema: &Schema,
        request: &QueryRequest,
        plan_node: &PlanNode,
        substrait_plan: &Plan,
        repro_params: &ReproducibilityParams,
        temp_dir: &TempDir,
        table_paths: &HashMap<String, String>,
    ) -> anyhow::Result<PathBuf> {
        let snapshot_dir = self.base_dir.join(snapshot_id);
        fs::create_dir_all(&snapshot_dir)?;

        // Save individual artifacts
        self.save_params_json(&snapshot_dir, repro_params)?;
        self.save_request_json(&snapshot_dir, request)?;
        self.save_plan_json(&snapshot_dir, substrait_plan)?;
        self.save_model_yaml(&snapshot_dir, schema)?;
        self.save_parquet_fixtures(&snapshot_dir, temp_dir, table_paths)?;

        Ok(snapshot_dir)
    }

    /// Save reproducibility parameters as JSON
    fn save_params_json(
        &self,
        snapshot_dir: &Path,
        repro_params: &ReproducibilityParams,
    ) -> anyhow::Result<()> {
        let params_path = snapshot_dir.join("params.json");
        let json = serde_json::to_string_pretty(repro_params)?;
        fs::write(params_path, json)?;
        Ok(())
    }

    /// Save query request as JSON
    fn save_request_json(
        &self,
        snapshot_dir: &Path,
        request: &QueryRequest,
    ) -> anyhow::Result<()> {
        let request_path = snapshot_dir.join("request.json");
        let json = serde_json::to_string_pretty(request)?;
        fs::write(request_path, json)?;
        Ok(())
    }

    /// Save Substrait plan as JSON
    fn save_plan_json(
        &self,
        snapshot_dir: &Path,
        substrait_plan: &Plan,
    ) -> anyhow::Result<()> {
        let plan_path = snapshot_dir.join("plan.json");
        let json = serde_json::to_string_pretty(substrait_plan)?;
        fs::write(plan_path, json)?;
        Ok(())
    }

    /// Save the semantic model as YAML
    fn save_model_yaml(
        &self,
        snapshot_dir: &Path,
        _schema: &Schema,
    ) -> anyhow::Result<()> {
        let model_path = snapshot_dir.join("model.yaml");
        // Save the embedded model YAML (deterministic source)
        let yaml = include_str!("../model.yaml");
        fs::write(model_path, yaml)?;
        Ok(())
    }

    /// Copy Parquet fixtures to snapshot directory
    fn save_parquet_fixtures(
        &self,
        snapshot_dir: &Path,
        temp_dir: &TempDir,
        table_paths: &HashMap<String, String>,
    ) -> anyhow::Result<()> {
        let fixtures_dir = snapshot_dir.join("fixtures");
        fs::create_dir_all(&fixtures_dir)?;

        for (table_name, temp_path) in table_paths {
            let dest_path = fixtures_dir.join(format!("{}.parquet", table_name));
            fs::copy(temp_path, &dest_path)?;
        }

        Ok(())
    }

    /// Compute a deterministic data-state fingerprint for hash inclusion
    pub async fn compute_data_fingerprint(
        &self,
        table_paths: &HashMap<String, String>,
    ) -> anyhow::Result<String> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        use datafusion::prelude::*;

        let mut hasher = DefaultHasher::new();

        // Include row counts and schema hashes for each table
        for (table_name, path) in table_paths {
            let ctx = SessionContext::new();
            let df = ctx.read_parquet(path, Default::default()).await?;
            let batches = df.clone().collect().await?;

            let row_count = batches.iter().map(|b| b.num_rows()).sum::<usize>();
            let schema_hash = compute_schema_hash(&batches);

            // Include max event_time if available
            let max_event_time: Option<DateTime<Utc>> = find_max_event_time(&df).await?;
            let max_event_timestamp = max_event_time
                .map(|dt| dt.timestamp())
                .unwrap_or(0);

            // Hash table metadata
            table_name.hash(&mut hasher);
            row_count.hash(&mut hasher);
            schema_hash.hash(&mut hasher);
            max_event_timestamp.hash(&mut hasher);
        }

        Ok(format!("{:x}", hasher.finish()))
    }
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

/// Find maximum event_time from a DataFrame
async fn find_max_event_time(df: &datafusion::dataframe::DataFrame) -> anyhow::Result<Option<DateTime<Utc>>> {
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