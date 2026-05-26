//! SemstraitEngine — the central orchestrator.
//!
//! Coordinates manifest, planner, and SQL emitter
//! to plan semantic queries and produce artifacts.

use crate::error::EngineError;
use crate::parse::RequestParser;
use crate::types::{ExplainResult, RawQueryRequest, ValidationResult};
use semstrait_catalog::{CatalogProvider, TableRef};
use semstrait_ir::{PlanArtifact, PlannerWarning};
use semstrait_manifest::{CompileSource, CompiledManifest, ManifestCompiler};
use semstrait_planner::SemanticPlanner;
use semstrait_adapter::EngineAdapter;
use semstrait_adapter::sql::{AnsiDialect, AnsiSqlEmitter, SqlEmitter};
use std::sync::Arc;

/// The central engine that orchestrates semantic query planning.
///
/// Supports:
/// - `validate()` — parse + validate request against manifest
/// - `explain()` — compile → plan → emit SQL
/// - `plan()` — compile → plan → produce PlanArtifact (Substrait or SQL)
pub struct SemstraitEngine {
    manifest: Option<CompiledManifest>,
    planner: SemanticPlanner,
    adapter: Option<Arc<dyn EngineAdapter>>,
}

impl SemstraitEngine {
    /// Create a new engine without a manifest. Only `validate()` works.
    pub fn new() -> Self {
        Self {
            manifest: None,
            planner: SemanticPlanner::builder().build(),
            adapter: None,
        }
    }

    /// Create an engine from a compiled manifest.
    pub fn with_manifest(manifest: CompiledManifest) -> Self {
        Self {
            manifest: Some(manifest),
            planner: SemanticPlanner::builder().build(),
            adapter: None,
        }
    }

    /// Create an engine with a manifest and an engine adapter.
    ///
    /// The adapter's `plan_builder()` is wired into the planner for
    /// engine-specific node construction. `explain()` uses the adapter's
    /// `debug_sql()` and `plan()` uses `adapt()`.
    pub fn with_adapter(
        manifest: CompiledManifest,
        adapter: Arc<dyn EngineAdapter>,
    ) -> Self {
        let mut planner_builder = SemanticPlanner::builder();
        if let Some(pb) = adapter.plan_builder() {
            planner_builder = planner_builder.with_plan_builder(pb);
        }
        let planner = planner_builder.build();

        Self {
            manifest: Some(manifest),
            planner,
            adapter: Some(adapter),
        }
    }

    /// Create an engine by compiling a model YAML string.
    pub async fn with_model(yaml: &str) -> Result<Self, EngineError> {
        let compiler = ManifestCompiler::new();
        let manifest = compiler
            .compile(CompileSource::Yaml(yaml.to_string()))
            .await?;
        Ok(Self::with_manifest(manifest))
    }

    /// Get a reference to the compiled manifest (if loaded).
    pub fn manifest(&self) -> Option<&CompiledManifest> {
        self.manifest.as_ref()
    }

    /// Emit SQL from a logical plan using ANSI dialect.
    ///
    /// Used as the fallback when no adapter is configured.
    fn emit_ansi_sql(plan: &semstrait_ir::LogicalPlan) -> Result<String, EngineError> {
        let emitter = AnsiSqlEmitter::new(AnsiDialect);
        Ok(emitter.emit(plan)?)
    }

    /// Validate that the request's engine field matches the configured adapter.
    ///
    /// The planner uses the adapter's plan_builder at construction time, so
    /// switching engines per-request is not supported. This check prevents
    /// silent mismatches where a client requests "datafusion" but gets ANSI output.
    fn validate_engine_field(&self, raw: &RawQueryRequest) -> Result<(), EngineError> {
        let requested = match &raw.engine {
            Some(e) if !e.is_empty() => e.as_str(),
            _ => return Ok(()), // no engine specified — use whatever is configured
        };

        // "ansi" / "canonical" always valid when no adapter is configured.
        if matches!(requested, "ansi" | "canonical") {
            if self.adapter.is_none() {
                return Ok(());
            }
            // Adapter is configured but client wants ANSI — that's fine, debug_sql uses ANSI anyway.
            return Ok(());
        }

        match &self.adapter {
            Some(adapter) if adapter.name() == requested => Ok(()),
            Some(adapter) => Err(EngineError::NotConfigured(format!(
                "engine mismatch: request specifies '{}' but server is configured with '{}'",
                requested,
                adapter.name(),
            ))),
            None => Err(EngineError::NotConfigured(format!(
                "engine '{}' requested but no adapter is configured (server uses ANSI canonical)",
                requested,
            ))),
        }
    }

    /// Validate a query request against the manifest.
    pub fn validate(&self, raw: &RawQueryRequest) -> ValidationResult {
        // Basic structural validation.
        if let Err(e) = RequestParser::parse(raw) {
            return ValidationResult {
                valid: false,
                errors: vec![e.to_string()],
                warnings: vec![],
            };
        }

        // If we have a manifest, validate names via full resolution.
        if let Some(manifest) = &self.manifest {
            match RequestParser::to_resolved(raw, manifest) {
                Ok(_) => ValidationResult {
                    valid: true,
                    errors: vec![],
                    warnings: vec![],
                },
                Err(e) => ValidationResult {
                    valid: false,
                    errors: vec![e.to_string()],
                    warnings: vec![],
                },
            }
        } else {
            // No manifest — structural validation only.
            ValidationResult {
                valid: true,
                errors: vec![],
                warnings: vec!["no manifest loaded; skipping semantic validation".to_string()],
            }
        }
    }

    /// Explain a query: compile, plan, emit SQL + human-readable plan tree.
    pub async fn explain(
        &self,
        raw: &RawQueryRequest,
    ) -> Result<ExplainResult, EngineError> {
        let manifest = self
            .manifest
            .as_ref()
            .ok_or_else(|| EngineError::NotConfigured("no manifest loaded".to_string()))?;

        self.validate_engine_field(raw)?;

        // Parse the raw request into a resolved query request.
        let request = RequestParser::to_resolved(raw, manifest)?;

        // Plan.
        let plan = self.planner.plan(&request, manifest)?;

        // Emit SQL via adapter or ANSI fallback.
        let sql = if let Some(adapter) = &self.adapter {
            Some(adapter.debug_sql(&plan)?)
        } else {
            Some(Self::emit_ansi_sql(&plan)?)
        };

        // Human-readable plan tree (Display impl on LogicalPlan).
        let plan_text = plan.to_string();

        Ok(ExplainResult {
            sql,
            plan_text,
        })
    }

    /// Plan a query and produce an engine-appropriate artifact.
    ///
    /// If an adapter is configured, uses `adapter.adapt()` to produce
    /// the engine-native artifact (Substrait for DataFusion, SQL for others).
    /// Without an adapter, falls back to ANSI SQL emission.
    pub async fn plan(
        &self,
        raw: &RawQueryRequest,
    ) -> Result<PlanArtifact, EngineError> {
        let manifest = self
            .manifest
            .as_ref()
            .ok_or_else(|| EngineError::NotConfigured("no manifest loaded".to_string()))?;

        self.validate_engine_field(raw)?;

        let request = RequestParser::to_resolved(raw, manifest)?;
        let plan = self.planner.plan(&request, manifest)?;

        if let Some(adapter) = &self.adapter {
            Ok(adapter.adapt(&plan)?)
        } else {
            let sql = Self::emit_ansi_sql(&plan)?;
            Ok(PlanArtifact::Sql(sql))
        }
    }

    /// Check for schema drift between the compiled manifest and the live catalog.
    ///
    /// Returns PLAN_W003 warnings for each dataset where the catalog schema
    /// differs from the compiled schema snapshot. Requires a catalog provider.
    ///
    /// This is a best-effort check: datasets without compiled schema snapshots
    /// or inaccessible in the catalog are silently skipped.
    pub async fn check_schema_drift(
        &self,
        catalog: &dyn CatalogProvider,
        namespace: &str,
    ) -> Vec<PlannerWarning> {
        let mut warnings = Vec::new();
        let manifest = match &self.manifest {
            Some(m) => m,
            None => return warnings,
        };

        let snapshot = match &manifest.catalog_snapshot {
            Some(s) => s,
            None => return warnings,
        };

        for (table_name, table_snap) in &snapshot.tables {
            let table_ref = TableRef::new(namespace, table_name);
            let live_columns = match catalog.get_schema(&table_ref).await {
                Ok(cols) => cols,
                Err(_) => continue,
            };

            let compiled = &table_snap.columns;
            let mut diffs = Vec::new();

            for cc in compiled {
                match live_columns.iter().find(|lc| lc.name == cc.name) {
                    None => diffs.push(format!("column '{}' removed", cc.name)),
                    Some(lc) => {
                        let live_type = format!("{:?}", lc.data_type);
                        let compiled_type = format!("{:?}", cc.data_type);
                        if live_type != compiled_type {
                            diffs.push(format!(
                                "column '{}' type changed: {} -> {}",
                                cc.name, compiled_type, live_type
                            ));
                        }
                        if lc.nullable != cc.nullable {
                            diffs.push(format!(
                                "column '{}' nullability changed",
                                cc.name
                            ));
                        }
                    }
                }
            }
            for lc in &live_columns {
                if !compiled.iter().any(|cc| cc.name == lc.name) {
                    diffs.push(format!("column '{}' added", lc.name));
                }
            }

            if !diffs.is_empty() {
                warnings.push(PlannerWarning::SchemaDrift {
                    dataset: table_name.clone(),
                    details: diffs.join("; "),
                });
            }
        }

        warnings
    }
}

impl Default for SemstraitEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Resolve an engine adapter by name.
///
/// Supported engines:
/// - `"datafusion"` — DataFusion adapter (Substrait output, DF-specific plan builder). Requires `datafusion` feature.
/// - `"ansi"` / `None` — No adapter, ANSI SQL canonical output.
///
/// Returns `None` for ANSI/canonical (no adapter needed).
/// Returns `Err` for unknown engine names or engines not compiled in.
pub fn resolve_adapter(engine: Option<&str>) -> Result<Option<Arc<dyn EngineAdapter>>, EngineError> {
    match engine {
        None | Some("ansi") | Some("canonical") => Ok(None),
        #[cfg(feature = "datafusion")]
        Some("datafusion") => {
            Ok(Some(Arc::new(semstrait_adapter::DataFusionAdapter)))
        }
        #[cfg(not(feature = "datafusion"))]
        Some("datafusion") => {
            Err(EngineError::NotConfigured(
                "engine 'datafusion' requires the 'datafusion' feature".to_string(),
            ))
        }
        Some(other) => Err(EngineError::NotConfigured(
            format!("unknown engine '{}' (supported: datafusion, ansi)", other),
        )),
    }
}

/// Shared engine state for API transports.
pub type SharedEngine = Arc<SemstraitEngine>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_valid_request() {
        let engine = SemstraitEngine::new();
        let raw = RawQueryRequest {
            from: Some("sales".to_string()),
            select: vec!["region".to_string(), "revenue".to_string()],
            ..Default::default()
        };

        let result = engine.validate(&raw);
        assert!(result.valid);
    }

    #[test]
    fn test_validate_invalid_request() {
        let engine = SemstraitEngine::new();
        let raw = RawQueryRequest {
            from: None,
            ..Default::default()
        };

        let result = engine.validate(&raw);
        assert!(!result.valid);
        assert!(!result.errors.is_empty());
    }

    #[tokio::test]
    async fn test_explain_no_manifest() {
        let engine = SemstraitEngine::new();
        let raw = RawQueryRequest {
            from: Some("sales".to_string()),
            select: vec!["region".to_string(), "revenue".to_string()],
            ..Default::default()
        };

        let result = engine.explain(&raw).await;
        assert!(matches!(result, Err(EngineError::NotConfigured(_))));
    }

    fn load_model(name: &str) -> String {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let path = format!(
            "{}/../../tests/fixtures/models/{}.yaml",
            manifest_dir, name
        );
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to load fixture '{}': {}", path, e))
    }

    #[tokio::test]
    async fn test_explain_with_manifest() {
        let yaml = load_model("orders_with_metrics");

        let engine = SemstraitEngine::with_model(&yaml)
            .await
            .expect("engine should compile manifest");

        let raw = RawQueryRequest {
            from: Some("orders".to_string()),
            select: vec!["date".to_string(), "region".to_string(), "revenue".to_string()],
            ..Default::default()
        };

        let result = engine.explain(&raw).await;
        assert!(result.is_ok(), "explain should succeed: {:?}", result.err());

        let explain = result.unwrap();
        assert!(explain.sql.is_some(), "should have SQL");
        let sql = explain.sql.unwrap();
        assert!(sql.contains("SELECT"), "SQL should contain SELECT: {}", sql);
        assert!(
            sql.contains("GROUP BY"),
            "SQL should contain GROUP BY: {}",
            sql
        );
        assert!(
            explain.plan_text.contains("TableScan:"),
            "plan_text should contain TableScan: {}",
            explain.plan_text
        );
    }

    #[tokio::test]
    async fn test_explain_with_inline_raw_filter_produces_where_clause() {
        // End-to-end: API parses RawQueryRequest with `raw_filters`, translates
        // through `resolve_raw_filter` into a CompiledFilter, planner injects
        // it at the scan layer (same engine as named DataKind filters per
        // `11 §6.4.2` / `19 §7.1`), adapter emits a WHERE in the SQL.
        let yaml = load_model("orders_with_metrics");

        let engine = SemstraitEngine::with_model(&yaml)
            .await
            .expect("engine should compile manifest");

        let raw = RawQueryRequest {
            from: Some("orders".to_string()),
            select: vec!["date".to_string(), "region".to_string(), "revenue".to_string()],
            raw_filters: vec![crate::types::RawFilter {
                field: "region".to_string(),
                operator: "eq".to_string(),
                value: serde_json::json!("US"),
            }],
            ..Default::default()
        };

        let result = engine.explain(&raw).await;
        assert!(
            result.is_ok(),
            "explain with inline raw filter should succeed: {:?}",
            result.err()
        );

        let sql = result.unwrap().sql.expect("should have SQL");
        let upper = sql.to_uppercase();
        assert!(
            upper.contains("WHERE"),
            "inline raw filter should emit a WHERE clause: {}",
            sql
        );
        // The string literal should be present.
        assert!(
            sql.contains("'US'") || upper.contains("'US'"),
            "WHERE should constrain to 'US': {}",
            sql
        );
    }

    #[tokio::test]
    async fn test_inline_raw_filter_unknown_field_rejected() {
        // Unknown field at request-resolution time produces a typed parse
        // error per `11 §6.4.2` validation contract.
        let yaml = load_model("orders_with_metrics");
        let engine = SemstraitEngine::with_model(&yaml).await.unwrap();

        let raw = RawQueryRequest {
            from: Some("orders".to_string()),
            select: vec!["date".to_string(), "revenue".to_string()],
            raw_filters: vec![crate::types::RawFilter {
                field: "nonexistent_field".to_string(),
                operator: "eq".to_string(),
                value: serde_json::json!("X"),
            }],
            ..Default::default()
        };

        let result = engine.explain(&raw).await;
        assert!(matches!(
            result,
            Err(EngineError::Parse(crate::error::ParseError::RawFilterFieldNotFound { .. }))
        ));
    }

    #[tokio::test]
    async fn test_inline_raw_filter_adhoc_resolves_against_planner_chosen_entity() {
        // Ad-hoc mode (no `from`) + raw_filters used to be rejected up-front.
        // It now resolves end-to-end: the parser stashes the raw_filter as a
        // `PendingInlineFilter`, and `plan_ad_hoc` lowers it against the
        // entity chosen by `find_covering_entities`. The lowered
        // `CompiledFilter` rides the scan-layer engine alongside named
        // DataKind filters per §6.4.2.
        let engine = SemstraitEngine::new();
        let raw = RawQueryRequest {
            from: None,
            select: vec!["revenue".to_string()],
            raw_filters: vec![crate::types::RawFilter {
                field: "region".to_string(),
                operator: "eq".to_string(),
                value: serde_json::json!("US"),
            }],
            ..Default::default()
        };

        // Structural validate (no manifest) accepts.
        let _ = engine.validate(&raw);

        // With a manifest-bearing engine, end-to-end resolution succeeds:
        // the `orders_with_metrics` model has `revenue` and `region` on the
        // same `orders` grainset, so `find_covering_entities(["revenue"])`
        // picks `orders` and the inline filter lowers against its interface.
        let yaml = load_model("orders_with_metrics");
        let engine_with_manifest = SemstraitEngine::with_model(&yaml).await.unwrap();
        let result = engine_with_manifest.explain(&raw).await;
        assert!(
            result.is_ok(),
            "ad-hoc inline raw filter should resolve: {:?}",
            result.err()
        );
    }

    #[tokio::test]
    async fn test_validate_against_manifest() {
        let yaml = load_model("orders_simple");

        let engine = SemstraitEngine::with_model(&yaml)
            .await
            .expect("engine should compile manifest");

        // Valid request.
        let raw = RawQueryRequest {
            from: Some("orders".to_string()),
            select: vec!["date".to_string(), "revenue".to_string()],
            ..Default::default()
        };
        let result = engine.validate(&raw);
        assert!(result.valid);

        // Invalid select name.
        let raw_bad = RawQueryRequest {
            from: Some("orders".to_string()),
            select: vec!["nonexistent".to_string()],
            ..Default::default()
        };
        let result_bad = engine.validate(&raw_bad);
        assert!(!result_bad.valid);
        assert!(result_bad.errors[0].contains("nonexistent"));
    }

    #[tokio::test]
    async fn test_explain_with_auto_column_mapping() {
        let yaml = r#"
semantic_model:
  name: auto_test
  grainsets:
    - name: orders
      dimensions:
        - name: order_date
          data_type: date
          type:
            temporal:
              grains:
                - day
      measures:
        - name: revenue
          data_type: float64
          agg: sum
      datasets:
        - name: orders_fact
          extras:
            column_mapping: auto
            storage:
              format: parquet
              paths:
                - db.orders_fact
"#;

        let engine = SemstraitEngine::with_model(yaml)
            .await
            .expect("engine should compile with auto mapping");

        let raw = RawQueryRequest {
            from: Some("orders".to_string()),
            select: vec!["order_date".to_string(), "revenue".to_string()],
            ..Default::default()
        };

        let result = engine.explain(&raw).await;
        assert!(result.is_ok(), "explain should succeed: {:?}", result.err());

        let sql = result.unwrap().sql.unwrap();
        // With auto mapping, physical names = semantic names (identity).
        assert!(sql.contains("order_date"), "SQL should use identity-mapped column: {}", sql);
        assert!(sql.contains("SELECT"), "SQL should contain SELECT: {}", sql);
    }

    #[tokio::test]
    async fn test_plan_not_configured() {
        let engine = SemstraitEngine::new();
        let raw = RawQueryRequest {
            from: Some("sales".to_string()),
            select: vec!["revenue".to_string()],
            ..Default::default()
        };

        let result = engine.plan(&raw).await;
        assert!(matches!(result, Err(EngineError::NotConfigured(_))));
    }

    #[tokio::test]
    async fn test_schema_drift_detection() {
        use semstrait_catalog::NullCatalogProvider;
        use semstrait_manifest::{CatalogSnapshot, TableSnapshot, ResolvedColumn};
        use semstrait_core::DataType;

        let yaml = load_model("orders_simple");
        let mut engine = SemstraitEngine::with_model(&yaml)
            .await
            .expect("engine should compile manifest");

        // Inject a catalog snapshot with a table that has known columns.
        if let Some(ref mut manifest) = engine.manifest {
            let mut tables = std::collections::HashMap::new();
            tables.insert(
                "orders".to_string(),
                TableSnapshot {
                    fqn: "default.orders".to_string(),
                    columns: vec![
                        ResolvedColumn {
                            name: "id".to_string(),
                            data_type: DataType::Integer,
                            nullable: false,
                            comment: None,
                            field_id: None,
                        },
                        ResolvedColumn {
                            name: "amount".to_string(),
                            data_type: DataType::Number,
                            nullable: true,
                            comment: None,
                            field_id: None,
                        },
                    ],
                    iceberg: None,
                },
            );
            manifest.catalog_snapshot = Some(CatalogSnapshot {
                tables,
                captured_at: chrono::Utc::now(),
            });
        }

        // NullCatalogProvider returns empty schemas, so all compiled columns look "removed".
        let warnings = engine
            .check_schema_drift(&NullCatalogProvider, "default")
            .await;
        assert!(!warnings.is_empty());
        assert!(matches!(
            &warnings[0],
            PlannerWarning::SchemaDrift { dataset, details }
            if dataset == "orders" && details.contains("removed")
        ));
    }
}
