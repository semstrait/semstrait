//! RequestParser — converts RawQueryRequest → validated/resolved request.

use crate::error::ParseError;
use crate::types::RawQueryRequest;
use semstrait_manifest::{CompiledFilter, CompiledInterface, CompiledManifest};
use semstrait_planner::inline_filter::lower_inline_filter;
use semstrait_planner::request::{
    OrderByClause, PendingInlineFilter, ResolvedQueryRequest, SortDirection,
};

/// Parses raw query requests against a compiled manifest.
pub struct RequestParser;

impl RequestParser {
    /// Basic structural validation (no manifest needed).
    ///
    /// Inline `raw_filters` are accepted here but not validated — full
    /// validation (field existence, operator admissibility, value type-check)
    /// happens in `to_resolved` against the live manifest.
    pub fn parse(raw: &RawQueryRequest) -> Result<ValidatedRequest, ParseError> {
        if raw.select.is_empty() {
            return Err(ParseError::Validation(
                "select must contain at least one column name or \"*\"".to_string(),
            ));
        }

        Ok(ValidatedRequest {
            entity_name: raw.from.clone(),
            select: raw.select.clone(),
            filters: raw.filters.clone(),
            grain: raw.grain.clone(),
            limit: raw.limit,
        })
    }

    /// Convert a RawQueryRequest into a fully resolved planner request,
    /// validating names against the compiled manifest.
    pub fn to_resolved(
        raw: &RawQueryRequest,
        manifest: &CompiledManifest,
    ) -> Result<ResolvedQueryRequest, ParseError> {
        // Basic validation.
        let _ = Self::parse(raw)?;

        // Convert order_by (entity-independent).
        let order_by = raw
            .order_by
            .iter()
            .map(|ob| OrderByClause {
                field: ob.field.clone(),
                direction: if ob.direction.to_lowercase() == "desc" {
                    SortDirection::Descending
                } else {
                    SortDirection::Ascending
                },
            })
            .collect();

        // If `from` is None, pass through to planner for ad-hoc resolution.
        // Select names are passed as-is — the planner classifies them. Any
        // inline raw_filters are stashed as `PendingInlineFilter`s for the
        // planner to finalise against the resolved entity (single-entity
        // path: `plan_ad_hoc`; multi-entity path: per-field attribution in
        // `build_ad_hoc_join_plan`). Lowering can't happen here because the
        // `CompiledInterface` needed for type-checking isn't known until
        // `find_covering_entities` resolves the target scope.
        let Some(ref from) = raw.from else {
            let pending_inline_filters: Vec<PendingInlineFilter> = raw
                .raw_filters
                .iter()
                .map(|rf| PendingInlineFilter {
                    field: rf.field.clone(),
                    operator: rf.operator.clone(),
                    value: rf.value.clone(),
                })
                .collect();

            return Ok(ResolvedQueryRequest {
                entity_name: String::new(),
                dimensions: raw.select.clone(), // planner will reclassify
                measures: vec![],
                filters: vec![],
                inline_filters: vec![],
                pending_inline_filters,
                grain: None,
                limit: raw.limit,
                order_by,
                session_variables: raw.session.clone(),
            });
        };

        // Resolve entity via CompiledDataKind.
        let data_kind = manifest
            .resolve(from)
            .ok_or_else(|| ParseError::EntityNotFound(from.clone()))?;
        let kind = data_kind.interface();

        // Expand "*" and classify select names.
        let select_names = expand_select(&raw.select, kind);
        let (dimensions, measures) = classify_select(&select_names, kind, from)?;

        // Resolve named filters against kind-level filters.
        for filter_name in &raw.filters {
            if !kind.filters.iter().any(|f| f.name == *filter_name) {
                return Err(ParseError::FilterNotFound {
                    entity: from.clone(),
                    name: filter_name.clone(),
                });
            }
        }

        // Translate inline raw filters into CompiledFilters via the shared
        // planner-side helper. Each becomes a request-scope, anonymous filter
        // with a synthetic `__inline_filter_N` name; they ride the same
        // scan-layer engine as named DataKind filters per
        // `docs/design/foundations/11_names_and_scopes.md §6.4.2`.
        // `InlineFilterError` is mapped to the corresponding `ParseError`
        // variant via `From<InlineFilterError> for ParseError`.
        let inline_filters: Vec<CompiledFilter> = raw
            .raw_filters
            .iter()
            .enumerate()
            .map(|(i, rf)| {
                let pending = PendingInlineFilter {
                    field: rf.field.clone(),
                    operator: rf.operator.clone(),
                    value: rf.value.clone(),
                };
                lower_inline_filter(&pending, kind, from, i).map_err(ParseError::from)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(ResolvedQueryRequest {
            entity_name: from.clone(),
            dimensions,
            measures,
            filters: vec![],
            inline_filters,
            pending_inline_filters: vec![],
            grain: None,
            limit: raw.limit,
            order_by,
            session_variables: raw.session.clone(),
        })
    }
}

/// Expand `["*"]` into all dimension + measure + metric + key names from the entity.
/// If no `*` is present, returns the select list as-is.
fn expand_select(select: &[String], kind: &CompiledInterface) -> Vec<String> {
    if select.len() == 1 && select[0] == "*" {
        let mut names: Vec<String> = Vec::new();
        names.extend(kind.dimensions.keys().cloned());
        // Include key columns not already in dimensions.
        if let Some(ref keys) = kind.keys {
            for key_col in keys.all_column_names() {
                if !kind.dimensions.contains_key(&key_col) {
                    names.push(key_col);
                }
            }
        }
        names.extend(kind.measures.keys().cloned());
        names.extend(kind.metrics.keys().cloned());
        names
    } else {
        select.to_vec()
    }
}

/// Classify select names into dimensions and measures/metrics.
/// Returns `(dimensions, measures)` where measures includes both measures and metrics.
fn classify_select(
    names: &[String],
    kind: &CompiledInterface,
    entity_name: &str,
) -> Result<(Vec<String>, Vec<String>), ParseError> {
    let mut dimensions = Vec::new();
    let mut measures = Vec::new();

    // Collect key column names for classification.
    let key_names: std::collections::HashSet<String> = kind
        .keys
        .as_ref()
        .map(|k| k.all_column_names().into_iter().collect())
        .unwrap_or_default();

    for name in names {
        if kind.dimensions.contains_key(name) {
            dimensions.push(name.clone());
        } else if key_names.contains(name) {
            // Keys contribute to GROUP BY — classify as dimension.
            dimensions.push(name.clone());
        } else if kind.measures.contains_key(name) || kind.metrics.contains_key(name) {
            measures.push(name.clone());
        } else {
            return Err(ParseError::UnknownSelectName {
                entity: entity_name.to_string(),
                name: name.clone(),
            });
        }
    }

    Ok((dimensions, measures))
}

/// A validated query request (structural validation only, no manifest).
#[derive(Debug, Clone)]
pub struct ValidatedRequest {
    pub entity_name: Option<String>,
    pub select: Vec<String>,
    pub filters: Vec<String>,
    pub grain: Option<String>,
    pub limit: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RawFilter;

    #[test]
    fn test_parse_valid_request() {
        let raw = RawQueryRequest {
            from: Some("sales".to_string()),
            select: vec!["region".to_string(), "revenue".to_string()],
            ..Default::default()
        };

        let result = RequestParser::parse(&raw);
        assert!(result.is_ok());
        let req = result.unwrap();
        assert_eq!(req.entity_name, Some("sales".to_string()));
        assert_eq!(req.select, vec!["region", "revenue"]);
    }

    #[test]
    fn test_parse_star_select() {
        let raw = RawQueryRequest {
            from: Some("sales".to_string()),
            select: vec!["*".to_string()],
            ..Default::default()
        };

        let result = RequestParser::parse(&raw);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_no_from() {
        let raw = RawQueryRequest {
            from: None,
            select: vec!["revenue".to_string()],
            ..Default::default()
        };

        // None passes structural validation — entity resolution happens in planner.
        let result = RequestParser::parse(&raw);
        assert!(result.is_ok());
        let req = result.unwrap();
        assert_eq!(req.entity_name, None);
    }

    #[test]
    fn test_parse_empty_select() {
        let raw = RawQueryRequest {
            from: Some("sales".to_string()),
            ..Default::default()
        };

        let result = RequestParser::parse(&raw);
        assert!(matches!(result, Err(ParseError::Validation(_))));
    }

    #[test]
    fn test_parse_accepts_raw_filters() {
        // `parse()` is structural and should not reject raw_filters; full
        // validation happens at `to_resolved` time against the manifest.
        let raw = RawQueryRequest {
            from: Some("sales".to_string()),
            select: vec!["revenue".to_string()],
            raw_filters: vec![RawFilter {
                field: "region".to_string(),
                operator: "=".to_string(),
                value: serde_json::json!("US"),
            }],
            ..Default::default()
        };

        let result = RequestParser::parse(&raw);
        assert!(result.is_ok(), "parse should accept raw_filters: {:?}", result.err());
    }

    // Lowering-helper unit tests live with the helper itself in
    // `semstrait-planner::inline_filter::tests`. The parser path is tested
    // for shape validation here, and for `to_resolved` end-to-end behaviour
    // via the e2e_pipeline scenarios in `crates/semstrait/tests/`.

    #[test]
    fn test_parse_adhoc_with_raw_filters_accepted() {
        // Structural validation of an ad-hoc request (no `from`) that
        // carries inline raw_filters. The pre-fix behaviour rejected this
        // at structural-parse time; the new behaviour accepts and defers
        // lowering to the planner.
        let raw = RawQueryRequest {
            from: None,
            select: vec!["date".to_string(), "revenue".to_string()],
            raw_filters: vec![
                RawFilter {
                    field: "date".to_string(),
                    operator: ">=".to_string(),
                    value: serde_json::json!("2026-05-01"),
                },
                RawFilter {
                    field: "date".to_string(),
                    operator: "<=".to_string(),
                    value: serde_json::json!("2026-05-31"),
                },
            ],
            ..Default::default()
        };

        assert!(RequestParser::parse(&raw).is_ok());
    }

    #[tokio::test]
    async fn test_to_resolved_adhoc_carries_raw_filters_as_pending() {
        // When `from` is omitted, `to_resolved` must stash raw_filters as
        // `PendingInlineFilter`s (not lower them) — the planner's
        // `plan_ad_hoc` finalises them against the entity it chooses.
        use semstrait_manifest::{CompileSource, ManifestCompiler};

        let yaml = r#"
semantic_model:
  name: m
  grainsets:
    - name: rows
      dimensions:
        - name: date
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
        - name: rows_ds
          extras:
            column_mapping:
              date: dt
              revenue: amt
            storage:
              format: parquet
              paths:
                - public.rows
"#;
        let manifest = ManifestCompiler::new()
            .compile(CompileSource::Yaml(yaml.into()))
            .await
            .expect("manifest compile");

        let raw = RawQueryRequest {
            from: None,
            select: vec!["date".to_string(), "revenue".to_string()],
            raw_filters: vec![
                RawFilter {
                    field: "date".to_string(),
                    operator: ">=".to_string(),
                    value: serde_json::json!("2026-05-01"),
                },
                RawFilter {
                    field: "date".to_string(),
                    operator: "<=".to_string(),
                    value: serde_json::json!("2026-05-31"),
                },
            ],
            ..Default::default()
        };

        let resolved = RequestParser::to_resolved(&raw, &manifest).expect("to_resolved");

        assert!(
            resolved.inline_filters.is_empty(),
            "inline_filters must stay empty when `from` is omitted; \
             pending_inline_filters carries the raw filters until the \
             planner lowers them"
        );
        assert_eq!(
            resolved.pending_inline_filters.len(),
            2,
            "both raw_filters should be carried as PendingInlineFilters"
        );
        // Order and content preservation (no operator canonicalisation,
        // no type coercion at this stage — strings stay strings).
        assert_eq!(resolved.pending_inline_filters[0].field, "date");
        assert_eq!(resolved.pending_inline_filters[0].operator, ">=");
        assert_eq!(
            resolved.pending_inline_filters[0].value,
            serde_json::json!("2026-05-01")
        );
        assert_eq!(resolved.pending_inline_filters[1].operator, "<=");

        // entity_name stays empty so the planner enters ad-hoc resolution.
        assert!(resolved.entity_name.is_empty());
    }

    #[tokio::test]
    async fn test_to_resolved_explicit_from_lowers_raw_filters_eagerly() {
        // With explicit `from`, the parser delegates to
        // `inline_filter::lower_inline_filter` immediately. The result
        // lands in `inline_filters`, not `pending_inline_filters`.
        use semstrait_manifest::{CompileSource, ManifestCompiler};

        let yaml = r#"
semantic_model:
  name: m
  grainsets:
    - name: orders
      dimensions:
        - name: region
          data_type: string
          type:
            categorical:
      measures:
        - name: revenue
          data_type: float64
          agg: sum
      datasets:
        - name: orders_ds
          extras:
            column_mapping:
              region: r
              revenue: amt
            storage:
              format: parquet
              paths:
                - public.orders
"#;
        let manifest = ManifestCompiler::new()
            .compile(CompileSource::Yaml(yaml.into()))
            .await
            .expect("manifest compile");

        let raw = RawQueryRequest {
            from: Some("orders".to_string()),
            select: vec!["region".to_string(), "revenue".to_string()],
            raw_filters: vec![RawFilter {
                field: "region".to_string(),
                operator: "eq".to_string(),
                value: serde_json::json!("US"),
            }],
            ..Default::default()
        };

        let resolved = RequestParser::to_resolved(&raw, &manifest).expect("to_resolved");

        assert_eq!(resolved.inline_filters.len(), 1);
        assert!(resolved.pending_inline_filters.is_empty());
        assert_eq!(resolved.inline_filters[0].name, "__inline_filter_0");
    }

    #[tokio::test]
    async fn test_to_resolved_explicit_from_maps_inline_filter_error() {
        // `InlineFilterError` from the shared helper should be mapped back
        // to the corresponding `ParseError::RawFilter*` variant via
        // `From<InlineFilterError> for ParseError`. We use an unknown field
        // here to trigger `FieldNotFound`.
        use semstrait_manifest::{CompileSource, ManifestCompiler};

        let yaml = r#"
semantic_model:
  name: m
  grainsets:
    - name: orders
      dimensions:
        - name: region
          data_type: string
          type:
            categorical:
      measures:
        - name: revenue
          data_type: float64
          agg: sum
      datasets:
        - name: orders_ds
          extras:
            column_mapping:
              region: r
              revenue: amt
            storage:
              format: parquet
              paths:
                - public.orders
"#;
        let manifest = ManifestCompiler::new()
            .compile(CompileSource::Yaml(yaml.into()))
            .await
            .expect("manifest compile");

        let raw = RawQueryRequest {
            from: Some("orders".to_string()),
            select: vec!["revenue".to_string()],
            raw_filters: vec![RawFilter {
                field: "nonexistent".to_string(),
                operator: "eq".to_string(),
                value: serde_json::json!("X"),
            }],
            ..Default::default()
        };

        let err = RequestParser::to_resolved(&raw, &manifest).unwrap_err();
        assert!(matches!(err, ParseError::RawFilterFieldNotFound { .. }));
    }
}
