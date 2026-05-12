//! Compilation pipeline steps 3-9.
//!
//! Steps 1 (parse) and 2 (resolve_refs) are handled by semstrait-model.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use indexmap::IndexMap;
use petgraph::algo::is_cyclic_directed;
use petgraph::graph::DiGraph;

use semstrait_catalog::{CatalogProvider, CatalogRegistry, StorageProvider};
use semstrait_core::Expr;
use semstrait_model::*;

use crate::compiled::*;
use crate::error::CompileError;

// ============================================================================
// Step 3: Resolve Sources
// ============================================================================

/// Result of source resolution (step 3).
///
/// Contains per-dataset resolved sources and catalog snapshot data.
/// Consumed by `emit()` to populate `ResolvedSource` fields on `DatasetBinding`
/// and attach `CatalogSnapshot` to the manifest.
#[derive(Debug, Default)]
pub(crate) struct SourceResolutionResult {
    /// Resolved sources per dataset, keyed by `dataset_display_name(&ds.name)`.
    ///
    /// INVARIANT: Both `resolve_sources()` and `compile_kind()` must use
    /// `dataset_display_name()` to derive the key. If the naming function
    /// changes, both sites must be updated in lockstep.
    pub resolved: HashMap<String, Vec<crate::acceleration::ResolvedSource>>,
    /// Catalog snapshot assembled from resolved table metadata.
    pub catalog_snapshot: Option<crate::catalog_snapshot::CatalogSnapshot>,
    /// Warnings accumulated during resolution.
    pub warnings: Vec<crate::acceleration::CompileWarning>,
}

/// Resolve all physical sources: expand globs/wildcards, fetch catalog metadata.
///
/// For `storage.tables`: looks up `extras.catalog.alias` in the registry,
/// resolves via `CatalogProvider`, populates location/schema/format.
///
/// For `storage.paths`: uses `StorageProvider` to expand globs,
/// populates format from `StorageConfig.format`.
///
/// This is the single step where all physical binding happens.
pub(crate) async fn resolve_sources(
    model: &SemanticModel,
    registry: Option<&CatalogRegistry>,
    legacy_catalog: Option<&dyn CatalogProvider>,
    storage: Option<&dyn StorageProvider>,
) -> Result<SourceResolutionResult, CompileError> {
    let mut result = SourceResolutionResult::default();
    let namespace = model.namespace.as_deref().unwrap_or("default");
    let mut table_snapshots: HashMap<String, crate::catalog_snapshot::TableSnapshot> =
        HashMap::new();

    // Resolve sources for all entities
    for dk in model.entities.values() {
        match dk {
            DataKind::Simple(dsk) => {
                // Standalone dataset with storage config
                if let Some(extras) = &dsk.extras {
                    if let Some(storage_config) = &extras.storage {
                        let sources = resolve_dataset_storage(
                            &dsk.name,
                            storage_config,
                            extras.catalog.as_ref(),
                            namespace,
                            registry,
                            legacy_catalog,
                            storage,
                            &mut table_snapshots,
                            &mut result.warnings,
                        )
                        .await?;
                        if !sources.is_empty() {
                            result.resolved.insert(dsk.name.clone(), sources);
                        }
                    }
                }
            }
            _ => {
                // Kind with children — resolve each child dataset
                if let Some(children) = dk.children() {
                    for entry in children {
                        match entry {
                            ChildEntry::Inline(ds) => {
                                let ds_name = dataset_display_name(&ds.name).to_string();
                                if let Some(storage_config) = &ds.extras.storage {
                                    let sources = resolve_dataset_storage(
                                        &ds_name,
                                        storage_config,
                                        ds.extras.catalog.as_ref(),
                                        namespace,
                                        registry,
                                        legacy_catalog,
                                        storage,
                                        &mut table_snapshots,
                                        &mut result.warnings,
                                    )
                                    .await?;
                                    if !sources.is_empty() {
                                        result.resolved.insert(ds_name, sources);
                                    }
                                }
                            }
                            ChildEntry::Ref(_) => {
                                // Nested kind reference — compiled separately as its own
                                // CompiledDataKind entry. No storage resolution needed here.
                            }
                        }
                    }
                }
            }
        }
    }

    // Assemble CatalogSnapshot
    if !table_snapshots.is_empty() {
        result.catalog_snapshot = Some(crate::catalog_snapshot::CatalogSnapshot {
            tables: table_snapshots,
            captured_at: chrono::Utc::now(),
        });
    }

    Ok(result)
}

/// Resolve storage sources for a single dataset.
///
/// Handles both `storage.paths` (via StorageProvider) and `storage.tables`
/// (via CatalogRegistry/CatalogProvider). Accumulates table snapshots and warnings.
#[allow(clippy::too_many_arguments)]
async fn resolve_dataset_storage(
    ds_name: &str,
    storage_config: &StorageConfig,
    catalog_ref: Option<&CatalogRef>,
    namespace: &str,
    registry: Option<&CatalogRegistry>,
    legacy_catalog: Option<&dyn CatalogProvider>,
    storage: Option<&dyn StorageProvider>,
    table_snapshots: &mut HashMap<String, crate::catalog_snapshot::TableSnapshot>,
    warnings: &mut Vec<crate::acceleration::CompileWarning>,
) -> Result<Vec<crate::acceleration::ResolvedSource>, CompileError> {
    let mut sources = Vec::new();

    // Resolve storage.paths via StorageProvider
    if !storage_config.paths.is_empty() {
        for path_pattern in &storage_config.paths {
            let expanded = if let Some(sp) = storage {
                if contains_glob_chars(path_pattern) {
                    let paths = sp
                        .expand_glob(path_pattern)
                        .await
                        .map_err(|e| CompileError::CatalogError(e.to_string()))?;
                    if paths.is_empty() {
                        warnings.push(crate::acceleration::CompileWarning {
                            code: "SRC_W001".to_string(),
                            message: format!(
                                "glob pattern '{}' in dataset '{}' matched no files",
                                path_pattern, ds_name
                            ),
                            location: ds_name.to_string(),
                        });
                    }
                    paths
                } else {
                    vec![path_pattern.clone()]
                }
            } else if contains_glob_chars(path_pattern) {
                return Err(CompileError::CatalogError(format!(
                    "wildcard pattern '{}' in dataset '{}' requires a storage provider",
                    path_pattern, ds_name
                )));
            } else {
                vec![path_pattern.clone()]
            };

            for path in expanded {
                let schema = if let Some(sp) = storage {
                    if let Some(fmt) = storage_config.format {
                        sp.read_schema(&path, fmt)
                            .await
                            .unwrap_or(None)
                            .map(|cols| {
                                cols.into_iter()
                                    .map(|c| crate::catalog_snapshot::ResolvedColumn {
                                        name: c.name,
                                        data_type: c.data_type,
                                        nullable: c.nullable,
                                        comment: c.comment,
                                        field_id: None,
                                    })
                                    .collect()
                            })
                    } else {
                        None
                    }
                } else {
                    None
                };

                sources.push(crate::acceleration::ResolvedSource {
                    reference: path_pattern.clone(),
                    source_type: crate::acceleration::SourceType::Path,
                    table_fqn: None,
                    location: Some(path),
                    format: storage_config.format,
                    catalog_alias: None,
                    schema,
                });
            }
        }
    }

    // Resolve storage.tables via CatalogRegistry or legacy catalog
    if !storage_config.tables.is_empty() {
        let catalog_alias = catalog_ref.map(|c| c.alias.as_str());
        let catalog_namespace = catalog_ref
            .and_then(|c| c.namespace.as_deref())
            .unwrap_or(namespace);

        // Look up the catalog provider
        let provider: Option<&dyn CatalogProvider> =
            if let (Some(alias), Some(reg)) = (catalog_alias, registry) {
                reg.get(alias).map(|arc| arc.as_ref())
            } else {
                legacy_catalog
            };

        for table_pattern in &storage_config.tables {
            let concrete_tables = if let Some(cat) = provider {
                if contains_glob_chars(table_pattern) {
                    // Split "adwords.*" → (namespace="adwords", glob="*").
                    // Bare glob "*" uses catalog_namespace as default.
                    let (ns, table_glob) =
                        split_table_pattern(table_pattern, catalog_namespace);
                    let glob = semstrait_core::GlobPattern::new(table_glob);
                    let tables = cat
                        .list_tables(ns, &glob)
                        .await
                        .map_err(|e| CompileError::CatalogError(e.to_string()))?;
                    if tables.is_empty() {
                        warnings.push(crate::acceleration::CompileWarning {
                            code: "SRC_W002".to_string(),
                            message: format!(
                                "table pattern '{}' in dataset '{}' matched no tables",
                                table_pattern, ds_name
                            ),
                            location: ds_name.to_string(),
                        });
                    }
                    tables
                        .into_iter()
                        .map(|t| t.fully_qualified())
                        .collect::<Vec<_>>()
                } else {
                    vec![table_pattern.clone()]
                }
            } else if contains_glob_chars(table_pattern) {
                return Err(CompileError::CatalogError(format!(
                    "wildcard pattern '{}' in dataset '{}' requires a catalog provider",
                    table_pattern, ds_name
                )));
            } else {
                vec![table_pattern.clone()]
            };

            for table_fqn in concrete_tables {
                let mut resolved = crate::acceleration::ResolvedSource {
                    reference: table_pattern.clone(),
                    source_type: crate::acceleration::SourceType::Table,
                    table_fqn: Some(table_fqn.clone()),
                    location: None,
                    format: None,
                    catalog_alias: catalog_alias.map(|s| s.to_string()),
                    schema: None,
                };

                // Fetch table metadata from catalog
                if let Some(cat) = provider {
                    let table_ref = parse_table_ref(&table_fqn, catalog_namespace);
                    match cat.load_table_metadata(&table_ref).await {
                        Ok(Some(meta)) => {
                            resolved.location = meta.location.clone();
                            resolved.format = Some(
                                meta.format
                                    .unwrap_or(semstrait_core::DataFormat::Iceberg),
                            );

                            let columns: Vec<crate::catalog_snapshot::ResolvedColumn> =
                                meta.columns
                                    .iter()
                                    .map(|c| crate::catalog_snapshot::ResolvedColumn {
                                        name: c.name.clone(),
                                        data_type: c.data_type.clone(),
                                        nullable: c.nullable,
                                        comment: c.comment.clone(),
                                        field_id: None,
                                    })
                                    .collect();

                            resolved.schema = Some(columns.clone());

                            // Build table snapshot for CatalogSnapshot
                            let partition_spec = meta
                                .partition_fields
                                .iter()
                                .map(|pf| {
                                    let transform =
                                        crate::catalog_snapshot::PartitionTransform::parse(
                                            &pf.transform,
                                        );
                                    let inferred_grain = transform
                                        .as_ref()
                                        .and_then(|t| t.inferred_grain());
                                    crate::catalog_snapshot::PartitionField {
                                        source_column: pf.source_column.clone(),
                                        transform: transform.unwrap_or(
                                            crate::catalog_snapshot::PartitionTransform::Identity,
                                        ),
                                        name: pf.name.clone(),
                                        inferred_grain,
                                    }
                                })
                                .collect();

                            let iceberg_meta = if meta.snapshot_id.is_some()
                                || !meta.partition_fields.is_empty()
                                || meta.format_version.is_some()
                            {
                                Some(crate::catalog_snapshot::IcebergMetadata {
                                    snapshot_id: meta.snapshot_id,
                                    partition_spec,
                                    format_version: meta.format_version,
                                    location: meta.location.clone(),
                                    properties: meta.properties.clone(),
                                })
                            } else {
                                None
                            };

                            table_snapshots.insert(
                                table_fqn.clone(),
                                crate::catalog_snapshot::TableSnapshot {
                                    fqn: table_fqn.clone(),
                                    columns,
                                    iceberg: iceberg_meta,
                                },
                            );
                        }
                        Ok(None) => {
                            // No extended metadata — try basic schema
                            if let Ok(cols) = cat.get_schema(&table_ref).await {
                                let columns: Vec<
                                    crate::catalog_snapshot::ResolvedColumn,
                                > = cols
                                    .iter()
                                    .map(|c| crate::catalog_snapshot::ResolvedColumn {
                                        name: c.name.clone(),
                                        data_type: c.data_type.clone(),
                                        nullable: c.nullable,
                                        comment: c.comment.clone(),
                                        field_id: None,
                                    })
                                    .collect();
                                resolved.schema = Some(columns.clone());

                                table_snapshots.insert(
                                    table_fqn.clone(),
                                    crate::catalog_snapshot::TableSnapshot {
                                        fqn: table_fqn.clone(),
                                        columns,
                                        iceberg: None,
                                    },
                                );
                            }
                        }
                        Err(e) => {
                            warnings.push(crate::acceleration::CompileWarning {
                                code: "CAT_W002".to_string(),
                                message: format!(
                                    "could not resolve table '{}': {}",
                                    table_fqn, e
                                ),
                                location: table_fqn.clone(),
                            });
                        }
                    }
                }

                sources.push(resolved);
            }
        }
    }

    Ok(sources)
}

/// Check if a string contains glob/wildcard characters (`*`, `?`, `[`).
fn contains_glob_chars(s: &str) -> bool {
    s.contains('*') || s.contains('?') || s.contains('[')
}

// ============================================================================
// Step 4: Validate Structure
// ============================================================================

/// Validate structural integrity of the model.
pub(crate) fn validate_structure(model: &SemanticModel) -> Result<(), CompileError> {
    let mut errors = Vec::new();

    // Check entity name uniqueness (all entities share a namespace)
    let mut seen_names = HashSet::new();
    for dk in model.entities.values() {
        let name = dk.name();
        if !seen_names.insert(name.to_string()) {
            errors.push(format!("duplicate entity name: '{}'", name));
        }

        // Non-dataset kinds must have at least one child dataset
        if let Some(children) = dk.children() {
            if children.is_empty() {
                errors.push(format!(
                    "{} '{}' must have at least one dataset",
                    dk.variant(), name
                ));
            }
        }

        // Joinsets must have relationships
        if dk.is_joinset() && dk.relationships().is_empty() {
            errors.push(format!(
                "joinset '{}' must have at least one relationship",
                name
            ));
        }

        // Check duplicate dimension/measure/metric names
        check_dim_uniqueness(&dk.interface().dimensions, name, &mut errors);
        check_measure_uniqueness(&dk.interface().measures, name, &mut errors);
        check_metric_uniqueness(&dk.interface().metrics, name, &mut errors);

        // Nesting matrix validation: check that ChildEntry::Ref targets
        // are allowed child kinds for this parent kind.
        if let Some(children) = dk.children() {
            let parent_variant = dk.variant_enum();
            for child_entry in children {
                if let ChildEntry::Ref(r) = child_entry {
                    use semstrait_model::DataKindVariant;
                    match (parent_variant, r.variant) {
                        (DataKindVariant::Grainset, DataKindVariant::Grainset) => {
                            errors.push(format!(
                                "grainset '{}' cannot nest grainset '{}' \
                                 (prohibited: flatten to a single grainset)",
                                name, r.ref_name
                            ));
                        }
                        (DataKindVariant::Joinset, DataKindVariant::Joinset) => {
                            errors.push(format!(
                                "joinset '{}' cannot nest joinset '{}' \
                                 (prohibited: creates ambiguous join graph)",
                                name, r.ref_name
                            ));
                        }
                        // unionset → unionset is allowed with warning (COMP_W010),
                        // emitted as a diagnostic during compilation, not a hard error.
                        _ => {} // all other combinations are valid
                    }
                }
            }
        }
    }

    // Nesting depth limit: max 2 levels in v1.
    for dk in model.entities.values() {
        if dk.children().is_some() {
            let depth = measure_nesting_depth(&model.entities, dk.name(), 0);
            if depth > 2 {
                errors.push(format!(
                    "{} '{}' exceeds maximum nesting depth of 2 (found {})",
                    dk.variant(), dk.name(), depth
                ));
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(CompileError::StructureValidation(errors))
    }
}

/// Measure the maximum nesting depth from a given kind, following Ref chains.
fn measure_nesting_depth(
    data_kinds: &BTreeMap<String, DataKind>,
    name: &str,
    current: usize,
) -> usize {
    let Some(dk) = data_kinds.get(name) else { return current };
    let Some(children) = dk.children() else { return current };

    let mut max_depth = current;
    for child in children {
        if let ChildEntry::Ref(r) = child {
            max_depth = max_depth.max(
                measure_nesting_depth(data_kinds, &r.ref_name, current + 1)
            );
        }
    }
    max_depth
}

fn check_dim_uniqueness(entries: &BTreeMap<String, DimensionEntry>, container: &str, errors: &mut Vec<String>) {
    let mut names = HashSet::new();
    for entry in entries.values() {
        if let DimensionEntry::Inline(d) = entry {
            if !names.insert(&d.name) {
                errors.push(format!(
                    "duplicate dimension '{}' in '{}'",
                    d.name, container
                ));
            }
        }
    }
}

fn check_measure_uniqueness(entries: &BTreeMap<String, MeasureEntry>, container: &str, errors: &mut Vec<String>) {
    let mut names = HashSet::new();
    for entry in entries.values() {
        if let MeasureEntry::Inline(m) = entry {
            if !names.insert(&m.name) {
                errors.push(format!(
                    "duplicate measure '{}' in '{}'",
                    m.name, container
                ));
            }
        }
    }
}

fn check_metric_uniqueness(entries: &BTreeMap<String, MetricEntry>, container: &str, errors: &mut Vec<String>) {
    let mut names = HashSet::new();
    for entry in entries.values() {
        if let MetricEntry::Inline(m) = entry {
            if !names.insert(&m.name) {
                errors.push(format!(
                    "duplicate metric '{}' in '{}'",
                    m.name, container
                ));
            }
        }
    }
}

// ============================================================================
// Step 4.6: Validate Temporal Equivalence
// ============================================================================

/// Validate that when both a kind and a dataset define a temporal type,
/// their temporal variant (timeseries/snapshot/scd) must match.
///
/// This must run BEFORE `expand_auto_mappings` because that step propagates
/// kind-level temporal defaults, which would overwrite dataset values.
pub(crate) fn validate_temporal_equivalence(
    model: &SemanticModel,
) -> Result<(), CompileError> {
    for dk in model.entities.values() {
        if dk.is_simple() { continue; }

        let kind_temporal = dk
            .complex_extras()
            .and_then(|e| e.temporal.as_ref());

        let kind_temporal = match kind_temporal {
            Some(t) => t,
            None => continue, // No kind-level temporal; nothing to conflict with.
        };

        for ds_entry in dk.children().unwrap_or(&[]) {
            if let ChildEntry::Inline(ds) = ds_entry {
                if let Some(ds_temporal) = &ds.extras.temporal {
                    let kind_variant = kind_temporal.temporal_type.variant_name();
                    let ds_variant = ds_temporal.temporal_type.variant_name();
                    if kind_variant != ds_variant {
                        return Err(CompileError::TemporalMismatch {
                            kind: dk.name().to_string(),
                            dataset: dataset_display_name(&ds.name).to_string(),
                            kind_type: kind_variant.to_string(),
                            dataset_type: ds_variant.to_string(),
                        });
                    }
                }
            }
        }
    }

    Ok(())
}

// ============================================================================
// Step 4.55: Validate temporal.dimension consistency across datasets
// ============================================================================

/// Ensure all datasets within a kind agree on `temporal.dimension`.
///
/// After `expand_auto_mappings` propagates kind-level temporal to datasets,
/// every dataset that specifies `temporal.dimension` must use the same value.
/// This makes `build_interface()`'s `find_map` provably correct — any pick
/// is the right pick because all datasets agree.
pub(crate) fn validate_temporal_dimension_consistency(
    model: &SemanticModel,
) -> Result<(), CompileError> {
    for dk in model.entities.values() {
        if dk.is_simple() { continue; }
        let mut seen: BTreeSet<String> = BTreeSet::new();
        for ds_entry in dk.children().unwrap_or(&[]) {
            if let ChildEntry::Inline(ds) = ds_entry {
                if let Some(ref temporal) = ds.extras.temporal {
                    if let Some(ref dim_name) = temporal.dimension {
                        seen.insert(dim_name.clone());
                    }
                }
            }
        }
        if seen.len() > 1 {
            return Err(CompileError::TemporalDimensionConflict {
                kind: dk.name().to_string(),
                values: seen.into_iter().collect::<Vec<_>>().join(", "),
            });
        }
    }
    Ok(())
}

// ============================================================================
// Step 4.7: Validate Storage Config
// ============================================================================

/// Validate storage config preconditions: paths/tables mutually exclusive,
/// at least one source when storage is defined, no empty strings.
pub(crate) fn validate_storage(model: &SemanticModel) -> Result<(), CompileError> {
    let mut errors = Vec::new();

    for dk in model.entities.values() {
        match dk {
            DataKind::Simple(dsk) => {
                if let Some(ref extras) = dsk.extras {
                    if let Some(ref storage) = extras.storage {
                        let ctx = format!("dataset '{}'", dsk.name);
                        validate_storage_config(storage, &ctx, &mut errors);
                    }
                }
            }
            _ => {
                for ds_entry in dk.children().unwrap_or(&[]) {
                    if let ChildEntry::Inline(ds) = ds_entry {
                        if let Some(ref storage) = ds.extras.storage {
                            let ctx = format!("{} '{}', dataset '{}'", dk.variant(), dk.name(), dataset_display_name(&ds.name));
                            validate_storage_config(storage, &ctx, &mut errors);
                        }
                    }
                }
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(CompileError::StructureValidation(errors))
    }
}

fn validate_storage_config(
    storage: &semstrait_model::StorageConfig,
    ctx: &str,
    errors: &mut Vec<String>,
) {
    if !storage.paths.is_empty() && !storage.tables.is_empty() {
        errors.push(format!("{ctx}: storage cannot mix paths and tables"));
    }
    if storage.paths.is_empty() && storage.tables.is_empty() {
        errors.push(format!("{ctx}: storage must specify at least one path or table"));
    }
    if !storage.paths.is_empty() && storage.format.is_none() {
        errors.push(format!("{ctx}: storage with paths requires a format (parquet, csv, iceberg)"));
    }
    if !storage.tables.is_empty() && storage.format.is_some() {
        errors.push(format!("{ctx}: storage with tables must not specify format (catalog determines it)"));
    }
    for src in storage.paths.iter().chain(storage.tables.iter()) {
        if src.trim().is_empty() {
            errors.push(format!("{ctx}: storage source must not be empty"));
        }
    }
}

// ============================================================================
// Step 4.8: Validate Metadata Dimensions
// ============================================================================

/// Validate that metadata dimensions have the required preconditions:
/// - `path.token` requires storage config with at least one path (file/object store).
/// - `partition.level` requires partition_defs on the dataset (or kind) extras.
/// - `partition.level` must not exceed the partition depth.
pub(crate) fn validate_metadata_dimensions(model: &SemanticModel) -> Result<(), CompileError> {
    let mut errors = Vec::new();

    for dk in model.entities.values() {
        if dk.is_simple() { continue; }
        // Collect kind-level dimensions that are metadata type.
        for dim in dk.interface().dimensions.values() {
            if let DimensionEntry::Inline(dim_def) = dim {
                if let DimensionType::Metadata(ref meta) = dim_def.dim_type {
                    // Check each dataset for metadata dimension preconditions.
                    for ds_entry in dk.children().unwrap_or(&[]) {
                        if let ChildEntry::Inline(ds) = ds_entry {
                            let ds_display = dataset_display_name(&ds.name);
                            validate_metadata_for_dataset(
                                dk.name(),
                                ds_display,
                                &dim_def.name,
                                meta,
                                &ds.extras,
                                &mut errors,
                            );
                        }
                    }
                }
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(CompileError::StructureValidation(errors))
    }
}

fn validate_metadata_for_dataset(
    kind_name: &str,
    ds_display: &str,
    dim_name: &str,
    meta: &MetadataDimension,
    extras: &InlineDatasetExtras,
    errors: &mut Vec<String>,
) {
    if let Some(ref path_ext) = meta.path {
        // path.token requires storage with paths OR tables (catalog tables resolve
        // to physical locations at compile time, providing extractable paths).
        let has_sources = extras.storage.as_ref().is_some_and(|s| {
            !s.paths.is_empty() || !s.tables.is_empty()
        });
        if !has_sources {
            errors.push(format!(
                "kind '{}', dataset '{}': metadata dimension '{}' uses path.token={} \
                 but dataset has no storage sources configured (paths or tables)",
                kind_name, ds_display, dim_name, path_ext.token
            ));
        }
    }

    if let Some(ref part_ext) = meta.partition {
        if part_ext.level == 0 {
            errors.push(format!(
                "kind '{}', dataset '{}': metadata dimension '{}' uses partition.level=0 \
                 but level is 1-indexed (must be >= 1)",
                kind_name, ds_display, dim_name
            ));
        }

        // partition.level requires partition_def on the dataset's storage config.
        // StorageConfig.partition_def is a single PartitionDef (depth=1).
        let storage_partition = extras
            .storage
            .as_ref()
            .and_then(|s| s.partition_def.as_ref())
            .map(|_| 1usize) // single partition_def = depth 1
            .unwrap_or(0);

        let partition_depth = storage_partition;

        if partition_depth == 0 {
            errors.push(format!(
                "kind '{}', dataset '{}': metadata dimension '{}' uses partition.level={} \
                 but dataset has no partition definitions",
                kind_name, ds_display, dim_name, part_ext.level
            ));
        } else if part_ext.level > partition_depth {
            errors.push(format!(
                "kind '{}', dataset '{}': metadata dimension '{}' uses partition.level={} \
                 but partition depth is only {}",
                kind_name, ds_display, dim_name, part_ext.level, partition_depth
            ));
        }
    }

    // At least one of path or partition must be specified.
    if meta.path.is_none() && meta.partition.is_none() {
        errors.push(format!(
            "kind '{}', dataset '{}': metadata dimension '{}' must specify \
             either 'path' or 'partition' extraction",
            kind_name, ds_display, dim_name
        ));
    }
}

// ============================================================================
// Step 4.5: Expand Auto Column Mappings
// ============================================================================

/// Expand `column_mapping: auto` / `inherited` into explicit identity mappings,
/// and merge kind-level defaults into each dataset's extras.
///
/// Handles three cases for a dataset's `column_mapping`:
///   - `Auto`:      1:1 identity from all kind interface names.
///   - `Inherited`: use `kind.extras.column_mapping`, falling back to identity.
///   - `Explicit`:  start from kind default (if any), then apply dataset overrides.
///
/// After this step every dataset has `ColumnMapping::Explicit`. `temporal` and
/// `catalog` defaults from `kind.extras` are also propagated (dataset value wins).
pub(crate) fn expand_auto_mappings(model: &mut SemanticModel) {
    for dk in model.entities.values_mut() {
        // Simple kinds: expand column_mapping on the dataset extras directly.
        // Simple kinds have no children — the kind IS the dataset.
        if let DataKind::Simple(dsk) = dk {
            let interface_names: Vec<String> = collect_mappable_names_simple(dsk);
            if let Some(ref mut extras) = dsk.extras {
                let effective = match &extras.column_mapping {
                    ColumnMapping::Auto | ColumnMapping::Inherited => {
                        // Identity mapping — each semantic name maps to itself.
                        interface_names
                            .iter()
                            .map(|n| (n.clone(), ColumnMappingValue::Simple(n.clone())))
                            .collect()
                    }
                    ColumnMapping::Explicit(ds_map) => {
                        // Start from identity, then apply user overrides.
                        let mut merged: HashMap<String, ColumnMappingValue> = interface_names
                            .iter()
                            .map(|n| (n.clone(), ColumnMappingValue::Simple(n.clone())))
                            .collect();
                        merged.extend(ds_map.clone());
                        merged
                    }
                };
                extras.column_mapping = ColumnMapping::Explicit(effective);
            }
            continue;
        }

        // Use mappable names (excludes metadata dimensions and metrics)
        // since those entities don't require physical column mapping.
        let interface_names: Vec<String> = collect_mappable_names(dk).collect();

        // Resolve the kind-level default mapping once per kind.
        let kind_default: Option<HashMap<String, ColumnMappingValue>> =
            dk.complex_extras().and_then(|e| e.column_mapping.as_ref()).map(|cm| {
                match cm {
                    ColumnMapping::Auto | ColumnMapping::Inherited => interface_names
                        .iter()
                        .map(|n| (n.clone(), ColumnMappingValue::Simple(n.clone())))
                        .collect(),
                    ColumnMapping::Explicit(m) => m.clone(),
                }
            });

        // Clone kind extras before mutable borrow of children.
        let kind_extras_temporal = dk.complex_extras().and_then(|e| e.temporal.clone());
        let kind_extras_catalog = dk.complex_extras().and_then(|e| e.catalog.clone());

        for ds_entry in dk.children_mut().unwrap() {
            if let ChildEntry::Inline(ds) = ds_entry {
                let effective: HashMap<String, ColumnMappingValue> = match &ds.extras.column_mapping {
                    ColumnMapping::Auto => {
                        // Identity map — same behaviour as before.
                        interface_names
                            .iter()
                            .map(|n| (n.clone(), ColumnMappingValue::Simple(n.clone())))
                            .collect()
                    }
                    ColumnMapping::Inherited => {
                        // Use kind default; fall back to identity if no kind default exists.
                        kind_default.clone().unwrap_or_else(|| {
                            interface_names
                                .iter()
                                .map(|n| (n.clone(), ColumnMappingValue::Simple(n.clone())))
                                .collect()
                        })
                    }
                    ColumnMapping::Explicit(ds_map) => {
                        // Merge: kind default is the base; dataset entries override.
                        let mut merged = kind_default.clone().unwrap_or_default();
                        merged.extend(ds_map.clone());
                        merged
                    }
                };
                // Flatten Anchored entries: insert anchor sub-names as Simple
                // mappings so that resolve_name can resolve them during planning.
                let mut anchor_expansions: Vec<(String, ColumnMappingValue)> = Vec::new();
                for value in effective.values() {
                    if let ColumnMappingValue::Anchored(anchors) = value {
                        for (anchor_name, physical_col) in anchors {
                            anchor_expansions.push((
                                anchor_name.clone(),
                                ColumnMappingValue::Simple(physical_col.clone()),
                            ));
                        }
                    }
                }
                let mut effective = effective;
                for (name, value) in anchor_expansions {
                    effective.entry(name).or_insert(value);
                }

                ds.extras.column_mapping = ColumnMapping::Explicit(effective);

                // Propagate temporal and catalog defaults (dataset value always wins).
                if ds.extras.temporal.is_none() {
                    ds.extras.temporal = kind_extras_temporal.clone();
                }
                if ds.extras.catalog.is_none() {
                    ds.extras.catalog = kind_extras_catalog.clone();
                }

                // Grain auto-propagation: when temporal.grain is set and the
                // temporal dimension's column_mapping points to the same physical
                // column as the temporal config's occurred_at/snapshotted_at,
                // auto-set the mapping grain. Explicit grain always wins.
                if let Some(ref temporal) = ds.extras.temporal {
                    if let (Some(grain), Some(ref dim_name)) = (temporal.grain, &temporal.dimension) {
                        let temporal_physical = match &temporal.temporal_type {
                            TemporalHistorization::Events(e) => Some(&e.occurred_at),
                            TemporalHistorization::Timeseries(t) => Some(&t.occurred_at),
                            TemporalHistorization::Snapshot(s) => Some(&s.snapshotted_at),
                            TemporalHistorization::Scd(_) => None,
                        };
                        if let Some(temporal_col) = temporal_physical {
                            if let ColumnMapping::Explicit(ref mut mapping) = ds.extras.column_mapping {
                                let should_propagate = match mapping.get(dim_name) {
                                    Some(ColumnMappingValue::Simple(col)) => col == temporal_col,
                                    Some(ColumnMappingValue::WithGrain { column, grain: existing }) =>
                                        existing.is_none() && column == temporal_col,
                                    _ => false,
                                };
                                if should_propagate {
                                    mapping.insert(dim_name.clone(), ColumnMappingValue::WithGrain {
                                        column: temporal_col.clone(),
                                        grain: Some(grain),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Step 4.9: Derive Dimension Grains
// ============================================================================

/// Derive temporal dimension grains from dataset temporal configs when the
/// dimension's grains list is empty.
///
/// For each kind with temporal dimensions that have empty grains:
/// - Collect `temporal.grain` values from all datasets with `temporal.dimension` matching
/// - Set dimension grains = all standard grains coarser-or-equal to finest dataset grain
/// - Emit COMP_I001 diagnostic
pub(crate) fn derive_dimension_grains(
    model: &mut SemanticModel,
    diagnostics: &mut Vec<crate::acceleration::CompileWarning>,
) {
    for dk in model.entities.values_mut() {
        if dk.is_simple() { continue; }

        // Collect dataset grains keyed by temporal.dimension name.
        let mut dim_grains: HashMap<String, Vec<TemporalGrain>> = HashMap::new();
        if let Some(children) = dk.children() {
            for ds_entry in children {
                if let ChildEntry::Inline(ds) = ds_entry {
                    if let Some(ref temporal) = ds.extras.temporal {
                        if let (Some(grain), Some(ref dim_name)) = (temporal.grain, &temporal.dimension) {
                            dim_grains.entry(dim_name.clone()).or_default().push(grain);
                        }
                    }
                }
            }
        }

        let dk_name = dk.name().to_string();

        // For temporal dims with empty grains, derive from datasets.
        for dim_entry in dk.interface_mut().dimensions.values_mut() {
            if let DimensionEntry::Inline(ref mut dim) = dim_entry {
                if let DimensionType::Temporal(ref mut td) = dim.dim_type {
                    if td.grains.is_empty() {
                        if let Some(dataset_grains) = dim_grains.get(&dim.name) {
                            let finest = dataset_grains.iter()
                                .copied()
                                .min_by_key(|g| g.coarseness());
                            if let Some(finest_grain) = finest {
                                let derived: Vec<TemporalGrain> = TemporalGrain::ALL.into_iter()
                                    .filter(|g| g.coarseness() >= finest_grain.coarseness())
                                    .collect();

                                diagnostics.push(crate::acceleration::CompileWarning {
                                    code: "COMP_I001".to_string(),
                                    message: format!(
                                        "auto-derived grains [{grains}] for temporal dimension '{dim}' from dataset temporal configs",
                                        grains = derived.iter().map(|g| format!("{:?}", g)).collect::<Vec<_>>().join(", "),
                                        dim = dim.name,
                                    ),
                                    location: format!("{} '{}'", "kind", dk_name),
                                });

                                td.grains = derived;
                            }
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Step 5: Validate Mappings
// ============================================================================

/// Validate that column_mapping keys in kind datasets correspond to
/// dimensions/measures/metrics declared in the kind's interface.
pub(crate) fn validate_mappings(model: &SemanticModel) -> Result<(), CompileError> {
    let mut errors = Vec::new();

    for dk in model.entities.values() {
        if dk.is_simple() { continue; }
        let dk_name = dk.name();
        let interface_names: HashSet<String> = collect_interface_names(dk).collect();

        // Check each dataset's column_mapping
        for ds_entry in dk.children().unwrap_or(&[]) {
            if let ChildEntry::Inline(ds) = ds_entry {
                let ds_display = dataset_display_name(&ds.name);

                // expand_auto_mappings (step 4.5) must run before this step
                // to convert all Auto/Inherited mappings to Explicit.
                let mapping = match &ds.extras.column_mapping {
                    ColumnMapping::Explicit(m) => m,
                    _ => {
                        return Err(CompileError::MappingValidation(vec![format!(
                            "{} '{}', dataset '{}': column_mapping is not Explicit \
                             (expand_auto_mappings must run before validate_mappings)",
                            dk.variant(), dk_name, ds_display
                        )]));
                    }
                };

                // Collect anchor sub-names from all Anchored entries (these are
                // synthetic keys added during flattening, not interface names).
                let mut anchor_subnames: HashSet<String> = HashSet::new();
                for value in mapping.values() {
                    if let ColumnMappingValue::Anchored(anchors) = value {
                        // Validate reserved names.
                        for anchor_name in anchors.keys() {
                            if anchor_name == "column" || anchor_name == "lit" {
                                errors.push(format!(
                                    "{} '{}', dataset '{}': anchor name '{}' is reserved \
                                     and cannot be used in Anchored column_mapping",
                                    dk.variant(), dk_name, ds_display, anchor_name
                                ));
                            }
                            anchor_subnames.insert(anchor_name.clone());
                        }
                    }
                }

                // Check that mapping keys reference existing interface names.
                // Skip anchor sub-names — they're injected by flattening, not interface names.
                for key in mapping.keys() {
                    if !interface_names.contains(key) && !anchor_subnames.contains(key) {
                        errors.push(format!(
                            "{} '{}', dataset '{}': column_mapping key '{}' \
                             does not match any dimension, measure, or metric in the interface",
                            dk.variant(), dk_name, ds_display, key
                        ));
                    }
                }

            }
        }

        // Union coverage: every mappable interface name must be mapped by at least
        // one dataset. Partial per-dataset mappings are valid — the planner handles
        // coverage at query time via grain groups and UNION ALL.
        let mappable_names: HashSet<String> = collect_mappable_names(dk).collect();
        let mut all_mapped: HashSet<String> = HashSet::new();
        for ds_entry in dk.children().unwrap_or(&[]) {
            if let ChildEntry::Inline(ds) = ds_entry {
                if let ColumnMapping::Explicit(m) = &ds.extras.column_mapping {
                    for key in m.keys() {
                        all_mapped.insert(key.clone());
                    }
                }
            }
        }
        for iname in &mappable_names {
            if !all_mapped.contains(iname) {
                errors.push(format!(
                    "{} '{}': interface name '{}' is not mapped by any dataset",
                    dk.variant(), dk_name, iname
                ));
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(CompileError::MappingValidation(errors))
    }
}

// ============================================================================
// Step 5b: Validate Grain Compatibility
// ============================================================================

/// Validate that dataset-level temporal grain specs are compatible with
/// the kind-level temporal dimension definitions.
///
/// For each kind with temporal dimensions:
/// - Each dataset's explicit grain (via `WithGrain { grain }`) must be present
///   in the kind-level dimension's `grains` list.
/// - If multiple temporal dimensions exist, each is validated independently.
pub(crate) fn validate_grain_compatibility(model: &SemanticModel) -> Result<(), CompileError> {
    let mut errors = Vec::new();

    for dk in model.entities.values() {
        if dk.is_simple() { continue; }
        // Collect temporal dimensions: name -> allowed grains
        let temporal_dims: Vec<(&str, &[TemporalGrain])> = dk
            .interface().dimensions
            .values()
            .filter_map(|d| match d {
                DimensionEntry::Inline(dim) => match &dim.dim_type {
                    DimensionType::Temporal(td) => Some((dim.name.as_str(), td.grains.as_slice())),
                    _ => None,
                },
                DimensionEntry::Ref(_) => None,
            })
            .collect();

        if temporal_dims.is_empty() {
            continue;
        }

        for ds_entry in dk.children().unwrap_or(&[]) {
            if let ChildEntry::Inline(ds) = ds_entry {
                let ds_display = dataset_display_name(&ds.name);

                for (dim_name, allowed_grains) in &temporal_dims {
                    if let Some(ColumnMappingValue::WithGrain {
                        grain: Some(grain), ..
                    }) = ds.extras.column_mapping.get(*dim_name)
                    {
                        if !allowed_grains.contains(grain) {
                            errors.push(format!(
                                "{} '{}', dataset '{}': temporal dimension '{}' \
                                 has grain '{:?}' which is not in the kind's allowed \
                                 grains {:?}",
                                dk.variant(), dk.name(), ds_display, dim_name, grain, allowed_grains
                            ));
                        }
                    }
                }
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(CompileError::MappingValidation(errors))
    }
}

// ============================================================================
// Step 6: Build Metric Graph
// ============================================================================

const MAX_METRIC_DEPTH: usize = 3;

/// Build metric dependency graph, detect cycles, enforce depth <= 3.
/// Returns a map of metric name -> depth.
pub(crate) fn build_metric_graph(
    model: &SemanticModel,
) -> Result<HashMap<String, usize>, CompileError> {
    let mut depths: HashMap<String, usize> = HashMap::new();

    // Collect all metric/measure names
    let mut metric_names: HashSet<String> = HashSet::new();
    let mut measure_names: HashSet<String> = HashSet::new();

    for m in &model.metrics {
        metric_names.insert(m.name.clone());
    }
    for m in &model.measures {
        measure_names.insert(m.name.clone());
    }
    for dk in model.entities.values() {
        for m in dk.interface().metrics.values() {
            if let MetricEntry::Inline(met) = m {
                metric_names.insert(met.name.clone());
            }
        }
        for m in dk.interface().measures.values() {
            if let MeasureEntry::Inline(mea) = m {
                measure_names.insert(mea.name.clone());
            }
        }
    }

    // Build graph nodes for all metrics and measures
    let all_names: Vec<String> = metric_names
        .iter()
        .chain(measure_names.iter())
        .cloned()
        .collect();
    let name_to_idx: HashMap<&str, usize> = all_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();

    let mut graph = DiGraph::<String, ()>::new();
    let nodes: Vec<_> = all_names.iter().map(|n| graph.add_node(n.clone())).collect();

    // Add edges: metric -> its dependencies
    let all_metrics = collect_all_metrics(model);
    for met in &all_metrics {
        if let Some(&src_idx) = name_to_idx.get(met.name.as_str()) {
            let deps = extract_identifiers_from_expr_source(&met.expr);
            for dep in deps {
                if let Some(&dst_idx) = name_to_idx.get(dep.as_str()) {
                    if src_idx != dst_idx {
                        graph.add_edge(nodes[src_idx], nodes[dst_idx], ());
                    }
                }
            }
        }
    }

    // Check for cycles
    if is_cyclic_directed(&graph) {
        return Err(CompileError::MetricCycle {
            cycle: vec!["(cycle detected in metric graph)".to_string()],
        });
    }

    // Measures have depth 0
    for name in &measure_names {
        depths.insert(name.clone(), 0);
    }

    // Iterative depth computation for metrics
    let mut changed = true;
    while changed {
        changed = false;
        for met in &all_metrics {
            let deps = extract_identifiers_from_expr_source(&met.expr);
            let max_dep_depth = deps
                .iter()
                .filter_map(|d| depths.get(d.as_str()))
                .max()
                .copied()
                .unwrap_or(0);
            let new_depth = if deps.is_empty() { 0 } else { max_dep_depth + 1 };

            match depths.get(&met.name) {
                Some(&existing) if new_depth <= existing => {}
                _ => {
                    depths.insert(met.name.clone(), new_depth);
                    changed = true;
                }
            }
        }
    }

    // Check depth limit
    for (name, depth) in &depths {
        if *depth > MAX_METRIC_DEPTH {
            return Err(CompileError::MetricDepthExceeded {
                metric: name.clone(),
                depth: *depth,
                max_depth: MAX_METRIC_DEPTH,
            });
        }
    }

    Ok(depths)
}

// ============================================================================
// Step 7: Build Relationship Graph
// ============================================================================

/// Build relationship graph for joinset anchor inference.
/// Returns a map of kind_name -> list of anchor dataset names.
pub(crate) fn build_rel_graph(
    model: &SemanticModel,
) -> Result<HashMap<String, Vec<String>>, CompileError> {
    let mut result: HashMap<String, Vec<String>> = HashMap::new();

    for dk in model.entities.values() {
        if !dk.is_joinset() {
            continue;
        }

        let mut graph = DiGraph::<String, ()>::new();
        let mut node_map: HashMap<String, petgraph::graph::NodeIndex> = HashMap::new();

        // Add nodes for each dataset
        for entry in dk.children().unwrap_or(&[]) {
            if let ChildEntry::Inline(ds) = entry {
                let name = dataset_display_name(&ds.name);
                if !node_map.contains_key(name) {
                    let owned = name.to_string();
                    let idx = graph.add_node(owned.clone());
                    node_map.insert(owned, idx);
                }
            }
        }

        // Add edges from relationships
        for rel in dk.relationships() {
            if let (Some(&from_idx), Some(&to_idx)) =
                (node_map.get(&rel.from), node_map.get(&rel.to))
            {
                graph.add_edge(from_idx, to_idx, ());
            }
        }

        // Infer anchors: nodes with in-degree 0
        let anchors: Vec<String> = graph
            .node_indices()
            .filter(|&n| {
                graph
                    .neighbors_directed(n, petgraph::Direction::Incoming)
                    .count()
                    == 0
            })
            .map(|n| graph[n].clone())
            .collect();

        result.insert(dk.name().to_string(), anchors);
    }

    Ok(result)
}

// ============================================================================
// Steps 8-9: Compile Expressions and Emit
// ============================================================================

/// Emit the final CompiledManifest (steps 8 + 9).
pub(crate) fn emit(
    model: SemanticModel,
    source_hash: String,
    metric_depths: &HashMap<String, usize>,
    resolution: SourceResolutionResult,
    extra_warnings: Vec<crate::acceleration::CompileWarning>,
) -> Result<CompiledManifest, CompileError> {
    let mut entities = IndexMap::new();
    let mut relationships = Vec::new();

    for dk in model.entities.values() {
        let compiled = compile_to_compiled_data_kind(dk, metric_depths, &resolution)?;
        entities.insert(dk.name().to_string(), compiled);
    }

    for rel in &model.relationships {
        relationships.push(CompiledRelationship {
            name: rel.name.clone(),
            from: rel.from.clone(),
            to: rel.to.clone(),
            join_type: rel.join_type,
            columns: rel.columns.clone(),
            cardinality: rel.cardinality,
        });
    }

    // Build global field index from entities.
    let field_index = build_field_index(&entities);

    // Build global relationship graph with shortest paths.
    let relationship_graph = build_relationship_graph(&entities, &relationships);

    // Build unified semantic graph (petgraph).
    let semantic_graph = build_semantic_graph(&entities, &relationships);

    // Merge resolution + derivation diagnostics.
    let mut diagnostics = crate::acceleration::CompileDiagnostics::default();
    diagnostics.warnings.extend(resolution.warnings);
    diagnostics.warnings.extend(extra_warnings);

    Ok(CompiledManifest {
        version: 3,
        compiled_at: chrono::DateTime::default(), // overwritten by compiler.compile()
        source_hash,
        entities,
        relationships,
        relationship_graph,
        field_index,
        semantic_graph,
        diagnostics,
        catalog_snapshot: resolution.catalog_snapshot,
        model_name: model.name,
        model_description: model.description,
    })
}

// ============================================================================
// Step 20-21: Global Graph Structures
// ============================================================================

/// Build global field index from compiled data kinds.
fn build_field_index(
    data_kinds: &IndexMap<String, crate::acceleration::CompiledDataKind>,
) -> crate::acceleration::FieldIndex {
    let mut providers: HashMap<String, Vec<String>> = HashMap::new();
    let mut all_dimensions: HashSet<String> = HashSet::new();
    let mut all_measures: HashSet<String> = HashSet::new();
    let mut all_metrics: HashSet<String> = HashSet::new();
    let mut all_keys: HashSet<String> = HashSet::new();

    for (name, dk) in data_kinds {
        let iface = dk.interface();
        for dim_name in iface.dimensions.keys() {
            providers
                .entry(dim_name.clone())
                .or_default()
                .push(name.clone());
            all_dimensions.insert(dim_name.clone());
        }
        for measure_name in iface.measures.keys() {
            providers
                .entry(measure_name.clone())
                .or_default()
                .push(name.clone());
            all_measures.insert(measure_name.clone());
        }
        for metric_name in iface.metrics.keys() {
            all_metrics.insert(metric_name.clone());
        }
        // Index key columns — add to providers only if not already a dimension.
        if let Some(keys) = iface.keys.as_ref() {
            for key_col in keys.all_column_names() {
                if !all_dimensions.contains(&key_col) {
                    providers
                        .entry(key_col.clone())
                        .or_default()
                        .push(name.clone());
                    all_keys.insert(key_col);
                }
            }
        }
    }

    crate::acceleration::FieldIndex {
        providers,
        all_dimensions,
        all_measures,
        all_metrics,
        all_keys,
    }
}

/// Build global relationship graph with pre-computed shortest paths.
fn build_relationship_graph(
    data_kinds: &IndexMap<String, crate::acceleration::CompiledDataKind>,
    relationships: &[CompiledRelationship],
) -> crate::acceleration::RelationshipGraph {
    use std::collections::VecDeque;

    let dataset_index: HashMap<String, usize> = data_kinds
        .keys()
        .enumerate()
        .map(|(i, name)| (name.clone(), i))
        .collect();

    let mut forward: HashMap<String, Vec<(String, usize)>> = HashMap::new();
    let mut reverse: HashMap<String, Vec<(String, usize)>> = HashMap::new();

    for (rel_idx, rel) in relationships.iter().enumerate() {
        forward
            .entry(rel.from.clone())
            .or_default()
            .push((rel.to.clone(), rel_idx));
        reverse
            .entry(rel.to.clone())
            .or_default()
            .push((rel.from.clone(), rel_idx));
    }

    // BFS from every dataset to compute shortest paths.
    let mut rel_graph = crate::acceleration::RelationshipGraph {
        forward,
        reverse,
        shortest_paths: HashMap::new(),
        dataset_index,
    };

    let ds_names: Vec<String> = data_kinds.keys().cloned().collect();
    for source_name in &ds_names {
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(String, Vec<usize>)> = VecDeque::new();
        visited.insert(source_name.clone());
        queue.push_back((source_name.clone(), vec![]));

        while let Some((current, path)) = queue.pop_front() {
            // Follow forward edges
            if let Some(edges) = rel_graph.forward.get(&current).cloned() {
                for (neighbor, rel_idx) in &edges {
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.clone());
                        let mut new_path = path.clone();
                        new_path.push(*rel_idx);
                        rel_graph.set_shortest_path(source_name, neighbor, new_path.clone());
                        queue.push_back((neighbor.clone(), new_path));
                    }
                }
            }
            // Follow reverse edges (relationships are bidirectional for BFS)
            if let Some(edges) = rel_graph.reverse.get(&current).cloned() {
                for (neighbor, rel_idx) in &edges {
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.clone());
                        let mut new_path = path.clone();
                        new_path.push(*rel_idx);
                        rel_graph.set_shortest_path(source_name, neighbor, new_path.clone());
                        queue.push_back((neighbor.clone(), new_path));
                    }
                }
            }
        }
    }

    rel_graph
}

/// Build unified semantic graph from compiled data kinds and relationships.
fn build_semantic_graph(
    data_kinds: &IndexMap<String, crate::acceleration::CompiledDataKind>,
    relationships: &[CompiledRelationship],
) -> crate::acceleration::SemanticGraph {
    use crate::acceleration::{FieldType, SemanticGraph};

    let mut graph = SemanticGraph::new();

    for (kind_name, dk) in data_kinds {
        let iface = dk.interface();
        // Add dataset nodes for each binding.
        for binding in dk.bindings() {
            graph.add_dataset(&binding.dataset_name, kind_name);

            // Add field edges for dimensions.
            for dim_name in iface.dimensions.keys() {
                graph.add_provides_field(&binding.dataset_name, dim_name, FieldType::Dimension);
            }
            // Add field edges for measures.
            for measure_name in iface.measures.keys() {
                graph.add_provides_field(&binding.dataset_name, measure_name, FieldType::Measure);
            }
        }
        // Add metric nodes (not tied to a specific dataset).
        for metric_name in iface.metrics.keys() {
            graph.add_field(metric_name, FieldType::Metric);
        }
        // Add key column nodes (only if not already a dimension or measure).
        if let Some(keys) = iface.keys.as_ref() {
            let dim_names: HashSet<&String> = iface.dimensions.keys().collect();
            let measure_names: HashSet<&String> = iface.measures.keys().collect();
            for key_col in keys.all_column_names() {
                if !dim_names.contains(&key_col) && !measure_names.contains(&key_col) {
                    for binding in dk.bindings() {
                        graph.add_provides_field(&binding.dataset_name, &key_col, FieldType::Key);
                    }
                }
            }
        }
    }

    // Add join edges from relationships.
    for (rel_idx, rel) in relationships.iter().enumerate() {
        graph.add_join(&rel.from, &rel.to, rel_idx);
    }

    graph
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Collect mappable names (dimensions + measures) from a kind.
/// Metrics are derived and do not require column mappings.
fn collect_mappable_names(dk: &DataKind) -> impl Iterator<Item = String> + '_ {
    dk.interface().dimensions
        .values()
        .filter_map(|d| match d {
            DimensionEntry::Inline(dim) => {
                // Metadata dimensions are extracted from source metadata,
                // not physical columns — they don't need column mapping.
                // Computed dimensions derive values from expressions over
                // other columns — they also don't need their own mapping.
                if matches!(dim.dim_type, DimensionType::Metadata(_)) || dim.expr.is_some() {
                    None
                } else {
                    Some(dim.name.clone())
                }
            }
            DimensionEntry::Ref(r) => Some(r.ref_name.clone()),
        })
        .chain(dk.interface().measures.values().map(|m| match m {
            MeasureEntry::Inline(mea) => mea.name.clone(),
            MeasureEntry::Ref(r) => r.ref_name.clone(),
        }))
}

/// Collect mappable names from a SimpleDataKind (standalone dataset).
/// Same logic as `collect_mappable_names` but takes &SimpleDataKind directly.
fn collect_mappable_names_simple(dsk: &SimpleDataKind) -> Vec<String> {
    let iface = &dsk.interface;
    let dims = iface.dimensions.values().filter_map(|d| match d {
        DimensionEntry::Inline(dim) => {
            if matches!(dim.dim_type, DimensionType::Metadata(_)) || dim.expr.is_some() {
                None
            } else {
                Some(dim.name.clone())
            }
        }
        DimensionEntry::Ref(r) => Some(r.ref_name.clone()),
    });
    let measures = iface.measures.values().map(|m| match m {
        MeasureEntry::Inline(mea) => mea.name.clone(),
        MeasureEntry::Ref(r) => r.ref_name.clone(),
    });
    dims.chain(measures).collect()
}

/// Collect interface names (dimensions + measures + metrics) from a data kind.
fn collect_interface_names(dk: &DataKind) -> impl Iterator<Item = String> + '_ {
    dk.interface().dimensions
        .values()
        .map(|d| match d {
            DimensionEntry::Inline(dim) => dim.name.clone(),
            DimensionEntry::Ref(r) => r.ref_name.clone(),
        })
        .chain(dk.interface().measures.values().map(|m| match m {
            MeasureEntry::Inline(mea) => mea.name.clone(),
            MeasureEntry::Ref(r) => r.ref_name.clone(),
        }))
        .chain(dk.interface().metrics.values().map(|m| match m {
            MetricEntry::Inline(met) => met.name.clone(),
            MetricEntry::Ref(r) => r.ref_name.clone(),
        }))
}

/// Derive a dimension's data_type from its expression when not explicitly declared.
///
/// Derivation chain:
/// 1. Declared → use it
/// 2. Expr is FunctionCall(f) + registry has Fixed(T) → use T
/// 3. Expr is FunctionCall(f) + registry has SameAsInput → cannot resolve (need declared)
/// 4. No expr, no declared → error
fn derive_dimension_data_type(
    dim_name: &str,
    declared: &Option<DataType>,
    compiled_expr: &Option<Expr>,
    registry: &crate::function_registry::FunctionRegistry,
) -> Result<semstrait_core::DataType, String> {
    // 1. Declared type always wins.
    if let Some(dt) = declared {
        return Ok(map_data_type(dt));
    }

    // 2. Try to derive from expression.
    if let Some(expr) = compiled_expr {
        if let Some(dt) = derive_type_from_expr(expr, registry) {
            return Ok(dt);
        }
        return Err(format!(
            "dimension '{}': cannot derive data_type from expression; specify data_type explicitly",
            dim_name
        ));
    }

    // 3. No expr, no declared.
    Err(format!(
        "dimension '{}': data_type is required (no expression to derive from)",
        dim_name
    ))
}

/// Derive a measure's data_type from its aggregation and expression when not explicitly declared.
///
/// Derivation chain:
/// 1. Declared → use it
/// 2. agg = COUNT/CountDistinct → Integer
/// 3. agg = AVG → Number
/// 4. agg = SUM/MIN/MAX with declared → input type (needs declared)
/// 5. Cannot derive → error
fn derive_measure_data_type(
    measure_name: &str,
    declared: &Option<DataType>,
    agg: semstrait_core::Aggregation,
) -> Result<semstrait_core::DataType, String> {
    use semstrait_core::Aggregation;

    // 1. Declared type always wins.
    if let Some(dt) = declared {
        return Ok(map_data_type(dt));
    }

    // 2. Derive from aggregation type.
    match agg {
        Aggregation::Count | Aggregation::CountDistinct => Ok(semstrait_core::DataType::Integer),
        Aggregation::Avg => Ok(semstrait_core::DataType::Number),
        Aggregation::Sum | Aggregation::Min | Aggregation::Max => {
            Err(format!(
                "measure '{}': cannot derive data_type for {}; specify data_type explicitly",
                measure_name,
                match agg {
                    Aggregation::Sum => "sum",
                    Aggregation::Min => "min",
                    Aggregation::Max => "max",
                    _ => unreachable!(),
                }
            ))
        }
    }
}

/// Derive a metric's data_type from leaf measures when not explicitly declared.
fn derive_metric_data_type(
    metric_name: &str,
    declared: &Option<DataType>,
    expr: &Expr,
    measures: &IndexMap<String, CompiledMeasure>,
    metrics: &IndexMap<String, CompiledMetric>,
) -> Result<semstrait_core::DataType, String> {
    // 1. Declared type always wins.
    if let Some(dt) = declared {
        return Ok(map_data_type(dt));
    }

    // 2. Try to derive from leaf measure types.
    // Collect all referenced leaf measure/metric types.
    let mut leaf_types = Vec::new();
    collect_leaf_types(expr, measures, metrics, &mut leaf_types);

    if leaf_types.len() == 1 {
        return Ok(leaf_types.into_iter().next().unwrap());
    }

    // Multiple types or none — can't derive.
    Err(format!(
        "metric '{}': cannot derive data_type; specify data_type explicitly",
        metric_name
    ))
}

/// Collect data types from leaf measures/metrics referenced in an expression.
fn collect_leaf_types(
    expr: &Expr,
    measures: &IndexMap<String, CompiledMeasure>,
    metrics: &IndexMap<String, CompiledMetric>,
    types: &mut Vec<semstrait_core::DataType>,
) {
    match expr {
        Expr::EntityRef(er) => {
            if let Some(m) = measures.get(&er.name) {
                if !types.contains(&m.data_type) {
                    types.push(m.data_type.clone());
                }
            } else if let Some(met) = metrics.get(&er.name) {
                if !types.contains(&met.data_type) {
                    types.push(met.data_type.clone());
                }
            }
        }
        Expr::BinaryOp(b) => {
            collect_leaf_types(&b.left, measures, metrics, types);
            collect_leaf_types(&b.right, measures, metrics, types);
        }
        Expr::Negate(u) | Expr::Not(u) | Expr::IsNull(u) | Expr::IsNotNull(u) => {
            collect_leaf_types(&u.expr, measures, metrics, types);
        }
        Expr::Case(c) => {
            for clause in &c.when_then {
                collect_leaf_types(&clause.result, measures, metrics, types);
            }
            if let Some(e) = &c.else_expr {
                collect_leaf_types(e, measures, metrics, types);
            }
        }
        _ => {}
    }
}

/// Try to derive a core DataType from an expression tree using the function registry.
fn derive_type_from_expr(
    expr: &Expr,
    registry: &crate::function_registry::FunctionRegistry,
) -> Option<semstrait_core::DataType> {
    use crate::function_registry::ReturnType;
    match expr {
        Expr::FunctionCall(fc) => {
            if let Some(spec) = registry.get(&fc.name) {
                match &spec.return_type {
                    ReturnType::Fixed(dt) => Some(dt.clone()),
                    ReturnType::SameAsInput => {
                        // Try to derive from first argument recursively.
                        fc.args.first().and_then(|arg| derive_type_from_expr(arg, registry))
                    }
                    ReturnType::Semantic => None, // Requires declared type.
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

fn compile_dimensions(entries: &BTreeMap<String, DimensionEntry>) -> Result<IndexMap<String, CompiledDimension>, CompileError> {
    let registry = crate::function_registry::FunctionRegistry::standard();
    let mut dimensions = IndexMap::new();
    let mut errors = Vec::new();

    for d in entries.values() {
        if let DimensionEntry::Inline(dim) = d {
            let (compiled_expr, expr_source) = if let Some(ref src) = dim.expr {
                let expr = resolve_expr_source(src, &dim.name)?;

                // Validate: computed dimensions must not contain aggregation.
                if contains_aggregation(&expr) {
                    errors.push(format!(
                        "dimension '{}': computed expression must not contain aggregation functions",
                        dim.name
                    ));
                    continue;
                }

                // Validate function calls against registry.
                errors.extend(validate_function_calls(&expr, &dim.name, registry));

                (Some(expr), Some(src.display_string()))
            } else {
                (None, None)
            };

            // Derive data_type: declared > expression-derived > error.
            let data_type = match derive_dimension_data_type(&dim.name, &dim.data_type, &compiled_expr, registry) {
                Ok(dt) => dt,
                Err(msg) => {
                    errors.push(msg);
                    continue;
                }
            };

            dimensions.insert(
                dim.name.clone(),
                CompiledDimension {
                    name: dim.name.clone(),
                    description: dim.description.clone(),
                    data_type,
                    dim_type: dim.dim_type.clone(),
                    expr: compiled_expr,
                    expr_source,
                },
            );
        }
    }

    if !errors.is_empty() {
        return Err(CompileError::ExprCompilation(errors));
    }
    Ok(dimensions)
}

/// SR-5a: Validate that all column references in expressions refer to names
/// exposed in the same interface (dimensions + measures + metrics).
fn validate_expr_scope(
    dimensions: &IndexMap<String, CompiledDimension>,
    measures: &IndexMap<String, CompiledMeasure>,
    metrics: &IndexMap<String, CompiledMetric>,
    kind_name: &str,
) -> Result<(), CompileError> {
    use std::collections::HashSet;

    // Build the allowed name set from the interface.
    let mut allowed: HashSet<&str> = HashSet::new();
    for name in dimensions.keys() { allowed.insert(name.as_str()); }
    for name in measures.keys() { allowed.insert(name.as_str()); }
    for name in metrics.keys() { allowed.insert(name.as_str()); }

    let mut errors = Vec::new();

    // Check computed dimension expressions.
    for dim in dimensions.values() {
        if let Some(ref expr) = dim.expr {
            for col_ref in expr.column_refs() {
                if !allowed.contains(col_ref.as_str()) {
                    errors.push(format!(
                        "dimension '{}' in '{}': expression references '{}' which is not exposed in the interface",
                        dim.name, kind_name, col_ref
                    ));
                }
            }
        }
    }

    // Note: measure base expressions (e.g. "amount") reference physical columns,
    // not semantic names — scope validation does not apply to them.

    // Check metric expressions (metrics reference measures/dimensions by name).
    for metric in metrics.values() {
        for col_ref in metric.expr.column_refs() {
            if !allowed.contains(col_ref.as_str()) {
                errors.push(format!(
                    "metric '{}' in '{}': expression references '{}' which is not exposed in the interface",
                    metric.name, kind_name, col_ref
                ));
            }
        }
    }

    if !errors.is_empty() {
        return Err(CompileError::ExprCompilation(errors));
    }
    Ok(())
}

/// Map model-layer DataType to core DataType (ANSI logical).
fn map_data_type(dt: &DataType) -> semstrait_core::DataType {
    match dt {
        DataType::I8 | DataType::I16 | DataType::I32 | DataType::I64 => {
            semstrait_core::DataType::Integer
        }
        DataType::F32 | DataType::F64 => semstrait_core::DataType::Number,
        DataType::Bool => semstrait_core::DataType::Boolean,
        DataType::String => semstrait_core::DataType::String,
        DataType::Date => semstrait_core::DataType::Date,
        DataType::Timestamp => semstrait_core::DataType::Timestamp { precision: 6 },
        DataType::Decimal { precision, scale } => semstrait_core::DataType::Decimal {
            precision: *precision,
            scale: *scale as i8,
        },
    }
}

/// Map model-layer AggregationType to core Aggregation.
fn map_aggregation_type(agg: &AggregationType) -> semstrait_core::Aggregation {
    match agg {
        AggregationType::Sum => semstrait_core::Aggregation::Sum,
        AggregationType::Avg => semstrait_core::Aggregation::Avg,
        AggregationType::Count => semstrait_core::Aggregation::Count,
        AggregationType::CountDistinct => semstrait_core::Aggregation::CountDistinct,
        AggregationType::Min => semstrait_core::Aggregation::Min,
        AggregationType::Max => semstrait_core::Aggregation::Max,
    }
}

/// Check if a parsed Expr contains any aggregation functions.
fn contains_aggregation(expr: &Expr) -> bool {
    let mut found = false;
    expr.walk(&mut |node| {
        if matches!(node, Expr::Aggregate(_)) {
            found = true;
        }
    });
    found
}

fn compile_measures(
    entries: &BTreeMap<String, MeasureEntry>,
) -> Result<IndexMap<String, CompiledMeasure>, CompileError> {
    let registry = crate::function_registry::FunctionRegistry::standard();
    let mut measures = IndexMap::new();
    let mut errors = Vec::new();

    for m in entries.values() {
        if let MeasureEntry::Inline(mea) = m {
            let filters = compile_measure_filters(&mea.filters)?;

            let (compiled_agg, compiled_expr, expr_source) = if let Some(ref agg) = mea.agg {
                // Declarative path: agg tag present.
                let core_agg = map_aggregation_type(agg);
                let expr_source = mea.expr.as_ref()
                    .map(|e| e.display_string())
                    .unwrap_or_else(|| mea.name.clone());

                if let Some(ref expr_src) = mea.expr {
                    // Parse/convert expr, validate no aggregation.
                    let parsed = resolve_expr_source(expr_src, &mea.name)?;
                    if contains_aggregation(&parsed) {
                        errors.push(format!(
                            "measure '{}': expr must not contain aggregation functions \
                             when 'agg' is specified; use horizontal expressions only",
                            mea.name
                        ));
                        continue;
                    }
                    (core_agg, parsed, expr_source)
                } else {
                    // No expr — the column is resolved from mapping by name.
                    (core_agg, Expr::entity_ref(&mea.name), expr_source)
                }
            } else if let Some(ref expr_src) = mea.expr {
                // Legacy auto-upgrade: extract aggregation from expr string.
                let parsed = resolve_expr_source(expr_src, &mea.name)?;
                let display = expr_src.display_string();
                if let Some((agg, inner_expr)) = try_extract_aggregation_from_expr(&parsed) {
                    (agg, inner_expr, display)
                } else {
                    // Expr has no recognizable aggregation — error.
                    errors.push(format!(
                        "measure '{}': 'agg' must be specified; expr does not contain \
                         a recognized aggregation function (SUM, COUNT, AVG, MIN, MAX, COUNT_DISTINCT)",
                        mea.name
                    ));
                    continue;
                }
            } else {
                // Neither agg nor expr specified — error.
                errors.push(format!(
                    "measure '{}': 'agg' must be specified",
                    mea.name
                ));
                continue;
            };

            // Validate function calls against registry.
            errors.extend(validate_function_calls(&compiled_expr, &mea.name, registry));

            // Derive additivity from agg when not explicitly specified.
            let additivity = mea.additivity.clone().or_else(|| {
                Some(crate::compiled::derive_additivity(compiled_agg))
            });

            // Derive data_type: declared > aggregation-derived > error.
            let data_type = match derive_measure_data_type(&mea.name, &mea.data_type, compiled_agg) {
                Ok(dt) => dt,
                Err(msg) => {
                    errors.push(msg);
                    continue;
                }
            };

            measures.insert(
                mea.name.clone(),
                CompiledMeasure {
                    name: mea.name.clone(),
                    description: mea.description.clone(),
                    data_type,
                    agg: compiled_agg,
                    expr: compiled_expr,
                    expr_source,
                    additivity,
                    constraints: mea.constraints.clone(),
                    filters,
                },
            );
        }
    }

    if !errors.is_empty() {
        return Err(CompileError::ExprCompilation(errors));
    }
    Ok(measures)
}

fn compile_metrics(
    entries: &BTreeMap<String, MetricEntry>,
    metric_depths: &HashMap<String, usize>,
    measures: &IndexMap<String, CompiledMeasure>,
) -> Result<IndexMap<String, CompiledMetric>, CompileError> {
    let registry = crate::function_registry::FunctionRegistry::standard();
    let mut errors = Vec::new();
    let mut metrics = IndexMap::new();
    for m in entries.values() {
        if let MetricEntry::Inline(met) = m {
            let expr = resolve_expr_source(&met.expr, &met.name)?;

            // Validate function calls against registry.
            errors.extend(validate_function_calls(&expr, &met.name, registry));
            let depth = metric_depths.get(&met.name).copied().unwrap_or(0);
            let filters = compile_measure_filters(&met.filters)?;
            let compiled_agg = met.agg.as_ref().map(map_aggregation_type);
            let metric_type = MetricType::infer(&expr);

            // Derive effective additivity from transitive leaf measures.
            let additivity = met.additivity.clone().or_else(|| {
                let leaf_additivity = collect_leaf_measure_additivity(&expr, measures, &metrics);
                if leaf_additivity.is_empty() {
                    None
                } else {
                    Some(crate::compiled::worst_case_additivity(leaf_additivity.iter()))
                }
            });

            // Derive data_type: declared > leaf-measure-derived > error.
            let data_type = match derive_metric_data_type(&met.name, &met.data_type, &expr, measures, &metrics) {
                Ok(dt) => dt,
                Err(msg) => {
                    errors.push(msg);
                    continue;
                }
            };

            metrics.insert(
                met.name.clone(),
                CompiledMetric {
                    name: met.name.clone(),
                    description: met.description.clone(),
                    data_type,
                    metric_type,
                    agg: compiled_agg,
                    expr,
                    expr_source: met.expr.display_string(),
                    additivity,
                    constraints: met.constraints.clone(),
                    filters,
                    depth,
                },
            );
        }
    }
    if !errors.is_empty() {
        return Err(CompileError::ExprCompilation(errors));
    }
    Ok(metrics)
}

/// Compile a model DataKind directly into a CompiledDataKind (acceleration type).
///
/// For standalone datasets: builds identity column mapping, single DatasetBinding.
/// For kinds (grainset/unionset/joinset): builds DatasetBindings from children,
/// acceleration structures (CoverageIndex, DimensionIndex, etc.).
fn compile_to_compiled_data_kind(
    dk: &DataKind,
    metric_depths: &HashMap<String, usize>,
    resolution: &SourceResolutionResult,
) -> Result<crate::acceleration::CompiledDataKind, CompileError> {
    use crate::acceleration::*;

    let iface = dk.interface();
    let measures = compile_measures(&iface.measures)?;
    let metrics = compile_metrics(&iface.metrics, metric_depths, &measures)?;
    let dimensions = compile_dimensions(&iface.dimensions)?;
    let filters = compile_measure_filters(&iface.filters.values().cloned().collect::<Vec<_>>())?;

    // Validate: measures with agg must not reference other measures (D-042).
    validate_measure_references(&measures)?;

    // SR-5a: Validate that computed dimension and metric expressions only
    // reference names exposed in the same interface.
    validate_expr_scope(&dimensions, &measures, &metrics, dk.name())?;

    // Build CompiledInterface (shared across all variants).
    let build_compiled_interface = |temporal_dim: Option<String>| -> CompiledInterface {
        CompiledInterface {
            name: dk.name().to_string(),
            description: iface.description.clone(),
            dimensions: dimensions.clone(),
            measures: measures.clone(),
            metrics: metrics.clone(),
            keys: iface.keys.clone(),
            filters: filters.clone(),
            temporal_dim,
        }
    };

    match dk {
        DataKind::Simple(dsk) => {
            // Standalone dataset: use column_mapping from extras (expanded in step 4.5).
            // Falls back to identity if extras is absent.
            let mapping: semstrait_model::ColumnMapping = dsk
                .extras
                .as_ref()
                .map(|e| e.column_mapping.clone())
                .unwrap_or_else(|| {
                    // No extras: build identity mapping from interface names.
                    let interface_names: Vec<&String> = dimensions.keys().chain(measures.keys()).collect();
                    semstrait_model::ColumnMapping::Explicit(
                        interface_names
                            .iter()
                            .map(|name| {
                                (
                                    (*name).clone(),
                                    semstrait_model::ColumnMappingValue::Simple((*name).clone()),
                                )
                            })
                            .collect(),
                    )
                });

            // Resolve sources from extras.storage if present.
            let resolved_sources = resolve_dataset_sources(&dsk.name, &dsk.extras, resolution);

            let binding = DatasetBinding {
                dataset_name: dsk.name.clone(),
                column_mapping: ResolvedColumnMapping::from_column_mapping(&mapping),
                resolved_sources,
            };

            let temporal_dim = dimensions
                .iter()
                .find(|(_, d)| matches!(d.dim_type, semstrait_model::DimensionType::Temporal(_)))
                .map(|(name, _)| name.clone());

            let interface = build_compiled_interface(temporal_dim);
            Ok(CompiledDataKind::Simple(Box::new(CompiledSimpleKind {
                interface,
                binding,
            })))
        }
        _ => {
            // Multi-dataset kinds: build bindings from children.
            let bindings = compile_dataset_bindings(dk, resolution);

            // Compile relationships (only relevant for joinsets, but harmless to collect for all).
            let compiled_rels: Vec<CompiledRelationship> = dk
                .relationships()
                .iter()
                .map(|rel| CompiledRelationship {
                    name: rel.name.clone(),
                    from: rel.from.clone(),
                    to: rel.to.clone(),
                    join_type: rel.join_type,
                    columns: rel.columns.clone(),
                    cardinality: rel.cardinality,
                })
                .collect();

            // Infer temporal_dim from children's temporal config, falling back to dimension scan.
            let temporal_dim = dk
                .children()
                .unwrap_or(&[])
                .iter()
                .find_map(|ds_entry| {
                    if let ChildEntry::Inline(ds) = ds_entry {
                        ds.extras.temporal.as_ref()?.dimension.clone()
                    } else {
                        None
                    }
                })
                .or_else(|| {
                    dimensions
                        .iter()
                        .find(|(_, d)| matches!(d.dim_type, semstrait_model::DimensionType::Temporal(_)))
                        .map(|(name, _)| name.clone())
                });

            let interface = build_compiled_interface(temporal_dim);

            // Single-dataset kinds → Dataset fast path.
            // A kind with 1 dataset is functionally a dataset for query purposes —
            // no grain routing, union, or join logic needed. The dataset planner
            // handles computed dimensions and simpler plans correctly.
            if bindings.len() == 1 {
                let binding = bindings.into_iter().next().unwrap();
                return Ok(CompiledDataKind::Simple(Box::new(CompiledSimpleKind {
                    interface,
                    binding,
                })));
            }

            // Build acceleration structures.
            let metric_order = MetricOrder::build(&metrics, &measures);
            let coverage_index = CoverageIndex::build(&dimensions, &measures, &bindings);
            let dimension_index = DimensionIndex::build(&dimensions, &bindings);

            match dk {
                DataKind::Complex(ComplexDataKind::Grainset(_)) => {
                    let grain_map = interface
                        .temporal_dim
                        .as_deref()
                        .map(|td| GrainMap::build(td, &bindings));

                    Ok(CompiledDataKind::Grainset(Box::new(CompiledGrainsetKind {
                        interface,
                        bindings,
                        coverage_index,
                        dimension_index,
                        metric_order,
                        grain_map,
                    })))
                }
                DataKind::Complex(ComplexDataKind::Unionset(u)) => {
                    Ok(CompiledDataKind::Unionset(Box::new(CompiledUnionsetKind {
                        interface,
                        mode: u.mode,
                        bindings,
                        coverage_index,
                        dimension_index,
                        metric_order,
                    })))
                }
                DataKind::Complex(ComplexDataKind::Joinset(j)) => {
                    let adjacency_index = AdjacencyIndex::build(&bindings, &compiled_rels);
                    Ok(CompiledDataKind::Joinset(Box::new(CompiledJoinsetKind {
                        interface,
                        associativity: j.associativity,
                        bindings,
                        relationships: compiled_rels,
                        coverage_index,
                        dimension_index,
                        metric_order,
                        adjacency_index,
                    })))
                }
                DataKind::Simple(_) => unreachable!("handled above"),
            }
        }
    }
}

/// Build DatasetBindings from a DataKind's children.
fn compile_dataset_bindings(
    dk: &DataKind,
    resolution: &SourceResolutionResult,
) -> Vec<crate::acceleration::DatasetBinding> {
    dk.children()
        .unwrap_or(&[])
        .iter()
        .filter_map(|ds_entry| {
            if let ChildEntry::Inline(ds) = ds_entry {
                let ds_name = dataset_display_name(&ds.name).to_string();
                let resolved_sources = resolve_child_dataset_sources(&ds_name, ds, resolution);
                let column_mapping =
                    crate::acceleration::ResolvedColumnMapping::from_column_mapping(
                        &ds.extras.column_mapping,
                    );
                Some(crate::acceleration::DatasetBinding {
                    dataset_name: ds_name,
                    column_mapping,
                    resolved_sources,
                })
            } else {
                None
            }
        })
        .collect()
}

/// Resolve sources for a standalone dataset from its extras.
fn resolve_dataset_sources(
    name: &str,
    extras: &Option<DatasetExtras>,
    resolution: &SourceResolutionResult,
) -> Vec<crate::acceleration::ResolvedSource> {
    if let Some(sources) = resolution.resolved.get(name) {
        return sources.clone();
    }
    extras
        .as_ref()
        .and_then(|e| e.storage.as_ref())
        .map(build_fallback_sources)
        .unwrap_or_default()
}

/// Resolve sources for a kind's child dataset.
fn resolve_child_dataset_sources(
    ds_name: &str,
    ds: &InlineDataset,
    resolution: &SourceResolutionResult,
) -> Vec<crate::acceleration::ResolvedSource> {
    if let Some(sources) = resolution.resolved.get(ds_name) {
        return sources.clone();
    }
    debug_assert!(
        ds.extras
            .storage
            .as_ref()
            .is_none_or(|s| s.paths.is_empty() && s.tables.is_empty()),
        "dataset '{}' has storage config but no entry in SourceResolutionResult",
        ds_name
    );
    ds.extras
        .storage
        .as_ref()
        .map(build_fallback_sources)
        .unwrap_or_default()
}

/// Build fallback ResolvedSources from raw StorageConfig (when no provider resolved them).
fn build_fallback_sources(
    storage: &StorageConfig,
) -> Vec<crate::acceleration::ResolvedSource> {
    let mut result = Vec::new();
    for p in &storage.paths {
        result.push(crate::acceleration::ResolvedSource {
            reference: p.clone(),
            source_type: crate::acceleration::SourceType::Path,
            table_fqn: None,
            location: None,
            format: storage.format,
            catalog_alias: None,
            schema: None,
        });
    }
    for t in &storage.tables {
        result.push(crate::acceleration::ResolvedSource {
            reference: t.clone(),
            source_type: crate::acceleration::SourceType::Table,
            table_fqn: None,
            location: None,
            format: None,
            catalog_alias: None,
            schema: None,
        });
    }
    result
}

/// Validate that measures with aggregation do not reference other measures.
/// Measures with `agg` can only reference physical columns, keys, or dimensions.
/// Deriving a measure from another measure creates meaningless two-stage aggregation;
/// use a metric instead (D-042).
fn validate_measure_references(
    measures: &IndexMap<String, CompiledMeasure>,
) -> Result<(), CompileError> {
    let mut errors = Vec::new();
    let measure_names: HashSet<&str> = measures.keys().map(|k| k.as_str()).collect();

    for (name, measure) in measures {
        let refs = collect_expr_column_refs(&measure.expr);
        for ref_name in &refs {
            // Self-reference is fine (measure referencing its own name = physical column).
            if ref_name != name && measure_names.contains(ref_name.as_str()) {
                errors.push(format!(
                    "measure '{}': references measure '{}' but has aggregation; \
                     use a metric to derive from other measures",
                    name, ref_name
                ));
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(CompileError::ExprCompilation(errors))
    }
}

/// Collect all column/entity reference names from an expression tree.
fn collect_expr_column_refs(expr: &Expr) -> Vec<String> {
    let mut refs = Vec::new();
    expr.walk(&mut |node| {
        match node {
            Expr::Column(col) => refs.push(col.name.clone()),
            Expr::EntityRef(er) => refs.push(er.name.clone()),
            _ => {}
        }
    });
    refs
}

fn compile_measure_filters(
    filters: &[MeasureFilter],
) -> Result<Vec<CompiledFilter>, CompileError> {
    let mut compiled = Vec::new();
    for mf in filters {
        let expr = resolve_expr_source(&mf.expr, &mf.name)?;
        compiled.push(CompiledFilter::from_measure_filter(mf, expr));
    }
    Ok(compiled)
}

/// Parse a DSL expression string into a Expr.
///
/// For v1, we parse common aggregation patterns (SUM, COUNT, etc.)
/// and store other expressions as entity refs. Full DSL parsing
/// will use the semstrait-core DSL lexer/parser when stabilized.
fn parse_expr(expr: &str, entity_name: &str) -> Result<Expr, CompileError> {
    let trimmed = expr.trim();

    if trimmed.is_empty() {
        return Err(CompileError::ExprCompilation(vec![format!(
            "empty expression for '{}'",
            entity_name
        )]));
    }

    // Reject raw SQL
    if looks_like_raw_sql(trimmed) {
        return Err(CompileError::RawSqlRejected {
            entity: entity_name.to_string(),
            expr: trimmed.to_string(),
        });
    }

    // Try parsing aggregation patterns: SUM(col), COUNT(col), etc.
    if let Some(parsed) = try_parse_aggregation(trimmed) {
        return Ok(parsed);
    }

    // Predicate parsers (lowest → highest precedence).
    // Logical OR
    if let Some(parsed) = try_parse_logical_or(trimmed) {
        return Ok(parsed);
    }
    // Logical AND
    if let Some(parsed) = try_parse_logical_and(trimmed) {
        return Ok(parsed);
    }
    // Comparison operators: =, !=, <, >, <=, >=
    if let Some(parsed) = try_parse_comparison(trimmed) {
        return Ok(parsed);
    }
    // IS NULL / IS NOT NULL
    if let Some(parsed) = try_parse_is_null(trimmed) {
        return Ok(parsed);
    }
    // NOT prefix
    if let Some(parsed) = try_parse_not_prefix(trimmed) {
        return Ok(parsed);
    }

    // Arithmetic: a op b (before entity ref, since
    // "{{ a }} - {{ b }}" starts with {{ and ends with }} but is arithmetic)
    if let Some(parsed) = try_parse_arithmetic(trimmed) {
        return Ok(parsed);
    }

    // Try entity ref: {{ name }}
    if trimmed.starts_with("{{") && trimmed.ends_with("}}") {
        let inner = trimmed[2..trimmed.len() - 2].trim();
        return Ok(Expr::entity_ref(inner));
    }

    // String literal: 'value'
    if trimmed.starts_with('\'') && trimmed.ends_with('\'') && trimmed.len() >= 2 {
        let inner = &trimmed[1..trimmed.len() - 1];
        return Ok(Expr::string(inner.to_string()));
    }

    // Boolean / null literals (before identifier — these are valid identifiers)
    match trimmed.to_lowercase().as_str() {
        "true" => return Ok(Expr::boolean(true)),
        "false" => return Ok(Expr::boolean(false)),
        "null" => return Ok(Expr::null()),
        _ => {}
    }

    // Bare identifier => entity ref
    if is_identifier(trimmed) {
        return Ok(Expr::entity_ref(trimmed));
    }

    // Numeric literal
    if let Ok(v) = trimmed.parse::<i64>() {
        return Ok(Expr::int(v));
    }
    if let Ok(v) = trimmed.parse::<f64>() {
        return Ok(Expr::float(v));
    }

    // Fallback: store as entity ref
    Ok(Expr::entity_ref(trimmed))
}

/// SQL keywords that indicate raw SQL (rejected in v1).
const SQL_KEYWORDS: &[&str] = &[
    "SELECT", "FROM", "WHERE", "INSERT", "UPDATE", "DELETE", "JOIN", "UNION",
    "GROUP BY", "ORDER BY", "HAVING", "LIMIT", "CREATE", "ALTER", "DROP",
];

fn looks_like_raw_sql(expr: &str) -> bool {
    let upper = expr.to_uppercase();
    // Use word-boundary matching: keyword must be preceded/followed by a non-alphanumeric
    // character (or be at the start/end of the string). This avoids false positives like
    // "deleted" matching "DELETE" or "updated_at" matching "UPDATE".
    SQL_KEYWORDS.iter().any(|kw| {
        let mut start = 0;
        while let Some(pos) = upper[start..].find(kw) {
            let abs_pos = start + pos;
            let end_pos = abs_pos + kw.len();
            let before_ok = abs_pos == 0
                || !upper.as_bytes()[abs_pos - 1].is_ascii_alphanumeric();
            let after_ok = end_pos == upper.len()
                || !upper.as_bytes()[end_pos].is_ascii_alphanumeric();
            if before_ok && after_ok {
                return true;
            }
            start = abs_pos + 1;
        }
        false
    })
}

/// Extract aggregation function and inner expression from a legacy parsed Expr.
///
/// Given a parsed Expr like `Aggregate(Sum, Column("amount"))`, returns
/// `(Aggregation::Sum, Column("amount"))`. Used for auto-upgrading legacy
/// measures that embed aggregation in `expr` instead of using `agg:`.
fn try_extract_aggregation_from_expr(expr: &Expr) -> Option<(semstrait_core::Aggregation, Expr)> {
    match expr {
        Expr::Aggregate(agg_expr) => {
            Some((agg_expr.function, *agg_expr.expr.clone()))
        }
        _ => None,
    }
}

/// Collect transitive leaf measure additivity values from a metric expression.
///
/// Walks the expression tree, resolving EntityRef/Column names against measures
/// (and recursively against already-compiled metrics). Returns the set of
/// AdditivityType values from all leaf measures reached.
fn collect_leaf_measure_additivity(
    expr: &Expr,
    measures: &IndexMap<String, CompiledMeasure>,
    metrics: &IndexMap<String, CompiledMetric>,
) -> Vec<AdditivityType> {
    let mut result = Vec::new();
    collect_leaf_additivity_inner(expr, measures, metrics, &mut result);
    result
}

fn collect_leaf_additivity_inner(
    expr: &Expr,
    measures: &IndexMap<String, CompiledMeasure>,
    metrics: &IndexMap<String, CompiledMetric>,
    result: &mut Vec<AdditivityType>,
) {
    match expr {
        Expr::Column(col) => {
            if let Some(m) = measures.get(&col.name) {
                if let Some(ref a) = m.additivity {
                    result.push(a.clone());
                }
            } else if let Some(met) = metrics.get(&col.name) {
                if let Some(ref a) = met.additivity {
                    result.push(a.clone());
                } else {
                    collect_leaf_additivity_inner(&met.expr, measures, metrics, result);
                }
            }
        }
        Expr::EntityRef(er) => {
            if let Some(m) = measures.get(&er.name) {
                if let Some(ref a) = m.additivity {
                    result.push(a.clone());
                }
            } else if let Some(met) = metrics.get(&er.name) {
                if let Some(ref a) = met.additivity {
                    result.push(a.clone());
                } else {
                    collect_leaf_additivity_inner(&met.expr, measures, metrics, result);
                }
            }
        }
        Expr::BinaryOp(bin) => {
            collect_leaf_additivity_inner(&bin.left, measures, metrics, result);
            collect_leaf_additivity_inner(&bin.right, measures, metrics, result);
        }
        Expr::Case(case) => {
            for wt in &case.when_then {
                collect_leaf_additivity_inner(&wt.condition, measures, metrics, result);
                collect_leaf_additivity_inner(&wt.result, measures, metrics, result);
            }
            if let Some(ref e) = case.else_expr {
                collect_leaf_additivity_inner(e, measures, metrics, result);
            }
        }
        _ => {}
    }
}

fn try_parse_aggregation(expr: &str) -> Option<Expr> {
    let upper = expr.to_uppercase();

    #[allow(clippy::type_complexity)]
    let agg_patterns: &[(&str, fn(Expr) -> Expr)] = &[
        ("SUM(", Expr::sum),
        ("COUNT_DISTINCT(", Expr::count_distinct),
        ("COUNT(", Expr::count),
        ("AVG(", Expr::avg),
        ("MIN(", Expr::min),
        ("MAX(", Expr::max),
    ];

    for (prefix, constructor) in agg_patterns {
        if upper.starts_with(prefix) && expr.ends_with(')') {
            let inner = expr[prefix.len()..expr.len() - 1].trim();
            let inner_expr = if is_identifier(inner) {
                Expr::column(inner)
            } else {
                Expr::entity_ref(inner)
            };
            return Some(constructor(inner_expr));
        }
    }

    None
}

// ── Inline DSL scan state ──────────────────────────────────────────────────
//
// Shared depth/quoting tracker for all try_parse_* byte-scan functions.
// Ensures operators inside parentheses or single-quoted string literals are
// never treated as top-level split points.

struct ScanState {
    paren_depth: i32,
    in_string: bool,
}

impl ScanState {
    fn new() -> Self {
        Self { paren_depth: 0, in_string: false }
    }

    /// Update state for the byte at position `i`. Returns `true` if the byte
    /// was consumed by quoting/nesting and the caller should skip operator checks.
    fn update(&mut self, b: u8) -> bool {
        if b == b'\'' {
            self.in_string = !self.in_string;
            return true;
        }
        if self.in_string {
            return true;
        }
        match b {
            b'(' => { self.paren_depth += 1; true }
            b')' => { self.paren_depth -= 1; true }
            _ => false,
        }
    }

    fn at_top_level(&self) -> bool {
        self.paren_depth == 0 && !self.in_string
    }
}

// ── Predicate parsers (lowest → highest precedence) ───────────────────────

/// Check if byte at `pos` is a word boundary in `bytes` (non-alphanumeric or string edge).
fn is_word_boundary(bytes: &[u8], pos: usize) -> bool {
    pos >= bytes.len() || !bytes[pos].is_ascii_alphanumeric() && bytes[pos] != b'_'
}

/// Find the last top-level occurrence of a case-insensitive keyword with word boundaries.
/// Returns the byte offset of the keyword start, or None.
fn find_last_keyword(expr: &str, keyword: &str) -> Option<usize> {
    let bytes = expr.as_bytes();
    let kw_len = keyword.len();
    let mut state = ScanState::new();
    let mut last_pos = None;

    let mut i = 0;
    while i < bytes.len() {
        if state.update(bytes[i]) {
            i += 1;
            continue;
        }
        if !state.at_top_level() {
            i += 1;
            continue;
        }
        if i + kw_len <= bytes.len()
            && expr[i..i + kw_len].eq_ignore_ascii_case(keyword)
            && (i == 0 || is_word_boundary(bytes, i - 1))
            && is_word_boundary(bytes, i + kw_len)
        {
            last_pos = Some(i);
        }
        i += 1;
    }
    last_pos
}

/// Try to parse a top-level `OR` (lowest precedence logical operator).
fn try_parse_logical_or(expr: &str) -> Option<Expr> {
    let pos = find_last_keyword(expr, "OR")?;
    let left = expr[..pos].trim();
    let right = expr[pos + 2..].trim();
    if left.is_empty() || right.is_empty() {
        return None;
    }
    Some(Expr::or(parse_predicate_operand(left), parse_predicate_operand(right)))
}

/// Try to parse a top-level `AND` (binds tighter than OR).
fn try_parse_logical_and(expr: &str) -> Option<Expr> {
    let pos = find_last_keyword(expr, "AND")?;
    let left = expr[..pos].trim();
    let right = expr[pos + 3..].trim();
    if left.is_empty() || right.is_empty() {
        return None;
    }
    Some(Expr::and(parse_predicate_operand(left), parse_predicate_operand(right)))
}

/// Try to parse a top-level comparison operator (`=`, `!=`, `<`, `>`, `<=`, `>=`).
fn try_parse_comparison(expr: &str) -> Option<Expr> {
    let bytes = expr.as_bytes();
    let mut state = ScanState::new();
    // Track the last comparison operator found: (position, operator_str_len, op_kind)
    let mut last_cmp: Option<(usize, usize, &str)> = None;

    let mut i = 0;
    while i < bytes.len() {
        if state.update(bytes[i]) {
            i += 1;
            continue;
        }
        if !state.at_top_level() {
            i += 1;
            continue;
        }

        // Multi-byte operators first (longest match)
        if bytes[i] == b'!' && bytes.get(i + 1) == Some(&b'=') {
            last_cmp = Some((i, 2, "!="));
            i += 2;
            continue;
        }
        if bytes[i] == b'<' && bytes.get(i + 1) == Some(&b'=') {
            last_cmp = Some((i, 2, "<="));
            i += 2;
            continue;
        }
        if bytes[i] == b'>' && bytes.get(i + 1) == Some(&b'=') {
            last_cmp = Some((i, 2, ">="));
            i += 2;
            continue;
        }
        // Single-byte (only if not part of multi-byte)
        if bytes[i] == b'=' {
            if i == 0 || !matches!(bytes[i - 1], b'!' | b'<' | b'>') {
                last_cmp = Some((i, 1, "="));
            }
        } else if bytes[i] == b'<' {
            // already checked <= above, so this is bare <
            last_cmp = Some((i, 1, "<"));
        } else if bytes[i] == b'>' {
            last_cmp = Some((i, 1, ">"));
        }
        i += 1;
    }

    let (pos, len, op_str) = last_cmp?;
    let left = expr[..pos].trim();
    let right = expr[pos + len..].trim();
    if left.is_empty() || right.is_empty() {
        return None;
    }

    let l = parse_predicate_operand(left);
    let r = parse_predicate_operand(right);
    Some(match op_str {
        "=" => Expr::eq(l, r),
        "!=" => Expr::ne(l, r),
        "<" => Expr::lt(l, r),
        ">" => Expr::gt(l, r),
        "<=" => Expr::lte(l, r),
        ">=" => Expr::gte(l, r),
        _ => unreachable!(),
    })
}

/// Try to parse `<expr> IS NOT NULL` or `<expr> IS NULL` suffix.
fn try_parse_is_null(expr: &str) -> Option<Expr> {
    let upper = expr.to_uppercase();
    if upper.ends_with(" IS NOT NULL") {
        let subject = expr[..expr.len() - " IS NOT NULL".len()].trim();
        if !subject.is_empty() {
            return Some(Expr::is_not_null(parse_predicate_operand(subject)));
        }
    } else if upper.ends_with(" IS NULL") {
        let subject = expr[..expr.len() - " IS NULL".len()].trim();
        if !subject.is_empty() {
            return Some(Expr::is_null(parse_predicate_operand(subject)));
        }
    }
    None
}

/// Try to parse `NOT <expr>` prefix.
fn try_parse_not_prefix(expr: &str) -> Option<Expr> {
    let upper = expr.to_uppercase();
    if upper.starts_with("NOT ") {
        let remainder = expr[4..].trim();
        if !remainder.is_empty() {
            return Some(Expr::not(parse_predicate_operand(remainder)));
        }
    }
    None
}

/// Recursive descent operand parser for the full predicate grammar.
///
/// Tries parsers from lowest to highest precedence:
/// OR → AND → comparison → IS NULL → NOT → arithmetic → atom.
fn parse_predicate_operand(s: &str) -> Expr {
    let trimmed = s.trim();

    // Strip balanced outer parens
    if trimmed.starts_with('(') && trimmed.ends_with(')') {
        let inner = &trimmed[1..trimmed.len() - 1];
        // Verify the parens are actually balanced (not "(a) + (b)")
        let mut depth = 0i32;
        let mut balanced = true;
        for (i, b) in inner.bytes().enumerate() {
            match b {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth < 0 {
                        balanced = false;
                        break;
                    }
                }
                _ => {}
            }
            // If depth goes negative before the end, the parens don't wrap the whole expr
            let _ = i;
        }
        if balanced && depth == 0 {
            return parse_predicate_operand(inner);
        }
    }

    // Aggregation (highest precedence in DSL context)
    if let Some(agg) = try_parse_aggregation(trimmed) {
        return agg;
    }

    // Logical OR (lowest)
    if let Some(parsed) = try_parse_logical_or(trimmed) {
        return parsed;
    }
    // Logical AND
    if let Some(parsed) = try_parse_logical_and(trimmed) {
        return parsed;
    }
    // Comparison
    if let Some(parsed) = try_parse_comparison(trimmed) {
        return parsed;
    }
    // IS NULL / IS NOT NULL
    if let Some(parsed) = try_parse_is_null(trimmed) {
        return parsed;
    }
    // NOT prefix
    if let Some(parsed) = try_parse_not_prefix(trimmed) {
        return parsed;
    }
    // Arithmetic
    if let Some(parsed) = try_parse_arithmetic(trimmed) {
        return parsed;
    }

    // Atom (delegate to existing parse_operand for literals, entity refs, identifiers)
    parse_operand(trimmed)
}

fn try_parse_arithmetic(expr: &str) -> Option<Expr> {
    let bytes = expr.as_bytes();
    let mut state = ScanState::new();
    let mut last_add_sub = None;
    let mut last_mul_div = None;

    for (i, &b) in bytes.iter().enumerate() {
        if state.update(b) {
            continue;
        }
        if !state.at_top_level() {
            continue;
        }
        match b {
            b'+' | b'-' if i > 0 => {
                // Require at least one space adjacent to +/- to distinguish
                // arithmetic operators from hyphens in identifier names like
                // `adwords-averageCost`. `revenue - cost` has spaces; column
                // names with hyphens do not.
                let prev_space = i > 0 && bytes[i - 1] == b' ';
                let next_space = i + 1 < bytes.len() && bytes[i + 1] == b' ';
                if prev_space || next_space {
                    last_add_sub = Some(i);
                }
            }
            b'*' | b'/' if i > 0 => {
                last_mul_div = Some(i);
            }
            _ => {}
        }
    }

    let split_pos = last_add_sub.or(last_mul_div)?;
    let op = bytes[split_pos] as char;
    let left = expr[..split_pos].trim();
    let right = expr[split_pos + 1..].trim();

    if left.is_empty() || right.is_empty() {
        return None;
    }

    let left_expr = parse_operand(left);
    let right_expr = parse_operand(right);

    Some(match op {
        '+' => Expr::add(left_expr, right_expr),
        '-' => Expr::subtract(left_expr, right_expr),
        '*' => Expr::multiply(left_expr, right_expr),
        '/' => Expr::safe_divide(left_expr, right_expr),
        _ => return None,
    })
}

fn parse_operand(s: &str) -> Expr {
    let trimmed = s.trim();

    // Strip outer parens
    if trimmed.starts_with('(') && trimmed.ends_with(')') {
        let inner = &trimmed[1..trimmed.len() - 1];
        if let Some(agg) = try_parse_aggregation(inner) {
            return agg;
        }
        if let Some(arith) = try_parse_arithmetic(inner) {
            return arith;
        }
    }

    if let Some(agg) = try_parse_aggregation(trimmed) {
        return agg;
    }

    if trimmed.starts_with("{{") && trimmed.ends_with("}}") {
        let inner = trimmed[2..trimmed.len() - 2].trim();
        return Expr::entity_ref(inner);
    }

    // String literal: 'value'
    if trimmed.starts_with('\'') && trimmed.ends_with('\'') && trimmed.len() >= 2 {
        let inner = &trimmed[1..trimmed.len() - 1];
        return Expr::string(inner.to_string());
    }

    // Boolean / null literals (before identifier check — these are valid identifiers)
    match trimmed.to_lowercase().as_str() {
        "true" => return Expr::boolean(true),
        "false" => return Expr::boolean(false),
        "null" => return Expr::null(),
        _ => {}
    }

    if let Ok(v) = trimmed.parse::<i64>() {
        return Expr::int(v);
    }
    if let Ok(v) = trimmed.parse::<f64>() {
        return Expr::float(v);
    }

    if is_identifier(trimmed) {
        return Expr::entity_ref(trimmed);
    }

    Expr::entity_ref(trimmed)
}

fn is_identifier(s: &str) -> bool {
    !s.is_empty()
        && s.chars().all(|c| c.is_alphanumeric() || c == '_')
        && s.chars().next().is_some_and(|c| c.is_alphabetic() || c == '_')
}

fn dataset_display_name(name: &DatasetName) -> &str {
    match name {
        DatasetName::Literal(n) => n.as_str(),
        DatasetName::Glob(g) => g.0.as_str(),
    }
}

/// Collect all metrics from all scopes in the model.
fn collect_all_metrics(model: &SemanticModel) -> Vec<&Metric> {
    let mut all = Vec::new();
    all.extend(model.metrics.iter());

    for dk in model.entities.values() {
        for m in dk.interface().metrics.values() {
            if let MetricEntry::Inline(met) = m {
                all.push(met);
            }
        }
    }

    // Deduplicate by name (keep first occurrence)
    let mut seen = HashSet::new();
    all.retain(|m| seen.insert(m.name.clone()));

    all
}

/// Extract identifiers from an expression string.
/// Filters out SQL keywords and numeric literals.
fn extract_identifiers_from_expr(expr: &str) -> Vec<String> {
    let sql_keywords: HashSet<&str> = [
        "sum", "avg", "count", "min", "max", "distinct", "case", "when", "then",
        "else", "end", "and", "or", "not", "null", "true", "false", "is", "in",
        "between", "like", "as", "if", "coalesce", "count_distinct",
    ]
    .into_iter()
    .collect();

    expr.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| !s.is_empty())
        .filter(|s| !sql_keywords.contains(&s.to_lowercase().as_str()))
        .filter(|s| s.parse::<f64>().is_err())
        .map(String::from)
        .collect()
}

/// Validate all `FunctionCall` nodes in an Expr tree against the registry.
///
/// Returns a list of validation error messages. Unknown functions are ignored
/// (they may be engine-specific); only arity mismatches are errors.
fn validate_function_calls(
    expr: &Expr,
    entity_name: &str,
    registry: &crate::function_registry::FunctionRegistry,
) -> Vec<String> {
    let mut errors = Vec::new();
    expr.walk(&mut |node| {
        if let Expr::FunctionCall(fc) = node {
            if let Err(msg) = registry.validate(&fc.name, fc.args.len()) {
                errors.push(format!("'{}': {}", entity_name, msg));
            }
        }
    });
    errors
}

/// Extract identifiers from an `ExprSource`.
///
/// For inline DSL strings, delegates to `extract_identifiers_from_expr`.
/// For declarative blocks, collects entity refs from the converted Expr tree.
fn extract_identifiers_from_expr_source(source: &semstrait_model::expr_block::ExprSource) -> Vec<String> {
    match source {
        semstrait_model::expr_block::ExprSource::Inline(s) => extract_identifiers_from_expr(s),
        semstrait_model::expr_block::ExprSource::Declarative(block) => {
            // Convert to Expr and collect EntityRef names.
            // Column references in declarative blocks may also be entity refs
            // in metric context — collect both.
            match block.to_expr() {
                Ok(expr) => collect_identifiers_from_expr(&expr),
                Err(_) => vec![],
            }
        }
    }
}

/// Collect identifier names from a compiled Expr tree (for metric dependency graph).
fn collect_identifiers_from_expr(expr: &Expr) -> Vec<String> {
    let mut names = Vec::new();
    expr.walk(&mut |node| {
        match node {
            Expr::EntityRef(e) => names.push(e.name.clone()),
            Expr::Column(c) => names.push(c.name.clone()),
            _ => {}
        }
    });
    names
}

/// Resolve an `ExprSource` to a core `Expr`.
///
/// Inline strings are parsed via the DSL parser.
/// Declarative blocks are converted directly via `ExprBlock::to_expr()`.
fn resolve_expr_source(
    source: &semstrait_model::expr_block::ExprSource,
    context_name: &str,
) -> Result<Expr, CompileError> {
    match source {
        semstrait_model::expr_block::ExprSource::Inline(s) => parse_expr(s, context_name),
        semstrait_model::expr_block::ExprSource::Declarative(block) => {
            block.to_expr().map_err(|e| {
                CompileError::ExprCompilation(vec![format!("'{}': {}", context_name, e)])
            })
        }
    }
}

/// Parse a table FQN string into a TableRef.
///
/// The table name is always the last dot-separated segment.
/// Everything before it is treated as namespace (supports nested namespaces like `ns1.ns2`).
/// A bare name with no dots uses `default_namespace`.
///
/// Examples:
///   "orders"             → namespace=default_namespace, table="orders"
///   "sales.orders"       → namespace="sales", table="orders"
///   "ns1.ns2.orders"     → namespace="ns1.ns2", table="orders"
/// Split a table pattern into (namespace, table_part).
///
/// Uses `rsplit_once('.')` — same logic as `parse_table_ref` — so that the
/// namespace portion is extracted from qualified patterns like `adwords.*`.
///
/// Examples:
///   "adwords.*"        → ("adwords", "*")
///   "ns1.ns2.*"        → ("ns1.ns2", "*")
///   "*"                → (default_namespace, "*")
///   "my_table"         → (default_namespace, "my_table")
fn split_table_pattern<'a>(pattern: &'a str, default_namespace: &'a str) -> (&'a str, &'a str) {
    match pattern.rsplit_once('.') {
        Some((prefix, suffix)) => (prefix, suffix),
        None => (default_namespace, pattern),
    }
}

fn parse_table_ref(fqn: &str, default_namespace: &str) -> semstrait_catalog::TableRef {
    match fqn.rsplit_once('.') {
        Some((prefix, table_name)) => {
            semstrait_catalog::TableRef::new(prefix, table_name)
        }
        None => {
            // Bare table name — use default namespace
            semstrait_catalog::TableRef::new(default_namespace, fqn)
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_identifiers() {
        let ids = extract_identifiers_from_expr("revenue / order_count");
        assert!(ids.contains(&"revenue".to_string()));
        assert!(ids.contains(&"order_count".to_string()));
    }

    #[test]
    fn test_extract_identifiers_with_aggregates() {
        let ids = extract_identifiers_from_expr("SUM(amount)");
        assert!(ids.contains(&"amount".to_string()));
        assert!(!ids.iter().any(|i| i.to_lowercase() == "sum"));
    }

    #[test]
    fn test_parse_expr_sum() {
        let expr = parse_expr("SUM(amount)", "revenue").unwrap();
        match &expr {
            Expr::Aggregate(agg) => {
                assert_eq!(agg.function, semstrait_core::Aggregation::Sum);
                match agg.expr.as_ref() {
                    Expr::Column(col) => assert_eq!(col.name, "amount"),
                    _ => panic!("expected Column inside Sum"),
                }
            }
            _ => panic!("expected Aggregate(Sum), got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_expr_count_distinct() {
        let expr = parse_expr("COUNT_DISTINCT(customer_id)", "unique_customers").unwrap();
        match &expr {
            Expr::Aggregate(agg) => {
                assert_eq!(agg.function, semstrait_core::Aggregation::CountDistinct);
            }
            _ => panic!("expected Aggregate(CountDistinct), got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_expr_entity_ref() {
        let expr = parse_expr("{{ revenue }}", "margin").unwrap();
        match &expr {
            Expr::EntityRef(e) => assert_eq!(e.name, "revenue"),
            _ => panic!("expected EntityRef"),
        }
    }

    #[test]
    fn test_parse_expr_arithmetic() {
        let expr = parse_expr("{{ revenue }} - {{ cost }}", "profit").unwrap();
        match &expr {
            Expr::BinaryOp(bin) => {
                assert_eq!(bin.op, semstrait_core::BinaryOp::Subtract);
            }
            _ => panic!("expected BinaryOp(Subtract), got {:?}", expr),
        }
    }

    // Hyphenated names must parse as a single EntityRef, not arithmetic.
    // Column names like `adwords-averageCost` are atomic identifiers — the
    // hyphen is part of the name, not a minus operator.
    #[test]
    fn test_parse_expr_hyphenated_name_is_entity_ref() {
        let expr = parse_expr("adwords-averageCost", "adwords-averageCost").unwrap();
        match &expr {
            Expr::EntityRef(e) => assert_eq!(e.name, "adwords-averageCost"),
            other => panic!("expected EntityRef(adwords-averageCost), got {:?}", other),
        }
    }

    // Spaced arithmetic must still work.
    #[test]
    fn test_parse_expr_spaced_arithmetic_still_works() {
        let expr = parse_expr("revenue - cost", "profit").unwrap();
        match &expr {
            Expr::BinaryOp(bin) => assert_eq!(bin.op, semstrait_core::BinaryOp::Subtract),
            other => panic!("expected BinaryOp(Subtract), got {:?}", other),
        }
    }

    // Multi-hyphen names must also be atomic.
    #[test]
    fn test_parse_expr_multi_hyphen_name_is_entity_ref() {
        let expr = parse_expr("adwords-conversions-cost-per-conversion", "adwords-conversions-cost-per-conversion").unwrap();
        match &expr {
            Expr::EntityRef(e) => assert_eq!(e.name, "adwords-conversions-cost-per-conversion"),
            other => panic!("expected EntityRef, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_expr_reject_raw_sql() {
        let result = parse_expr("SELECT sum(amount) FROM orders", "bad_metric");
        assert!(matches!(result, Err(CompileError::RawSqlRejected { .. })));
    }

    #[test]
    fn test_validate_structure_duplicate_dataset() {
        let model = SemanticModel {
            name: "test".to_string(),
            description: None,
            ai_context: None,
            labels: vec![],
            namespace: None,
            entities: BTreeMap::from([
                ("orders".to_string(), DataKind::Simple(SimpleDataKind {
                    name: "orders".to_string(),
                    interface: SemanticInterface::default(),
                    extras: None,
                })),
                ("orders_dup".to_string(), DataKind::Simple(SimpleDataKind {
                    name: "orders".to_string(),
                    interface: SemanticInterface::default(),
                    extras: None,
                })),
            ]),
            relationships: vec![],
            dimensions: vec![],
            measures: vec![],
            metrics: vec![],
        };

        let result = validate_structure(&model);
        assert!(matches!(result, Err(CompileError::StructureValidation(_))));
    }

    #[test]
    fn test_validate_structure_empty_kind() {
        let model = SemanticModel {
            name: "test".to_string(),
            description: None,
            ai_context: None,
            labels: vec![],
            namespace: None,
            entities: BTreeMap::from([
                ("empty_kind".to_string(), DataKind::Complex(ComplexDataKind::Grainset(GrainsetSpec {
                    name: "empty_kind".to_string(),
                    interface: SemanticInterface::default(),
                    children: vec![],
                    extras: None,
                }))),
            ]),
            relationships: vec![],
            dimensions: vec![],
            measures: vec![],
            metrics: vec![],
        };

        let result = validate_structure(&model);
        assert!(matches!(result, Err(CompileError::StructureValidation(_))));
    }

    #[test]
    fn test_is_identifier() {
        assert!(is_identifier("revenue"));
        assert!(is_identifier("order_count"));
        assert!(is_identifier("_private"));
        assert!(!is_identifier("123abc"));
        assert!(!is_identifier(""));
        assert!(!is_identifier("a b"));
    }

    #[test]
    fn test_split_table_pattern_qualified_glob() {
        let (ns, table) = split_table_pattern("adwords.*", "default");
        assert_eq!(ns, "adwords");
        assert_eq!(table, "*");
    }

    #[test]
    fn test_split_table_pattern_nested_namespace() {
        let (ns, table) = split_table_pattern("ns1.ns2.*", "default");
        assert_eq!(ns, "ns1.ns2");
        assert_eq!(table, "*");
    }

    #[test]
    fn test_split_table_pattern_bare_glob() {
        let (ns, table) = split_table_pattern("*", "my_namespace");
        assert_eq!(ns, "my_namespace");
        assert_eq!(table, "*");
    }

    #[test]
    fn test_split_table_pattern_bare_name() {
        let (ns, table) = split_table_pattern("my_table", "default");
        assert_eq!(ns, "default");
        assert_eq!(table, "my_table");
    }

    #[test]
    fn test_split_table_pattern_qualified_name() {
        let (ns, table) = split_table_pattern("sales.orders", "default");
        assert_eq!(ns, "sales");
        assert_eq!(table, "orders");
    }

    #[test]
    fn test_parse_table_ref_simple() {
        let r = parse_table_ref("orders", "default");
        assert_eq!(r.namespace, "default");
        assert_eq!(r.name, "orders");
    }

    #[test]
    fn test_parse_table_ref_qualified() {
        let r = parse_table_ref("sales.orders", "default");
        assert_eq!(r.namespace, "sales");
        assert_eq!(r.name, "orders");
    }

    #[test]
    fn test_parse_table_ref_nested_namespace() {
        let r = parse_table_ref("ns1.ns2.orders", "default");
        assert_eq!(r.namespace, "ns1.ns2");
        assert_eq!(r.name, "orders");
    }

    /// SR-5a: Expression referencing name not in interface → error.
    #[test]
    fn test_validate_expr_scope_unknown_ref() {
        use indexmap::IndexMap;
        use semstrait_core::expr::Expr;
        use crate::compiled::CompiledDimension;

        let mut dimensions = IndexMap::new();
        dimensions.insert("country".to_string(), CompiledDimension {
            name: "country".to_string(),
            description: None,
            data_type: semstrait_core::DataType::String,
            dim_type: semstrait_model::DimensionType::Categorical(semstrait_model::CategoricalDimension { enum_values: None }),
            expr: Some(Expr::column("nonexistent_col")), // references unknown name
            expr_source: Some("nonexistent_col".to_string()),
        });

        let measures = IndexMap::new();
        let metrics = IndexMap::new();

        let result = super::validate_expr_scope(&dimensions, &measures, &metrics, "test_kind");
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(err.contains("nonexistent_col"), "error: {}", err);
    }

    /// SR-5a: Expression referencing valid interface name → OK.
    #[test]
    fn test_validate_expr_scope_valid_ref() {
        use indexmap::IndexMap;
        use semstrait_core::expr::Expr;
        use crate::compiled::CompiledDimension;

        let mut dimensions = IndexMap::new();
        dimensions.insert("raw_name".to_string(), CompiledDimension {
            name: "raw_name".to_string(),
            description: None,
            data_type: semstrait_core::DataType::String,
            dim_type: semstrait_model::DimensionType::Categorical(semstrait_model::CategoricalDimension { enum_values: None }),
            expr: None,
            expr_source: None,
        });
        dimensions.insert("name_upper".to_string(), CompiledDimension {
            name: "name_upper".to_string(),
            description: None,
            data_type: semstrait_core::DataType::String,
            dim_type: semstrait_model::DimensionType::Categorical(semstrait_model::CategoricalDimension { enum_values: None }),
            expr: Some(Expr::column("raw_name")), // references valid dim
            expr_source: Some("UPPER(raw_name)".to_string()),
        });

        let measures = IndexMap::new();
        let metrics = IndexMap::new();

        let result = super::validate_expr_scope(&dimensions, &measures, &metrics, "test_kind");
        assert!(result.is_ok());
    }

    // ── Predicate parsing tests ────────────────────────────────────────

    #[test]
    fn test_parse_expr_simple_equality() {
        let expr = parse_expr("status = 'active'", "filter").unwrap();
        match &expr {
            Expr::BinaryOp(bin) => {
                assert_eq!(bin.op, semstrait_core::BinaryOp::Eq);
                assert!(matches!(&*bin.left, Expr::EntityRef(e) if e.name == "status"));
                assert!(matches!(&*bin.right, Expr::Literal(semstrait_core::expr::Literal::String { value }) if value == "active"));
            }
            _ => panic!("expected BinaryOp(Eq), got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_expr_not_equal() {
        let expr = parse_expr("status != 'cancelled'", "filter").unwrap();
        match &expr {
            Expr::BinaryOp(bin) => {
                assert_eq!(bin.op, semstrait_core::BinaryOp::NotEq);
                assert!(matches!(&*bin.left, Expr::EntityRef(e) if e.name == "status"));
                assert!(matches!(&*bin.right, Expr::Literal(semstrait_core::expr::Literal::String { value }) if value == "cancelled"));
            }
            _ => panic!("expected BinaryOp(NotEq), got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_expr_gt_numeric() {
        let expr = parse_expr("amount > 100", "filter").unwrap();
        match &expr {
            Expr::BinaryOp(bin) => {
                assert_eq!(bin.op, semstrait_core::BinaryOp::Gt);
                assert!(matches!(&*bin.left, Expr::EntityRef(e) if e.name == "amount"));
                assert!(matches!(&*bin.right, Expr::Literal(semstrait_core::expr::Literal::Integer { value: 100 })));
            }
            _ => panic!("expected BinaryOp(Gt), got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_expr_lte_float() {
        let expr = parse_expr("price <= 9.99", "filter").unwrap();
        match &expr {
            Expr::BinaryOp(bin) => {
                assert_eq!(bin.op, semstrait_core::BinaryOp::LtEq);
                assert!(matches!(&*bin.left, Expr::EntityRef(e) if e.name == "price"));
            }
            _ => panic!("expected BinaryOp(LtEq), got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_expr_and_combinator() {
        let expr = parse_expr("status = 'active' AND amount > 0", "filter").unwrap();
        match &expr {
            Expr::BinaryOp(bin) => {
                assert_eq!(bin.op, semstrait_core::BinaryOp::And);
                assert!(matches!(&*bin.left, Expr::BinaryOp(inner) if inner.op == semstrait_core::BinaryOp::Eq));
                assert!(matches!(&*bin.right, Expr::BinaryOp(inner) if inner.op == semstrait_core::BinaryOp::Gt));
            }
            _ => panic!("expected BinaryOp(And), got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_expr_or_combinator() {
        let expr = parse_expr("type = 'a' OR type = 'b'", "filter").unwrap();
        match &expr {
            Expr::BinaryOp(bin) => {
                assert_eq!(bin.op, semstrait_core::BinaryOp::Or);
                assert!(matches!(&*bin.left, Expr::BinaryOp(inner) if inner.op == semstrait_core::BinaryOp::Eq));
                assert!(matches!(&*bin.right, Expr::BinaryOp(inner) if inner.op == semstrait_core::BinaryOp::Eq));
            }
            _ => panic!("expected BinaryOp(Or), got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_expr_precedence_and_binds_tighter_than_or() {
        // "a = 1 OR b = 2 AND c = 3" should parse as OR(a=1, AND(b=2, c=3))
        let expr = parse_expr("a = 1 OR b = 2 AND c = 3", "filter").unwrap();
        match &expr {
            Expr::BinaryOp(bin) => {
                assert_eq!(bin.op, semstrait_core::BinaryOp::Or);
                // Right side should be AND
                assert!(matches!(&*bin.right, Expr::BinaryOp(inner) if inner.op == semstrait_core::BinaryOp::And));
            }
            _ => panic!("expected BinaryOp(Or), got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_expr_not_prefix() {
        let expr = parse_expr("NOT deleted", "filter").unwrap();
        assert!(matches!(&expr, Expr::Not(inner) if matches!(&*inner.expr, Expr::EntityRef(e) if e.name == "deleted")));
    }

    #[test]
    fn test_parse_expr_is_null() {
        let expr = parse_expr("end_date IS NULL", "filter").unwrap();
        assert!(matches!(&expr, Expr::IsNull(inner) if matches!(&*inner.expr, Expr::EntityRef(e) if e.name == "end_date")));
    }

    #[test]
    fn test_parse_expr_is_not_null() {
        let expr = parse_expr("start_date IS NOT NULL", "filter").unwrap();
        assert!(matches!(&expr, Expr::IsNotNull(inner) if matches!(&*inner.expr, Expr::EntityRef(e) if e.name == "start_date")));
    }

    #[test]
    fn test_parse_expr_boolean_literal() {
        let expr = parse_expr("is_active = true", "filter").unwrap();
        match &expr {
            Expr::BinaryOp(bin) => {
                assert_eq!(bin.op, semstrait_core::BinaryOp::Eq);
                assert!(matches!(&*bin.right, Expr::Literal(semstrait_core::expr::Literal::Boolean { value: true })));
            }
            _ => panic!("expected BinaryOp(Eq), got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_expr_entity_ref_in_comparison() {
        let expr = parse_expr("{{ status }} != 'cancelled'", "filter").unwrap();
        match &expr {
            Expr::BinaryOp(bin) => {
                assert_eq!(bin.op, semstrait_core::BinaryOp::NotEq);
                assert!(matches!(&*bin.left, Expr::EntityRef(e) if e.name == "status"));
            }
            _ => panic!("expected BinaryOp(NotEq), got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_expr_arithmetic_regression() {
        // Existing arithmetic must still work after predicate parser additions.
        let expr = parse_expr("{{ revenue }} - {{ cost }}", "profit").unwrap();
        match &expr {
            Expr::BinaryOp(bin) => {
                assert_eq!(bin.op, semstrait_core::BinaryOp::Subtract);
            }
            _ => panic!("expected BinaryOp(Subtract), got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_expr_string_with_operator_chars() {
        // Operator chars inside string literals must not cause splits.
        let expr = parse_expr("name = 'greater > less'", "filter").unwrap();
        match &expr {
            Expr::BinaryOp(bin) => {
                assert_eq!(bin.op, semstrait_core::BinaryOp::Eq);
                assert!(matches!(&*bin.left, Expr::EntityRef(e) if e.name == "name"));
                assert!(matches!(&*bin.right, Expr::Literal(semstrait_core::expr::Literal::String { value }) if value == "greater > less"));
            }
            _ => panic!("expected BinaryOp(Eq), got {:?}", expr),
        }
    }
}
