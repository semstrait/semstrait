//! Test helper functions for constructing test fixtures.
//!
//! These helpers create minimal CompiledManifest and ResolvedQueryRequest
//! instances for unit testing. All fixtures construct CompiledDataKind directly.

use indexmap::IndexMap;
use std::collections::HashMap;

use semstrait_manifest::{
    CompiledDimension, CompiledManifest, CompiledMeasure, DimensionType,
};
use semstrait_manifest::acceleration::{
    CompiledDataKind, DatasetBinding, CompiledSimpleKind,
    CompiledInterface, ResolvedColumnMapping,
};

use crate::request::ResolvedQueryRequest;

/// Build a CompiledInterface from dimensions, measures, and metrics.
fn build_interface(
    name: &str,
    dimensions: IndexMap<String, CompiledDimension>,
    measures: IndexMap<String, CompiledMeasure>,
) -> CompiledInterface {
    let temporal_dim = dimensions
        .iter()
        .find(|(_, d)| matches!(d.dim_type, DimensionType::Temporal(_)))
        .map(|(name, _)| name.clone());

    CompiledInterface {
        name: name.to_string(),
        description: None,
        dimensions,
        measures,
        metrics: IndexMap::new(),
        keys: None,
        filters: vec![],
        temporal_dim,
    }
}

/// Build a DatasetBinding from a name and physical column mapping pairs.
fn build_binding(
    name: &str,
    physical_pairs: Vec<(&str, &str)>,
    sources: Vec<&str>,
) -> DatasetBinding {
    let mut physical = IndexMap::new();
    for (semantic, phys) in physical_pairs {
        physical.insert(semantic.to_string(), phys.to_string());
    }
    DatasetBinding {
        dataset_name: name.to_string(),
        column_mapping: ResolvedColumnMapping {
            physical,
            literals: HashMap::new(),
            temporal: HashMap::new(),
            anchored: HashMap::new(),
        },
        resolved_sources: sources
            .into_iter()
            .map(semstrait_manifest::ResolvedSource::path)
            .collect(),
    }
}

/// Create a basic test manifest with a single Dataset kind "orders".
///
/// The kind has dimensions [date, region, customer, user_id] and measure [revenue].
/// It has one dataset "orders_daily" that covers all fields.
pub fn make_test_manifest() -> CompiledManifest {
    make_test_manifest_with_constraints(None, None)
}

/// Create a test manifest with optional constraints on the "revenue" measure.
pub fn make_test_manifest_with_constraints(
    dim_constraints: Option<semstrait_manifest::DimensionConstraints>,
    agg_constraints: Option<semstrait_manifest::AggregationConstraints>,
) -> CompiledManifest {
    let constraints = if dim_constraints.is_some() || agg_constraints.is_some() {
        Some(semstrait_manifest::MeasureConstraints {
            dimensions: dim_constraints,
            aggregations: agg_constraints,
        })
    } else {
        None
    };

    let mut dimensions = IndexMap::new();
    for name in &["date", "region", "customer", "user_id"] {
        dimensions.insert(
            name.to_string(),
            CompiledDimension {
                name: name.to_string(),
                description: None,
                data_type: semstrait_core::DataType::String,
                dim_type: DimensionType::Categorical(semstrait_manifest::CategoricalDimension {
                    enum_values: None,
                }),
                expr: None,
                expr_source: None,
            },
        );
    }

    let mut measures = IndexMap::new();
    measures.insert(
        "revenue".to_string(),
        CompiledMeasure {
            name: "revenue".to_string(),
            description: None,
            data_type: semstrait_core::DataType::Number,
            agg: semstrait_core::Aggregation::Sum,
            expr: semstrait_core::Expr::entity_ref("amount"),
            expr_source: "amount".to_string(),
            additivity: None,
            constraints,
            filters: vec![],
        },
    );

    let interface = build_interface("orders", dimensions, measures);

    let binding = build_binding(
        "orders_daily",
        vec![
            ("date", "order_date"),
            ("region", "region_name"),
            ("customer", "customer_name"),
            ("user_id", "user_id"),
            ("revenue", "amount"),
        ],
        vec![],
    );

    // Single dataset → CompiledDataKind::Simple (fast path).
    let data_kind = CompiledDataKind::Simple(Box::new(CompiledSimpleKind { interface, binding }));

    let mut entities = IndexMap::new();
    entities.insert("orders".to_string(), data_kind);

    CompiledManifest {
        version: 3,
        compiled_at: chrono::Utc::now(),
        source_hash: "test".to_string(),
        relationships: vec![],
        model_name: "test_model".to_string(),
        model_description: None,
        entities,
        relationship_graph: semstrait_manifest::RelationshipGraph::default(),
        field_index: semstrait_manifest::FieldIndex::default(),
        semantic_graph: semstrait_manifest::SemanticGraph::default(),
        diagnostics: semstrait_manifest::CompileDiagnostics::default(),
        catalog_snapshot: None,
    }
}

/// Create a basic test request.
pub fn make_test_request(
    kind_name: &str,
    dimensions: Vec<&str>,
    measures: Vec<&str>,
) -> ResolvedQueryRequest {
    ResolvedQueryRequest {
        entity_name: kind_name.to_string(),
        dimensions: dimensions.into_iter().map(String::from).collect(),
        measures: measures.into_iter().map(String::from).collect(),
        filters: vec![],
        inline_filters: vec![],
        pending_inline_filters: vec![],
        grain: None,
        limit: None,
        order_by: vec![],
        session_variables: HashMap::new(),
    }
}

/// Create a test manifest with a computed dimension "market" = UPPER(region).
///
/// The kind "orders" has dimensions [date, region, market] and measure [revenue].
/// "market" has `expr: Some(UPPER(Column("region")))` — a computed dimension that
/// should be emitted as a ProjectNode expression, not scanned.
pub fn make_computed_dim_manifest() -> CompiledManifest {
    let mut dimensions = IndexMap::new();
    for name in &["date", "region"] {
        dimensions.insert(
            name.to_string(),
            CompiledDimension {
                name: name.to_string(),
                description: None,
                data_type: semstrait_core::DataType::String,
                dim_type: DimensionType::Categorical(semstrait_manifest::CategoricalDimension {
                    enum_values: None,
                }),
                expr: None,
                expr_source: None,
            },
        );
    }
    // Computed dimension: market = UPPER(region)
    dimensions.insert(
        "market".to_string(),
        CompiledDimension {
            name: "market".to_string(),
            description: None,
            data_type: semstrait_core::DataType::String,
            dim_type: DimensionType::Categorical(semstrait_manifest::CategoricalDimension {
                enum_values: None,
            }),
            expr: Some(semstrait_core::Expr::function_call(
                "UPPER",
                vec![semstrait_core::Expr::entity_ref("region")],
            )),
            expr_source: Some("UPPER(region)".to_string()),
        },
    );

    let mut measures = IndexMap::new();
    measures.insert(
        "revenue".to_string(),
        CompiledMeasure {
            name: "revenue".to_string(),
            description: None,
            data_type: semstrait_core::DataType::Number,
            agg: semstrait_core::Aggregation::Sum,
            expr: semstrait_core::Expr::entity_ref("amount"),
            expr_source: "amount".to_string(),
            additivity: None,
            constraints: None,
            filters: vec![],
        },
    );

    let interface = build_interface("orders", dimensions, measures);

    let binding = build_binding(
        "orders_daily",
        vec![
            ("date", "order_date"),
            ("region", "region_name"),
            ("revenue", "amount"),
        ],
        vec![],
    );

    let data_kind = CompiledDataKind::Simple(Box::new(CompiledSimpleKind { interface, binding }));

    let mut entities = IndexMap::new();
    entities.insert("orders".to_string(), data_kind);

    CompiledManifest {
        version: 3,
        compiled_at: chrono::Utc::now(),
        source_hash: "test".to_string(),
        relationships: vec![],
        model_name: "test_model".to_string(),
        model_description: None,
        entities,
        relationship_graph: semstrait_manifest::RelationshipGraph::default(),
        field_index: semstrait_manifest::FieldIndex::default(),
        semantic_graph: semstrait_manifest::SemanticGraph::default(),
        diagnostics: semstrait_manifest::CompileDiagnostics::default(),
        catalog_snapshot: None,
    }
}
