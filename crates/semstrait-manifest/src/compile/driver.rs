use std::collections::BTreeMap;

use semstrait_catalog::CatalogProvider;
use semstrait_common::diagnostic::{Diagnostic, Diagnostics};
use semstrait_common::io::{FileSystem, Fingerprintable};
use semstrait_common::EntityId;
use semstrait_model::SemanticModel;

use super::assemble::{condition_handles, seal_unit, ManifestAssembly};
use crate::error::CompileError;
use crate::identity::mint_model_id;
use crate::manifest::bitmap::SemanticBitmap;
use crate::manifest::metadata::SemanticManifestMetadata;
use crate::manifest::physical_source::StorageTarget;
use crate::manifest::semantic_definition::SemanticDefinition;
use crate::manifest::serving::Serving;
use crate::manifest::source_alias::SourceAliases;
use crate::manifest::top::{ManifestProvenance, SemanticManifest};

use super::{
    assemble, catalog, definitions, keysets, relationships, seal, semi_additive, sources, sugar,
    temporal,
};

const MANIFEST_VERSION: u64 = 4;

const SEMSTRAIT_VERSION: &str = env!("CARGO_PKG_VERSION");

#[allow(
    clippy::result_large_err,
    reason = "compile's ratified rich diagnostic return shape is intentional"
)]
pub async fn compile(
    model: SemanticModel,
    catalog: Option<&dyn CatalogProvider>,
    fs: &dyn FileSystem,
    provenance: ManifestProvenance,
) -> Result<
    (SemanticManifest, Diagnostics<CompileError>),
    (Diagnostic<CompileError>, Diagnostics<CompileError>),
> {
    let mut warnings: Diagnostics<CompileError> = Vec::new();

    let model_hash = model.fingerprint();
    let source_uri = source_model_uri(&model);

    let model = sugar::desugar(model);

    let plan = match assemble::plan_semantic_sources(&model) {
        Ok(plan) => plan,
        Err(error) => return Err((Diagnostic::new(error), warnings)),
    };

    let dialect = catalog.and_then(CatalogProvider::dialect);

    let leaves = assemble::leaf_declarations(&plan);
    let roster = match catalog::fetch_physical_sources(&leaves, catalog, fs).await {
        Ok((roster, mut step_warnings)) => {
            warnings.append(&mut step_warnings);
            roster
        }
        Err(error) => return Err((error, warnings)),
    };
    let sources::SourceRoster {
        resolved: resolved_sources,
        by_dataset: declared_sources,
    } = roster;

    let (plan, leaf_reads) = assemble::reduce_multi_source_leaves(plan, &declared_sources);

    let definitions = match definitions::build_semantic_definitions(&model) {
        Ok(definitions) => definitions,
        Err(error) => return Err((Diagnostic::new(error), warnings)),
    };
    let pool_bodies = definitions::pool_bodies(&model);
    let pool_filters = definitions::pool_filters(&model);

    let aliases = SourceAliases::derive(
        plan.iter().map(|unit| (&unit.id, unit.form.source_role())),
        resolved_sources.keys(),
    );

    let keyset_stage = keysets::Stage::new(&plan, &definitions);
    let mut keysets = keysets::KeysetPlan::default();

    let reduction_columns = match assemble::reduction_columns(&plan, &leaf_reads, &resolved_sources)
    {
        Ok(reduction_columns) => reduction_columns,
        Err(error) => return Err((Diagnostic::new(error), warnings)),
    };

    let column_types = match assemble::column_types(
        &plan,
        &definitions,
        &pool_bodies,
        dialect,
        &leaf_reads,
        &resolved_sources,
        &reduction_columns,
    ) {
        Ok(column_types) => column_types,
        Err(error) => return Err((Diagnostic::new(error), warnings)),
    };

    let mut assembly = ManifestAssembly::new();
    for unit in assemble::seal_order(&plan) {
        match seal_unit(
            unit,
            assemble::Vocabulary {
                definitions: &definitions,
                pool_bodies: &pool_bodies,
                pool_filters: &pool_filters,
                columns: &column_types,
                reduction_columns: &reduction_columns,
            },
            &keysets,
            &leaf_reads,
            &resolved_sources,
            &aliases,
            &mut assembly,
        ) {
            Ok(sealed) => {
                if let Err(error) = assembly.push_source(unit, sealed, &aliases) {
                    return Err((Diagnostic::new(error), warnings));
                }
                temporal::derive_sealed_shape(&mut assembly.semantic_sources, &unit.id);
            }
            Err(error) => {
                let mut diagnostic = Diagnostic::new(error);
                if let Ok((source, _)) = assemble::leaf_source(unit, &leaf_reads, &resolved_sources)
                {
                    if let (StorageTarget::Path { .. }, Some(format)) =
                        (&source.storage.target, source.storage.format)
                    {
                        diagnostic = diagnostic.with_note(format!(
                            "`{}` reads `{}` as {format}",
                            unit.name,
                            source.locator()
                        ));
                    }
                }
                return Err((diagnostic, warnings));
            }
        }
        let served =
            Serving::new(&assembly.semantic_sources, &assembly.signatures).served(&unit.id);
        if let Err(error) = keyset_stage.record(unit, &served, &mut keysets) {
            return Err((Diagnostic::new(error), warnings));
        }
    }
    if let Err(error) = assembly.check_column_qualifiers(&aliases) {
        return Err((Diagnostic::new(error), warnings));
    }

    if let Err(error) = temporal::derive_composed_shapes(&mut assembly.semantic_sources) {
        return Err((Diagnostic::new(error), warnings));
    }

    if let Err(error) = semi_additive::check_semi_additive_sources(
        &model,
        &plan,
        &definitions,
        &assembly.semantic_sources,
        &assembly.signatures,
        &assembly.expressions,
    ) {
        return Err((Diagnostic::new(error), warnings));
    }

    keysets::attribute_signatures(&keysets, &mut assembly.signatures);
    keysets::attribute_grainsets(&keysets, &mut assembly.semantic_sources);

    let sources = match sources::complete_sources(
        &resolved_sources,
        &aliases,
        &assembly.signatures,
        &assembly.expressions,
        &column_types,
        dialect,
    ) {
        Ok(sources) => sources,
        Err(error) => return Err((Diagnostic::new(error), warnings)),
    };

    if let Some(error) = relationships::unserved_condition_leaves(
        &assembly.semantic_sources,
        &assembly.signatures,
        &definitions,
        &assembly.expressions,
    )
    .into_iter()
    .next()
    {
        return Err((Diagnostic::new(error), warnings));
    }

    let referenced = assembly
        .signatures
        .values()
        .flat_map(|signature| {
            std::iter::once(signature.expression_id).chain(signature.filter_expr_id)
        })
        .chain(
            assembly
                .semantic_sources
                .values()
                .flat_map(condition_handles),
        );
    if let Some(error) = seal::sweep_orphan_expressions(&assembly.expressions, referenced) {
        return Err((Diagnostic::new(error), warnings));
    }

    match assemble_manifest(
        model_hash,
        source_uri,
        provenance,
        sources,
        definitions,
        keysets,
        assembly,
    ) {
        Ok(manifest) => Ok((manifest, warnings)),
        Err(error) => Err((Diagnostic::new(error), warnings)),
    }
}

fn source_model_uri(model: &SemanticModel) -> Option<String> {
    if model.name.is_empty() {
        None
    } else {
        Some(model.name.clone())
    }
}

fn assemble_manifest(
    model_hash: [u8; 32],
    source_uri: Option<String>,
    provenance: ManifestProvenance,
    sources: BTreeMap<EntityId, crate::manifest::physical_source::PhysicalSource>,
    definitions: Vec<SemanticDefinition>,
    keysets: keysets::KeysetPlan,
    assembly: ManifestAssembly,
) -> Result<SemanticManifest, CompileError> {
    let model_id = mint_model_id(&model_hash);

    let metadata = SemanticManifestMetadata::new(SEMSTRAIT_VERSION, source_uri, BTreeMap::new());

    let mut manifest =
        SemanticManifest::new(MANIFEST_VERSION, provenance, model_hash, model_id, metadata);

    let registry = SemanticBitmap::freeze(definitions)?;
    let mut semantic_sources = assembly.semantic_sources;
    assemble::complete_semantic_sources(&mut semantic_sources, &registry, &assembly.signatures)?;
    if let Some(error) = temporal::distinct_grains(&semantic_sources, &registry)
        .into_iter()
        .next()
    {
        return Err(error);
    }

    manifest.semantic_definitions = registry;
    manifest.semantic_signatures = assembly.signatures;
    manifest.semantic_sources = semantic_sources;
    manifest.sources = sources;
    manifest.expressions = assembly.expressions;
    manifest.keysets = keysets.roster;
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::error::ShapeAxis;
    use crate::manifest::expression_layer::ExprLayer;
    use crate::manifest::SemanticSignature;
    use std::time::{Duration, UNIX_EPOCH};

    fn without_provenance(mut manifest: SemanticManifest) -> SemanticManifest {
        manifest.created_at = UNIX_EPOCH;
        manifest.updated_at = UNIX_EPOCH;
        manifest.metadata = SemanticManifestMetadata::new("", None, BTreeMap::new());
        manifest
    }

    fn test_provenance() -> ManifestProvenance {
        ManifestProvenance::new(
            UNIX_EPOCH + Duration::from_secs(1_700_000_000),
            UNIX_EPOCH + Duration::from_secs(1_800_000_000),
        )
    }

    use semstrait_common::diagnostic::Severity;
    use semstrait_common::io::InMemoryFileSystem;
    use semstrait_model::expr::measure as measure_body;
    use semstrait_model::{Agg, Dimension, Measure};
    use semstrait_sql::datatype::DataType;
    use semstrait_sql::expr::{Call, CastFailure, Expr, NamedReference};

    fn agg(name: &str) -> Agg {
        Agg::of_name(name).expect("a catalog aggregate")
    }

    fn aggregate_call(expr: &crate::manifest::StoredExpr) -> Option<(&'static str, bool)> {
        match expr.expr() {
            Expr::Function(call @ Call::Aggregate(aggregate, _)) => {
                Some((call.name(), aggregate.distinct()))
            }
            _ => None,
        }
    }

    fn happy_model() -> SemanticModel {
        let mut model = SemanticModel::default();
        model.name = "sales".to_owned();

        let region = Dimension::builder("region")
            .data_type(DataType::String)
            .categorical()
            .build();
        let revenue = Measure::builder("revenue")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .build();
        model
            .dimensions
            .insert(region.name().as_str().to_owned(), region);
        model
            .measures
            .insert(revenue.name().as_str().to_owned(), revenue);

        let orders = semstrait_model::Dataset::builder("orders")
            .table("warehouse.orders")
            .dimension(semstrait_model::DimensionSite::r#ref("region"))
            .measure(semstrait_model::MeasureSite::r#ref("revenue"))
            .build();
        model.datasets.insert("orders".to_owned(), orders);
        model
    }

    fn realisation_of(manifest: &SemanticManifest, name: &str) -> SemanticSignature {
        let definition = manifest
            .semantic_definitions
            .by_name(name)
            .unwrap_or_else(|| panic!("no definition named `{name}`"));
        manifest
            .semantic_sources
            .values()
            .flat_map(|source| &source.semantic_signature_ids)
            .filter_map(|id| manifest.semantic_signatures.get(id))
            .find(|signature| signature.definition_id == definition.id)
            .cloned()
            .unwrap_or_else(|| panic!("`{name}` is realised nowhere"))
    }

    fn realised_expr_id(
        manifest: &SemanticManifest,
        name: &str,
    ) -> crate::manifest::expression::PhysicalExprId {
        realisation_of(manifest, name).expression_id
    }

    fn memory_fs() -> InMemoryFileSystem {
        InMemoryFileSystem::new()
    }

    #[tokio::test]
    async fn happy_path_one_dataset_compiles() {
        let fs = memory_fs();
        let (manifest, warnings) = compile(happy_model(), None, &fs, test_provenance())
            .await
            .expect("happy-path model must compile");

        assert_eq!(manifest.semantic_definitions.len(), 2);
        assert_eq!(manifest.semantic_signatures.len(), 2);
        assert_eq!(manifest.semantic_sources.len(), 1);
        assert_eq!(manifest.sources.len(), 1);
        assert_eq!(manifest.expressions.len(), 2);
        assert!(
            warnings.is_empty(),
            "expected no warnings, got {warnings:?}"
        );

        assert!(
            manifest.validate().is_ok(),
            "a compiled manifest must satisfy its own validator"
        );

        let column = || {
            Expr::Column(NamedReference::qualified("ps0", "region").expect("an alias and a name"))
        };
        let cast = |input, target| Expr::Cast {
            input: Box::new(input),
            target,
            on_failure: CastFailure::Error,
        };
        let signatures: Vec<_> = manifest.semantic_signatures.values().collect();
        let region_expr_id = realised_expr_id(&manifest, "region");
        assert_eq!(
            manifest
                .expressions
                .reassemble(&region_expr_id, &signatures, &[]),
            Some(cast(column(), DataType::String))
        );
        assert_eq!(
            manifest.expressions.physical[&region_expr_id].layer,
            ExprLayer::Scalar
        );
        let revenue_expr_id = realised_expr_id(&manifest, "revenue");
        let Some(Expr::Cast { input, .. }) =
            manifest
                .expressions
                .reassemble(&revenue_expr_id, &signatures, &[])
        else {
            panic!("revenue's root is the cast to its declared type");
        };
        assert_eq!(
            manifest.expressions.physical[&revenue_expr_id].layer,
            ExprLayer::PostAggregate
        );
        let Expr::Function(call @ Call::Aggregate(..)) = *input else {
            panic!("the cast's input is revenue's aggregate call");
        };
        assert_eq!(call.name(), "sum");
    }

    fn single_measure_model(measure: Measure) -> SemanticModel {
        let mut model = SemanticModel::default();
        model.name = "metrics".to_owned();
        let name = measure.name().as_str().to_owned();
        model.measures.insert(name.clone(), measure);
        let events = semstrait_model::Dataset::builder("events")
            .table("warehouse.events")
            .measure(semstrait_model::MeasureSite::r#ref(name))
            .build();
        model.datasets.insert("events".to_owned(), events);
        model
    }

    fn pooled_measure_aggregate(
        manifest: &SemanticManifest,
    ) -> &crate::manifest::expression::ManifestExpression {
        let id = realised_expr_id(manifest, "m");
        manifest
            .expressions
            .get(&id)
            .expect("pooled aggregate present")
    }

    #[tokio::test]
    async fn body_free_count_distinct_lowers_to_count_with_distinct() {
        let fs = memory_fs();
        let model = single_measure_model(
            Measure::builder("m")
                .data_type(DataType::BigInt)
                .agg(agg("count_distinct"))
                .build(),
        );
        let (manifest, warnings) = compile(model, None, &fs, test_provenance())
            .await
            .expect("count_distinct measure compiles");
        assert!(
            warnings.is_empty(),
            "no warnings expected, got {warnings:?}"
        );
        let pooled = pooled_measure_aggregate(&manifest);
        assert_eq!(pooled.layer, ExprLayer::Aggregate);
        assert_eq!(
            aggregate_call(&pooled.expr),
            Some(("count", true)),
            "CountDistinct must be count with DISTINCT, got {:?}",
            pooled.expr
        );
        assert!(manifest.validate().is_ok());
    }

    #[tokio::test]
    async fn body_free_median_measure_lowers_to_the_registrys_median() {
        let fs = memory_fs();
        let model = single_measure_model(
            Measure::builder("m")
                .data_type(DataType::Double)
                .agg(agg("median"))
                .build(),
        );
        let (manifest, warnings) = compile(model, None, &fs, test_provenance())
            .await
            .expect("a Median measure lowers to the registry's `median`");
        assert!(warnings.is_empty(), "no warnings expected: {warnings:?}");

        let definition = manifest
            .semantic_definitions
            .by_name("m")
            .expect("the Measure is registered");
        let median = agg("median");
        assert_eq!(
            definition.role_definition.aggregation(),
            Some((median.function(), median.distinct()))
        );

        let pooled = pooled_measure_aggregate(&manifest);
        assert_eq!(pooled.layer, ExprLayer::Aggregate);
        assert_eq!(aggregate_call(&pooled.expr), Some(("median", false)));
    }

    #[tokio::test]
    async fn a_self_referential_measure_is_refused_as_cyclic() {
        let mut cyclic_model = happy_model();
        let revenue = Measure::builder("revenue")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .expr(measure_body("revenue").expect("a name"))
            .build();
        cyclic_model
            .measures
            .insert(revenue.name().as_str().to_owned(), revenue);
        let fs = memory_fs();
        let (error, _) = compile(cyclic_model, None, &fs, test_provenance())
            .await
            .expect_err("a self-referential Measure must be refused as cyclic");
        match &error.kind {
            CompileError::CyclicReference { cycle } => {
                assert!(cycle.contains(&"revenue".to_owned()), "names it: {cycle:?}");
            }
            other => panic!("expected CyclicReference, got {other:?}"),
        }
        assert_eq!(error.severity, Severity::Error);
    }

    #[tokio::test]
    async fn unknown_reference_in_body() {
        let mut model = happy_model();
        let bucket = Dimension::builder("tier")
            .data_type(DataType::String)
            .categorical()
            .expr(measure_body("nonexistent").expect("a name"))
            .build();
        model
            .dimensions
            .insert(bucket.name().as_str().to_owned(), bucket);
        let orders = semstrait_model::Dataset::builder("orders")
            .table("warehouse.orders")
            .dimension(semstrait_model::DimensionSite::r#ref("region"))
            .dimension(semstrait_model::DimensionSite::r#ref("tier"))
            .measure(semstrait_model::MeasureSite::r#ref("revenue"))
            .build();
        model.datasets.insert("orders".to_owned(), orders);

        let fs = memory_fs();
        let (error, _) = compile(model, None, &fs, test_provenance())
            .await
            .expect_err("a reference to no definition is refused");
        match &error.kind {
            CompileError::UnknownReference { scope, .. } => {
                assert_eq!(
                    *scope,
                    crate::error::ReferenceScope::DataKind(
                        crate::identity::mint_semantic_source_id("orders")
                    ),
                    "the scope carries the source's EntityId, matching the id the two\
                     lowering paths (the reference graph and the substitution walk) both use"
                );
            }
            other => panic!("expected UnknownReference, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn determinism_same_model_twice() {
        let model = happy_model();
        let fs = memory_fs();
        let (first, _) = compile(model.clone(), None, &fs, test_provenance())
            .await
            .expect("first compile");
        let (second, _) = compile(model, None, &fs, test_provenance())
            .await
            .expect("second compile");
        assert!(first.validate().is_ok(), "first compile is valid");
        assert!(second.validate().is_ok(), "second compile is valid");
        let first = without_provenance(first);
        let second = without_provenance(second);
        assert_eq!(
            first, second,
            "identical inputs must yield equal load-bearing state"
        );
        let a = serde_json::to_vec(&first).expect("serialize first");
        let b = serde_json::to_vec(&second).expect("serialize second");
        assert_eq!(a, b, "identical inputs must serialize to identical bytes");
    }

    #[test]
    fn model_hash_excludes_root_id_but_not_content() {
        let base = happy_model();
        let base_hash = base.fingerprint();

        let mut id = base.clone();
        id.id = Some("0197cb6e-a63a-7d53-8d6f-35f4b1d67d4e".to_owned());
        assert_eq!(
            id.fingerprint(),
            base_hash,
            "`id` is the documented exclusion; it must not perturb model_hash"
        );

        let mut name = base.clone();
        name.name = format!("{}-renamed", base.name);
        assert_ne!(name.fingerprint(), base_hash, "`name` is content");

        let mut description = base.clone();
        description.description = Some("annotated".to_owned());
        assert_ne!(
            description.fingerprint(),
            base_hash,
            "`description` is content"
        );

        let mut ai_context = base.clone();
        ai_context.ai_context = Some(
            semstrait_model::AiContext::builder()
                .description("ask about revenue")
                .build(),
        );
        assert_ne!(
            ai_context.fingerprint(),
            base_hash,
            "`ai_context` is content"
        );

        let mut labels = base.clone();
        labels.labels.push("tier-1".to_owned());
        assert_ne!(labels.fingerprint(), base_hash, "`labels` is content");

        let mut datasets = base.clone();
        datasets.datasets.insert(
            "extra-dataset-id".to_owned(),
            semstrait_model::Dataset::builder("returns")
                .table("warehouse.returns")
                .build(),
        );
        assert_ne!(datasets.fingerprint(), base_hash, "`datasets` is content");

        let mut grainsets = base.clone();
        grainsets.grainsets.insert(
            "extra-grainset-id".to_owned(),
            semstrait_model::Grainset::builder("daily").build(),
        );
        assert_ne!(grainsets.fingerprint(), base_hash, "`grainsets` is content");

        let mut unionsets = base.clone();
        unionsets.unionsets.insert(
            "extra-unionset-id".to_owned(),
            semstrait_model::Unionset::builder("all_orders").build(),
        );
        assert_ne!(unionsets.fingerprint(), base_hash, "`unionsets` is content");

        let mut joinsets = base.clone();
        joinsets.joinsets.insert(
            "extra-joinset-id".to_owned(),
            semstrait_model::Joinset::builder("orders_joined").build(),
        );
        assert_ne!(joinsets.fingerprint(), base_hash, "`joinsets` is content");

        let mut dimensions = base.clone();
        dimensions.dimensions.insert(
            "extra-dimension-id".to_owned(),
            Dimension::builder("channel")
                .data_type(DataType::String)
                .categorical()
                .build(),
        );
        assert_ne!(
            dimensions.fingerprint(),
            base_hash,
            "`dimensions` is content"
        );

        let mut measures = base.clone();
        measures.measures.insert(
            "extra-measure-id".to_owned(),
            Measure::builder("cost")
                .data_type(DataType::Integer)
                .agg(agg("sum"))
                .build(),
        );
        assert_ne!(measures.fingerprint(), base_hash, "`measures` is content");

        let mut metrics = base.clone();
        metrics.metrics.insert(
            "extra-metric-id".to_owned(),
            semstrait_model::Metric::builder("margin")
                .data_type(DataType::Double)
                .build(),
        );
        assert_ne!(metrics.fingerprint(), base_hash, "`metrics` is content");

        let mut relationships = base.clone();
        relationships.relationships.insert(
            "extra-relationship-id".to_owned(),
            semstrait_model::Relationship::builder()
                .name("orders_self")
                .from("orders")
                .to("orders")
                .on_key("id")
                .cardinality(semstrait_model::Cardinality::ManyToOne)
                .build()
                .expect("relationship build"),
        );
        assert_ne!(
            relationships.fingerprint(),
            base_hash,
            "`relationships` is content"
        );
    }

    #[tokio::test]
    async fn storageless_dataset_fails_fast_not_dangling_ok() {
        let mut model = happy_model();
        model.datasets.clear();
        let orders = semstrait_model::Dataset::builder("orders").build();
        model.datasets.insert("orders".to_owned(), orders);

        let fs = memory_fs();
        let (error, _warnings) = compile(model, None, &fs, test_provenance())
            .await
            .expect_err("a storage-less Dataset must fail fast, not compile to a dangling Ok");
        match error.kind {
            CompileError::DatasetWithoutSource { dataset_id } => {
                assert_eq!(dataset_id, "orders", "the offending Dataset is named");
            }
            other => panic!("expected DatasetWithoutSource, got {other:?}"),
        }
        assert_eq!(error.severity, Severity::Error);
    }

    #[tokio::test]
    async fn an_explicit_mapping_naming_an_unexposed_semantics_is_rejected() {
        use semstrait_model::SemanticMapping;
        let mut model = happy_model();
        model.datasets.clear();
        let mapping = SemanticMapping::builder()
            .column("region", "region_col")
            .build();
        assert!(!mapping.is_auto(), "precondition: the mapping is Explicit");
        let orders = semstrait_model::Dataset::builder("orders")
            .table("warehouse.orders")
            .semantic_mapping(mapping)
            .build();
        model.datasets.insert("orders".to_owned(), orders);

        let fs = memory_fs();
        let (error, _warnings) = compile(model, None, &fs, test_provenance())
            .await
            .expect_err("a mapping entry for an unexposed semantics must fail fast");
        match &error.kind {
            CompileError::TypeInferenceFailure { node, reason } => {
                assert!(node.contains("orders"), "names the Dataset: {node}");
                assert!(
                    reason.contains("does not expose"),
                    "explains the refusal: {reason}"
                );
            }
            other => panic!("expected TypeInferenceFailure, got {other:?}"),
        }
        assert_eq!(error.severity, Severity::Error);
    }

    #[tokio::test]
    async fn an_exposed_semantics_left_unmapped_by_an_explicit_block_is_rejected() {
        use semstrait_model::SemanticMapping;
        let mut model = happy_model();
        model.datasets.clear();
        let mapping = SemanticMapping::builder()
            .column("region", "region_col")
            .build();
        let orders = semstrait_model::Dataset::builder("orders")
            .table("warehouse.orders")
            .dimension(semstrait_model::DimensionSite::r#ref("region"))
            .measure(semstrait_model::MeasureSite::r#ref("revenue"))
            .semantic_mapping(mapping)
            .build();
        model.datasets.insert("orders".to_owned(), orders);

        let fs = memory_fs();
        let (error, _warnings) = compile(model, None, &fs, test_provenance())
            .await
            .expect_err("an unmapped exposed semantics must fail fast");
        match &error.kind {
            CompileError::TypeInferenceFailure { node, reason } => {
                assert!(node.contains("orders"), "names the Dataset: {node}");
                assert!(
                    reason.contains("`revenue`"),
                    "names the unmapped semantics: {reason}"
                );
            }
            other => panic!("expected TypeInferenceFailure, got {other:?}"),
        }
        assert_eq!(error.severity, Severity::Error);
    }

    #[tokio::test]
    async fn a_joinset_composing_nothing_is_not_a_deferred_form() {
        let fs = memory_fs();
        let mut model = happy_model();
        model.joinsets.insert(
            "joined".to_owned(),
            semstrait_model::Joinset::builder("joined").build(),
        );
        let (error, _) = compile(model, None, &fs, test_provenance())
            .await
            .expect_err("a Joinset with no composition must be refused");
        match &error.kind {
            CompileError::RelationshipResolution {
                relationship,
                reason,
            } => {
                assert_eq!(relationship, "joined", "the Joinset is named");
                assert!(
                    reason.contains("composes nothing"),
                    "the reason is the missing composition: {reason}"
                );
            }
            other => panic!("expected RelationshipResolution, got {other:?}"),
        }
        assert_eq!(error.severity, Severity::Error);
    }

    #[tokio::test]
    async fn one_name_claimed_by_two_carriers_is_refused() {
        let mut model = happy_model();
        let amount_dim = Dimension::builder("amount")
            .data_type(DataType::String)
            .categorical()
            .build();
        let amount_measure = Measure::builder("amount")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .build();
        model
            .dimensions
            .insert(amount_dim.name().as_str().to_owned(), amount_dim);
        model
            .measures
            .insert(amount_measure.name().as_str().to_owned(), amount_measure);

        let fs = memory_fs();
        let (error, _warnings) = compile(model, None, &fs, test_provenance())
            .await
            .expect_err("one name held by two carriers must fail fast");
        match &error.kind {
            CompileError::SemanticKindMismatch {
                authored_kind,
                registered_kind,
                name,
            } => {
                assert_eq!(name, "amount", "names the contested name");
                assert_ne!(
                    authored_kind, registered_kind,
                    "both roles are named so the author sees the pair"
                );
            }
            other => panic!("expected SemanticKindMismatch, got {other:?}"),
        }
        assert_eq!(error.severity, Severity::Error);
    }

    #[tokio::test]
    async fn a_dataset_and_a_semantics_may_share_a_name() {
        let mut model = happy_model();
        let orders_dim = Dimension::builder("orders")
            .data_type(DataType::String)
            .categorical()
            .build();
        model
            .dimensions
            .insert(orders_dim.name().as_str().to_owned(), orders_dim);

        let fs = memory_fs();
        let (manifest, _warnings) = compile(model, None, &fs, test_provenance())
            .await
            .expect("a Dataset and a semantics in different id lanes must compile");

        let dimension = manifest
            .semantic_definitions
            .by_name("orders")
            .expect("the Dimension is registered");
        let source = manifest
            .semantic_sources
            .values()
            .find(|entry| entry.name.as_str() == "orders")
            .expect("the Dataset keeps its own id lane");
        assert_ne!(
            dimension.id, source.id,
            "a definition and a semantic source of one name are two entities"
        );
        assert_ne!(
            dimension.id, "orders",
            "a definition's id is minted, never the raw name"
        );
        assert_ne!(
            source.id, "orders",
            "a semantic source's id is minted from its name, never the raw name"
        );
        assert!(manifest.validate().is_ok());
    }

    #[tokio::test]
    async fn a_shape_disagreeing_occurrence_of_one_name_is_refused() {
        let mut model = happy_model();
        let region_site = Dimension::builder("region")
            .data_type(DataType::Integer)
            .categorical()
            .build();
        let orders = semstrait_model::Dataset::builder("orders")
            .table("warehouse.orders")
            .dimension(semstrait_model::DimensionSite::inline(region_site))
            .measure(semstrait_model::MeasureSite::r#ref("revenue"))
            .build();
        model.datasets.insert("orders".to_owned(), orders);

        let fs = memory_fs();
        let (error, _warnings) = compile(model, None, &fs, test_provenance())
            .await
            .expect_err("a shape-disagreeing occurrence must fail fast");
        match &error.kind {
            CompileError::SemanticShapeMismatch { name, axis, .. } => {
                assert_eq!(name, "region", "names the contested name");
                assert_eq!(*axis, ShapeAxis::DataType, "names the disagreeing axis");
            }
            other => panic!("expected SemanticShapeMismatch, got {other:?}"),
        }
        assert_eq!(error.severity, Severity::Error);
    }
}
