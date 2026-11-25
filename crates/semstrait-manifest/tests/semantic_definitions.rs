mod common;

use common::golden::{self, Golden};
use common::{agg, compile_diagnostic, dimension, test_provenance};
use std::collections::{BTreeMap, BTreeSet};

use semstrait_common::diagnostic::Diagnostic;
use semstrait_common::io::InMemoryFileSystem;
use semstrait_common::SemanticsName;
use semstrait_ir::{AdditivityType, SemanticRole};
use semstrait_manifest::manifest::semantic_definition::SemanticRoleDefinition;
use semstrait_manifest::{
    compile, CompileError, DimensionType, SemanticDefinition, SemanticManifest, SemanticSignature,
    ShapeAxis,
};
use semstrait_model::expr::{dim, lit};
use semstrait_model::{
    AiContext, BucketSpec, DataKindFilter, Dimension, DimensionSite, Measure, MeasureSite,
    SemanticModel, UnresolvedSemanticExpr,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{ExprOperationExt, Literal};

fn memory_fs() -> InMemoryFileSystem {
    InMemoryFileSystem::new()
}

fn predicate() -> UnresolvedSemanticExpr {
    dim("region")
        .expect("a name")
        .eq(dim("region").expect("a name"))
}

fn definitions_model() -> SemanticModel {
    let mut model = SemanticModel::default();
    model.name = "sales".to_owned();

    let region = Dimension::builder("region")
        .data_type(DataType::String)
        .categorical()
        .description("the sales region")
        .ai_context(
            AiContext::builder()
                .description("ask about regions")
                .build(),
        )
        .build();
    let tier = Dimension::builder("tier")
        .data_type(DataType::String)
        .categorical()
        .build();
    for d in [region, tier] {
        model
            .dimensions
            .insert(d.entity_id().expect("builder mints id").to_owned(), d);
    }
    let revenue = Measure::builder("revenue")
        .data_type(DataType::Integer)
        .agg(agg("sum"))
        .full()
        .build();
    model.measures.insert(
        revenue.entity_id().expect("builder mints id").to_owned(),
        revenue,
    );

    let channel = Dimension::builder("channel")
        .data_type(DataType::String)
        .categorical()
        .build();

    let orders = semstrait_model::Dataset::builder("orders")
        .table("warehouse.orders")
        .dimension(DimensionSite::r#ref("region").with_key())
        .dimension(DimensionSite::inline(channel))
        .measure(MeasureSite::r#ref("revenue"))
        .filter(DataKindFilter::builder("recent").expr(predicate()).build())
        .build();
    model.datasets.insert("orders".to_owned(), orders);
    model
}

async fn compiled(model: SemanticModel) -> SemanticManifest {
    let fs = memory_fs();
    let (manifest, _warnings) = compile(model, None, &fs, test_provenance())
        .await
        .expect("the model must compile");
    manifest
}

async fn refused(model: SemanticModel) -> CompileError {
    compile_diagnostic(model).await.kind
}

fn definition<'m>(manifest: &'m SemanticManifest, name: &str) -> &'m SemanticDefinition {
    manifest
        .semantic_definitions
        .entries
        .values()
        .find(|d| d.name.as_str() == name)
        .unwrap_or_else(|| panic!("no definition named `{name}`"))
}

fn signature<'m>(manifest: &'m SemanticManifest, name: &str) -> Option<&'m SemanticSignature> {
    let id = &definition(manifest, name).id;
    let found: Vec<&SemanticSignature> = manifest
        .semantic_sources
        .values()
        .flat_map(|source| &source.semantic_signature_ids)
        .filter_map(|signature_id| manifest.semantic_signatures.get(signature_id))
        .filter(|signature| &signature.definition_id == id)
        .collect();
    assert!(
        found.len() <= 1,
        "`{name}` has {} signatures; the pair (source, semantics) is unique",
        found.len()
    );
    found.into_iter().next()
}

fn owner<'m>(manifest: &'m SemanticManifest, signature_id: &str) -> &'m str {
    let mut owners = manifest.semantic_sources.values().filter(|source| {
        source
            .semantic_signature_ids
            .iter()
            .any(|id| id == signature_id)
    });
    let found = owners.next().expect("a signature has an owner");
    assert!(
        owners.next().is_none(),
        "exactly one source lists a signature"
    );
    found.id.as_str()
}

#[tokio::test]
async fn a_definition_carries_the_ruled_roster() {
    let manifest = compiled(definitions_model()).await;
    let region = definition(&manifest, "region");

    assert_eq!(region.name, SemanticsName::from("region"));
    assert_eq!(region.data_type, Some(DataType::String));
    assert_eq!(region.role(), SemanticRole::Dimension);
    assert_eq!(region.description.as_deref(), Some("the sales region"));
    let stored = serde_json::to_value(region).expect("serialize SemanticDefinition");
    assert_eq!(
        stored["ai_context"]["description"], "ask about regions",
        "authored AI context rides the definition's stored JSON (ADR 0043): {stored}"
    );
    assert_eq!(
        manifest.semantic_definitions.entries[&region.id].name,
        region.name
    );
}

#[tokio::test]
async fn a_type_written_by_an_alias_or_as_an_object_is_stored_by_its_canonical_name() {
    let yaml = r#"
semantic_model:
  name: sales
  dimensions:
    - name: units
      data_type: long
      type: categorical
    - name: price
      data_type: {NUMERIC: {precision: 18, scale: 4}}
      type: categorical
    - name: tags
      data_type: {array: {element: {varchar: {length: 10}}}}
      type: categorical
  measures:
    - name: revenue
      data_type: FLOAT
      agg: sum
  datasets:
    - name: orders
      dimensions:
        - ref: units
        - ref: price
        - ref: tags
      measures:
        - ref: revenue
      extras:
        storage:
          tables: [warehouse.orders]
"#;
    let (model, _warnings) = semstrait_model::parse(yaml)
        .expect("the model parses")
        .build()
        .expect("the model builds");
    let manifest = compiled(model).await;

    assert_eq!(
        definition(&manifest, "units").data_type,
        Some(DataType::BigInt)
    );
    assert_eq!(
        definition(&manifest, "revenue").data_type,
        Some(DataType::Real)
    );
    assert_eq!(
        definition(&manifest, "price").data_type,
        Some(DataType::Decimal {
            precision: 18,
            scale: 4
        })
    );
    let stored =
        serde_json::to_value(&manifest.semantic_definitions).expect("the definitions serialize");
    let mut names: Vec<&str> = stored["entries"]
        .as_object()
        .expect("the entries are an object")
        .values()
        .filter_map(|entry| entry["data_type"].as_str())
        .collect();
    names.sort_unstable();
    assert_eq!(
        names,
        ["array<varchar(10)>", "bigint", "decimal(18,4)", "real"]
    );
}

#[tokio::test]
async fn a_signature_carries_the_ruled_roster() {
    let manifest = compiled(definitions_model()).await;
    let signature = signature(&manifest, "region").expect("an exposed dimension is realised");

    assert_eq!(signature.definition_id, definition(&manifest, "region").id);
    assert!(
        manifest
            .semantic_sources
            .contains_key(owner(&manifest, &signature.id)),
        "the owning source resolves in `semantic_sources`"
    );
    assert!(
        manifest.expressions.get(&signature.expression_id).is_some(),
        "`expression_id` names a pool entry"
    );
    assert_eq!(manifest.semantic_signatures[&signature.id].id, signature.id);
    assert_eq!(signature.filter_expr_id, None);
}

#[tokio::test]
async fn an_inline_only_interface_site_reaches_the_registry() {
    let manifest = compiled(definitions_model()).await;
    let channel = definition(&manifest, "channel");
    assert_eq!(channel.role(), SemanticRole::Dimension);
    assert_eq!(channel.data_type, Some(DataType::String));
    assert!(signature(&manifest, "channel").is_some());
}

#[tokio::test]
async fn a_named_filter_is_a_definition_with_a_filter_role() {
    let manifest = compiled(definitions_model()).await;
    let recent = definition(&manifest, "recent");
    assert_eq!(recent.role(), SemanticRole::Filter);
    assert!(matches!(
        recent.role_definition,
        SemanticRoleDefinition::Filter
    ));
    assert_eq!(recent.data_type, None);
    assert!(
        signature(&manifest, "recent").is_some(),
        "a named filter's predicate is realised, so the source owns a signature for it"
    );
}

#[tokio::test]
async fn a_realisation_comes_from_the_definitions_own_site() {
    let manifest = compiled(definitions_model()).await;
    let tier = definition(&manifest, "tier");
    assert_eq!(tier.role(), SemanticRole::Dimension);
    assert!(
        signature(&manifest, "tier").is_none(),
        "an unexposed semantics must not be realised against a source it never named"
    );
    for entry in manifest.expressions.physical.values() {
        let rendered = format!("{:?}", entry.expr);
        assert!(
            !rendered.contains("tier"),
            "no expression may mention an unexposed semantics: {rendered}"
        );
    }
}

#[tokio::test]
async fn the_additivity_and_aggregation_families_ride_the_role_body() {
    let manifest = compiled(definitions_model()).await;
    let revenue = definition(&manifest, "revenue");
    assert_eq!(revenue.role(), SemanticRole::Measure);
    let sum = agg("sum");
    assert_eq!(
        revenue.role_definition.aggregation(),
        Some((sum.function(), sum.distinct()))
    );
    assert_eq!(
        revenue.role_definition.additivity(),
        Some(&AdditivityType::Full)
    );

    let region = definition(&manifest, "region");
    assert!(matches!(
        region.role_definition.category(),
        Some(DimensionType::Categorical)
    ));
}

#[tokio::test]
async fn the_median_family_is_accepted_and_lowers() {
    let mut model = definitions_model();
    let p50 = Measure::builder("p50_basket")
        .data_type(DataType::Double)
        .agg(agg("median"))
        .build();
    let id = p50.entity_id().expect("builder mints id").to_owned();
    model.measures.insert(id, p50);
    let orders = model.datasets.values().next().expect("one dataset").clone();
    let orders = expose_measure(orders, "p50_basket");
    model.datasets.clear();
    model.datasets.insert("orders".to_owned(), orders);

    let manifest = compiled(model).await;
    let p50 = definition(&manifest, "p50_basket");
    let median = agg("median");
    assert_eq!(
        p50.role_definition.aggregation(),
        Some((median.function(), median.distinct())),
        "the aggregate rides the role body"
    );
    assert!(
        signature(&manifest, "p50_basket").is_some(),
        "`median` lowers, so it is realised"
    );
}

fn expose_measure(dataset: semstrait_model::Dataset, name: &str) -> semstrait_model::Dataset {
    let mut interface = dataset.semantic_interface.clone();
    interface = interface.with_measure(MeasureSite::r#ref(name));
    let mut out = dataset;
    out.semantic_interface = interface;
    out
}

#[tokio::test]
async fn keyness_reaches_the_artifact_as_the_keyset_attribution_list() {
    let manifest = compiled(definitions_model()).await;

    let region = signature(&manifest, "region").expect("realised");
    assert_eq!(
        region.keysets.len(),
        1,
        "the keyed site attributes exactly its own keyset, got {:?}",
        region.keysets
    );
    let keyset = manifest
        .keysets
        .get(&region.keysets[0])
        .expect("the keyset resolves in the roster");
    assert_eq!(keyset.name.as_str(), "default");

    let channel = signature(&manifest, "channel").expect("realised");
    assert!(
        channel.keysets.is_empty(),
        "an unkeyed site declares no keyset, so the list is empty"
    );
}

#[tokio::test]
async fn exactly_one_signature_per_source_and_semantics() {
    let manifest = compiled(definitions_model()).await;
    let mut seen: BTreeSet<(&str, &str)> = BTreeSet::new();
    for source in manifest.semantic_sources.values() {
        for signature_id in &source.semantic_signature_ids {
            let signature = &manifest.semantic_signatures[signature_id];
            assert!(
                seen.insert((source.id.as_str(), signature.definition_id.as_str())),
                "the pair ({}, {}) appears twice",
                source.id,
                signature.definition_id
            );
        }
    }
    assert_eq!(seen.len(), 4, "got {seen:?}");
}

#[tokio::test]
async fn bit_positions_are_dense_and_in_canonical_id_order() {
    let manifest = compiled(definitions_model()).await;
    let registry = &manifest.semantic_definitions;
    let n = registry.entries.len();
    assert_eq!(n, 5, "region, tier, channel, revenue, recent");

    let occupied: BTreeSet<u32> = registry.entries.values().map(|d| d.bit_position).collect();
    let dense: BTreeSet<u32> = (0..u32::try_from(n).expect("small")).collect();
    assert_eq!(
        occupied, dense,
        "positions are exactly {{0..n-1}} — no holes"
    );

    let in_key_order: Vec<u32> = registry.entries.values().map(|d| d.bit_position).collect();
    let mut ascending = in_key_order.clone();
    ascending.sort_unstable();
    assert_eq!(
        in_key_order, ascending,
        "position order must follow canonical id order"
    );
}

#[tokio::test]
async fn compiling_the_same_model_twice_freezes_the_same_positions() {
    let model = definitions_model();
    let first = compiled(model.clone()).await;
    let second = compiled(model).await;
    let positions = |m: &SemanticManifest| -> BTreeMap<String, u32> {
        m.semantic_definitions
            .entries
            .values()
            .map(|d| (d.name.as_str().to_owned(), d.bit_position))
            .collect()
    };
    assert_eq!(positions(&first), positions(&second));
}

#[tokio::test]
async fn no_reverse_name_to_bit_index_is_persisted() {
    let manifest = compiled(definitions_model()).await;
    let json = serde_json::to_string(&manifest).expect("serialize");
    for forbidden in [
        "semantic_bitmap_index",
        "bit_positions",
        "semantics_by_name",
        "by_name",
        "name_to_bit",
    ] {
        assert!(
            !json.contains(forbidden),
            "`{forbidden}` must not be persisted anywhere in the artifact"
        );
    }
    assert!(json.contains("\"semantic_definitions\""));
    assert!(json.contains("\"semantic_signatures\""));
    assert!(json.contains("\"bit_position\""));
}

#[tokio::test]
async fn a_compiled_manifest_satisfies_its_own_validator() {
    let manifest = compiled(definitions_model()).await;
    match manifest.validate() {
        Ok(warnings) => assert!(warnings.is_empty(), "no advisory faults: {warnings:?}"),
        Err(faults) => panic!("a compiled manifest must satisfy its own validator: {faults:?}"),
    }
}

fn model_with_pooled_dimension(pool: Dimension) -> SemanticModel {
    let mut model = SemanticModel::default();
    model.name = "sales".to_owned();
    model
        .dimensions
        .insert(pool.entity_id().expect("builder mints id").to_owned(), pool);
    model
}

fn dataset_with_dimension_site(model: &mut SemanticModel, site: Dimension) {
    let orders = semstrait_model::Dataset::builder("orders")
        .table("warehouse.orders")
        .dimension(DimensionSite::inline(site))
        .build();
    model.datasets.insert("orders".to_owned(), orders);
}

fn model_with_pooled_measure(pool: Measure) -> SemanticModel {
    let mut model = SemanticModel::default();
    model.name = "sales".to_owned();
    model
        .measures
        .insert(pool.entity_id().expect("builder mints id").to_owned(), pool);
    model
}

fn dataset_with_measure_site(model: &mut SemanticModel, site: Measure) {
    let orders = semstrait_model::Dataset::builder("orders")
        .table("warehouse.orders")
        .measure(MeasureSite::inline(site))
        .build();
    model.datasets.insert("orders".to_owned(), orders);
}

#[tokio::test]
async fn two_occurrences_agreeing_on_every_axis_still_compile() {
    let mut model = model_with_pooled_dimension(
        Dimension::builder("region")
            .data_type(DataType::String)
            .categorical()
            .build(),
    );
    dataset_with_dimension_site(
        &mut model,
        Dimension::builder("region")
            .data_type(DataType::String)
            .categorical()
            .build(),
    );
    let manifest = compiled(model).await;
    assert_eq!(
        definition(&manifest, "region").data_type,
        Some(DataType::String)
    );
}

#[tokio::test]
async fn a_bucketed_dimensions_stored_category_pins_its_buckets() {
    let model = model_with_pooled_dimension(
        Dimension::builder("price_bucket")
            .data_type(DataType::String)
            .bucketed([
                BucketSpec::builder("low")
                    .lower(Literal::BigInt(0))
                    .upper(Literal::BigInt(50))
                    .build(),
                BucketSpec::builder("mid")
                    .lower(Literal::Double(0.5))
                    .upper(Literal::Double(99.5))
                    .build(),
                BucketSpec::builder("high")
                    .lower(Literal::Array {
                        element: DataType::BigInt,
                        values: vec![Literal::BigInt(1), Literal::BigInt(2)],
                    })
                    .upper(Literal::Array {
                        element: DataType::BigInt,
                        values: vec![Literal::BigInt(3), Literal::BigInt(4)],
                    })
                    .build(),
                BucketSpec::builder("wide")
                    .lower(Literal::Array {
                        element: DataType::Decimal {
                            precision: 10,
                            scale: 2,
                        },
                        values: vec![
                            Literal::Decimal {
                                value: "1.5".to_owned(),
                                precision: 10,
                                scale: 2,
                            },
                            Literal::Null(DataType::Decimal {
                                precision: 10,
                                scale: 2,
                            }),
                        ],
                    })
                    .build(),
            ])
            .build(),
    );
    let manifest = compiled(model).await;
    let role_definition = &definition(&manifest, "price_bucket").role_definition;
    let stored = serde_json::to_value(role_definition).expect("serialize Dimension");
    assert_eq!(
        stored,
        serde_json::json!({
            "type": "dimension",
            "content": {
                "category": {
                    "bucketed": {
                        "buckets": [
                            { "name": "low", "lower": { "bigint": 0 }, "upper": { "bigint": 50 } },
                            { "name": "mid", "lower": { "double": 0.5 }, "upper": { "double": 99.5 } },
                            {
                                "name": "high",
                                "lower": { "array": { "element": "bigint", "values": [1, 2] } },
                                "upper": { "array": { "element": "bigint", "values": [3, 4] } }
                            },
                            {
                                "name": "wide",
                                "lower": {
                                    "array": { "element": "decimal(10,2)", "values": ["1.5", null] }
                                }
                            }
                        ]
                    }
                }
            }
        }),
        "the manifest's own `DimensionType::Bucketed` still stores its buckets unchanged: {stored}"
    );
    let read_back: SemanticRoleDefinition =
        serde_json::from_value(stored).expect("deserialize the stored role definition back");
    assert_eq!(
        &read_back, role_definition,
        "the stored JSON reads back to the same buckets, null array elements included"
    );
}

#[tokio::test]
async fn a_non_additive_measures_stored_additivity_pins_non() {
    let mut model = SemanticModel::default();
    model.name = "sales".to_owned();
    let distinct_customers = Measure::builder("distinct_customers")
        .data_type(DataType::Integer)
        .agg(agg("count"))
        .non()
        .build();
    model.measures.insert(
        distinct_customers
            .entity_id()
            .expect("builder mints id")
            .to_owned(),
        distinct_customers,
    );
    let manifest = compiled(model).await;
    assert_eq!(
        definition(&manifest, "distinct_customers")
            .role_definition
            .additivity(),
        Some(&AdditivityType::Non)
    );
    let stored = serde_json::to_value(&definition(&manifest, "distinct_customers").role_definition)
        .expect("serialize Measure");
    assert_eq!(
        stored["content"]["additivity"], "non",
        "the manifest's own `AdditivityType::Non` still serializes `non`: {stored}"
    );
}

#[tokio::test]
async fn a_dimension_occurrence_disagreeing_on_role_is_refused() {
    let mut model = model_with_pooled_dimension(
        Dimension::builder("amount")
            .data_type(DataType::String)
            .categorical()
            .build(),
    );
    dataset_with_measure_site(
        &mut model,
        Measure::builder("amount")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .build(),
    );
    match refused(model).await {
        CompileError::SemanticKindMismatch {
            name,
            authored_kind,
            registered_kind,
        } => {
            assert_eq!(name, "amount");
            assert_ne!(
                authored_kind, registered_kind,
                "both roles are named so the author sees the pair"
            );
        }
        other => panic!("expected SemanticKindMismatch, got {other:?}"),
    }
}

fn model_with_typed_region(
    data_type: DataType,
    body: Option<UnresolvedSemanticExpr>,
) -> SemanticModel {
    let mut model = SemanticModel::default();
    model.name = "sales".to_owned();
    let region = Dimension::builder("region")
        .data_type(data_type)
        .maybe_expr(body)
        .categorical()
        .build();
    dataset_with_dimension_site(&mut model, region);
    model
}

#[tokio::test]
async fn a_built_semantics_whose_declared_type_is_not_valid_is_refused() {
    for body in [None, Some(lit("north"))] {
        let error = refused(model_with_typed_region(
            DataType::VarChar { length: 0 },
            body,
        ))
        .await;
        let message = error.to_string();
        match error {
            CompileError::TypeInferenceFailure { node, .. } => assert_eq!(node, "region"),
            other => panic!("expected TypeInferenceFailure, got {other:?}"),
        }
        assert_eq!(
            message,
            "type inference failed at `region`: its semantics declares data_type `varchar(0)`, \
             which is not a valid data type"
        );
    }
    let manifest = compiled(model_with_typed_region(
        DataType::VarChar { length: 10 },
        None,
    ))
    .await;
    assert_eq!(
        definition(&manifest, "region").data_type,
        Some(DataType::VarChar { length: 10 })
    );
}

#[tokio::test]
async fn a_dimension_occurrence_disagreeing_on_data_type_is_refused() {
    let mut model = model_with_pooled_dimension(
        Dimension::builder("region")
            .data_type(DataType::String)
            .categorical()
            .build(),
    );
    dataset_with_dimension_site(
        &mut model,
        Dimension::builder("region")
            .data_type(DataType::Integer)
            .categorical()
            .build(),
    );
    let error = refused(model).await;
    let message = error.to_string();
    match error {
        CompileError::SemanticShapeMismatch { name, axis, .. } => {
            assert_eq!(name, "region");
            assert_eq!(axis, ShapeAxis::DataType);
        }
        other => panic!("expected SemanticShapeMismatch, got {other:?}"),
    }
    assert_eq!(
        message,
        "shape mismatch for `region` on data type: authored as integer, \
         already registered as string"
    );
}

#[tokio::test]
async fn a_dimension_occurrence_disagreeing_on_category_is_refused() {
    let mut model = model_with_pooled_dimension(
        Dimension::builder("region")
            .data_type(DataType::String)
            .categorical()
            .build(),
    );
    dataset_with_dimension_site(
        &mut model,
        Dimension::builder("region")
            .data_type(DataType::String)
            .temporal([])
            .build(),
    );
    let error = refused(model).await;
    let message = error.to_string();
    match error {
        CompileError::SemanticShapeMismatch { name, axis, .. } => {
            assert_eq!(name, "region");
            assert_eq!(axis, ShapeAxis::Category);
        }
        other => panic!("expected SemanticShapeMismatch, got {other:?}"),
    }
    assert_eq!(
        message,
        r#"shape mismatch for `region` on category: authored as {"temporal":{"granularities":[]}}, already registered as categorical"#
    );
}

#[tokio::test]
async fn a_measure_occurrence_disagreeing_on_aggregation_is_refused() {
    let mut model = model_with_pooled_measure(
        Measure::builder("revenue")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .build(),
    );
    dataset_with_measure_site(
        &mut model,
        Measure::builder("revenue")
            .data_type(DataType::Integer)
            .agg(agg("avg"))
            .build(),
    );
    let error = refused(model).await;
    let message = error.to_string();
    match error {
        CompileError::SemanticShapeMismatch { name, axis, .. } => {
            assert_eq!(name, "revenue");
            assert_eq!(axis, ShapeAxis::Aggregation);
        }
        other => panic!("expected SemanticShapeMismatch, got {other:?}"),
    }
    assert_eq!(
        message,
        "shape mismatch for `revenue` on aggregation: authored as avg, \
         already registered as sum"
    );
}

#[tokio::test]
async fn a_measure_occurrence_disagreeing_on_a_count_distinct_aggregation_is_refused() {
    let mut model = model_with_pooled_measure(
        Measure::builder("revenue")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .build(),
    );
    dataset_with_measure_site(
        &mut model,
        Measure::builder("revenue")
            .data_type(DataType::Integer)
            .agg(agg("count_distinct"))
            .build(),
    );
    let error = refused(model).await;
    let message = error.to_string();
    match error {
        CompileError::SemanticShapeMismatch { name, axis, .. } => {
            assert_eq!(name, "revenue");
            assert_eq!(axis, ShapeAxis::Aggregation);
        }
        other => panic!("expected SemanticShapeMismatch, got {other:?}"),
    }
    assert_eq!(
        message,
        "shape mismatch for `revenue` on aggregation: authored as count_distinct, \
         already registered as sum"
    );
}

#[tokio::test]
async fn a_measure_occurrence_disagreeing_on_additivity_is_refused() {
    let mut model = model_with_pooled_measure(
        Measure::builder("revenue")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .full()
            .build(),
    );
    dataset_with_measure_site(
        &mut model,
        Measure::builder("revenue")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .semi(
                semstrait_model::SemiAdditivity::builder()
                    .axes(vec![])
                    .strategy(semstrait_model::SemiAdditivityStrategy::Latest)
                    .build(),
            )
            .build(),
    );
    let error = refused(model).await;
    let message = error.to_string();
    match error {
        CompileError::SemanticShapeMismatch { name, axis, .. } => {
            assert_eq!(name, "revenue");
            assert_eq!(axis, ShapeAxis::Additivity);
        }
        other => panic!("expected SemanticShapeMismatch, got {other:?}"),
    }
    assert_eq!(
        message,
        r#"shape mismatch for `revenue` on additivity: authored as {"semi":{"axes":[],"strategy":"latest"}}, already registered as full"#
    );
}

#[tokio::test]
async fn two_semi_additive_occurrences_disagreeing_on_axes_are_refused() {
    let mut model = model_with_pooled_measure(
        Measure::builder("revenue")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .semi(
                semstrait_model::SemiAdditivity::builder()
                    .axes(vec![SemanticsName::new("account")])
                    .strategy(semstrait_model::SemiAdditivityStrategy::Latest)
                    .build(),
            )
            .build(),
    );
    dataset_with_measure_site(
        &mut model,
        Measure::builder("revenue")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .semi(
                semstrait_model::SemiAdditivity::builder()
                    .axes(vec![])
                    .strategy(semstrait_model::SemiAdditivityStrategy::Latest)
                    .build(),
            )
            .build(),
    );
    let error = refused(model).await;
    let message = error.to_string();
    match error {
        CompileError::SemanticShapeMismatch { name, axis, .. } => {
            assert_eq!(name, "revenue");
            assert_eq!(axis, ShapeAxis::Additivity);
        }
        other => panic!("expected SemanticShapeMismatch, got {other:?}"),
    }
    assert_eq!(
        message,
        r#"shape mismatch for `revenue` on additivity: authored as {"semi":{"axes":[],"strategy":"latest"}}, already registered as {"semi":{"axes":["account"],"strategy":"latest"}}"#
    );
}

#[tokio::test]
async fn two_semi_additive_occurrences_disagreeing_on_by_are_refused() {
    let mut model = model_with_pooled_measure(
        Measure::builder("revenue")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .semi(
                semstrait_model::SemiAdditivity::builder()
                    .axes(vec![])
                    .by(SemanticsName::new("order_date"))
                    .strategy(semstrait_model::SemiAdditivityStrategy::Latest)
                    .build(),
            )
            .build(),
    );
    dataset_with_measure_site(
        &mut model,
        Measure::builder("revenue")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .semi(
                semstrait_model::SemiAdditivity::builder()
                    .axes(vec![])
                    .by(SemanticsName::new("ship_date"))
                    .strategy(semstrait_model::SemiAdditivityStrategy::Latest)
                    .build(),
            )
            .build(),
    );
    let error = refused(model).await;
    let message = error.to_string();
    match error {
        CompileError::SemanticShapeMismatch { name, axis, .. } => {
            assert_eq!(name, "revenue");
            assert_eq!(axis, ShapeAxis::Additivity);
        }
        other => panic!("expected SemanticShapeMismatch, got {other:?}"),
    }
    assert_eq!(
        message,
        r#"shape mismatch for `revenue` on additivity: authored as {"semi":{"axes":[],"by":"ship_date","strategy":"latest"}}, already registered as {"semi":{"axes":[],"by":"order_date","strategy":"latest"}}"#
    );
}

#[tokio::test]
async fn two_definition_names_equal_without_regard_to_case_are_refused() {
    duplicate_definition_name_refusal().await.assert_committed();
    let diagnostic = compile_diagnostic(two_dimensions("Ärger", "ärger")).await;
    match &diagnostic.kind {
        CompileError::DuplicateDefinitionName { first, second } => {
            assert_eq!([first.as_str(), second.as_str()], ["Ärger", "ärger"]);
        }
        other => panic!("expected DuplicateDefinitionName, got {other:?}"),
    }
}

#[tokio::test]
async fn two_definition_names_that_the_plans_case_fold_keeps_apart_compile() {
    compiled(two_dimensions("ΟΔΟΣ", "οδος")).await;
}

fn two_dimensions(first: &str, second: &str) -> SemanticModel {
    let mut model = SemanticModel::default();
    model.name = "sales".to_owned();
    let orders = semstrait_model::Dataset::builder("orders")
        .table("warehouse.orders")
        .dimension(DimensionSite::inline(dimension(first, DataType::String)))
        .dimension(DimensionSite::inline(dimension(second, DataType::String)))
        .build();
    model.datasets.insert("orders".to_owned(), orders);
    model
}

async fn duplicate_definition_name_refusal() -> Golden {
    let diagnostic = compile_diagnostic(two_dimensions("Region", "region")).await;
    match &diagnostic.kind {
        CompileError::DuplicateDefinitionName { first, second } => {
            assert_eq!([first.as_str(), second.as_str()], ["Region", "region"]);
        }
        other => panic!("expected DuplicateDefinitionName, got {other:?}"),
    }
    Golden::message(
        "two_definition_names_equal_without_regard_to_case_are_refused",
        diagnostic.to_string(),
    )
}

#[tokio::test]
async fn a_definition_name_with_a_reserved_prefix_is_refused() {
    reserved_definition_name_refusal().await.assert_committed();
    for prefix in ["__semstrait_", "__agg_"] {
        reserved_prefix_refusal(prefix).await;
    }
}

async fn reserved_prefix_refusal(prefix: &str) -> Diagnostic<CompileError> {
    let mut model = SemanticModel::default();
    model.name = "sales".to_owned();
    let name = format!("{prefix}filter_revenue");
    dataset_with_dimension_site(&mut model, dimension(&name, DataType::String));
    let diagnostic = compile_diagnostic(model).await;
    match &diagnostic.kind {
        CompileError::ReservedDefinitionName {
            name: got_name,
            prefix: got_prefix,
        } => {
            assert_eq!(got_name, &name);
            assert_eq!(*got_prefix, prefix);
        }
        other => panic!("expected ReservedDefinitionName, got {other:?}"),
    }
    diagnostic
}

async fn reserved_definition_name_refusal() -> Golden {
    let diagnostic = reserved_prefix_refusal("__plan_").await;
    Golden::message(
        "a_definition_name_with_a_reserved_prefix_is_refused",
        diagnostic.to_string(),
    )
}

#[tokio::test]
async fn generate_goldens() {
    golden::generate(async {
        vec![
            duplicate_definition_name_refusal().await,
            reserved_definition_name_refusal().await,
        ]
    })
    .await;
}
