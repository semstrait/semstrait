use std::collections::BTreeSet;

use semstrait_ir::PhysicalExpr;
use semstrait_manifest::{CompileError, SemanticManifest, SemanticSignature};
use semstrait_model::{
    Dataset, Dimension, DimensionSite, Measure, MeasureSite, SemanticMapping, SemanticMappingValue,
    SemanticModel,
};
use semstrait_sql::datatype::{DataType, TypeOrFamily};
use semstrait_sql::expr::{Expr, Literal, TypeError, UntypedLiteral};

mod common;
use common::*;

fn two_leaves_mapping_one_name() -> SemanticModel {
    let mut model = SemanticModel::default();
    model.name = "people".to_owned();
    insert_dimension(
        &mut model,
        Dimension::builder("first_name")
            .data_type(DataType::String)
            .categorical()
            .build(),
    );
    insert_dataset(
        &mut model,
        Dataset::builder("contacts")
            .table("warehouse.contacts")
            .dimension(DimensionSite::r#ref("first_name"))
            .semantic_mapping(
                SemanticMapping::builder()
                    .column("first_name", "contact_name")
                    .build(),
            )
            .build(),
    );
    insert_dataset(
        &mut model,
        Dataset::builder("users")
            .table("warehouse.users")
            .dimension(DimensionSite::r#ref("first_name"))
            .semantic_mapping(
                SemanticMapping::builder()
                    .column("first_name", "first_name")
                    .build(),
            )
            .build(),
    );
    model
}

fn owned<'m>(manifest: &'m SemanticManifest, source_name: &str) -> Vec<&'m SemanticSignature> {
    let entry = manifest
        .semantic_sources
        .values()
        .find(|entry| entry.name.as_str() == source_name)
        .unwrap_or_else(|| panic!("no semantic source named `{source_name}`"));
    entry
        .semantic_signature_ids
        .iter()
        .map(|id| {
            manifest
                .semantic_signatures
                .get(id)
                .unwrap_or_else(|| panic!("`{source_name}` lists an unheld signature `{id}`"))
        })
        .collect()
}

#[tokio::test]
async fn two_leaves_mapping_one_name_own_one_signature_each_over_the_one_definition() {
    let manifest = compile_model(two_leaves_mapping_one_name()).await;

    let named: Vec<_> = manifest
        .semantic_definitions
        .entries
        .values()
        .filter(|definition| definition.name.as_str() == "first_name")
        .collect();
    assert_eq!(named.len(), 1, "one name is one definition, got {named:?}");
    let definition_id = named[0].id.clone();

    let contacts = owned(&manifest, "contacts");
    let users = owned(&manifest, "users");
    assert_eq!(contacts.len(), 1);
    assert_eq!(users.len(), 1);
    assert_eq!(contacts[0].definition_id, definition_id);
    assert_eq!(users[0].definition_id, definition_id);

    assert_ne!(
        contacts[0].id, users[0].id,
        "a signature is per source, so the two ids differ"
    );
    assert_ne!(
        contacts[0].expression_id, users[0].expression_id,
        "the two implementations read different columns"
    );
    assert_eq!(
        tree(&manifest, contacts[0]),
        cast(
            column(&source_alias(&manifest, "contacts"), "contact_name"),
            DataType::String
        )
    );
    assert_eq!(
        tree(&manifest, users[0]),
        cast(
            column(&source_alias(&manifest, "users"), "first_name"),
            DataType::String
        )
    );

    for signature in manifest.semantic_signatures.values() {
        assert!(signature.filter_expr_id.is_none());
    }
}

#[tokio::test]
async fn a_measures_signature_roots_its_family_over_the_mapped_column() {
    let mut model = SemanticModel::default();
    model.name = "sales".to_owned();
    insert_measure(
        &mut model,
        Measure::builder("revenue")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .build(),
    );
    insert_dataset(
        &mut model,
        Dataset::builder("orders")
            .table("warehouse.orders")
            .measure(MeasureSite::r#ref("revenue"))
            .semantic_mapping(
                SemanticMapping::builder()
                    .column("revenue", "amount")
                    .build(),
            )
            .build(),
    );

    let manifest = compile_model(model).await;
    let orders = owned(&manifest, "orders");
    assert_eq!(orders.len(), 1);
    let amount = cast(column("ps0", "amount"), DataType::Integer);
    assert_eq!(
        tree(&manifest, orders[0]),
        cast(sum(amount), DataType::Integer)
    );
}

#[tokio::test]
async fn every_sources_mask_sets_exactly_its_signatures_bits() {
    let manifest = compile_model(two_leaves_mapping_one_name()).await;
    for entry in manifest.semantic_sources.values() {
        let listed: BTreeSet<u32> = owned(&manifest, entry.name.as_str())
            .iter()
            .map(|signature| {
                manifest.semantic_definitions.entries[&signature.definition_id].bit_position
            })
            .collect();
        let set_bits: BTreeSet<u32> = manifest
            .semantic_definitions
            .entries
            .values()
            .filter(|definition| entry.semantic_bitmask.contains(definition.bit_position))
            .map(|definition| definition.bit_position)
            .collect();
        assert_eq!(
            listed,
            set_bits,
            "`{}`'s mask and its signatures disagree",
            entry.name.as_str()
        );
    }
}

#[tokio::test]
async fn signature_identity_is_identical_across_two_compiles() {
    let first = compile_model(two_leaves_mapping_one_name()).await;
    let second = compile_model(two_leaves_mapping_one_name()).await;
    assert_eq!(first.semantic_signatures, second.semantic_signatures);
    for (id, entry) in &first.semantic_sources {
        assert_eq!(
            entry.semantic_signature_ids,
            second.semantic_sources[id].semantic_signature_ids
        );
    }
}

#[tokio::test]
async fn a_compile_time_key_site_reaches_its_signatures_keysets() {
    let mut model = SemanticModel::default();
    model.name = "sales".to_owned();
    insert_dimension(
        &mut model,
        Dimension::builder("region")
            .data_type(DataType::String)
            .categorical()
            .build(),
    );
    insert_dataset(
        &mut model,
        Dataset::builder("orders")
            .table("warehouse.orders")
            .dimension(DimensionSite::r#ref("region").with_key())
            .build(),
    );

    let manifest = compile_model(model).await;
    let orders = owned(&manifest, "orders");
    assert_eq!(orders.len(), 1);
    assert!(
        !orders[0].keysets.is_empty(),
        "a keyed site attributes at least one keyset onto its signature"
    );
    for keyset_id in &orders[0].keysets {
        assert!(
            manifest.keysets.contains_key(keyset_id),
            "`{keyset_id}` must resolve in the roster"
        );
    }
}

#[tokio::test]
async fn a_set_bit_whose_signature_no_source_lists_is_refused() {
    let mut manifest = compile_model(two_leaves_mapping_one_name()).await;
    assert!(
        manifest.validate().is_ok(),
        "precondition: compile is clean"
    );

    let owner = manifest
        .semantic_sources
        .values_mut()
        .find(|entry| !entry.semantic_signature_ids.is_empty())
        .expect("a leaf owns signatures");
    owner
        .semantic_signature_ids
        .pop()
        .expect("the list is non-empty");

    assert!(manifest.validate().is_err());
}

#[tokio::test]
async fn a_signature_two_sources_both_list_is_refused() {
    let mut manifest = compile_model(two_leaves_mapping_one_name()).await;

    let borrowed = owned(&manifest, "contacts")
        .first()
        .expect("`contacts` owns a signature")
        .id
        .clone();
    let thief = manifest
        .semantic_sources
        .values_mut()
        .find(|entry| entry.name.as_str() == "users")
        .expect("the fixture holds a second leaf");
    thief.semantic_signature_ids.push(borrowed);
    thief.semantic_signature_ids.sort();

    assert!(manifest.validate().is_err());
}

fn literal_mapped_dimension(declared: DataType, value: SemanticMappingValue) -> SemanticModel {
    let mut model = SemanticModel::default();
    model.name = "constants".to_owned();
    insert_dimension(
        &mut model,
        Dimension::builder("tag")
            .data_type(declared)
            .categorical()
            .build(),
    );
    insert_dataset(
        &mut model,
        Dataset::builder("rows")
            .table("warehouse.rows")
            .dimension(DimensionSite::r#ref("tag"))
            .semantic_mapping(
                SemanticMapping::builder()
                    .with_semantic("tag", value)
                    .build(),
            )
            .build(),
    );
    model
}

fn long(value: i64) -> PhysicalExpr {
    Expr::Literal(Literal::BigInt(value))
}

fn untyped(literal: Literal) -> SemanticMappingValue {
    SemanticMappingValue::Untyped(UntypedLiteral(literal))
}

fn sum(input: PhysicalExpr) -> PhysicalExpr {
    aggregate("sum", false, input, None)
}

fn type_error(error: CompileError) -> TypeError {
    match error {
        CompileError::TypeError { error, .. } => error,
        other => panic!("expected the TypeError variant, got {other:?}"),
    }
}

#[tokio::test]
async fn a_tagged_string_literal_mapped_to_an_integer_dimension_reaches_the_signature_cast() {
    let model = literal_mapped_dimension(
        DataType::Integer,
        SemanticMappingValue::Literal(Literal::String("abc".to_owned())),
    );
    let manifest = compile_model(model).await;

    let rows = owned(&manifest, "rows");
    assert_eq!(rows.len(), 1);
    assert_eq!(
        tree(&manifest, rows[0]),
        cast(
            Expr::Literal(Literal::String("abc".to_owned())),
            DataType::Integer,
        )
    );
}

#[tokio::test]
async fn an_untyped_string_literal_mapped_to_an_integer_dimension_is_refused() {
    let model = literal_mapped_dimension(
        DataType::Integer,
        untyped(Literal::String("abc".to_owned())),
    );
    assert!(matches!(
        type_error(compile_error(model).await),
        TypeError::UntypedLiteral {
            context: DataType::Integer,
            ..
        }
    ));
}

#[tokio::test]
async fn an_untyped_int_literal_mapped_to_an_integer_dimension_takes_the_integer_type() {
    let model = literal_mapped_dimension(DataType::Integer, untyped(Literal::BigInt(7)));
    let manifest = compile_model(model).await;

    let rows = owned(&manifest, "rows");
    assert_eq!(rows.len(), 1);
    assert_eq!(tree(&manifest, rows[0]), Expr::Literal(Literal::Integer(7)));
}

#[tokio::test]
async fn a_tagged_bigint_literal_mapped_to_an_integer_dimension_is_refused() {
    let model = literal_mapped_dimension(
        DataType::Integer,
        SemanticMappingValue::Literal(Literal::BigInt(7)),
    );
    match compile_error(model).await {
        CompileError::NarrowingLiteral {
            node,
            literal,
            declared,
        } => {
            assert_eq!(node, "tag");
            assert_eq!(literal, Literal::BigInt(7));
            assert_eq!(declared, DataType::Integer);
        }
        other => panic!("expected the narrowing literal's refusal, got {other:?}"),
    }
}

#[tokio::test]
async fn an_int_literal_mapped_to_a_bigint_dimension_stays_a_bare_literal() {
    let model = literal_mapped_dimension(DataType::BigInt, untyped(Literal::BigInt(7)));
    let manifest = compile_model(model).await;

    let rows = owned(&manifest, "rows");
    assert_eq!(rows.len(), 1);
    assert_eq!(tree(&manifest, rows[0]), long(7));
}

fn literal_summed_measure(declared: DataType, value: SemanticMappingValue) -> SemanticModel {
    let mut model = SemanticModel::default();
    model.name = "sales".to_owned();
    insert_measure(
        &mut model,
        Measure::builder("units")
            .data_type(declared)
            .agg(agg("sum"))
            .build(),
    );
    insert_dataset(
        &mut model,
        Dataset::builder("orders")
            .table("warehouse.orders")
            .measure(MeasureSite::r#ref("units"))
            .semantic_mapping(
                SemanticMapping::builder()
                    .with_semantic("units", value)
                    .build(),
            )
            .build(),
    );
    model
}

#[tokio::test]
async fn a_measure_summing_an_int_literal_casts_the_aggregates_type_to_the_declared_type() {
    let model = literal_summed_measure(DataType::Integer, untyped(Literal::BigInt(1)));
    let manifest = compile_model(model).await;

    let orders = owned(&manifest, "orders");
    assert_eq!(orders.len(), 1);
    assert_eq!(
        tree(&manifest, orders[0]),
        cast(sum(long(1)), DataType::Integer),
    );
}

#[tokio::test]
async fn a_measure_summing_an_int_literal_into_a_bigint_stays_a_bare_aggregate() {
    let model = literal_summed_measure(DataType::BigInt, untyped(Literal::BigInt(1)));
    let manifest = compile_model(model).await;

    let orders = owned(&manifest, "orders");
    assert_eq!(orders.len(), 1);
    assert_eq!(tree(&manifest, orders[0]), sum(long(1)));
}

#[tokio::test]
async fn a_measure_summing_a_string_literal_is_refused_at_the_compile_seam() {
    let model = literal_summed_measure(
        DataType::Integer,
        untyped(Literal::String("abc".to_owned())),
    );

    match type_error(compile_error(model).await) {
        TypeError::Argument { name, got, .. } => {
            assert_eq!(name, "sum");
            assert_eq!(got, TypeOrFamily::Type(DataType::String));
        }
        other => panic!("expected sum's argument refused, got {other:?}"),
    }
}

#[tokio::test]
async fn a_mapped_null_decimal_and_timestamp_literal_compiles() {
    let decimal = Literal::Decimal {
        value: "1.5".to_owned(),
        precision: 38,
        scale: 1,
    };
    let timestamp = Literal::Timestamp {
        value: "2026-05-19T10:00:00".to_owned(),
        precision: 6,
    };
    for (declared, value, stored) in [
        (
            DataType::BigInt,
            SemanticMappingValue::Literal(Literal::Null(DataType::BigInt)),
            Literal::Null(DataType::BigInt),
        ),
        (
            DataType::String,
            untyped(Literal::Null(DataType::String)),
            Literal::Null(DataType::String),
        ),
        (
            decimal.data_type(),
            SemanticMappingValue::Literal(decimal.clone()),
            decimal,
        ),
        (
            timestamp.data_type(),
            SemanticMappingValue::Literal(timestamp.clone()),
            timestamp,
        ),
    ] {
        let manifest = compile_model(literal_mapped_dimension(declared, value)).await;
        let rows = owned(&manifest, "rows");
        assert_eq!(rows.len(), 1);
        assert_eq!(tree(&manifest, rows[0]), Expr::Literal(stored));
    }
}

#[tokio::test]
async fn the_corpus_actions_literal_sums_a_bigint_zero_in_either_form() {
    let declared = DataType::Decimal {
        precision: 18,
        scale: 2,
    };
    for value in [
        untyped(Literal::BigInt(0)),
        SemanticMappingValue::Literal(Literal::BigInt(0)),
    ] {
        let manifest = compile_model(literal_summed_measure(declared.clone(), value)).await;
        let orders = owned(&manifest, "orders");
        assert_eq!(orders.len(), 1);
        assert_eq!(
            tree(&manifest, orders[0]),
            cast(sum(long(0)), declared.clone())
        );
    }
}
