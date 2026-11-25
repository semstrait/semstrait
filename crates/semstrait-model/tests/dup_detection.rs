mod common;

use common::agg;
use semstrait_model::{
    ComplexExtras, Dataset, Dimension, DimensionType, Grainset, InMemoryFs, LeafExtras, Measure,
    ModelBuildErrorKind, SemanticModel, ValidateErrorKind,
};
use semstrait_sql::datatype::DataType;

#[test]
fn builder_emits_duplicate_data_kind_name_when_two_datasets_share_name() {
    let d1 = Dataset::builder("orders")
        .extras(LeafExtras::default())
        .build();
    let d2 = Dataset::builder("orders")
        .extras(LeafExtras::default())
        .build();

    let err = SemanticModel::builder()
        .name("dup")
        .dataset(d1)
        .dataset(d2)
        .build()
        .unwrap_err();

    let occurrences = err
        .iter()
        .find_map(|d| match &d.kind {
            ModelBuildErrorKind::Validate(ValidateErrorKind::DuplicateDataKindName {
                name,
                occurrences,
            }) if name == "orders" => Some(occurrences.clone()),
            _ => None,
        })
        .expect("duplicate-data-kind diagnostic for `orders`");
    assert_eq!(
        occurrences.len(),
        2,
        "both occurrences must be tracked, got {:?}",
        occurrences
    );
}

#[test]
fn builder_emits_duplicate_data_kind_name_across_dataset_and_grainset() {
    let dataset = Dataset::builder("orders")
        .extras(LeafExtras::default())
        .build();
    let grainset = Grainset::builder("orders")
        .extras(ComplexExtras::default())
        .build();

    let err = SemanticModel::builder()
        .name("dup")
        .dataset(dataset)
        .grainset(grainset)
        .build()
        .unwrap_err();

    assert!(err.iter().any(|d| matches!(
        &d.kind,
        ModelBuildErrorKind::Validate(ValidateErrorKind::DuplicateDataKindName {
            name, ..
        }) if name == "orders"
    )));
}

#[test]
fn builder_emits_duplicate_shared_semantics_name_when_two_dimensions_share_name() {
    let dim1 = Dimension::builder("country")
        .data_type(DataType::String)
        .dim_type(DimensionType::categorical())
        .build();
    let dim2 = Dimension::builder("country")
        .data_type(DataType::String)
        .dim_type(DimensionType::categorical())
        .build();

    let err = SemanticModel::builder()
        .name("dup-dims")
        .dimension(dim1)
        .dimension(dim2)
        .build()
        .unwrap_err();

    let occurrences = err
        .iter()
        .find_map(|d| match &d.kind {
            ModelBuildErrorKind::Validate(ValidateErrorKind::DuplicateSharedSemanticsName {
                carrier,
                name,
                occurrences,
            }) if carrier == "dimensions" && name == "country" => Some(occurrences.clone()),
            _ => None,
        })
        .expect("duplicate-shared-semantics diagnostic for dimensions/`country`");
    assert_eq!(occurrences.len(), 2);
}

#[test]
fn builder_emits_duplicate_shared_semantics_name_when_two_measures_share_name() {
    let m1 = Measure::builder("revenue")
        .data_type(DataType::Decimal {
            precision: 18,
            scale: 2,
        })
        .agg(agg("sum"))
        .build();
    let m2 = Measure::builder("revenue")
        .data_type(DataType::Decimal {
            precision: 18,
            scale: 2,
        })
        .agg(agg("max"))
        .build();

    let err = SemanticModel::builder()
        .name("dup-measures")
        .measure(m1)
        .measure(m2)
        .build()
        .unwrap_err();

    assert!(err.iter().any(|d| matches!(
        &d.kind,
        ModelBuildErrorKind::Validate(ValidateErrorKind::DuplicateSharedSemanticsName {
            carrier, name, ..
        }) if carrier == "measures" && name == "revenue"
    )));
}

#[test]
fn builder_emits_duplicate_entity_id_when_two_entities_share_authored_id() {
    const SHARED: &str = "00000000-0000-7000-8000-00000000d12d";
    let mut north = Dimension::builder("north")
        .data_type(DataType::String)
        .dim_type(DimensionType::categorical())
        .build();
    *north.id_mut() = Some(SHARED.to_owned());
    let mut south = Dimension::builder("south")
        .data_type(DataType::String)
        .dim_type(DimensionType::categorical())
        .build();
    *south.id_mut() = Some(SHARED.to_owned());

    let err = SemanticModel::builder()
        .name("dup-id")
        .dimension(north)
        .dimension(south)
        .build()
        .unwrap_err();

    let occurrences = err
        .iter()
        .find_map(|d| match &d.kind {
            ModelBuildErrorKind::Validate(ValidateErrorKind::DuplicateEntityId {
                id,
                occurrences,
            }) if id == SHARED => Some(occurrences.clone()),
            _ => None,
        })
        .expect("duplicate-entity-id diagnostic for the shared id");
    assert_eq!(
        occurrences.len(),
        2,
        "both inserts must be tracked despite the map overwrite, got {occurrences:?}"
    );
}

#[test]
fn single_file_duplicate_still_carries_original_location() {
    let yaml = r#"
semantic_model:
  name: dup-single
  datasets:
    - name: orders
      extras:
        semantic_mapping: auto
    - name: orders
      extras:
        semantic_mapping: auto
"#;

    let mut fs = InMemoryFs::new();
    fs.insert("model.yaml", yaml);

    let diags = SemanticModel::loader()
        .with_fs(fs)
        .from_yaml_file("model.yaml")
        .build()
        .unwrap_err();

    let occurrences = diags
        .iter()
        .find_map(|d| match &d.kind {
            ModelBuildErrorKind::Validate(ValidateErrorKind::DuplicateDataKindName {
                name,
                occurrences,
            }) if name == "orders" => Some(occurrences.clone()),
            _ => None,
        })
        .expect("duplicate-data-kind diagnostic from single-file parse");
    assert_eq!(occurrences.len(), 2);
    assert!(occurrences
        .iter()
        .all(|loc| loc.source.as_str() == "model.yaml"));
}

#[test]
fn loader_emits_duplicate_when_two_files_share_dataset_name() {
    let yaml_a = r#"
semantic_model:
  name: cross-dup
  datasets:
    - name: orders
      extras:
        semantic_mapping: auto
"#;
    let yaml_b = r#"
semantic_model:
  name: cross-dup
  datasets:
    - name: orders
      extras:
        semantic_mapping: auto
"#;

    let mut fs = InMemoryFs::new();
    fs.insert("a.yaml", yaml_a);
    fs.insert("b.yaml", yaml_b);

    let diags = SemanticModel::loader()
        .with_fs(fs)
        .from_yaml_file("a.yaml")
        .from_yaml_file("b.yaml")
        .build()
        .unwrap_err();

    let occurrences = diags
        .iter()
        .find_map(|d| match &d.kind {
            ModelBuildErrorKind::Validate(ValidateErrorKind::DuplicateDataKindName {
                name,
                occurrences,
            }) if name == "orders" => Some(occurrences.clone()),
            _ => None,
        })
        .expect("duplicate-data-kind diagnostic from cross-file accumulation");
    assert_eq!(occurrences.len(), 2);

    let sources: Vec<&str> = occurrences.iter().map(|loc| loc.source.as_str()).collect();
    assert!(
        sources.contains(&"a.yaml") && sources.contains(&"b.yaml"),
        "locations must point at both source files, got {:?}",
        sources
    );
}
