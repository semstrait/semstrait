mod common;

use common::{agg, by_name, fixture_fs, identity_mapping, lake_folder, seeded_fs, test_provenance};

use semstrait_common::Granularity;
use semstrait_ir::{ScdType as IrScdType, TemporalType};
use semstrait_manifest::{compile, CompileError, RoleDefinition, SemanticManifest, SemanticSource};
use semstrait_model::{
    Dataset, Dimension, DimensionSite, Joinset, Measure, MeasureSite, NestedDataset, Relationship,
    ScdType, SemanticModel, StorageFormat, TemporalShape, Unionset,
};
use semstrait_sql::datatype::DataType;

const NAMES: [&str; 6] = [
    "region",
    "order_ts",
    "ship_ts",
    "valid_from",
    "valid_to",
    "revenue",
];

fn leaf(name: &str, shape: Option<TemporalShape>) -> NestedDataset {
    let mut built = NestedDataset::builder(name)
        .path(lake_folder(name))
        .format(StorageFormat::Parquet)
        .semantic_mapping(identity_mapping(&NAMES));
    if let Some(shape) = shape {
        built = built.temporal(shape);
    }
    built.build()
}

macro_rules! interface {
    ($built:expr) => {
        $built
            .dimension(DimensionSite::r#ref("region"))
            .dimension(DimensionSite::r#ref("order_ts"))
            .dimension(DimensionSite::r#ref("ship_ts"))
            .dimension(DimensionSite::r#ref("valid_from"))
            .dimension(DimensionSite::r#ref("valid_to"))
            .measure(MeasureSite::r#ref("revenue"))
    };
}

fn unionset(name: &str, branches: Vec<NestedDataset>, own: Option<TemporalShape>) -> Unionset {
    let mut built = interface!(Unionset::builder(name)).union_all();
    for branch in branches {
        built = built.dataset(branch);
    }
    if let Some(shape) = own {
        built = built.temporal(shape);
    }
    built.build()
}

fn joinset(
    name: &str,
    constituents: Vec<NestedDataset>,
    relationships: Vec<Relationship>,
) -> Joinset {
    joinset_with(name, constituents, relationships, None)
}

fn joinset_stating(
    name: &str,
    constituents: Vec<NestedDataset>,
    relationships: Vec<Relationship>,
    stated: TemporalShape,
) -> Joinset {
    joinset_with(name, constituents, relationships, Some(stated))
}

fn joinset_with(
    name: &str,
    constituents: Vec<NestedDataset>,
    relationships: Vec<Relationship>,
    stated: Option<TemporalShape>,
) -> Joinset {
    let mut built = interface!(Joinset::builder(name));
    for constituent in constituents {
        built = built.dataset(constituent);
    }
    for relationship in relationships {
        built = built.relationship(relationship);
    }
    if let Some(shape) = stated {
        built = built.temporal(shape);
    }
    built.build()
}

fn related(name: &str, from: &str, to: &str) -> Relationship {
    Relationship::builder()
        .name(name)
        .from(from)
        .to(to)
        .on(["region"])
        .many_to_one()
        .build()
        .expect("the fixture relationship must build")
}

fn model(
    datasets: Vec<Dataset>,
    unionsets: Vec<Unionset>,
    joinsets: Vec<Joinset>,
    relationships: Vec<Relationship>,
) -> SemanticModel {
    let mut built = SemanticModel::builder().name("sales");
    for axis in ["order_ts", "ship_ts", "valid_from", "valid_to"] {
        built = built.dimension(
            Dimension::builder(axis)
                .data_type(DataType::Date)
                .categorical()
                .build(),
        );
    }
    built = built
        .dimension(
            Dimension::builder("region")
                .data_type(DataType::String)
                .categorical()
                .build(),
        )
        .measure(
            Measure::builder("revenue")
                .data_type(DataType::Double)
                .agg(agg("sum"))
                .build(),
        );
    for one in datasets {
        built = built.dataset(one);
    }
    for one in unionsets {
        built = built.unionset(one);
    }
    for one in joinsets {
        built = built.joinset(one);
    }
    for one in relationships {
        built = built.relationship(one);
    }
    built.build().expect("the fixture model must build").0
}

async fn compiled(model: SemanticModel, files: &[&str]) -> SemanticManifest {
    let fs = fixture_fs(files).await;
    let (manifest, warnings) = compile(model, None, &fs, test_provenance())
        .await
        .expect("the fixture model must compile");
    assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
    assert!(
        manifest.validate().is_ok(),
        "compile must not emit an artifact its own validator refuses: {:?}",
        manifest.validate()
    );
    manifest
}

async fn refused(model: SemanticModel, files: &[&str]) -> CompileError {
    let fs = fixture_fs(files).await;
    let (error, _) = compile(model, None, &fs, test_provenance())
        .await
        .expect_err("the fixture model must be refused");
    error.kind
}

fn shape<'m>(manifest: &'m SemanticManifest, name: &str) -> &'m semstrait_ir::TemporalShape {
    by_name(manifest, name)
        .temporal
        .as_ref()
        .unwrap_or_else(|| panic!("`{name}` must publish a time axis"))
}

fn axis(manifest: &SemanticManifest, name: &str) -> semstrait_common::EntityId {
    manifest
        .semantic_definitions
        .by_name(name)
        .unwrap_or_else(|| panic!("`{name}` must be registered"))
        .id
        .clone()
}

#[tokio::test]
async fn a_unionset_publishes_the_axis_its_constituents_agree_on() {
    let manifest = compiled(
        model(
            vec![],
            vec![unionset(
                "all_orders",
                vec![
                    leaf(
                        "eu",
                        Some(TemporalShape::events("order_ts", Some(Granularity::Day))),
                    ),
                    leaf(
                        "us",
                        Some(TemporalShape::events("order_ts", Some(Granularity::Day))),
                    ),
                ],
                None,
            )],
            vec![],
            vec![],
        ),
        &["eu", "us"],
    )
    .await;

    let stacked = shape(&manifest, "all_orders");
    assert_eq!(
        stacked.temporal_type,
        TemporalType::events(axis(&manifest, "order_ts")),
        "the agreed axis reaches the stacked surface, resolved to its minted id"
    );
    assert_eq!(
        stacked.granularity,
        Some(Granularity::Day),
        "the grain the constituents agree on is carried"
    );
}

#[tokio::test]
async fn two_constituents_on_different_clocks_are_reported() {
    let error = refused(
        model(
            vec![],
            vec![unionset(
                "all_orders",
                vec![
                    leaf(
                        "eu",
                        Some(TemporalShape::events("order_ts", Some(Granularity::Day))),
                    ),
                    leaf(
                        "us",
                        Some(TemporalShape::events("ship_ts", Some(Granularity::Day))),
                    ),
                ],
                None,
            )],
            vec![],
            vec![],
        ),
        &["eu", "us"],
    )
    .await;

    let CompileError::ComposedTemporalDisagreement { composite, shapes } = error else {
        panic!("expected the Unionset disagreement, got {error:?}");
    };
    assert_eq!(
        composite,
        by_name_id("all_orders"),
        "names the stacked surface"
    );
    assert_eq!(
        shapes.len(),
        2,
        "carries every disagreeing constituent and its shape, got {shapes:?}"
    );
}

fn by_name_id(name: &str) -> semstrait_common::EntityId {
    semstrait_manifest::mint_semantic_source_id(name)
}

#[tokio::test]
async fn the_grain_is_no_part_of_the_agreement() {
    let manifest = compiled(
        model(
            vec![],
            vec![unionset(
                "all_orders",
                vec![
                    leaf(
                        "eu",
                        Some(TemporalShape::events("order_ts", Some(Granularity::Day))),
                    ),
                    leaf(
                        "us",
                        Some(TemporalShape::events("order_ts", Some(Granularity::Month))),
                    ),
                ],
                None,
            )],
            vec![],
            vec![],
        ),
        &["eu", "us"],
    )
    .await;

    let stacked = shape(&manifest, "all_orders");
    assert_eq!(
        stacked.temporal_type,
        TemporalType::events(axis(&manifest, "order_ts")),
        "the axis still agrees, so the shape is published"
    );
    assert_eq!(
        stacked.granularity, None,
        "differing grains guarantee no single grain, and that is not an error"
    );
}

#[tokio::test]
async fn a_stated_shape_joins_the_unanimity_set() {
    let manifest = compiled(
        model(
            vec![],
            vec![unionset(
                "all_orders",
                vec![leaf("eu", None), leaf("us", None)],
                Some(TemporalShape::events("order_ts", None)),
            )],
            vec![],
            vec![],
        ),
        &["eu", "us"],
    )
    .await;

    assert_eq!(
        shape(&manifest, "all_orders").temporal_type,
        TemporalType::events(axis(&manifest, "order_ts")),
        "the stated shape reaches the artifact, resolved"
    );
    assert!(
        by_name(&manifest, "all_orders.eu").temporal.is_none(),
        "and it is not cascaded onto a constituent that states none"
    );
}

#[tokio::test]
async fn a_stated_shape_does_not_override_a_constituent() {
    let error = refused(
        model(
            vec![],
            vec![unionset(
                "all_orders",
                vec![
                    leaf(
                        "eu",
                        Some(TemporalShape::events("ship_ts", Some(Granularity::Day))),
                    ),
                    leaf("us", None),
                ],
                Some(TemporalShape::events("order_ts", None)),
            )],
            vec![],
            vec![],
        ),
        &["eu", "us"],
    )
    .await;

    assert!(
        matches!(
            error,
            CompileError::ComposedTemporalDisagreement { ref shapes, .. } if shapes.len() == 2
        ),
        "the stated shape and the constituent's are two participants, got {error:?}"
    );
}

#[tokio::test]
async fn a_reduction_carries_the_authored_shape_onto_the_stacked_surface() {
    let fs = seeded_fs([
        "s3://lake/orders/eu/part-0.parquet",
        "s3://lake/orders/us/part-0.parquet",
    ])
    .await;
    let dataset = interface!(Dataset::builder("orders"))
        .temporal(TemporalShape::timeseries(
            "order_ts",
            Some(Granularity::Day),
        ))
        .path("s3://lake/orders/*/*.parquet")
        .format(StorageFormat::Parquet)
        .build();
    let (manifest, _) = compile(
        model(vec![dataset], vec![], vec![], vec![]),
        None,
        &fs,
        test_provenance(),
    )
    .await
    .expect("the reduction must compile");
    assert!(matches!(
        by_name(&manifest, "orders").role_definition,
        RoleDefinition::Unionset { .. }
    ));

    let stacked = shape(&manifest, "orders");
    assert_eq!(
        stacked.temporal_type,
        TemporalType::timeseries(axis(&manifest, "order_ts")),
        "the authored axis reaches the entry the author's name resolves to"
    );
    assert_eq!(
        stacked.granularity,
        Some(Granularity::Day),
        "and so does the authored grain, which is what a grain-routed group reads"
    );
}

#[tokio::test]
async fn a_star_join_takes_the_highest_shape_rank() {
    let manifest = compiled(
        model(
            vec![],
            vec![],
            vec![joinset(
                "sales",
                vec![
                    leaf(
                        "orders",
                        Some(TemporalShape::events("order_ts", Some(Granularity::Day))),
                    ),
                    leaf(
                        "customers",
                        Some(TemporalShape::scd(
                            ScdType::Type2,
                            "valid_from",
                            "valid_to",
                            Some(Granularity::Day),
                        )),
                    ),
                ],
                vec![related("r1", "orders", "customers")],
            )],
            vec![],
        ),
        &["orders", "customers"],
    )
    .await;

    assert_eq!(
        shape(&manifest, "sales").temporal_type,
        TemporalType::events(axis(&manifest, "order_ts")),
        "`Events` outranks `Scd`, so the joined surface publishes the event axis"
    );
    assert_eq!(
        shape(&manifest, "sales").granularity,
        Some(Granularity::Day),
        "and the grain both operands agree on is carried"
    );
}

#[tokio::test]
async fn an_operand_with_no_shape_abstains() {
    let manifest = compiled(
        model(
            vec![],
            vec![],
            vec![joinset(
                "sales",
                vec![
                    leaf(
                        "orders",
                        Some(TemporalShape::events("order_ts", Some(Granularity::Day))),
                    ),
                    leaf("customers", None),
                ],
                vec![related("r1", "orders", "customers")],
            )],
            vec![],
        ),
        &["orders", "customers"],
    )
    .await;

    assert_eq!(
        shape(&manifest, "sales").temporal_type,
        TemporalType::events(axis(&manifest, "order_ts")),
        "an unclassified operand is not a shape to disagree with"
    );
    assert_eq!(
        shape(&manifest, "sales").granularity,
        Some(Granularity::Day),
        "and it guarantees no grain to disagree about either"
    );
}

#[tokio::test]
async fn equal_rank_over_two_clocks_yields_absence() {
    let manifest = compiled(
        model(
            vec![],
            vec![],
            vec![joinset(
                "sales",
                vec![
                    leaf(
                        "orders",
                        Some(TemporalShape::events("order_ts", Some(Granularity::Day))),
                    ),
                    leaf(
                        "shipments",
                        Some(TemporalShape::events("ship_ts", Some(Granularity::Day))),
                    ),
                ],
                vec![related("r1", "orders", "shipments")],
            )],
            vec![],
        ),
        &["orders", "shipments"],
    )
    .await;

    assert_eq!(
        shape(&manifest, "sales.orders").temporal_type,
        TemporalType::events(axis(&manifest, "order_ts")),
    );
    assert_eq!(
        shape(&manifest, "sales.shipments").temporal_type,
        TemporalType::events(axis(&manifest, "ship_ts")),
    );
    assert!(
        by_name(&manifest, "sales").temporal.is_none(),
        "the joined rows record time in two ways, so there is no one shape to publish"
    );
}

#[tokio::test]
async fn every_enumerated_unit_carries_a_derived_shape() {
    let manifest = compiled(
        model(
            vec![],
            vec![],
            vec![joinset(
                "sales",
                vec![
                    leaf(
                        "orders",
                        Some(TemporalShape::events("order_ts", Some(Granularity::Day))),
                    ),
                    leaf(
                        "customers",
                        Some(TemporalShape::scd(
                            ScdType::Type2,
                            "valid_from",
                            "valid_to",
                            Some(Granularity::Day),
                        )),
                    ),
                    leaf("regions", None),
                ],
                vec![
                    related("r1", "orders", "customers"),
                    related("r2", "orders", "regions"),
                ],
            )],
            vec![],
        ),
        &["orders", "customers", "regions"],
    )
    .await;

    let customers_shape = shape(&manifest, "sales.customers");
    assert_eq!(
        customers_shape.temporal_type,
        TemporalType::scd(
            IrScdType::Type2,
            axis(&manifest, "valid_from"),
            axis(&manifest, "valid_to"),
        ),
    );
    let stored = serde_json::to_value(customers_shape).expect("serialize TemporalShape");
    assert_eq!(
        stored["temporal_type"]["content"]["scd_type"], "type2",
        "the manifest's own `ScdType` still serializes `type2`: {stored}"
    );
    assert_eq!(
        stored["granularity"], "day",
        "the manifest's own `TemporalShape.granularity` reaches the artifact unchanged: {stored}"
    );

    let events = TemporalType::events(axis(&manifest, "order_ts"));
    let composed: Vec<&SemanticSource> = manifest
        .semantic_sources
        .values()
        .filter(|entry| !entry.role_definition.operands().is_empty())
        .collect();
    assert!(
        composed.len() > 1,
        "the fixture must enumerate more than the authored unit, got {}",
        composed.len()
    );
    for entry in composed {
        let shape = entry.temporal.as_ref().unwrap_or_else(|| {
            panic!(
                "the enumerated unit `{}` must publish the axis its operands reached",
                entry.name.as_str()
            )
        });
        assert_eq!(
            shape.temporal_type,
            events,
            "unit `{}` takes the highest rank among its operands",
            entry.name.as_str()
        );
    }
}

#[tokio::test]
async fn the_derived_shape_is_invariant_to_authored_direction() {
    let mut shapes = Vec::new();
    for (from, to) in [("orders", "customers"), ("customers", "orders")] {
        let manifest = compiled(
            model(
                vec![],
                vec![],
                vec![joinset(
                    "sales",
                    vec![
                        leaf(
                            "orders",
                            Some(TemporalShape::events("order_ts", Some(Granularity::Day))),
                        ),
                        leaf(
                            "customers",
                            Some(TemporalShape::snapshot("ship_ts", Some(Granularity::Day))),
                        ),
                    ],
                    vec![related("r1", from, to)],
                )],
                vec![],
            ),
            &["orders", "customers"],
        )
        .await;
        shapes.push(shape(&manifest, "sales").clone());
    }
    assert_eq!(
        shapes[0], shapes[1],
        "the same operands joined the other way round publish the same shape"
    );
}

#[tokio::test]
async fn absence_records_no_reason() {
    let manifest = compiled(
        model(
            vec![],
            vec![unionset(
                "all_orders",
                vec![
                    leaf(
                        "eu",
                        Some(TemporalShape::events("order_ts", Some(Granularity::Day))),
                    ),
                    leaf(
                        "us",
                        Some(TemporalShape::events("order_ts", Some(Granularity::Month))),
                    ),
                ],
                None,
            )],
            vec![],
            vec![],
        ),
        &["eu", "us"],
    )
    .await;

    let encoded = serde_json::to_value(shape(&manifest, "all_orders"))
        .expect("a persisted shape must serialise");
    let fields: Vec<&String> = encoded
        .as_object()
        .expect("a shape encodes as an object")
        .keys()
        .collect();
    assert_eq!(
        fields,
        vec!["temporal_type", "granularity"],
        "an absent grain records no reason, and neither does an absent shape"
    );
}

#[tokio::test]
async fn ordering_is_recovered_from_coarseness_and_not_persisted() {
    let manifest = compiled(
        model(
            vec![],
            vec![],
            vec![joinset(
                "sales",
                vec![
                    leaf(
                        "daily",
                        Some(TemporalShape::events("order_ts", Some(Granularity::Day))),
                    ),
                    leaf(
                        "monthly",
                        Some(TemporalShape::events("order_ts", Some(Granularity::Month))),
                    ),
                ],
                vec![related("r1", "daily", "monthly")],
            )],
            vec![],
        ),
        &["daily", "monthly"],
    )
    .await;

    let day = shape(&manifest, "sales.daily")
        .granularity
        .expect("a grain");
    let month = shape(&manifest, "sales.monthly")
        .granularity
        .expect("a grain");
    assert!(
        month.coarseness() > day.coarseness(),
        "the order the levels used to persist is recovered from the carried grains"
    );
    assert_eq!(
        by_name(&manifest, "sales")
            .temporal
            .as_ref()
            .map(|s| s.granularity),
        Some(None),
        "and the composed surface guarantees no single grain, since they differ"
    );
}

fn disagreement_shapes_on_sales(
    error: CompileError,
) -> Vec<(semstrait_common::EntityId, TemporalType)> {
    let CompileError::ComposedTemporalDisagreement { composite, shapes } = error else {
        panic!("expected the Joinset disagreement, got {error:?}");
    };
    assert_eq!(composite, by_name_id("sales"));
    shapes
}

fn assert_disagreement_on_sales(error: &CompileError) {
    assert!(
        matches!(error, CompileError::ComposedTemporalDisagreement { composite, .. } if *composite == by_name_id("sales")),
        "expected the Joinset disagreement, got {error:?}"
    );
}

fn daily(
    shape: fn(&'static str, Option<Granularity>) -> TemporalShape,
    axis: &'static str,
) -> TemporalShape {
    shape(axis, Some(Granularity::Day))
}

#[tokio::test]
async fn a_joinset_over_shapeless_constituents_carries_the_shape_it_states() {
    let manifest = compiled(
        model(
            vec![],
            vec![],
            vec![joinset_stating(
                "sales",
                vec![leaf("orders", None), leaf("customers", None)],
                vec![related("r1", "orders", "customers")],
                TemporalShape::events("order_ts", None),
            )],
            vec![],
        ),
        &["orders", "customers"],
    )
    .await;

    assert_eq!(
        shape(&manifest, "sales").temporal_type,
        TemporalType::events(axis(&manifest, "order_ts")),
        "no operand carries a shape, so the stated one is the only source"
    );
    assert_eq!(shape(&manifest, "sales").granularity, None);
}

#[tokio::test]
async fn a_joinset_in_a_tie_carries_the_tied_shape_it_states() {
    let manifest = compiled(
        model(
            vec![],
            vec![],
            vec![joinset_stating(
                "sales",
                vec![
                    leaf("orders", Some(daily(TemporalShape::events, "order_ts"))),
                    leaf("shipments", Some(daily(TemporalShape::events, "ship_ts"))),
                ],
                vec![related("r1", "orders", "shipments")],
                TemporalShape::events("order_ts", None),
            )],
            vec![],
        ),
        &["orders", "shipments"],
    )
    .await;

    assert_eq!(
        shape(&manifest, "sales").temporal_type,
        TemporalType::events(axis(&manifest, "order_ts")),
        "the stated shape equals a shape at the top rank, so compile selects nothing itself"
    );
}

#[tokio::test]
async fn a_joinset_in_a_tie_refuses_a_shape_that_is_neither_tied_shape() {
    let error = refused(
        model(
            vec![],
            vec![],
            vec![joinset_stating(
                "sales",
                vec![
                    leaf("orders", Some(daily(TemporalShape::events, "order_ts"))),
                    leaf("shipments", Some(daily(TemporalShape::events, "ship_ts"))),
                ],
                vec![related("r1", "orders", "shipments")],
                TemporalShape::events("valid_from", None),
            )],
            vec![],
        ),
        &["orders", "shipments"],
    )
    .await;

    let shapes = disagreement_shapes_on_sales(error);
    assert_eq!(
        shapes.len(),
        3,
        "both constituents and the stated shape: {shapes:?}"
    );
}

#[tokio::test]
async fn a_joinset_refuses_a_stated_shape_of_another_kind_than_its_constituents() {
    let error = refused(
        model(
            vec![],
            vec![],
            vec![joinset_stating(
                "sales",
                vec![
                    leaf("orders", Some(daily(TemporalShape::timeseries, "order_ts"))),
                    leaf(
                        "shipments",
                        Some(daily(TemporalShape::timeseries, "order_ts")),
                    ),
                ],
                vec![related("r1", "orders", "shipments")],
                TemporalShape::snapshot("order_ts", None),
            )],
            vec![],
        ),
        &["orders", "shipments"],
    )
    .await;

    assert_disagreement_on_sales(&error);
}

#[tokio::test]
async fn a_joinset_refuses_the_shape_of_a_constituent_below_the_top_rank() {
    let scd = TemporalShape::scd(
        ScdType::Type2,
        "valid_from",
        "valid_to",
        Some(Granularity::Day),
    );
    let error = refused(
        model(
            vec![],
            vec![],
            vec![joinset_stating(
                "sales",
                vec![
                    leaf("orders", Some(daily(TemporalShape::events, "order_ts"))),
                    leaf("customers", Some(scd.clone())),
                ],
                vec![related("r1", "orders", "customers")],
                TemporalShape::scd(ScdType::Type2, "valid_from", "valid_to", None),
            )],
            vec![],
        ),
        &["orders", "customers"],
    )
    .await;

    assert_disagreement_on_sales(&error);
}
