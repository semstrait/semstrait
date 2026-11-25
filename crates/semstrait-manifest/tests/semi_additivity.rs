mod common;
use common::golden::{self, Golden};
use common::{
    agg, compile_error, compile_error_on, compile_model, compile_model_on, constituent, dated,
    dimension, fixture_fs, identity_mapping, insert_dimension, insert_measure, lake_folder,
    measure, orders_model, refusal_golden, seeded_fs,
};

use semstrait_common::Granularity;
use semstrait_common::SemanticsName;
use semstrait_ir::SemanticRole;
use semstrait_manifest::CompileError;
use semstrait_model::expr::lit;
use semstrait_model::{
    CrossFilter, Dimension, DimensionSite, Joinset, Measure, MeasureSite, NestedDataset,
    NestedGrainset, Optional, Relationship, SemanticModel, SemanticModelBuilder, SemiAdditivity,
    SemiAdditivityStrategy, StorageFormat, TemporalShape, Unionset,
};
use semstrait_sql::datatype::DataType;

fn temporal_dimension(name: &str) -> Dimension {
    Dimension::builder(name)
        .data_type(DataType::String)
        .temporal([Granularity::Day])
        .build()
}

fn semi_balance(by: Option<&str>, axes: &[&str]) -> Measure {
    Measure::builder("balance")
        .data_type(DataType::Double)
        .agg(agg("sum"))
        .semi(
            SemiAdditivity::builder()
                .maybe_by(by.map(SemanticsName::new))
                .axes(axes.iter().map(|axis| SemanticsName::new(*axis)).collect())
                .strategy(SemiAdditivityStrategy::Latest)
                .build(),
        )
        .build()
}

fn by_no_default(semantic_source: &str) -> CompileError {
    CompileError::SemiAdditiveByNoDefault {
        semantics: "balance".into(),
        semantic_source: semantic_source.into(),
    }
}

#[tokio::test]
async fn a_semi_additive_measures_axes_and_by_store_the_definition_ids_they_name() {
    let model = orders_model(
        vec![
            dimension("account", DataType::String),
            temporal_dimension("order_date"),
        ],
        vec![semi_balance(Some("order_date"), &["account"])],
        |orders| orders.semantic_mapping(identity_mapping(&["account", "order_date", "balance"])),
    );
    let manifest = compile_model(model).await;
    let balance = manifest
        .semantic_definitions
        .by_name("balance")
        .expect("`balance` is defined");
    let account = manifest
        .semantic_definitions
        .by_name("account")
        .expect("`account` is defined");
    let order_date = manifest
        .semantic_definitions
        .by_name("order_date")
        .expect("`order_date` is defined");
    let json = serde_json::to_value(balance).expect("a definition serializes");
    let semi = &json["role_definition"]["content"]["additivity"]["semi"];
    assert_eq!(
        semi["axes"],
        serde_json::json!([account.id.as_str()]),
        "a non-empty `axes` stores the axis definitions' ids: {json}"
    );
    assert_eq!(
        semi["by"],
        serde_json::json!(order_date.id.as_str()),
        "`by` stores the definition id it names: {json}"
    );
}

#[tokio::test]
async fn a_semi_additive_measures_empty_axes_stores_an_empty_list_not_omitted() {
    let model = orders_model(
        vec![temporal_dimension("order_date")],
        vec![semi_balance(Some("order_date"), &[])],
        |orders| orders.semantic_mapping(identity_mapping(&["order_date", "balance"])),
    );
    let manifest = compile_model(model).await;
    let balance = manifest
        .semantic_definitions
        .by_name("balance")
        .expect("`balance` is defined");
    let json = serde_json::to_value(balance).expect("a definition serializes");
    let semi = &json["role_definition"]["content"]["additivity"]["semi"];
    assert_eq!(
        semi["axes"],
        serde_json::json!([]),
        "an empty `axes` stores `[]`, not omitted: {json}"
    );
}

#[tokio::test]
async fn by_resolves_to_an_id_for_a_measure_no_source_serves() {
    let mut model = orders_model(vec![temporal_dimension("order_date")], vec![], |orders| {
        orders.semantic_mapping(identity_mapping(&["order_date"]))
    });
    insert_measure(&mut model, semi_balance(Some("order_date"), &[]));
    let manifest = compile_model(model).await;
    let balance = manifest
        .semantic_definitions
        .by_name("balance")
        .expect("`balance` is defined");
    let order_date = manifest
        .semantic_definitions
        .by_name("order_date")
        .expect("`order_date` is defined");
    let json = serde_json::to_value(balance).expect("a definition serializes");
    let semi = &json["role_definition"]["content"]["additivity"]["semi"];
    assert_eq!(
        semi["by"],
        serde_json::json!(order_date.id.as_str()),
        "`by` stores the definition id it names even though no source serves the measure: {json}"
    );
}

#[tokio::test]
async fn by_fails_naming_no_definition_for_a_measure_no_source_serves() {
    let mut model = orders_model(vec![temporal_dimension("order_date")], vec![], |orders| {
        orders.semantic_mapping(identity_mapping(&["order_date"]))
    });
    insert_measure(&mut model, semi_balance(Some("ghost"), &[]));
    let expected = CompileError::SemiAdditiveByUnknown {
        semantics: "balance".into(),
        by: "ghost".into(),
    };
    assert_eq!(compile_error(model).await, expected);
}

#[tokio::test]
async fn by_defaults_to_the_one_events_time_axis() {
    let model = orders_model(
        vec![temporal_dimension("order_date")],
        vec![semi_balance(None, &[])],
        |orders| {
            orders
                .semantic_mapping(identity_mapping(&["order_date", "balance"]))
                .temporal(TemporalShape::events("order_date", Some(Granularity::Day)))
        },
    );
    let manifest = compile_model(model).await;
    let balance = manifest
        .semantic_definitions
        .by_name("balance")
        .expect("`balance` is defined");
    let json = serde_json::to_value(balance).expect("a definition serializes");
    let semi = &json["role_definition"]["content"]["additivity"]["semi"];
    assert!(semi.is_object(), "`semi` is stored: {json}");
    assert!(
        semi.get("by").is_none(),
        "compile checks the default; it does not write `by`: {json}"
    );
}

fn events_constituent(name: &str, event_time: &str, names: &[&str]) -> NestedDataset {
    NestedDataset::builder(name)
        .path(lake_folder(name))
        .format(StorageFormat::Parquet)
        .semantic_mapping(identity_mapping(names))
        .temporal(TemporalShape::events(event_time, Some(Granularity::Day)))
        .build()
}

fn full_join_on_customer_id(from: &str, to: &str) -> Relationship {
    Relationship::builder()
        .name(format!("{from}_to_{to}"))
        .from(from)
        .to(to)
        .on(["customer_id"])
        .many_to_many()
        .optional(Optional::Both)
        .cross_filter(CrossFilter::None)
        .build()
        .expect("the fixture relationship must build")
}

#[tokio::test]
async fn by_default_fails_with_no_temporal_shape() {
    by_default_no_temporal_shape_refusal()
        .await
        .assert_committed();
}

async fn by_default_no_temporal_shape_refusal() -> Golden {
    let model = orders_model(
        vec![dimension("account", DataType::String)],
        vec![semi_balance(None, &[])],
        |orders| orders.semantic_mapping(identity_mapping(&["account", "balance"])),
    );
    refusal_golden(
        "by_default_no_temporal_shape",
        model,
        by_no_default("orders"),
    )
    .await
}

#[tokio::test]
async fn by_default_fails_with_two_differing_time_axes() {
    by_default_two_differing_axes_refusal()
        .await
        .assert_committed();
}

async fn by_default_two_differing_axes_refusal() -> Golden {
    let joinset = Joinset::builder("jx")
        .dataset(events_constituent(
            "x",
            "event_x",
            &["customer_id", "balance", "event_x"],
        ))
        .dataset(events_constituent(
            "y",
            "event_y",
            &["customer_id", "balance", "event_y"],
        ))
        .dimension(DimensionSite::r#ref("customer_id"))
        .dimension(DimensionSite::r#ref("event_x"))
        .dimension(DimensionSite::r#ref("event_y"))
        .measure(MeasureSite::r#ref("balance"))
        .relationship(full_join_on_customer_id("x", "y"))
        .build();
    let model = SemanticModel::builder()
        .name("sales")
        .dimension(dimension("customer_id", DataType::String))
        .dimension(temporal_dimension("event_x"))
        .dimension(temporal_dimension("event_y"))
        .measure(semi_balance(None, &[]))
        .joinset(joinset)
        .build()
        .expect("the fixture model must build")
        .0;
    refusal_golden(
        "by_default_two_differing_time_axes",
        model,
        by_no_default("jx"),
    )
    .await
}

#[tokio::test]
async fn by_default_fails_reading_no_dataset() {
    by_default_no_dataset_read_refusal()
        .await
        .assert_committed();
}

async fn by_default_no_dataset_read_refusal() -> Golden {
    let joinset = Joinset::builder("jx")
        .dataset(events_constituent(
            "a",
            "snapped_at",
            &["customer_id", "snapped_at"],
        ))
        .dataset(events_constituent(
            "b",
            "snapped_at",
            &["customer_id", "snapped_at"],
        ))
        .dimension(DimensionSite::r#ref("customer_id"))
        .dimension(DimensionSite::r#ref("snapped_at"))
        .measure(MeasureSite::ref_with_expr("balance", lit(1)))
        .relationship(
            Relationship::builder()
                .name("a_to_b")
                .from("a")
                .to("b")
                .on(["customer_id"])
                .many_to_one()
                .build()
                .expect("the fixture relationship must build"),
        )
        .build();
    let model = SemanticModel::builder()
        .name("sales")
        .dimension(dimension("customer_id", DataType::String))
        .dimension(temporal_dimension("snapped_at"))
        .measure(semi_balance(None, &[]))
        .joinset(joinset)
        .build()
        .expect("the fixture model must build")
        .0;
    refusal_golden("by_default_no_dataset_read", model, by_no_default("jx")).await
}

async fn stacked_over(paths: &[&str]) -> CompileError {
    let fs = seeded_fs(paths.iter().copied()).await;
    let unionset = Unionset::builder("stacked")
        .dataset(
            NestedDataset::builder("r")
                .path("s3://lake/r/*/*.parquet")
                .format(StorageFormat::Parquet)
                .semantic_mapping(identity_mapping(&["snap_date"]))
                .temporal(TemporalShape::events("snap_date", Some(Granularity::Day)))
                .build(),
        )
        .dataset(events_constituent("s", "snap_date", &["snap_date"]))
        .dimension(DimensionSite::r#ref("snap_date"))
        .measure(MeasureSite::ref_with_expr("balance", lit(1)))
        .union_all()
        .build();
    let model = SemanticModel::builder()
        .name("sales")
        .dimension(temporal_dimension("snap_date"))
        .measure(semi_balance(None, &[]))
        .unionset(unionset)
        .build()
        .expect("the fixture model must build")
        .0;
    compile_error_on(model, &fs).await
}

#[tokio::test]
async fn by_default_below_a_composer_is_layout_independent() {
    let one_source = ["s3://lake/r/a/part-0.parquet"];
    let two_sources = [
        "s3://lake/r/a/part-0.parquet",
        "s3://lake/r/b/part-0.parquet",
    ];
    assert_eq!(stacked_over(&one_source).await, by_no_default("stacked"));
    assert_eq!(stacked_over(&two_sources).await, by_no_default("stacked"));
}

#[tokio::test]
async fn by_fails_naming_a_non_temporal_dimension() {
    by_invalid_refusal().await.assert_committed();
}

async fn by_invalid_refusal() -> Golden {
    let model = orders_model(
        vec![dimension("account", DataType::String)],
        vec![semi_balance(Some("account"), &[])],
        |orders| orders.semantic_mapping(identity_mapping(&["account", "balance"])),
    );
    refusal_golden("by_invalid_non_temporal", model, by_invalid("account")).await
}

fn by_invalid(by: &str) -> CompileError {
    CompileError::SemiAdditiveByInvalid {
        semantics: "balance".into(),
        by: by.into(),
        semantic_source: "orders".into(),
    }
}

#[tokio::test]
async fn by_fails_naming_one_of_its_own_axes() {
    by_in_axes_refusal().await.assert_committed();
}

async fn by_in_axes_refusal() -> Golden {
    let model = orders_model(
        vec![temporal_dimension("order_date")],
        vec![semi_balance(Some("order_date"), &["order_date"])],
        |orders| orders.semantic_mapping(identity_mapping(&["order_date", "balance"])),
    );
    refusal_golden("by_invalid_in_axes", model, by_invalid("order_date")).await
}

#[tokio::test]
async fn axis_fails_naming_a_dimension_the_source_does_not_read() {
    dimension_not_read_refusal().await.assert_committed();
}

async fn dimension_not_read_refusal() -> Golden {
    let mut model = orders_model(
        vec![temporal_dimension("order_date")],
        vec![semi_balance(Some("order_date"), &["region"])],
        |orders| orders.semantic_mapping(identity_mapping(&["order_date", "balance"])),
    );
    insert_dimension(&mut model, dimension("region", DataType::String));
    let expected = CompileError::SemiAdditiveDimensionNotRead {
        semantics: "balance".into(),
        dimension: "region".into(),
        semantic_source: "orders".into(),
    };
    refusal_golden("dimension_not_read", model, expected).await
}

#[tokio::test]
async fn axis_fails_mapped_by_two_datasets_with_no_single_time_axis() {
    dimension_ambiguous_refusal().await.assert_committed();
}

async fn dimension_ambiguous_refusal() -> Golden {
    let joinset = Joinset::builder("jx")
        .dataset(events_constituent(
            "x",
            "by_dim",
            &["customer_id", "balance", "by_dim", "region"],
        ))
        .dataset(constituent(
            "y",
            &["customer_id", "balance", "by_dim", "region"],
        ))
        .dimension(DimensionSite::r#ref("customer_id"))
        .dimension(DimensionSite::r#ref("by_dim"))
        .dimension(DimensionSite::r#ref("region"))
        .measure(MeasureSite::r#ref("balance"))
        .relationship(full_join_on_customer_id("x", "y"))
        .build();
    let model = SemanticModel::builder()
        .name("sales")
        .dimension(dimension("customer_id", DataType::String))
        .dimension(temporal_dimension("by_dim"))
        .dimension(dimension("region", DataType::String))
        .measure(semi_balance(Some("by_dim"), &["region"]))
        .joinset(joinset)
        .build()
        .expect("the fixture model must build")
        .0;
    let expected = CompileError::SemiAdditiveDimensionAmbiguous {
        semantics: "balance".into(),
        dimension: "region".into(),
        semantic_source: "jx".into(),
    };
    refusal_golden("dimension_ambiguous", model, expected).await
}

#[tokio::test]
async fn by_fails_unserved_by_a_constituent_that_reads_it() {
    by_not_served_refusal().await.assert_committed();
}

async fn by_not_served_refusal() -> Golden {
    let joinset = Joinset::builder("jx")
        .dataset(constituent("x", &["customer_id", "balance", "order_date"]))
        .dataset(constituent("y", &["customer_id", "balance"]))
        .dimension(DimensionSite::r#ref("customer_id"))
        .dimension(DimensionSite::r#ref("order_date"))
        .measure(MeasureSite::r#ref("balance"))
        .relationship(full_join_on_customer_id("x", "y"))
        .build();
    let model = SemanticModel::builder()
        .name("sales")
        .dimension(dimension("customer_id", DataType::String))
        .dimension(temporal_dimension("order_date"))
        .measure(semi_balance(Some("order_date"), &[]))
        .joinset(joinset)
        .build()
        .expect("the fixture model must build")
        .0;
    let expected = CompileError::SemiAdditiveByNotServed {
        semantics: "balance".into(),
        by: "order_date".into(),
        semantic_source: "jx.y".into(),
    };
    refusal_golden("by_not_served", model, expected).await
}

#[tokio::test]
async fn axis_fails_naming_no_semantics() {
    axis_unknown_refusal().await.assert_committed();
}

async fn axis_unknown_refusal() -> Golden {
    let model = orders_model(vec![], vec![semi_balance(None, &["ghost"])], |orders| {
        orders.semantic_mapping(identity_mapping(&["balance"]))
    });
    let expected = CompileError::SemiAdditiveAxisUnknown {
        semantics: "balance".into(),
        axis: "ghost".into(),
    };
    refusal_golden("axis_unknown", model, expected).await
}

#[tokio::test]
async fn axis_fails_naming_a_non_dimension_role() {
    axis_not_a_dimension_refusal().await.assert_committed();
}

async fn axis_not_a_dimension_refusal() -> Golden {
    let model = orders_model(
        vec![],
        vec![
            semi_balance(None, &["revenue"]),
            measure("revenue", DataType::Double, "sum"),
        ],
        |orders| orders.semantic_mapping(identity_mapping(&["balance", "revenue"])),
    );
    let expected = CompileError::SemiAdditiveAxisNotADimension {
        semantics: "balance".into(),
        axis: "revenue".into(),
        role: SemanticRole::Measure,
    };
    refusal_golden("axis_not_a_dimension", model, expected).await
}

#[tokio::test]
async fn axis_fails_unserved_by_a_branch_that_signs_the_measure() {
    axis_not_served_refusal().await.assert_committed();
}

async fn axis_not_served_refusal() -> Golden {
    let unionset = Unionset::builder("stacked")
        .dataset(events_constituent(
            "b1",
            "order_date",
            &["balance", "order_date", "snap_date"],
        ))
        .dataset(constituent("b2", &["balance", "order_date"]))
        .dimension(DimensionSite::r#ref("order_date"))
        .dimension(DimensionSite::r#ref("snap_date"))
        .measure(MeasureSite::r#ref("balance"))
        .union_all()
        .build();
    let model = SemanticModel::builder()
        .name("sales")
        .dimension(temporal_dimension("order_date"))
        .dimension(dimension("snap_date", DataType::String))
        .measure(semi_balance(Some("order_date"), &["snap_date"]))
        .unionset(unionset)
        .build()
        .expect("the fixture model must build")
        .0;
    let expected = CompileError::SemiAdditiveAxisNotServed {
        semantics: "balance".into(),
        axis: "snap_date".into(),
        semantic_source: "stacked.b2".into(),
    };
    refusal_golden("axis_not_served", model, expected).await
}

#[tokio::test]
async fn by_fails_naming_no_definition_for_a_served_measure() {
    by_unknown_refusal().await.assert_committed();
}

async fn by_unknown_refusal() -> Golden {
    let model = orders_model(vec![], vec![semi_balance(Some("ghost"), &[])], |orders| {
        orders.semantic_mapping(identity_mapping(&["balance"]))
    });
    let expected = CompileError::SemiAdditiveByUnknown {
        semantics: "balance".into(),
        by: "ghost".into(),
    };
    refusal_golden("by_unknown", model, expected).await
}

fn targets(yearly: &[&str]) -> NestedGrainset {
    NestedGrainset::builder("targets")
        .dataset(dated(
            "targets_monthly",
            &["campaign_id", "date", "balance"],
            Granularity::Month,
        ))
        .dataset(dated("targets_yearly", yearly, Granularity::Year))
        .temporal(TemporalShape::events("date", Some(Granularity::Year)))
        .build()
}

async fn compiles_over_targets(
    composition: impl Fn(SemanticModelBuilder, NestedGrainset) -> SemanticModelBuilder,
) {
    let fs = fixture_fs(&["daily", "targets_monthly", "targets_yearly"]).await;
    for yearly in [
        &["campaign_id", "date"][..],
        &["campaign_id", "date", "balance"],
    ] {
        for by in [None, Some("date")] {
            let date = Dimension::builder("date")
                .data_type(DataType::Date)
                .temporal([Granularity::Day, Granularity::Month, Granularity::Year])
                .build();
            let model = composition(
                SemanticModel::builder()
                    .name("sales")
                    .dimension(dimension("campaign_id", DataType::String))
                    .dimension(date)
                    .measure(semi_balance(by, &[])),
                targets(yearly),
            )
            .build()
            .expect("the fixture model must build")
            .0;
            compile_model_on(model, &fs).await;
        }
    }
}

#[tokio::test]
async fn by_resolves_through_a_joinsets_grainset_operand() {
    compiles_over_targets(|model, targets| {
        model.joinset(
            Joinset::builder("j")
                .dataset(dated("daily", &["campaign_id", "date"], Granularity::Day))
                .grainset(targets)
                .dimension(DimensionSite::r#ref("campaign_id"))
                .dimension(DimensionSite::r#ref("date"))
                .measure(MeasureSite::r#ref("balance"))
                .relationship(
                    Relationship::builder()
                        .name("daily_to_targets")
                        .from("daily")
                        .to("targets")
                        .on(["campaign_id"])
                        .many_to_one()
                        .build()
                        .expect("the fixture relationship must build"),
                )
                .build(),
        )
    })
    .await;
}

#[tokio::test]
async fn by_resolves_through_a_unionsets_grainset_branch() {
    compiles_over_targets(|model, targets| {
        model.unionset(
            Unionset::builder("u")
                .grainset(targets)
                .dataset(dated("daily", &["campaign_id", "date"], Granularity::Day))
                .dimension(DimensionSite::r#ref("campaign_id"))
                .dimension(DimensionSite::r#ref("date"))
                .measure(MeasureSite::r#ref("balance"))
                .union_all()
                .build(),
        )
    })
    .await;
}

#[tokio::test]
async fn generate_goldens() {
    golden::generate(async {
        vec![
            by_default_no_temporal_shape_refusal().await,
            by_default_two_differing_axes_refusal().await,
            by_default_no_dataset_read_refusal().await,
            by_invalid_refusal().await,
            by_in_axes_refusal().await,
            dimension_not_read_refusal().await,
            dimension_ambiguous_refusal().await,
            by_not_served_refusal().await,
            axis_unknown_refusal().await,
            axis_not_a_dimension_refusal().await,
            axis_not_served_refusal().await,
            by_unknown_refusal().await,
        ]
    })
    .await;
}
