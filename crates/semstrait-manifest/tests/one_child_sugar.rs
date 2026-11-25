mod common;

use common::{
    agg, by_name, compile_error_on, compile_model_on, constituent, dated, fixture_fs,
    has_signature, identity_mapping, lake_folder, names, signature,
};
use semstrait_common::io::Fingerprintable;
use semstrait_common::Granularity;
use semstrait_ir::{TemporalShape as IrShape, TemporalType};
use semstrait_manifest::{
    mint_model_id, mint_semantic_source_id, CompileError, RoleDefinition, SemanticManifest,
};
use semstrait_model::expr::{dim, lit, measure};
use semstrait_model::{
    ComplexExtras, DataKindFilter, Dimension, DimensionSite, Grainset, Measure, MeasureSite,
    Metric, MetricSite, NestedDataset, NestedGrainset, NestedJoinset, NestedUnionset, Relationship,
    SemanticModel, SemanticModelBuilder, StorageFormat, TemporalShape, Unionset,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::ExprOperationExt;

const NAMES: [&str; 3] = ["region", "date", "revenue"];

fn vocabulary() -> SemanticModelBuilder {
    SemanticModel::builder()
        .name("sales")
        .dimension(
            Dimension::builder("region")
                .data_type(DataType::String)
                .categorical()
                .build(),
        )
        .dimension(
            Dimension::builder("date")
                .data_type(DataType::Date)
                .categorical()
                .build(),
        )
        .measure(
            Measure::builder("revenue")
                .data_type(DataType::Double)
                .agg(agg("sum"))
                .build(),
        )
}

fn metric_vocabulary() -> SemanticModelBuilder {
    vocabulary().metric(
        Metric::builder("doubled")
            .data_type(DataType::Double)
            .expr(measure("revenue").expect("a name") + measure("revenue").expect("a name"))
            .build(),
    )
}

fn eu_only() -> DataKindFilter {
    DataKindFilter::builder("eu_only")
        .expr(ExprOperationExt::eq(
            dim("region").expect("a name"),
            lit("eu"),
        ))
        .build()
}

fn daily(name: &str) -> semstrait_model::NestedDataset {
    dated(name, &NAMES, Granularity::Day)
}

fn declared_grainset() -> Grainset {
    Grainset::builder("ads")
        .dimension(DimensionSite::r#ref("region").with_key())
        .dimension(DimensionSite::r#ref("date"))
        .measure(MeasureSite::r#ref("revenue"))
        .metric(MetricSite::r#ref("doubled"))
        .filter(eu_only())
        .dataset(daily("daily"))
        .build()
}

async fn one_child_grainset() -> SemanticManifest {
    let model = metric_vocabulary()
        .grainset(declared_grainset())
        .build()
        .expect("the fixture model must build")
        .0;
    compile_model_on(model, &fixture_fs(&["daily"]).await).await
}

#[tokio::test]
async fn a_one_child_grainset_is_stored_as_its_child_under_its_name() {
    let manifest = one_child_grainset().await;

    assert_eq!(names(&manifest), vec!["ads"]);
    let stored = by_name(&manifest, "ads");
    assert!(stored.public);
    assert!(
        matches!(stored.role_definition, RoleDefinition::Dataset { .. }),
        "{:?}",
        stored.role_definition
    );
}

#[tokio::test]
async fn a_one_branch_unionset_of_mode_all_is_stored_as_its_branch_under_its_name() {
    let unionset = Unionset::builder("ads")
        .dimension(DimensionSite::r#ref("region"))
        .dimension(DimensionSite::r#ref("date"))
        .measure(MeasureSite::r#ref("revenue"))
        .union_all()
        .dataset(constituent("only", &NAMES))
        .build();
    let model = vocabulary()
        .unionset(unionset)
        .build()
        .expect("the fixture model must build")
        .0;
    let manifest = compile_model_on(model, &fixture_fs(&["only"]).await).await;

    assert_eq!(names(&manifest), vec!["ads"]);
    assert!(matches!(
        by_name(&manifest, "ads").role_definition,
        RoleDefinition::Dataset { .. }
    ));
}

#[tokio::test]
async fn the_composition_s_dimensions_and_measures_are_on_the_stored_child() {
    let manifest = one_child_grainset().await;

    for name in ["region", "date", "revenue"] {
        assert!(has_signature(&manifest, "ads", name), "`ads` lacks {name}");
    }
}

#[tokio::test]
async fn the_composition_s_metric_is_on_the_stored_child() {
    let manifest = one_child_grainset().await;

    assert!(has_signature(&manifest, "ads", "doubled"));
}

#[tokio::test]
async fn the_composition_s_named_filter_is_on_the_stored_child() {
    let manifest = one_child_grainset().await;

    assert!(has_signature(&manifest, "ads", "eu_only"));
}

#[tokio::test]
async fn the_composition_s_key_is_on_the_stored_child() {
    let manifest = one_child_grainset().await;

    assert!(
        !signature(&manifest, "ads", "region").keysets.is_empty(),
        "the key dimension carries its keyset"
    );
    assert!(!manifest.keysets.is_empty());
}

#[tokio::test]
async fn the_composition_s_named_keyset_is_on_the_stored_child() {
    let grainset = Grainset::builder("ads")
        .dimension(
            DimensionSite::r#ref("region")
                .with_keysets(["by_region"])
                .expect("a keyset"),
        )
        .dimension(DimensionSite::r#ref("date"))
        .measure(MeasureSite::r#ref("revenue"))
        .dataset(daily("daily"))
        .build();
    let model = vocabulary()
        .grainset(grainset)
        .build()
        .expect("the fixture model must build")
        .0;
    let manifest = compile_model_on(model, &fixture_fs(&["daily"]).await).await;

    let keyset = manifest
        .keysets
        .iter()
        .find_map(|(id, keyset)| (keyset.name.as_str() == "by_region").then_some(id))
        .expect("the manifest holds the keyset `by_region`");
    assert!(signature(&manifest, "ads", "region")
        .keysets
        .contains(keyset));
}

#[tokio::test]
async fn the_composition_s_temporal_shape_is_on_the_stored_child() {
    let unionset = Unionset::builder("ads")
        .dimension(DimensionSite::r#ref("region"))
        .dimension(DimensionSite::r#ref("date"))
        .measure(MeasureSite::r#ref("revenue"))
        .temporal(TemporalShape::events("date", None))
        .union_all()
        .dataset(constituent("only", &NAMES))
        .build();
    let model = vocabulary()
        .unionset(unionset)
        .build()
        .expect("the fixture model must build")
        .0;
    let manifest = compile_model_on(model, &fixture_fs(&["only"]).await).await;

    assert!(by_name(&manifest, "ads").temporal.is_some());
}

#[tokio::test]
async fn a_nested_one_child_grainset_takes_its_name_under_its_parent() {
    let inner = NestedGrainset::builder("ads")
        .dataset(daily("daily"))
        .build();
    let unionset = Unionset::builder("all_ads")
        .dimension(DimensionSite::r#ref("region"))
        .dimension(DimensionSite::r#ref("date"))
        .measure(MeasureSite::r#ref("revenue"))
        .union_all()
        .dataset(daily("search"))
        .grainset(inner)
        .build();
    let model = vocabulary()
        .unionset(unionset)
        .build()
        .expect("the fixture model must build")
        .0;
    let manifest = compile_model_on(model, &fixture_fs(&["daily", "search"]).await).await;

    assert_eq!(
        names(&manifest),
        vec!["all_ads", "all_ads.ads", "all_ads.search"]
    );
}

#[tokio::test]
async fn a_one_child_composition_over_a_nested_composition_stores_that_composition() {
    let inner = fine_and_coarse("grains");
    let unionset = Unionset::builder("ads")
        .extras(ComplexExtras::default())
        .dimension(DimensionSite::r#ref("region"))
        .dimension(DimensionSite::r#ref("date"))
        .measure(MeasureSite::r#ref("revenue"))
        .union_all()
        .grainset(inner)
        .build();
    let model = vocabulary()
        .unionset(unionset)
        .build()
        .expect("the fixture model must build")
        .0;
    let manifest = compile_model_on(model, &fixture_fs(&["fine", "coarse"]).await).await;

    assert!(matches!(
        by_name(&manifest, "ads").role_definition,
        RoleDefinition::Grainset { .. }
    ));
    assert_eq!(names(&manifest), vec!["ads", "ads.coarse", "ads.fine"]);
}

#[tokio::test]
async fn the_child_s_granularity_survives_a_composition_that_states_none() {
    let grainset = Grainset::builder("ads")
        .extras(ComplexExtras::default().with_temporal(TemporalShape::events("date", None)))
        .dimension(DimensionSite::r#ref("region"))
        .dimension(DimensionSite::r#ref("date"))
        .measure(MeasureSite::r#ref("revenue"))
        .dataset(daily("daily"))
        .build();
    let model = vocabulary()
        .grainset(grainset)
        .build()
        .expect("the fixture model must build")
        .0;
    let manifest = compile_model_on(model, &fixture_fs(&["daily"]).await).await;

    let temporal = by_name(&manifest, "ads")
        .temporal
        .as_ref()
        .expect("a temporal shape");
    assert_eq!(temporal.granularity, Some(Granularity::Day));
}

#[tokio::test]
async fn a_chain_of_one_child_compositions_is_stored_as_its_leaf() {
    let inner = NestedGrainset::builder("adwords")
        .dataset(daily("daily"))
        .build();
    let unionset = Unionset::builder("sources")
        .dimension(DimensionSite::r#ref("region"))
        .dimension(DimensionSite::r#ref("date"))
        .measure(MeasureSite::r#ref("revenue"))
        .union_all()
        .grainset(inner)
        .build();
    let model = vocabulary()
        .unionset(unionset)
        .build()
        .expect("the fixture model must build")
        .0;
    let manifest = compile_model_on(model, &fixture_fs(&["daily"]).await).await;

    assert_eq!(names(&manifest), vec!["sources"]);
    assert!(matches!(
        by_name(&manifest, "sources").role_definition,
        RoleDefinition::Dataset { .. }
    ));
}

fn month_under_an_inner_unionset(stated: Option<Granularity>) -> SemanticModel {
    let inner = NestedUnionset::builder("inner")
        .temporal(TemporalShape::events("date", stated))
        .union_all()
        .dataset(dated("monthly", &NAMES, Granularity::Month))
        .build();
    let grainset = Grainset::builder("all_ads")
        .dimension(DimensionSite::r#ref("region"))
        .dimension(DimensionSite::r#ref("date"))
        .measure(MeasureSite::r#ref("revenue"))
        .dataset(daily("other"))
        .unionset(inner)
        .build();
    vocabulary()
        .grainset(grainset)
        .build()
        .expect("the fixture model must build")
        .0
}

#[tokio::test]
async fn a_one_branch_unionset_stating_a_finer_granularity_than_its_branch_keeps_the_branch_s_granularity(
) {
    let fs = fixture_fs(&["monthly", "other"]).await;

    let manifest =
        compile_model_on(month_under_an_inner_unionset(Some(Granularity::Day)), &fs).await;

    let temporal = by_name(&manifest, "all_ads.inner")
        .temporal
        .as_ref()
        .expect("a temporal shape");
    assert_eq!(temporal.granularity, Some(Granularity::Month));
}

#[tokio::test]
async fn a_composition_stating_a_coarser_granularity_than_its_child_keeps_the_child_s_granularity()
{
    let inner = NestedUnionset::builder("inner")
        .temporal(TemporalShape::events("date", Some(Granularity::Month)))
        .union_all()
        .dataset(daily("fine"))
        .build();
    let grainset = Grainset::builder("all_ads")
        .dimension(DimensionSite::r#ref("region"))
        .dimension(DimensionSite::r#ref("date"))
        .measure(MeasureSite::r#ref("revenue"))
        .dataset(dated("yearly", &NAMES, Granularity::Year))
        .unionset(inner)
        .build();
    let model = vocabulary()
        .grainset(grainset)
        .build()
        .expect("the fixture model must build")
        .0;
    let manifest = compile_model_on(model, &fixture_fs(&["fine", "yearly"]).await).await;

    let temporal = by_name(&manifest, "all_ads.inner")
        .temporal
        .as_ref()
        .expect("a temporal shape");
    assert_eq!(temporal.granularity, Some(Granularity::Day));
}

#[tokio::test]
async fn a_composition_stating_another_shape_kind_than_its_child_keeps_the_child_s_shape() {
    let child = NestedDataset::builder("daily")
        .path(lake_folder("daily"))
        .format(StorageFormat::Parquet)
        .semantic_mapping(identity_mapping(&NAMES))
        .temporal(TemporalShape::timeseries("date", Some(Granularity::Day)))
        .build();
    let grainset = Grainset::builder("ads")
        .extras(ComplexExtras::default().with_temporal(TemporalShape::events("date", None)))
        .dimension(DimensionSite::r#ref("region"))
        .dimension(DimensionSite::r#ref("date"))
        .measure(MeasureSite::r#ref("revenue"))
        .dataset(child)
        .build();
    let model = vocabulary()
        .grainset(grainset)
        .build()
        .expect("the fixture model must build")
        .0;
    let manifest = compile_model_on(model, &fixture_fs(&["daily"]).await).await;

    let date = manifest
        .semantic_definitions
        .by_name("date")
        .expect("`date` is registered")
        .id
        .clone();
    assert_eq!(
        by_name(&manifest, "ads").temporal,
        Some(IrShape::new(
            TemporalType::timeseries(date),
            Some(Granularity::Day)
        ))
    );
}

#[tokio::test]
async fn a_one_branch_unionset_over_a_joinset_leaves_the_joinset_s_own_check_to_refuse() {
    let relationship = Relationship::builder()
        .name("a_to_b")
        .from("a")
        .to("b")
        .on(["region"])
        .many_to_one()
        .build()
        .expect("the fixture relationship must build");
    let joined = NestedJoinset::builder("joined")
        .temporal(TemporalShape::events("date", Some(Granularity::Day)))
        .dataset(dated("a", &NAMES, Granularity::Month))
        .dataset(constituent("b", &["region"]))
        .relationship(relationship)
        .build();
    let inner = NestedUnionset::builder("inner")
        .temporal(TemporalShape::events("date", Some(Granularity::Month)))
        .union_all()
        .joinset(joined)
        .build();
    let grainset = Grainset::builder("all_ads")
        .dimension(DimensionSite::r#ref("region"))
        .dimension(DimensionSite::r#ref("date"))
        .measure(MeasureSite::r#ref("revenue"))
        .dataset(daily("other"))
        .unionset(inner)
        .build();
    let model = vocabulary()
        .grainset(grainset)
        .build()
        .expect("the fixture model must build")
        .0;

    let error = compile_error_on(model, &fixture_fs(&["a", "b", "other"]).await).await;

    assert_eq!(
        error,
        CompileError::StatedGranularityNotServed {
            composite: mint_semantic_source_id("all_ads.inner"),
            stated: Granularity::Day,
            constituents: vec![(
                mint_semantic_source_id("all_ads.inner.a"),
                Granularity::Month
            )],
        }
    );
}

#[tokio::test]
async fn a_one_branch_unionset_stating_its_branch_s_granularity_or_none_compiles() {
    for stated in [Some(Granularity::Month), None] {
        let fs = fixture_fs(&["monthly", "other"]).await;

        let manifest = compile_model_on(month_under_an_inner_unionset(stated), &fs).await;

        let temporal = by_name(&manifest, "all_ads.inner")
            .temporal
            .as_ref()
            .expect("a temporal shape");
        assert_eq!(temporal.granularity, Some(Granularity::Month), "{stated:?}");
    }
}

fn fine_and_coarse(name: &str) -> NestedGrainset {
    NestedGrainset::builder(name)
        .dataset(daily("fine"))
        .dataset(dated("coarse", &NAMES, Granularity::Month))
        .build()
}

fn unionset_under_a_one_child_grainset() -> NestedGrainset {
    let inner = NestedUnionset::builder("inner")
        .union_all()
        .dataset(constituent("a", &NAMES))
        .dataset(constituent("b", &NAMES))
        .build();
    NestedGrainset::builder("mid").unionset(inner).build()
}

#[tokio::test]
async fn a_root_chain_of_one_child_compositions_stores_the_inner_unionset_under_the_root_name() {
    let mid = unionset_under_a_one_child_grainset();
    let outer = Unionset::builder("outer")
        .dimension(DimensionSite::r#ref("region"))
        .dimension(DimensionSite::r#ref("date"))
        .measure(MeasureSite::r#ref("revenue"))
        .union_all()
        .grainset(mid)
        .build();
    let model = vocabulary()
        .unionset(outer)
        .build()
        .expect("the fixture model must build")
        .0;

    let manifest = compile_model_on(model, &fixture_fs(&["a", "b"]).await).await;

    assert_eq!(names(&manifest), vec!["outer", "outer.a", "outer.b"]);
    assert!(matches!(
        by_name(&manifest, "outer").role_definition,
        RoleDefinition::Unionset { .. }
    ));
}

#[tokio::test]
async fn a_root_chain_of_one_child_compositions_stores_the_inner_grainset_under_the_root_name() {
    let inner = fine_and_coarse("inner");
    let mid = NestedUnionset::builder("mid")
        .union_all()
        .grainset(inner)
        .build();
    let outer = Grainset::builder("outer")
        .dimension(DimensionSite::r#ref("region"))
        .dimension(DimensionSite::r#ref("date"))
        .measure(MeasureSite::r#ref("revenue"))
        .unionset(mid)
        .build();
    let model = vocabulary()
        .grainset(outer)
        .build()
        .expect("the fixture model must build")
        .0;

    let manifest = compile_model_on(model, &fixture_fs(&["fine", "coarse"]).await).await;

    assert_eq!(
        names(&manifest),
        vec!["outer", "outer.coarse", "outer.fine"]
    );
    assert!(matches!(
        by_name(&manifest, "outer").role_definition,
        RoleDefinition::Grainset { .. }
    ));
}

#[tokio::test]
async fn a_grainset_chain_ending_in_a_unionset_stays_under_a_unionset_parent() {
    let mid = unionset_under_a_one_child_grainset();
    let outer = Unionset::builder("outer")
        .dimension(DimensionSite::r#ref("region"))
        .dimension(DimensionSite::r#ref("date"))
        .measure(MeasureSite::r#ref("revenue"))
        .union_all()
        .dataset(constituent("x", &NAMES))
        .grainset(mid)
        .build();
    let model = vocabulary()
        .unionset(outer)
        .build()
        .expect("the fixture model must build")
        .0;

    let manifest = compile_model_on(model, &fixture_fs(&["a", "b", "x"]).await).await;

    assert_eq!(
        names(&manifest),
        vec![
            "outer",
            "outer.mid",
            "outer.mid.inner",
            "outer.mid.inner.a",
            "outer.mid.inner.b",
            "outer.x"
        ]
    );
    assert!(matches!(
        by_name(&manifest, "outer").role_definition,
        RoleDefinition::Unionset { .. }
    ));
    assert!(matches!(
        by_name(&manifest, "outer.mid").role_definition,
        RoleDefinition::Grainset { .. }
    ));
    assert!(matches!(
        by_name(&manifest, "outer.mid.inner").role_definition,
        RoleDefinition::Unionset { .. }
    ));
}

#[tokio::test]
async fn a_manifest_keys_on_the_authored_model_not_its_one_child_form() {
    let unionset = Unionset::builder("ads")
        .dimension(DimensionSite::r#ref("region"))
        .dimension(DimensionSite::r#ref("date"))
        .measure(MeasureSite::r#ref("revenue"))
        .union_all()
        .dataset(constituent("only", &NAMES))
        .build();
    let model = vocabulary()
        .unionset(unionset)
        .build()
        .expect("the fixture model must build")
        .0;
    let authored = model.fingerprint();
    let manifest = compile_model_on(model, &fixture_fs(&["only"]).await).await;

    assert_eq!(manifest.model_hash, authored);
    assert_eq!(manifest.model_id, mint_model_id(&authored));
}

#[tokio::test]
async fn a_one_branch_unionset_holding_a_grainset_stays_under_a_grainset_parent() {
    let mid = NestedUnionset::builder("mid")
        .temporal(TemporalShape::events("date", Some(Granularity::Month)))
        .union_all()
        .grainset(
            NestedGrainset::builder("inner")
                .temporal(TemporalShape::events("date", Some(Granularity::Month)))
                .dataset(daily("fine"))
                .dataset(dated("coarse", &NAMES, Granularity::Month))
                .build(),
        )
        .build();
    let outer = Grainset::builder("outer")
        .dimension(DimensionSite::r#ref("region"))
        .dimension(DimensionSite::r#ref("date"))
        .measure(MeasureSite::r#ref("revenue"))
        .dataset(daily("x"))
        .unionset(mid)
        .build();
    let model = vocabulary()
        .grainset(outer)
        .build()
        .expect("the fixture model must build")
        .0;

    let manifest = compile_model_on(model, &fixture_fs(&["fine", "coarse", "x"]).await).await;

    assert_eq!(
        names(&manifest),
        vec![
            "outer",
            "outer.mid",
            "outer.mid.inner",
            "outer.mid.inner.coarse",
            "outer.mid.inner.fine",
            "outer.x"
        ]
    );
    assert!(matches!(
        by_name(&manifest, "outer").role_definition,
        RoleDefinition::Grainset { .. }
    ));
    assert!(matches!(
        by_name(&manifest, "outer.mid").role_definition,
        RoleDefinition::Unionset { .. }
    ));
    assert!(matches!(
        by_name(&manifest, "outer.mid.inner").role_definition,
        RoleDefinition::Grainset { .. }
    ));
}

#[tokio::test]
async fn a_one_child_grainset_holding_a_joinset_stays_under_a_joinset_parent() {
    let relationship = |from: &str, to: &str| {
        Relationship::builder()
            .name(format!("{from}_to_{to}"))
            .from(from)
            .to(to)
            .on(["region"])
            .many_to_one()
            .build()
            .expect("the fixture relationship must build")
    };
    let inner = NestedJoinset::builder("inner")
        .dataset(constituent("a", &NAMES))
        .dataset(constituent("b", &["region"]))
        .relationship(relationship("a", "b"))
        .build();
    let mid = NestedGrainset::builder("mid").joinset(inner).build();
    let outer = semstrait_model::Joinset::builder("outer")
        .dimension(DimensionSite::r#ref("region"))
        .dimension(DimensionSite::r#ref("date"))
        .measure(MeasureSite::r#ref("revenue"))
        .dataset(constituent("x", &NAMES))
        .dataset(constituent("y", &["region"]))
        .grainset(mid)
        .relationship(relationship("x", "y"))
        .relationship(relationship("y", "mid"))
        .build();
    let model = vocabulary()
        .joinset(outer)
        .build()
        .expect("the fixture model must build")
        .0;

    let manifest = compile_model_on(model, &fixture_fs(&["a", "b", "x", "y"]).await).await;

    assert_eq!(
        names(&manifest),
        vec![
            "outer",
            "outer.mid",
            "outer.mid.inner",
            "outer.mid.inner.a",
            "outer.mid.inner.b",
            "outer.x",
            "outer.x_to_y",
            "outer.y",
            "outer.y_to_mid"
        ]
    );
    assert!(matches!(
        by_name(&manifest, "outer").role_definition,
        RoleDefinition::Joinset { .. }
    ));
    assert!(matches!(
        by_name(&manifest, "outer.mid").role_definition,
        RoleDefinition::Grainset { .. }
    ));
    assert!(matches!(
        by_name(&manifest, "outer.mid.inner").role_definition,
        RoleDefinition::Joinset { .. }
    ));
}

#[tokio::test]
async fn a_joinset_chain_through_a_one_child_unionset_keeps_its_top_grainset() {
    let relationship = |from: &str, to: &str| {
        Relationship::builder()
            .name(format!("{from}_to_{to}"))
            .from(from)
            .to(to)
            .on(["region"])
            .many_to_one()
            .build()
            .expect("the fixture relationship must build")
    };
    let inner = NestedJoinset::builder("inner")
        .dataset(constituent("a", &NAMES))
        .dataset(constituent("b", &["region"]))
        .relationship(relationship("a", "b"))
        .build();
    let between = NestedUnionset::builder("u1")
        .union_all()
        .joinset(inner)
        .build();
    let mid = NestedGrainset::builder("g1").unionset(between).build();
    let outer = semstrait_model::Joinset::builder("outer")
        .dimension(DimensionSite::r#ref("region"))
        .dimension(DimensionSite::r#ref("date"))
        .measure(MeasureSite::r#ref("revenue"))
        .dataset(constituent("x", &NAMES))
        .dataset(constituent("y", &["region"]))
        .grainset(mid)
        .relationship(relationship("x", "y"))
        .relationship(relationship("y", "g1"))
        .build();
    let model = vocabulary()
        .joinset(outer)
        .build()
        .expect("the fixture model must build")
        .0;

    let manifest = compile_model_on(model, &fixture_fs(&["a", "b", "x", "y"]).await).await;

    assert_eq!(
        names(&manifest),
        vec![
            "outer",
            "outer.g1",
            "outer.g1.u1",
            "outer.g1.u1.a",
            "outer.g1.u1.b",
            "outer.x",
            "outer.x_to_y",
            "outer.y",
            "outer.y_to_g1"
        ]
    );
    assert!(matches!(
        by_name(&manifest, "outer").role_definition,
        RoleDefinition::Joinset { .. }
    ));
    assert!(matches!(
        by_name(&manifest, "outer.g1").role_definition,
        RoleDefinition::Grainset { .. }
    ));
    assert!(matches!(
        by_name(&manifest, "outer.g1.u1").role_definition,
        RoleDefinition::Joinset { .. }
    ));
}

#[tokio::test]
async fn a_nested_one_child_grainset_s_temporal_shape_fills_a_child_with_none() {
    let inner = NestedGrainset::builder("ads")
        .temporal(TemporalShape::events("date", Some(Granularity::Day)))
        .dataset(constituent("undated", &NAMES))
        .build();
    let unionset = Unionset::builder("all_ads")
        .dimension(DimensionSite::r#ref("region"))
        .dimension(DimensionSite::r#ref("date"))
        .measure(MeasureSite::r#ref("revenue"))
        .union_all()
        .dataset(daily("search"))
        .grainset(inner)
        .build();
    let model = vocabulary()
        .unionset(unionset)
        .build()
        .expect("the fixture model must build")
        .0;
    let manifest = compile_model_on(model, &fixture_fs(&["undated", "search"]).await).await;

    let temporal = by_name(&manifest, "all_ads.ads")
        .temporal
        .as_ref()
        .expect("the nested child carries the Grainset's shape");
    assert_eq!(temporal.granularity, Some(Granularity::Day));
}

#[tokio::test]
async fn a_nested_chain_of_one_child_compositions_below_a_staying_parent_is_stored_as_its_leaf() {
    let innermost = NestedUnionset::builder("inner")
        .union_all()
        .dataset(daily("daily"))
        .build();
    let middle = NestedGrainset::builder("chain").unionset(innermost).build();
    let unionset = Unionset::builder("all_ads")
        .dimension(DimensionSite::r#ref("region"))
        .dimension(DimensionSite::r#ref("date"))
        .measure(MeasureSite::r#ref("revenue"))
        .union_all()
        .dataset(daily("search"))
        .grainset(middle)
        .build();
    let model = vocabulary()
        .unionset(unionset)
        .build()
        .expect("the fixture model must build")
        .0;
    let manifest = compile_model_on(model, &fixture_fs(&["daily", "search"]).await).await;

    assert_eq!(
        names(&manifest),
        vec!["all_ads", "all_ads.chain", "all_ads.search"]
    );
    assert!(matches!(
        by_name(&manifest, "all_ads.chain").role_definition,
        RoleDefinition::Dataset { .. }
    ));
}

#[tokio::test]
async fn a_kept_composition_over_constituents_that_disagree_without_a_granularity_is_refused() {
    let inner = NestedUnionset::builder("inner")
        .union_all()
        .dataset(daily("d"))
        .dataset(dated("m", &NAMES, Granularity::Month))
        .build();
    let kept = NestedGrainset::builder("g").unionset(inner).build();
    let outer = Unionset::builder("outer")
        .dimension(DimensionSite::r#ref("region"))
        .dimension(DimensionSite::r#ref("date"))
        .measure(MeasureSite::r#ref("revenue"))
        .union_all()
        .dataset(daily("x"))
        .grainset(kept)
        .build();
    let model = vocabulary()
        .unionset(outer)
        .build()
        .expect("the fixture model must build")
        .0;

    let error = compile_error_on(model, &fixture_fs(&["d", "m", "x"]).await).await;

    assert_eq!(
        error,
        CompileError::ComposedConstituentMissingGranularity {
            composite: mint_semantic_source_id("outer.g"),
            constituent: mint_semantic_source_id("outer.g.inner"),
        }
    );
}

#[tokio::test]
async fn a_kept_composition_under_a_nested_composition_stays_at_every_depth() {
    let inner = NestedUnionset::builder("inner")
        .union_all()
        .dataset(constituent("a", &NAMES))
        .dataset(constituent("b", &NAMES))
        .build();
    let kept = NestedGrainset::builder("g").unionset(inner).build();
    let nested = NestedUnionset::builder("u")
        .union_all()
        .dataset(constituent("c", &NAMES))
        .grainset(kept)
        .build();
    let outer = Grainset::builder("outer")
        .dimension(DimensionSite::r#ref("region"))
        .dimension(DimensionSite::r#ref("date"))
        .measure(MeasureSite::r#ref("revenue"))
        .dataset(daily("x"))
        .unionset(nested)
        .build();
    let model = vocabulary()
        .grainset(outer)
        .build()
        .expect("the fixture model must build")
        .0;

    let manifest = compile_model_on(model, &fixture_fs(&["a", "b", "c", "x"]).await).await;

    assert!(matches!(
        by_name(&manifest, "outer").role_definition,
        RoleDefinition::Grainset { .. }
    ));
    assert!(matches!(
        by_name(&manifest, "outer.u").role_definition,
        RoleDefinition::Unionset { .. }
    ));
    assert!(matches!(
        by_name(&manifest, "outer.u.g").role_definition,
        RoleDefinition::Grainset { .. }
    ));
    assert!(matches!(
        by_name(&manifest, "outer.u.g.inner").role_definition,
        RoleDefinition::Unionset { .. }
    ));
}
