use std::time::UNIX_EPOCH;

use semstrait_common::io::InMemoryFileSystem;
use semstrait_common::Granularity;
use semstrait_common::SemanticsName;
use semstrait_common::{Diagnostic, Diagnostics};
use semstrait_ir::SemanticPlan;
use semstrait_manifest::{compile, ManifestProvenance, SemanticManifest};
use semstrait_model::expr::{dim, lit};
use semstrait_model::{
    Agg, Dataset, Dimension, DimensionSite, Measure, MeasureSite, SemanticModel,
};
use semstrait_planner::{
    Advisory, DimensionModifier, Direction, OrderKey, PlanError, Planner, Request, RequestContext,
    Selection, Timezone, UnknownTimezone, Weekday,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::ExprOperationExt;

async fn sales() -> SemanticManifest {
    let categorical = |name: &str| {
        Dimension::builder(name)
            .data_type(DataType::String)
            .categorical()
            .build()
    };
    let revenue = Measure::builder("revenue")
        .data_type(DataType::Double)
        .agg(Agg::of_name("sum").expect("a catalog aggregate"))
        .build();
    let orders = Dataset::builder("orders")
        .table("warehouse.orders")
        .dimension(DimensionSite::r#ref("channel"))
        .dimension(DimensionSite::r#ref("region"))
        .measure(MeasureSite::r#ref("revenue"))
        .build();
    let (model, _) = SemanticModel::builder()
        .name("sales")
        .dimension(categorical("channel"))
        .dimension(categorical("region"))
        .measure(revenue)
        .dataset(orders)
        .build()
        .expect("the model builds");
    let provenance = ManifestProvenance::new(UNIX_EPOCH, UNIX_EPOCH);
    let (manifest, _) = compile(model, None, &InMemoryFileSystem::new(), provenance)
        .await
        .expect("the model compiles");
    manifest
}

#[test]
fn every_part_of_a_request_and_its_context_reads_back_as_set() {
    let select = vec![
        Selection::Field("channel".into()),
        Selection::Dimension {
            name: "ordered_at".into(),
            modifier: Some(DimensionModifier::Granularity(Granularity::Month)),
        },
        Selection::Measure {
            name: "revenue".into(),
        },
        Selection::Metric {
            name: "region_label".into(),
        },
    ];
    let predicate = dim("region").expect("a name").eq(lit("eu"));
    let order_by = vec![
        OrderKey::new("revenue", Direction::Desc),
        OrderKey::new("channel", Direction::Asc),
    ];

    let request = Request::new(select.clone())
        .with_from("orders")
        .with_predicate(predicate.clone())
        .with_filters(vec!["web_only".into()])
        .with_order_by(order_by.clone())
        .with_limit(10)
        .with_offset(20);

    assert_eq!(request.from, Some("orders".into()));
    assert_eq!(request.select, select);
    assert_eq!(request.predicate, Some(predicate));
    assert_eq!(request.filters, vec![SemanticsName::from("web_only")]);
    assert_eq!(request.order_by, order_by);
    assert_eq!(request.order_by[0].name, "revenue".into());
    assert_eq!(request.order_by[0].direction, Direction::Desc);
    assert_eq!((request.limit, request.offset), (Some(10), Some(20)));

    let default = RequestContext::default();
    assert_eq!(default.timezone.name(), "UTC");
    assert_eq!(default.week_start, Weekday::Monday);
    let set = default
        .with_timezone(Timezone::new("America/New_York").expect("an IANA name"))
        .with_week_start(Weekday::Sunday);
    assert_eq!(set.timezone.name(), "America/New_York");
    assert_eq!(set.week_start, Weekday::Sunday);
}

#[test]
fn a_known_timezone_builds_a_context_and_an_unknown_one_is_refused() {
    let stockholm = Timezone::new("Europe/Stockholm").expect("an IANA name");
    let context = RequestContext::default().with_timezone(stockholm);
    assert_eq!(context.timezone.name(), "Europe/Stockholm");

    assert!(matches!(
        Timezone::new("Mars/Olympus_Mons"),
        Err(UnknownTimezone { name, .. }) if name == "Mars/Olympus_Mons"
    ));
}

#[tokio::test]
async fn a_caller_names_the_result_type_and_matches_each_kind_with_a_wildcard() {
    let manifest = sales().await;
    let request = Request::new(vec![Selection::Field("channel".into())]);

    #[allow(
        clippy::type_complexity,
        reason = "the result type, as ADR 0064 spells it"
    )]
    let result: Result<
        (SemanticPlan, Diagnostics<Advisory>),
        (Diagnostic<PlanError>, Diagnostics<Advisory>),
    > = Planner::new().plan(&manifest, &request, &RequestContext::default());

    let (refusal, advisories) = match result {
        Ok((_, advisories)) => (None, advisories),
        Err((error, advisories)) => (Some(error.kind), advisories),
    };
    let refused = refusal.map(|kind| match kind {
        PlanError::GrainNotServable { .. } => "a missed grain",
        _ => "another refusal",
    });
    let advised: Vec<&str> = (advisories.iter())
        .map(|advisory| match advisory.kind {
            Advisory::AmbiguousCandidates { .. } => "a tie",
            _ => "another advisory",
        })
        .collect();
    assert_ne!(refused, Some("a missed grain"));
    assert!(!advised.contains(&"a tie"));
}
