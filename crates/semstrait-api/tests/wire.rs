use std::fmt::Debug;

use semstrait_api::{
    ContextPart, Key, Order, RequestPart, Selected, SemstraitError, WeekStart, WireError,
};
use semstrait_common::Granularity;
use semstrait_common::{Diagnose, Diagnostic, Severity};
use semstrait_model::model::serde::parse_expr;
use semstrait_planner::{
    DimensionModifier, Direction, OrderKey, Request, RequestContext, Selection, Timezone, Weekday,
};
use serde::de::DeserializeOwned;
use serde::Serialize;

type Reader = fn(&str) -> Result<(Request, RequestContext), WireError>;

const READERS: [Reader; 2] = [RequestPart::read_yaml, RequestPart::read_json];

const DOCUMENT_YAML: &str = "
from: orders
select:
  - region
  - {dimension: order_date, granularity: month}
  - {measure: revenue}
  - {metric: margin}
where: {eq: [channel, {lit: web}]}
filters: [paid_only]
order_by: [{name: revenue, direction: desc}]
limit: 100
offset: 0
context: {timezone: Europe/Stockholm, week_start: sunday}
";

const DOCUMENT_JSON: &str = r#"{
  "from": "orders",
  "select": [
    "region",
    {"dimension": "order_date", "granularity": "month"},
    {"measure": "revenue"},
    {"metric": "margin"}
  ],
  "where": {"eq": ["channel", {"lit": "web"}]},
  "filters": ["paid_only"],
  "order_by": [{"name": "revenue", "direction": "desc"}],
  "limit": 100,
  "offset": 0,
  "context": {"timezone": "Europe/Stockholm", "week_start": "sunday"}
}"#;

const REFUSED_WHERE: &str = r#"{"select": ["region"], "where": {"nope": ["channel"]}}"#;

const UNKNOWN_TIMEZONE: &str = r#"{"select": ["region"], "context": {"timezone": "Mars/Olympus"}}"#;

#[test]
fn each_part_of_a_yaml_and_a_json_document_maps_to_what_the_planners_constructors_build() {
    let predicate = serde_yaml::from_str("{eq: [channel, {lit: web}]}").expect("YAML");
    let request = Request::new(vec![
        Selection::Field("region".into()),
        Selection::Dimension {
            name: "order_date".into(),
            modifier: Some(DimensionModifier::Granularity(Granularity::Month)),
        },
        Selection::Measure {
            name: "revenue".into(),
        },
        Selection::Metric {
            name: "margin".into(),
        },
    ])
    .with_from("orders")
    .with_predicate(parse_expr(&predicate).expect("the model's expression form"))
    .with_filters(vec!["paid_only".into()])
    .with_order_by(vec![OrderKey::new("revenue", Direction::Desc)])
    .with_limit(100)
    .with_offset(0);
    let context = RequestContext::default()
        .with_timezone(Timezone::new("Europe/Stockholm").expect("an IANA timezone"))
        .with_week_start(Weekday::Sunday);

    assert_eq!(
        RequestPart::read_yaml(DOCUMENT_YAML),
        Ok((request.clone(), context))
    );
    assert_eq!(
        RequestPart::read_json(DOCUMENT_JSON),
        Ok((request, context))
    );
}

#[test]
fn a_where_of_each_value_kind_gives_the_request_that_with_predicate_builds_from_it() {
    for expression in ["\"channel\"", "true", "7", "-7", "2.5", r#"{"lit": "web"}"#] {
        let predicate = serde_yaml::from_str(expression).expect("YAML");
        let request = Request::new(vec![Selection::Field("region".into())])
            .with_predicate(parse_expr(&predicate).expect("the model's expression form"));
        let document = format!(r#"{{"select": ["region"], "where": {expression}}}"#);

        for read in READERS {
            let (planned, _) = read(&document).expect("the document reads");
            assert_eq!(planned, request);
        }
    }
}

#[test]
fn a_document_with_no_context_is_planned_in_utc_with_weeks_from_monday() {
    for read in READERS {
        let (request, context) = read(r#"{"select": ["region"]}"#).expect("the document reads");

        assert_eq!(
            request,
            Request::new(vec![Selection::Field("region".into())])
        );
        assert_eq!(
            (context.timezone.name(), context.week_start),
            ("UTC", Weekday::Monday)
        );
    }
}

fn round_trips<T: Serialize + DeserializeOwned + PartialEq + Debug>(yaml: &str) {
    let value: T = serde_yaml::from_str(yaml).expect("the wire type reads");

    let yaml = serde_yaml::to_string(&value).expect("the wire type writes YAML");
    assert_eq!(serde_yaml::from_str::<T>(&yaml).expect("YAML"), value);
    let json = serde_json::to_string(&value).expect("the wire type writes JSON");
    assert_eq!(serde_json::from_str::<T>(&json).expect("JSON"), value);
}

#[test]
fn each_wire_type_round_trips_through_yaml_and_json() {
    round_trips::<RequestPart>(DOCUMENT_YAML);
    for selected in [
        "region",
        "{dimension: order_date}",
        "{dimension: order_date, granularity: month}",
        "{measure: revenue}",
        "{metric: margin}",
    ] {
        round_trips::<Selected>(selected);
    }
    round_trips::<Key>("{name: revenue, direction: desc}");
    round_trips::<Order>("asc");
    round_trips::<ContextPart>("{timezone: Europe/Stockholm, week_start: sunday}");
    round_trips::<WeekStart>("sunday");
}

#[test]
fn an_unknown_key_is_refused_in_the_document_a_typed_selection_an_order_by_entry_and_context() {
    for (document, named) in [
        (r#"{"select": ["region"], "colour": 1}"#, true),
        (
            r#"{"select": [{"measure": "revenue", "colour": 1}]}"#,
            false,
        ),
        (
            r#"{"select": ["region"], "order_by": [{"name": "region", "direction": "asc", "colour": 1}]}"#,
            true,
        ),
        (r#"{"select": ["region"], "context": {"colour": 1}}"#, true),
    ] {
        for read in READERS {
            let refused = read(document);

            let Err(WireError::Document { message }) = refused else {
                panic!("{document} is refused, not {refused:?}");
            };
            assert!(
                !named || message.contains("unknown field `colour`"),
                "{message}"
            );
        }
    }
}

#[test]
fn a_refused_where_is_a_refused_document_whose_message_holds_the_refusal_and_its_position() {
    let [yaml, json] = READERS;
    for (read, refusal) in [
        (
            yaml,
            "where: unknown reserved tag: `nope` at line 1 column 33",
        ),
        (json, "unknown reserved tag: `nope` at line 1 column 53"),
    ] {
        let kind = WireError::Document {
            message: refusal.into(),
        };
        assert_eq!(read(REFUSED_WHERE), Err(kind.clone()));
        let diagnostic = Diagnostic::new(SemstraitError::Wire(kind));

        assert_eq!(diagnostic.severity, Severity::Error);
        assert_eq!(
            diagnostic.kind.message(),
            format!("the Request document is refused: {refusal}")
        );
        assert!(diagnostic.kind.cause().is_none());
    }
}

#[test]
fn an_unknown_timezone_gives_the_wire_kind_whose_cause_is_the_refusal() {
    let unknown_timezone = Timezone::new("Mars/Olympus").expect_err("no IANA timezone");
    let kind = WireError::Timezone(unknown_timezone.clone());
    for read in READERS {
        assert_eq!(read(UNKNOWN_TIMEZONE), Err(kind.clone()));
    }
    let diagnostic = Diagnostic::new(SemstraitError::Wire(kind));

    assert_eq!(diagnostic.severity, Severity::Error);
    assert_eq!(diagnostic.kind.message(), unknown_timezone.to_string());
    let carried = diagnostic.kind.cause().map(ToString::to_string);
    assert_eq!(carried, Some(unknown_timezone.to_string()));
}
