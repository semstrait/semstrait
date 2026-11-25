use semstrait_common::io::InMemoryFileSystem;
use semstrait_ir::PhysicalExpr;
use semstrait_manifest::{compile, CompileError};
use semstrait_model::SemanticModel;
use semstrait_sql::datatype::{DataType, TypeFamily};
use semstrait_sql::expr::{Literal, TypeError};
use semstrait_sql::functions::ParameterType;

mod common;
use common::golden::{self, Golden};
use common::*;

type Declared = (&'static str, &'static str, &'static str);

const COLUMNS: &[(&str, &str)] = &[
    ("ts", "timestamp(6)"),
    ("event_timestamp", "timestamp(6)"),
    ("d", "date"),
    ("created_at", "date"),
    ("name", "string"),
];

fn model(dimensions: &[Declared]) -> SemanticModel {
    let declared: String = (dimensions.iter())
        .map(|(name, data_type, expr)| {
            format!(
                "    - name: {name}\n      data_type: {data_type}\n      type: categorical\n      \
                 expr: {expr}\n"
            )
        })
        .collect();
    let sites: String = (dimensions.iter())
        .map(|(name, ..)| format!("        - ref: {name}\n"))
        .collect();
    model_from_yaml(&format!(
        "semantic_model:\n  name: events\n  dimensions:\n{declared}  datasets:\n    - name: \
         events\n      extras:\n        semantic_mapping: auto\n        storage:\n          \
         tables: [warehouse.orders]\n      dimensions:\n{sites}"
    ))
}

async fn stored<const N: usize>(dimensions: [Declared; N]) -> [PhysicalExpr; N] {
    let model = model(&dimensions);
    let names = dimensions.map(|(name, ..)| name);
    stored_with(model, Some(&catalog(COLUMNS)), "events", names).await
}

fn stored_call(function: &str, value_of: (&str, &str), rest: &[&str]) -> PhysicalExpr {
    let (enumerated, value) = value_of;
    let args = [string(enumerated), orders_column(value)];
    physical_call(
        function,
        args.into_iter()
            .chain(rest.iter().map(|arg| string(arg)))
            .collect(),
    )
}

fn date_trunc(grain: &str, value: &str, rest: &[&str]) -> PhysicalExpr {
    stored_call("date_trunc", (grain, value), rest)
}

#[tokio::test]
async fn date_trunc_takes_the_declared_defaults_of_the_parameters_that_it_leaves_out() {
    let stored = stored([
        ("ts_month", "timestamp", "{date_trunc: [month, {col: ts}]}"),
        ("d_week", "date", "{date_trunc: [week, {col: d}]}"),
        (
            "d_sunday_week",
            "date",
            "{date_trunc: [week, {col: d}, {lit: sunday}]}",
        ),
        (
            "ts_week",
            "timestamp",
            "{date_trunc: {granularity: week, value: {col: ts}, timezone: {lit: UTC}}}",
        ),
    ])
    .await;
    assert_eq!(
        stored,
        [
            date_trunc("month", "ts", &["UTC"]),
            date_trunc("week", "d", &["monday"]),
            date_trunc("week", "d", &["sunday"]),
            date_trunc("week", "ts", &["UTC", "monday"]),
        ]
    );
}

#[tokio::test]
async fn an_hour_bucket_of_a_timestamp_takes_utc_and_a_month_bucket_of_a_date_no_timezone() {
    let stored = stored([
        (
            "reading_hour",
            "timestamp",
            "{date_trunc: [{lit: hour}, {col: event_timestamp}]}",
        ),
        (
            "event_month",
            "date",
            "{date_trunc: [{lit: month}, {col: created_at}]}",
        ),
    ])
    .await;
    assert_eq!(
        stored,
        [
            date_trunc("hour", "event_timestamp", &["UTC"]),
            date_trunc("month", "created_at", &[]),
        ]
    );
}

#[tokio::test]
async fn a_week_bucket_starts_on_monday_unless_it_writes_sunday() {
    let stored = stored([
        ("d_week", "date", "{date_trunc: [week, {col: d}]}"),
        ("ts_week", "timestamp", "{date_trunc: [week, {col: ts}]}"),
        (
            "d_sunday_week",
            "date",
            "{date_trunc: {granularity: week, value: {col: d}, week_start: sunday}}",
        ),
        (
            "ts_sunday_week",
            "timestamp",
            "{date_trunc: {granularity: week, value: {col: ts}, week_start: sunday}}",
        ),
    ])
    .await;
    assert_eq!(
        stored,
        [
            date_trunc("week", "d", &["monday"]),
            date_trunc("week", "ts", &["UTC", "monday"]),
            date_trunc("week", "d", &["sunday"]),
            date_trunc("week", "ts", &["UTC", "sunday"]),
        ]
    );
}

#[tokio::test]
async fn extract_gives_a_bigint_component_of_a_date_or_a_timestamp() {
    let stored = stored([
        (
            "event_dow",
            "bigint",
            "{extract: [{lit: day_of_week}, {col: created_at}]}",
        ),
        (
            "reading_minute",
            "bigint",
            "{extract: [{lit: minute}, {col: event_timestamp}]}",
        ),
    ])
    .await;
    assert_eq!(
        stored,
        [
            stored_call("extract", ("day_of_week", "created_at"), &[]),
            stored_call("extract", ("minute", "event_timestamp"), &[]),
        ]
    );
}

fn compile_refusals() -> [(&'static str, Declared, CompileError); 6] {
    let not_utc = |node: &str, zone: &str| CompileError::DateTruncTimezoneNotUtc {
        node: node.to_owned(),
        timezone: Literal::String(zone.to_owned()),
    };
    let type_error = |node: &str, error| CompileError::TypeError {
        node: node.to_owned(),
        error,
    };
    [
        (
            "a_timezone_other_than_utc_is_refused_at_compile",
            (
                "ts_month",
                "timestamp",
                "{date_trunc: [month, {col: ts}, {lit: Europe/Stockholm}]}",
            ),
            not_utc("ts_month", "Europe/Stockholm"),
        ),
        (
            "date_trunc_of_a_string_is_refused_at_compile",
            ("name_month", "date", "{date_trunc: [month, {col: name}]}"),
            type_error(
                "name_month",
                TypeError::Argument {
                    name: "date_trunc",
                    parameter: "value",
                    takes: &ParameterType::Var(Some(&TypeFamily::DateAndTimestamp)),
                    got: DataType::String.into(),
                },
            ),
        ),
        (
            "date_trunc_of_a_date_to_the_hour_is_refused_at_compile",
            ("d_hour", "date", "{date_trunc: [hour, {col: d}]}"),
            type_error(
                "d_hour",
                TypeError::EnumRefused {
                    name: "date_trunc",
                    parameter: "granularity",
                    value: "hour".to_owned(),
                    got: DataType::Date.into(),
                },
            ),
        ),
        (
            "date_trunc_in_new_york_is_refused_at_compile",
            (
                "reading_hour",
                "timestamp",
                "{date_trunc: [hour, {col: event_timestamp}, {lit: America/New_York}]}",
            ),
            not_utc("reading_hour", "America/New_York"),
        ),
        (
            "extract_of_a_string_is_refused_at_compile",
            ("name_year", "bigint", "{extract: [year, {col: name}]}"),
            type_error(
                "name_year",
                TypeError::Argument {
                    name: "extract",
                    parameter: "value",
                    takes: &ParameterType::Family(&TypeFamily::DateAndTimestamp),
                    got: DataType::String.into(),
                },
            ),
        ),
        (
            "extract_of_a_date_s_minute_is_refused_at_compile",
            ("d_minute", "bigint", "{extract: [minute, {col: d}]}"),
            type_error(
                "d_minute",
                TypeError::EnumRefused {
                    name: "extract",
                    parameter: "component",
                    value: "minute".to_owned(),
                    got: DataType::Date.into(),
                },
            ),
        ),
    ]
}

async fn refusal_goldens() -> Vec<Golden> {
    let mut goldens = Vec::new();
    for (case, dimension, expected) in compile_refusals() {
        let fs = InMemoryFileSystem::new();
        let (refusal, _) = compile(
            model(&[dimension]),
            Some(&catalog(COLUMNS)),
            &fs,
            test_provenance(),
        )
        .await
        .expect_err("the model must be refused at compile");
        assert_eq!(refusal.kind, expected, "{case}");
        goldens.push(Golden::message(case, refusal.to_string()));
    }
    goldens
}

#[tokio::test]
async fn each_fault_that_needs_types_is_refused_at_compile() {
    for golden in refusal_goldens().await {
        golden.assert_committed();
    }
}

#[tokio::test]
async fn generate_goldens() {
    golden::generate(refusal_goldens()).await;
}
