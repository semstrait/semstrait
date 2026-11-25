use std::fmt;

use serde::de::{DeserializeOwned, DeserializeSeed, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{json, Value};

mod common;

use common::function;
use semstrait_common::{Dialect, ValidateError};
use semstrait_sql::datatype::{DataType, TypeFamily};
use semstrait_sql::expr::{
    AggregateAllowed, AggregateCall, Call, CallClause, CallError, CastFailure, Expr,
    ExprOperationExt, FunctionRef, JsonForm, JsonFormError, JsonReader, Leaf, LikeKind, Literal,
    NamedReference, NullOrdering, NullTreatment, OptionSetting, Over, ScalarCall, SortDir, SortKey,
    WindowBound, WindowCall, WindowFrame, WindowFrameKind,
};
use semstrait_sql::functions::{
    registry, ArgumentForm, ClauseRule, ExactType, ParameterDecl, ParameterType, Presence,
    Signature,
};
use semstrait_sql::DocumentDialect;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
struct Id(String);

impl Leaf for Id {}

fn column(parts: &[&str]) -> Expr<Id> {
    Expr::Column(NamedReference::new(parts.iter().copied()).expect("one part or more"))
}

fn amount() -> Expr<Id> {
    column(&["ps0", "amount"])
}

fn amount_form() -> Value {
    json!({"type": "column", "content": ["ps0", "amount"]})
}

fn bigint(value: i64) -> Expr<Id> {
    Expr::Literal(Literal::BigInt(value))
}

fn bigint_form(value: i64) -> Value {
    json!({"type": "literal", "content": {"bigint": value}})
}

fn string(value: &str) -> Expr<Id> {
    Expr::Literal(Literal::String(value.into()))
}

fn string_form(value: &str) -> Value {
    json!({"type": "literal", "content": {"string": value}})
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
enum Slot {
    Keyed {
        key: String,
    },
    Typed {
        #[serde(rename = "type")]
        kind: String,
    },
}

impl Leaf for Slot {}

fn id(id: &str) -> Expr<Id> {
    Expr::Leaf(Id(id.into()))
}

fn call(name: &str, args: Vec<Expr<Id>>) -> Expr<Id> {
    let no_over = Over {
        partition_by: Vec::new(),
        order_by: Vec::new(),
        window_frame: None,
    };
    match function(name) {
        FunctionRef::Scalar(_) => scalar(name, args, Vec::new()),
        FunctionRef::Aggregate(..) => aggregate(name, args, false, None, Vec::new(), None),
        FunctionRef::Window(_) => window(name, args, None, no_over),
        other => panic!("no function type of the catalog: {other:?}"),
    }
}

fn scalar(name: &str, args: Vec<Expr<Id>>, options: Vec<OptionSetting>) -> Expr<Id> {
    let FunctionRef::Scalar(function) = function(name) else {
        panic!("`{name}` is no scalar function");
    };
    let call = ScalarCall::builder(function, args).options(options).build();
    Expr::Function(Call::Scalar(call.expect("fits")))
}

fn aggregate(
    name: &str,
    args: Vec<Expr<Id>>,
    distinct: bool,
    filter: Option<Expr<Id>>,
    order_by: Vec<SortKey<Id, AggregateAllowed>>,
    nulls: Option<NullTreatment>,
) -> Expr<Id> {
    let FunctionRef::Aggregate(function, aggregation) = function(name) else {
        panic!("`{name}` is no aggregate function");
    };
    let call = AggregateCall::builder(function, args)
        .distinct(distinct)
        .maybe_filter(filter)
        .order_by(order_by)
        .maybe_nulls(nulls)
        .build();
    Expr::Function(Call::Aggregate(call.expect("fits"), aggregation))
}

fn window(
    name: &str,
    args: Vec<Expr<Id>>,
    nulls: Option<NullTreatment>,
    over: Over<Id, AggregateAllowed>,
) -> Expr<Id> {
    let FunctionRef::Window(function) = function(name) else {
        panic!("`{name}` is no window function");
    };
    let call = WindowCall::builder(function, args)
        .maybe_nulls(nulls)
        .over(over)
        .build();
    Expr::Function(Call::Window(call.expect("fits")))
}

fn write<L: Leaf + Serialize>(expr: &Expr<L>) -> Value {
    serde_json::to_value(JsonForm::new(expr, DocumentDialect::Canonical)).expect("writes")
}

fn read<L: Leaf + DeserializeOwned>(form: Value) -> Result<Expr<L>, serde_json::Error> {
    JsonReader::new(DocumentDialect::Canonical).deserialize(form)
}

fn read_text<L: Leaf + DeserializeOwned>(text: &str) -> Result<Expr<L>, serde_json::Error> {
    JsonReader::new(DocumentDialect::Canonical)
        .deserialize(&mut serde_json::Deserializer::from_str(text))
}

#[track_caller]
fn round_trip<L: Leaf + Serialize + DeserializeOwned>(expr: &Expr<L>, form: Value) {
    assert_eq!(write(expr), form);
    reads_back(expr);
}

#[track_caller]
fn reads_back<L: Leaf + Serialize + DeserializeOwned>(expr: &Expr<L>) {
    let form = write(expr);
    let text = form.to_string();
    let reversed = reversed(form.clone());
    for text in [&text, &reversed.to_string()] {
        assert_eq!(
            &read_text::<L>(text).expect("its text reads"),
            expr,
            "{text}"
        );
    }
    let from_reversed = read::<L>(reversed);
    assert_eq!(
        &from_reversed.expect("its keys read in any order"),
        expr,
        "{text}"
    );
    assert_eq!(&read::<L>(form).expect("its value reads"), expr, "{text}");
}

fn reversed(form: Value) -> Value {
    match form {
        Value::Object(fields) => (fields.into_iter().rev())
            .map(|(key, value)| (key, reversed(value)))
            .collect(),
        Value::Array(items) => items.into_iter().map(reversed).collect(),
        other => other,
    }
}

#[test]
fn a_column_is_its_tag_and_its_name_path() {
    round_trip(&amount(), amount_form());
}

#[test]
fn an_operation_node_is_its_tag_and_its_own_fields() {
    round_trip(
        &(column(&["ps0", "revenue"]) - column(&["ps0", "cost"])),
        json!({"type": "binary_op", "content": {
            "op": "subtract",
            "left": {"type": "column", "content": ["ps0", "revenue"]},
            "right": {"type": "column", "content": ["ps0", "cost"]},
        }}),
    );
}

#[test]
fn an_aggregate_call_s_tag_is_its_function_type_and_its_body_its_name_and_arguments() {
    round_trip(
        &call("sum", vec![amount()]),
        json!({"type": "aggregate", "content": {"name": "sum", "args": [amount_form()]}}),
    );
}

#[test]
fn a_scalar_call_writes_its_name_arguments_and_options_and_no_field_at_its_default() {
    let name = || column(&["ps0", "name"]);
    let name_form = json!({"type": "column", "content": ["ps0", "name"]});
    round_trip(
        &call("upper", vec![name()]),
        json!({"type": "scalar", "content": {"name": "upper", "args": [name_form]}}),
    );
    round_trip(
        &call("current_date", Vec::new()),
        json!({"type": "scalar", "content": {"name": "current_date"}}),
    );
    round_trip(
        &call("concat", vec![name(), name()]),
        json!({"type": "scalar", "content": {"name": "concat", "args": [name_form, name_form]}}),
    );
    let ignore_nulls = OptionSetting {
        name: "null_handling".into(),
        value: "IGNORE_NULLS".into(),
    };
    round_trip(
        &scalar("concat", vec![name(), name()], vec![ignore_nulls]),
        json!({"type": "scalar", "content": {
            "name": "concat",
            "args": [name_form, name_form],
            "options": {"null_handling": "IGNORE_NULLS"},
        }}),
    );
    let accept_nulls = OptionSetting {
        name: "null_handling".into(),
        value: "ACCEPT_NULLS".into(),
    };
    let at_canonical = scalar("concat", vec![name(), name()], vec![accept_nulls]);
    assert_eq!(at_canonical, call("concat", vec![name(), name()]));
    round_trip(
        &at_canonical,
        json!({"type": "scalar", "content": {"name": "concat", "args": [name_form, name_form]}}),
    );
}

#[test]
fn a_call_built_with_the_full_check_reads_back_as_an_equal_call() {
    let upper = Call::new(function("upper"), vec![column(&["ps0", "name"])]);
    reads_back(&Expr::Function(upper.expect("upper takes one string")));
}

fn key(expr: Expr<Id>, dir: SortDir) -> SortKey<Id, AggregateAllowed> {
    SortKey { expr, dir }
}

#[test]
fn an_aggregate_call_writes_each_clause_that_it_holds() {
    let paid = || column(&["ps0", "paid"]);
    let paid_form = json!({"type": "column", "content": ["ps0", "paid"]});
    let asc_last = SortDir::Asc {
        nulls: NullOrdering::Last,
    };
    round_trip(
        &aggregate(
            "count",
            vec![amount()],
            true,
            Some(paid()),
            Vec::new(),
            None,
        ),
        json!({"type": "aggregate", "content": {
            "name": "count", "args": [amount_form()], "distinct": true, "filter": paid_form,
        }}),
    );
    let separator = || Expr::Literal(Literal::String(",".into()));
    round_trip(
        &aggregate(
            "string_agg",
            vec![column(&["ps0", "name"]), separator()],
            false,
            None,
            vec![key(amount(), asc_last)],
            None,
        ),
        json!({"type": "aggregate", "content": {
            "name": "string_agg",
            "args": [
                {"type": "column", "content": ["ps0", "name"]},
                {"type": "literal", "content": {"string": ","}},
            ],
            "order_by": [{"expr": amount_form(), "dir": {"asc": {"nulls": "last"}}}],
        }}),
    );
    round_trip(
        &aggregate(
            "percentile_cont",
            vec![Expr::Literal(Literal::Double(0.5))],
            false,
            None,
            vec![key(amount(), asc_last)],
            None,
        ),
        json!({"type": "aggregate", "content": {
            "name": "percentile_cont",
            "args": [{"type": "literal", "content": {"double": "0.5"}}],
            "order_by": [{"expr": amount_form(), "dir": {"asc": {"nulls": "last"}}}],
        }}),
    );
}

#[test]
fn a_window_call_writes_its_null_treatment_and_its_over_with_no_part_at_its_default() {
    let day = || column(&["ps0", "day"]);
    let day_form = json!({"type": "column", "content": ["ps0", "day"]});
    let by_day = || {
        vec![key(
            day(),
            SortDir::Asc {
                nulls: NullOrdering::First,
            },
        )]
    };
    let by_day_form = json!([{"expr": day_form, "dir": {"asc": {"nulls": "first"}}}]);
    round_trip(
        &window(
            "lag",
            vec![amount()],
            Some(NullTreatment::IgnoreNulls),
            Over {
                partition_by: vec![column(&["ps0", "channel"])],
                order_by: by_day(),
                window_frame: None,
            },
        ),
        json!({"type": "window", "content": {
            "name": "lag",
            "args": [amount_form()],
            "nulls": "ignore_nulls",
            "over": {
                "partition_by": [{"type": "column", "content": ["ps0", "channel"]}],
                "order_by": by_day_form,
            },
        }}),
    );
    round_trip(
        &window(
            "first_value",
            vec![amount()],
            None,
            Over {
                partition_by: Vec::new(),
                order_by: by_day(),
                window_frame: Some(WindowFrame::new(
                    WindowFrameKind::Rows,
                    WindowBound::UnboundedPreceding,
                    WindowBound::CurrentRow,
                )),
            },
        ),
        json!({"type": "window", "content": {
            "name": "first_value",
            "args": [amount_form()],
            "over": {
                "order_by": by_day_form,
                "window_frame": {
                    "kind": "rows",
                    "start": {"type": "unbounded_preceding"},
                    "end": {"type": "current_row"},
                },
            },
        }}),
    );
    round_trip(
        &call("row_number", Vec::new()),
        json!({"type": "window", "content": {"name": "row_number"}}),
    );
}

#[test]
fn a_data_type_is_its_canonical_name_and_unknown_is_refused_both_ways() {
    let cast = |target| Expr::Cast {
        input: Box::new(amount()),
        target,
        on_failure: CastFailure::Error,
    };
    let cast_form = |target: &str| {
        json!({"type": "cast", "content": {
            "input": amount_form(), "target": target, "on_failure": "error",
        }})
    };
    round_trip(&cast(DataType::BigInt), cast_form("bigint"));
    let unknown = cast(DataType::Unknown);
    assert!(serde_json::to_value(JsonForm::new(&unknown, DocumentDialect::Canonical)).is_err());
    assert!(read::<Id>(cast_form("unknown")).is_err());
    let null: Expr<Id> = Expr::Literal(Literal::Null(DataType::Unknown));
    assert!(serde_json::to_value(JsonForm::new(&null, DocumentDialect::Canonical)).is_err());
    assert!(read::<Id>(json!({"type": "literal", "content": {"null": "unknown"}})).is_err());
}

#[test]
fn a_call_s_name_is_its_declaration_s_name_in_the_document_s_dialect() {
    let datafusion = DocumentDialect::Dialect(Dialect::DataFusion);
    let in_datafusion = |expr: &Expr<Id>| {
        serde_json::to_value(JsonForm::new(expr, datafusion)).map_err(|error| error.to_string())
    };
    let read_in = |dialect, form: Value| {
        JsonReader::<Id>::new(dialect)
            .deserialize(form)
            .map_err(|error| error.to_string())
    };
    let approx = call("approx_count_distinct", vec![id("#1")]);
    let form = json!({"type": "aggregate", "content": {"name": "approx_distinct", "args": ["#1"]}});
    assert_eq!(in_datafusion(&approx), Ok(form.clone()));
    assert_eq!(read_in(datafusion, form.clone()), Ok(approx));
    let unknown = |name: &str, dialect| {
        Err(JsonFormError::UnknownFunction {
            name: name.into(),
            dialect,
        }
        .to_string())
    };
    assert_eq!(
        read_in(DocumentDialect::Canonical, form),
        unknown("approx_distinct", DocumentDialect::Canonical)
    );
    assert_eq!(
        read_in(
            datafusion,
            json!({"type": "aggregate", "content": {"name": "approx_count_distinct", "args": ["#1"]}}),
        ),
        unknown("approx_count_distinct", datafusion)
    );
    let group = Expr::Literal(Literal::Integer(1));
    let extract = call("regexp_extract", vec![id("#1"), id("#2"), group]);
    assert_eq!(
        in_datafusion(&extract),
        Err(JsonFormError::NoNameInDialect {
            function: "regexp_extract",
            dialect: datafusion,
        }
        .to_string())
    );
}

#[test]
fn a_document_s_dialect_is_written_and_read_by_its_name() {
    for (dialect, name) in [
        (DocumentDialect::Canonical, "canonical"),
        (DocumentDialect::Dialect(Dialect::DataFusion), "datafusion"),
        (DocumentDialect::Dialect(Dialect::Substrait), "substrait"),
    ] {
        assert_eq!(serde_json::to_value(dialect).ok(), Some(json!(name)));
        assert_eq!(serde_json::from_value(json!(name)).ok(), Some(dialect));
        assert_eq!(dialect.to_string(), name);
    }
    assert!(serde_json::from_value::<DocumentDialect>(json!("Canonical")).is_err());
}

fn simplest(function: FunctionRef<AggregateAllowed>) -> Expr<Id> {
    let args = |signature: &Signature| {
        let required = |parameter: &&ParameterDecl| parameter.presence == Presence::Required;
        (signature.parameters.iter().chain(&signature.variadic))
            .filter(required)
            .enumerate()
            .map(
                |(index, parameter)| match (&parameter.form, &parameter.ty) {
                    (ArgumentForm::Value, _) => id(&format!("#{index}")),
                    (ArgumentForm::Enum(values), _) => string(values[0]),
                    (
                        ArgumentForm::Constant,
                        ParameterType::Exact(ExactType::Type(DataType::Integer)),
                    ) => Expr::Literal(Literal::Integer(1)),
                    (
                        ArgumentForm::Constant,
                        ParameterType::Exact(ExactType::Type(DataType::Double)),
                    ) => Expr::Literal(Literal::Double(0.5)),
                    (ArgumentForm::Constant, ParameterType::Family(TypeFamily::String)) => {
                        string(",")
                    }
                    (form, ty) => panic!("no argument of the form {form:?} of {ty:?} here"),
                },
            )
            .collect()
    };
    let required = |rule: ClauseRule| rule == ClauseRule::Required;
    let dir = SortDir::Asc {
        nulls: NullOrdering::Unspecified,
    };
    let keys = |written: bool| written.then(|| key(id("#key"), dir)).into_iter().collect();
    let nulls = |rule| required(rule).then_some(NullTreatment::RespectNulls);
    match function {
        FunctionRef::Scalar(function) => {
            let declaration = &function.canonical;
            scalar(declaration.name, args(&declaration.signature), Vec::new())
        }
        FunctionRef::Aggregate(function, _) => {
            let declaration = &function.declarations.canonical;
            let clauses = &declaration.specific.clauses;
            aggregate(
                declaration.name,
                args(&declaration.signature),
                required(clauses.distinct),
                required(clauses.filter).then(|| id("#filter")),
                keys(required(clauses.order_by) || required(clauses.within_group)),
                nulls(clauses.nulls),
            )
        }
        FunctionRef::Window(function) => {
            let declaration = &function.canonical;
            let clauses = &declaration.specific.clauses;
            let over = Over {
                partition_by: (required(clauses.partition_by).then(|| id("#partition")))
                    .into_iter()
                    .collect(),
                order_by: keys(required(clauses.order_by)),
                window_frame: required(clauses.window_frame).then(|| {
                    WindowFrame::new(
                        WindowFrameKind::Rows,
                        WindowBound::UnboundedPreceding,
                        WindowBound::CurrentRow,
                    )
                }),
            };
            let args = args(&declaration.signature);
            window(declaration.name, args, nulls(clauses.nulls), over)
        }
        other => panic!("no function type of the catalog: {other:?}"),
    }
}

#[test]
fn every_catalog_function_s_simplest_call_reads_back() {
    for function in registry().all() {
        reads_back(&simplest(function));
    }
}

#[track_caller]
fn refused(form: Value, error: impl std::fmt::Display) {
    let refusal = read::<Id>(form).expect_err("refused");
    assert_eq!(refusal.to_string(), error.to_string());
}

#[test]
fn reading_refuses_a_call_name_that_no_function_has_in_the_document_s_dialect() {
    let unknown = |name: &str| JsonFormError::UnknownFunction {
        name: name.into(),
        dialect: DocumentDialect::Canonical,
    };
    refused(
        json!({"type": "scalar", "content": {"name": "frobnicate", "args": ["#1"]}}),
        unknown("frobnicate"),
    );
    refused(
        json!({"type": "scalar", "content": {"name": "ceiling", "args": ["#1"]}}),
        unknown("ceiling"),
    );
}

#[test]
fn reading_refuses_a_tag_that_is_not_the_declaration_s_function_type() {
    let cases = [
        ("scalar", "sum"),
        ("aggregate", "upper"),
        ("window", "sum"),
        ("aggregate", "row_number"),
    ];
    for (tag, name) in cases {
        refused(
            json!({"type": tag, "content": {"name": name, "args": ["#1"]}}),
            JsonFormError::NotItsFunctionType {
                name: name.into(),
                tag,
            },
        );
    }
}

#[test]
fn reading_refuses_a_case_with_no_when_and_an_in_with_no_item() {
    refused(
        json!({"type": "case", "content": {"whens": []}}),
        ValidateError::EmptyCase,
    );
    refused(
        json!({"type": "in_list", "content": {"value": "#1", "list": [], "negated": false}}),
        ValidateError::EmptyInList,
    );
}

#[test]
fn reading_refuses_a_key_written_twice() {
    let cases = [
        (
            r##"{"type": "is_null", "type": "is_null", "content": "#1"}"##,
            "duplicate field `type`".to_owned(),
        ),
        (
            r##"{"type": "is_null", "content": "#1", "content": "#2"}"##,
            "duplicate field `content`".to_owned(),
        ),
        (
            r##"{"type": "binary_op", "content": {
                "op": "subtract", "op": "add", "left": "#1", "right": "#2"}}"##,
            "duplicate field `op`".to_owned(),
        ),
        (
            r##"{"type": "scalar", "content": {"name": "concat", "args": ["#1", "#2"],
                "options": {"null_handling": "IGNORE_NULLS", "null_handling": "ACCEPT_NULLS"}}}"##,
            JsonFormError::RepeatedOption {
                name: "null_handling".into(),
            }
            .to_string(),
        ),
    ];
    for (text, refusal) in cases {
        let error = read_text::<Id>(text).expect_err(text).to_string();
        assert!(error.starts_with(&refusal), "{text}: {error}");
    }
}

#[test]
fn reading_refuses_a_wrong_number_of_arguments() {
    refused(
        json!({"type": "scalar", "content": {"name": "upper", "args": ["#1", "#2"]}}),
        CallError::TooManyArguments {
            function: "upper",
            got: 2,
        },
    );
    refused(
        json!({"type": "scalar", "content": {"name": "upper"}}),
        CallError::TooFewArguments {
            function: "upper",
            min: 1,
            max: Some(1),
        },
    );
    refused(
        json!({"type": "scalar", "content": {"name": "concat", "args": ["#1"]}}),
        CallError::TooFewArguments {
            function: "concat",
            min: 2,
            max: None,
        },
    );
}

#[test]
fn reading_refuses_a_leaf_at_an_enum_or_a_constant_parameter() {
    refused(
        json!({"type": "scalar", "content": {"name": "round", "args": ["#x", "#2"]}}),
        CallError::NotALiteral {
            function: "round",
            parameter: "scale",
        },
    );
    refused(
        json!({"type": "scalar", "content": {"name": "extract", "args": ["#year", "#x"]}}),
        CallError::NotALiteral {
            function: "extract",
            parameter: "component",
        },
    );
}

#[test]
fn reading_refuses_an_enum_value_that_the_declaration_does_not_list() {
    refused(
        json!({"type": "scalar", "content": {
            "name": "extract", "args": [string_form("yeer"), "#x"],
        }}),
        CallError::EnumValue {
            function: "extract",
            parameter: "component",
            value: Literal::String("yeer".into()),
        },
    );
}

#[test]
fn reading_refuses_a_refused_or_no_effect_clause_and_a_missing_required_one() {
    let key = json!({"expr": "#9", "dir": {"asc": {"nulls": "unspecified"}}});
    let cases = [
        (
            json!({"name": "corr", "args": ["#1", "#2"], "distinct": true}),
            CallError::ClauseRefused {
                function: "corr",
                clause: CallClause::Distinct,
            },
        ),
        (
            json!({"name": "max", "args": ["#1"], "distinct": true}),
            CallError::ClauseRefused {
                function: "max",
                clause: CallClause::Distinct,
            },
        ),
        (
            json!({"name": "sum", "args": ["#1"], "order_by": [key]}),
            CallError::ClauseRefused {
                function: "sum",
                clause: CallClause::OrderBy,
            },
        ),
        (
            json!({
                "name": "percentile_cont",
                "args": [{"type": "literal", "content": {"double": "0.5"}}],
            }),
            CallError::ClauseRequired {
                function: "percentile_cont",
                clause: CallClause::WithinGroup,
            },
        ),
    ];
    for (content, error) in cases {
        refused(json!({"type": "aggregate", "content": content}), error);
    }
    refused(
        json!({"type": "window", "content": {"name": "rank"}}),
        CallError::ClauseRequired {
            function: "rank",
            clause: CallClause::OrderBy,
        },
    );
    refused(
        json!({"type": "window", "content": {
            "name": "rank", "nulls": "ignore_nulls", "over": {"order_by": [key]},
        }}),
        CallError::ClauseRefused {
            function: "rank",
            clause: CallClause::Nulls,
        },
    );
}

#[test]
fn a_call_s_body_holds_only_its_function_type_s_fields() {
    let refusal = read::<Id>(json!({"type": "scalar", "content": {
        "name": "upper", "args": ["#1"], "distinct": true,
    }}))
    .expect_err("a scalar call holds no DISTINCT");
    assert!(
        refusal.to_string().starts_with("unknown field `distinct`"),
        "{refusal}"
    );
    let refusal = read::<Id>(json!({"type": "window", "content": {
        "name": "row_number", "filter": "#1",
    }}))
    .expect_err("a window call holds no FILTER");
    assert!(
        refusal.to_string().starts_with("unknown field `filter`"),
        "{refusal}"
    );
}

#[test]
fn a_leaf_is_written_in_its_own_form_with_no_tag_and_reads_back() {
    round_trip(
        &(id("#2") - id("#4")),
        json!({"type": "binary_op", "content": {"op": "subtract", "left": "#2", "right": "#4"}}),
    );
    round_trip(
        &id("#2").is_null(),
        json!({"type": "is_null", "content": "#2"}),
    );
    let keyed = Slot::Keyed {
        key: "channel".into(),
    };
    let typed = Slot::Typed {
        kind: "partition".into(),
    };
    round_trip(&Expr::Leaf(keyed.clone()), json!({"key": "channel"}));
    round_trip(&Expr::Leaf(typed.clone()), json!({"type": "partition"}));
    round_trip(
        &Expr::InList {
            value: Box::new(Expr::Leaf(keyed)),
            list: vec![Expr::Leaf(typed)],
            negated: false,
        },
        json!({"type": "in_list", "content": {
            "value": {"key": "channel"}, "list": [{"type": "partition"}], "negated": false,
        }}),
    );
}

#[test]
fn each_operation_node_and_a_literal_keep_their_own_tag_and_fields() {
    let cases = [
        (bigint(5), bigint_form(5)),
        (
            !amount(),
            json!({"type": "unary_op", "content": {"op": "not", "operand": amount_form()}}),
        ),
        (
            Expr::Cast {
                input: Box::new(amount()),
                target: DataType::Decimal {
                    precision: 18,
                    scale: 2,
                },
                on_failure: CastFailure::Null,
            },
            json!({"type": "cast", "content": {
                "input": amount_form(), "target": "decimal(18,2)", "on_failure": "null",
            }}),
        ),
        (
            Expr::InList {
                value: Box::new(amount()),
                list: vec![bigint(1), bigint(2)],
                negated: false,
            },
            json!({"type": "in_list", "content": {
                "value": amount_form(), "list": [bigint_form(1), bigint_form(2)], "negated": false,
            }}),
        ),
        (
            Expr::Between {
                value: Box::new(amount()),
                low: Box::new(bigint(1)),
                high: Box::new(bigint(9)),
                negated: true,
            },
            json!({"type": "between", "content": {
                "value": amount_form(), "low": bigint_form(1), "high": bigint_form(9),
                "negated": true,
            }}),
        ),
        (
            Expr::Like {
                value: Box::new(column(&["ps0", "name"])),
                pattern: Box::new(Expr::Literal(Literal::String("a%".into()))),
                kind: LikeKind::ILike,
            },
            json!({"type": "like", "content": {
                "value": {"type": "column", "content": ["ps0", "name"]},
                "pattern": {"type": "literal", "content": {"string": "a%"}},
                "kind": "i_like",
            }}),
        ),
    ];
    for (expr, form) in cases {
        round_trip(&expr, form);
    }
}

fn case(whens: Vec<(Expr<Id>, Expr<Id>)>, else_: Option<Expr<Id>>) -> Expr<Id> {
    Expr::Case {
        whens,
        else_: else_.map(Box::new),
    }
}

#[test]
fn a_case_is_its_whens_with_each_then_and_its_else() {
    round_trip(
        &case(
            vec![(amount().gt(bigint(1)), string("a"))],
            Some(string("b")),
        ),
        json!({"type": "case", "content": {
            "whens": [{
                "when": {"type": "binary_op", "content": {
                    "op": "gt", "left": amount_form(), "right": bigint_form(1),
                }},
                "then": string_form("a"),
            }],
            "else": string_form("b"),
        }}),
    );
}

#[test]
fn a_case_with_no_else_writes_no_else() {
    round_trip(
        &case(vec![(amount().is_null(), string("a"))], None),
        json!({"type": "case", "content": {"whens": [{
            "when": {"type": "is_null", "content": amount_form()},
            "then": string_form("a"),
        }]}}),
    );
}

#[test]
fn a_case_s_else_reads_from_a_literal_form() {
    let form = json!({"type": "case", "content": {
        "whens": [{"when": "#1", "then": "#2"}],
        "else": "#3",
    }});
    assert_eq!(
        read::<Id>(form).expect("reads"),
        case(vec![(id("#1"), id("#2"))], Some(id("#3")))
    );
}

#[test]
fn a_case_nests_in_a_when_a_then_and_an_else() {
    let country = || column(&["ps0", "country"]);
    let spain = case(
        vec![(
            column(&["ps0", "region"]).eq(string("Canarias")),
            string("IC"),
        )],
        Some(string("ES")),
    );
    let market = case(
        vec![
            (country().eq(string("Germany")), string("DE")),
            (country().eq(string("Spain")), spain),
        ],
        Some(string("OTHER")),
    );
    reads_back(&market);
    let known = case(
        vec![(country().is_null(), Expr::Literal(Literal::Boolean(false)))],
        Some(Expr::Literal(Literal::Boolean(true))),
    );
    reads_back(&case(
        vec![(known, market)],
        Some(case(vec![(id("#1"), id("#2"))], None)),
    ));
}

#[test]
fn reading_refuses_a_case_in_another_shape() {
    let cases = [
        (json!({"whens": [["#1", "#2"]]}), "invalid type: sequence"),
        (
            json!({"whens": [{"when": "#1", "then": "#2"}], "else_": "#3"}),
            "unknown field `else_`",
        ),
        (
            json!({"whens": [{"when": "#1", "then": "#2", "else": "#3"}]}),
            "unknown field `else`",
        ),
    ];
    for (content, refusal) in cases {
        let read = read::<Id>(json!({"type": "case", "content": content}));
        let error = read.expect_err(&content.to_string()).to_string();
        assert!(error.starts_with(refusal), "{content}: {error}");
    }
}

#[test]
fn a_node_reads_whatever_the_order_of_its_type_and_its_content() {
    assert_eq!(
        read::<Id>(reversed(amount_form())).expect("reads"),
        amount()
    );
    let text = r##"{"content": {"name": "sum", "args": ["#3"]}, "type": "aggregate"}"##;
    assert_eq!(read_text(text).expect("reads"), call("sum", vec![id("#3")]));
    let dir = json!({"asc": {"nulls": "last"}, "junk": 1});
    let content = json!({
        "name": "string_agg",
        "args": ["#1", string_form(",")],
        "order_by": [{"expr": "#3", "dir": dir}],
    });
    for form in [
        json!({"type": "aggregate", "content": content}),
        json!({"content": content, "type": "aggregate"}),
    ] {
        read::<Id>(form.clone()).expect_err(&form.to_string());
    }
}

#[test]
fn a_link_leaf_reads_back_whatever_the_order_of_its_node_s_keys() {
    let text = r#"{"content":{"right":"09712050-49b8-534f-98f1-34d64cf76e3c","left":"15184182-69e5-55a9-9292-b09b5731f128","op":"subtract"},"type":"binary_op"}"#;
    let subtraction =
        id("15184182-69e5-55a9-9292-b09b5731f128") - id("09712050-49b8-534f-98f1-34d64cf76e3c");
    assert_eq!(read_text::<Id>(text).expect("reads"), subtraction);
    reads_back(&subtraction);
}

#[test]
fn the_form_reads_through_any_serde_deserializer() {
    use serde::de::value::{Error, MapDeserializer, StrDeserializer};
    let reader = JsonReader::<Id>::new(DocumentDialect::Canonical);
    assert_eq!(
        reader.deserialize(StrDeserializer::<Error>::new("#1")),
        Ok(id("#1"))
    );
    for entries in [
        [("type", "is_null"), ("content", "#1")],
        [("content", "#1"), ("type", "is_null")],
    ] {
        let node = reader.deserialize(MapDeserializer::<_, Error>::new(entries.into_iter()));
        assert_eq!(node, Ok(id("#1").is_null()));
    }
}

struct Trees;

impl<'de> Visitor<'de> for Trees {
    type Value = Vec<Expr<Id>>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a list of trees")
    }

    fn visit_seq<S: SeqAccess<'de>>(self, mut seq: S) -> Result<Self::Value, S::Error> {
        let reader = JsonReader::new(DocumentDialect::Canonical);
        let mut trees = Vec::new();
        while let Some(tree) = seq.next_element_seed(reader)? {
            trees.push(tree);
        }
        Ok(trees)
    }
}

#[test]
fn a_refusal_in_a_document_names_a_line_of_the_document() {
    let streamed = r##"[
        "#1",
        {"type": "binary_op", "content": {"op": "frob", "left": "#1", "right": "#2"}}]"##;
    let held = r##"[
        "#1",
        {"content": {"op": "frob", "left": "#1", "right": "#2"},
         "type": "binary_op"}]"##;
    for (text, place) in [(streamed, (3, 54)), (held, (4, 29))] {
        let mut document = serde_json::Deserializer::from_str(text);
        let refusal = document
            .deserialize_seq(Trees)
            .expect_err("`frob` is no operator");
        assert_eq!((refusal.line(), refusal.column()), place, "{refusal}");
    }
}
