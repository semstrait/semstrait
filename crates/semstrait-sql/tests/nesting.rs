use serde::de::DeserializeSeed;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

mod common;

use common::{aggregate_function, ascending, function, window_function};
use semstrait_common::{Tree, ValidateError};
use semstrait_sql::expr::{
    AggregateAllowed, AggregateCall, Call, CallError, Expr, ExprOperationExt, JsonForm, JsonReader,
    Leaf, Literal, NamedReference, Over, WindowCall,
};
use semstrait_sql::DocumentDialect;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Id(String);

impl Leaf for Id {}

struct Case {
    sql: &'static str,
    outer: fn(Expr<Id>) -> Result<Expr<Id>, CallError>,
    outer_form: fn(Value) -> Value,
    nested: Expr<Id>,
    refusal: Option<CallError>,
}

fn column(name: &str) -> Expr<Id> {
    Expr::Column(NamedReference::new([name]).expect("one part"))
}

fn x() -> Expr<Id> {
    column("x")
}

fn x_form() -> Value {
    json!({"type": "column", "content": ["x"]})
}

fn call(name: &str, args: Vec<Expr<Id>>) -> Result<Expr<Id>, CallError> {
    Call::new(function(name), args).map(Expr::Function)
}

fn of_x(name: &str) -> Expr<Id> {
    call(name, vec![x()]).expect("a call of one column builds")
}

fn call_form(tag: &str, name: &str, arg: Value) -> Value {
    json!({"type": tag, "content": {"name": name, "args": [arg]}})
}

fn aggregate(call: AggregateCall<Id, AggregateAllowed>) -> Expr<Id> {
    Expr::Function(Call::Aggregate(call, AggregateAllowed))
}

fn first_value_over(over: Over<Id, AggregateAllowed>) -> Result<Expr<Id>, CallError> {
    WindowCall::builder(window_function("first_value"), vec![x()])
        .over(over)
        .build()
        .map(|call| Expr::Function(Call::Window(call)))
}

fn key_form(expr: Value) -> Value {
    json!({"expr": expr, "dir": {"asc": {"nulls": "unspecified"}}})
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            sql: "sum(sum(x))",
            outer: |arg| call("sum", vec![arg]),
            outer_form: |arg| call_form("aggregate", "sum", arg),
            nested: of_x("sum"),
            refusal: Some(CallError::AggregateInAggregate {
                function: "sum",
                nested: "sum",
            }),
        },
        Case {
            sql: "sum(1 + sum(x))",
            outer: |arg| call("sum", vec![arg]),
            outer_form: |arg| call_form("aggregate", "sum", arg),
            nested: Expr::Literal(Literal::BigInt(1)) + of_x("sum"),
            refusal: Some(CallError::AggregateInAggregate {
                function: "sum",
                nested: "sum",
            }),
        },
        Case {
            sql: "count(x) FILTER (WHERE sum(x) > 0)",
            outer: |filter| {
                AggregateCall::builder(aggregate_function("count"), vec![x()])
                    .filter(filter)
                    .build()
                    .map(aggregate)
            },
            outer_form: |filter| {
                json!({"type": "aggregate", "content": {
                    "name": "count", "args": [x_form()], "filter": filter,
                }})
            },
            nested: of_x("sum").gt(Expr::Literal(Literal::BigInt(0))),
            refusal: Some(CallError::AggregateInAggregate {
                function: "count",
                nested: "sum",
            }),
        },
        Case {
            sql: "sum(first_value(x))",
            outer: |arg| call("sum", vec![arg]),
            outer_form: |arg| call_form("aggregate", "sum", arg),
            nested: of_x("first_value"),
            refusal: Some(CallError::WindowInAggregate {
                function: "sum",
                nested: "first_value",
            }),
        },
        Case {
            sql: "first_value(first_value(x))",
            outer: |arg| call("first_value", vec![arg]),
            outer_form: |arg| call_form("window", "first_value", arg),
            nested: of_x("first_value"),
            refusal: Some(CallError::WindowInWindow {
                function: "first_value",
                nested: "first_value",
            }),
        },
        Case {
            sql: "first_value(sum(x))",
            outer: |arg| call("first_value", vec![arg]),
            outer_form: |arg| call_form("window", "first_value", arg),
            nested: of_x("sum"),
            refusal: None,
        },
        Case {
            sql: "string_agg(x, ',' ORDER BY sum(x))",
            outer: |key| {
                let separator = Expr::Literal(Literal::String(",".to_owned()));
                AggregateCall::builder(aggregate_function("string_agg"), vec![x(), separator])
                    .order_by(vec![ascending(key)])
                    .build()
                    .map(aggregate)
            },
            outer_form: |key| {
                let separator = json!({"type": "literal", "content": {"string": ","}});
                json!({"type": "aggregate", "content": {
                    "name": "string_agg",
                    "args": [x_form(), separator],
                    "order_by": [key_form(key)],
                }})
            },
            nested: of_x("sum"),
            refusal: Some(CallError::AggregateInAggregate {
                function: "string_agg",
                nested: "sum",
            }),
        },
        Case {
            sql: "sum(abs(first_value(x)))",
            outer: |arg| call("sum", vec![arg]),
            outer_form: |arg| call_form("aggregate", "sum", arg),
            nested: call("abs", vec![of_x("first_value")]).expect("abs of a value builds"),
            refusal: Some(CallError::WindowInAggregate {
                function: "sum",
                nested: "first_value",
            }),
        },
        Case {
            sql: "first_value(x) OVER (PARTITION BY first_value(x))",
            outer: |key| {
                first_value_over(Over {
                    partition_by: vec![key],
                    ..Over::default()
                })
            },
            outer_form: |key| {
                json!({"type": "window", "content": {
                    "name": "first_value",
                    "args": [x_form()],
                    "over": {"partition_by": [key]},
                }})
            },
            nested: of_x("first_value"),
            refusal: Some(CallError::WindowInWindow {
                function: "first_value",
                nested: "first_value",
            }),
        },
        Case {
            sql: "first_value(x) OVER (ORDER BY sum(x))",
            outer: |key| {
                first_value_over(Over {
                    order_by: vec![ascending(key)],
                    ..Over::default()
                })
            },
            outer_form: |key| {
                json!({"type": "window", "content": {
                    "name": "first_value",
                    "args": [x_form()],
                    "over": {"order_by": [key_form(key)]},
                }})
            },
            nested: of_x("sum"),
            refusal: None,
        },
    ]
}

fn slot() -> Expr<Id> {
    column("slot")
}

#[test]
fn the_call_s_constructor_refuses_each_nesting_of_the_rule_and_builds_the_others() {
    for case in cases() {
        let built = (case.outer)(case.nested);
        assert_eq!(built.err(), case.refusal, "{}", case.sql);
    }
}

#[test]
fn with_new_children_refuses_each_nesting_of_the_rule_and_builds_the_others() {
    for case in cases() {
        let outer = (case.outer)(slot()).expect("a column at each place builds");
        let children = (outer.children().into_iter())
            .map(|child| {
                if *child == slot() {
                    case.nested.clone()
                } else {
                    child.clone()
                }
            })
            .collect();
        let expected = match case.refusal {
            Some(refusal) => Err(ValidateError::CallRefused {
                reason: refusal.to_string(),
            }),
            None => Ok((case.outer)(case.nested).expect(case.sql)),
        };
        assert_eq!(outer.with_new_children(children), expected, "{}", case.sql);
    }
}

#[test]
fn reading_refuses_each_nesting_of_the_rule_and_builds_the_others() {
    let reader = JsonReader::<Id>::new(DocumentDialect::Canonical);
    for case in cases() {
        let nested = JsonForm::new(&case.nested, DocumentDialect::Canonical);
        let nested = serde_json::to_value(nested).expect("the nested operand writes");
        let read = reader.deserialize((case.outer_form)(nested));
        match case.refusal {
            Some(refusal) => {
                let read = read.expect_err(case.sql).to_string();
                assert_eq!(read, refusal.to_string(), "{}", case.sql);
            }
            None => {
                let built = (case.outer)(case.nested).expect(case.sql);
                assert_eq!(read.expect(case.sql), built, "{}", case.sql);
            }
        }
    }
}
