use serde::de::DeserializeSeed;
use serde::Deserialize;
use serde_json::{json, Value};

use semstrait_sql::expr::{
    AggregateAllowed, CallError, Expr, JsonReader, Leaf, NamedReference, Over, OverError,
    WindowBound, WindowCall, WindowFrame, WindowFrameKind,
};
use semstrait_sql::DocumentDialect;
use WindowBound::{CurrentRow, Following, Preceding, UnboundedFollowing, UnboundedPreceding};
use WindowFrameKind::{Groups, Range, Rows};

mod common;

use common::{ascending, window_function};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Id(String);

impl Leaf for Id {}

fn day() -> Expr<Id> {
    Expr::Column(NamedReference::new(["ps0", "day"]).expect("two parts"))
}

fn day_form() -> Value {
    json!({"type": "column", "content": ["ps0", "day"]})
}

fn refused() -> [(WindowFrame, usize, OverError); 9] {
    [
        (
            WindowFrame::new(Rows, UnboundedFollowing, UnboundedFollowing),
            1,
            OverError::StartsUnboundedFollowing,
        ),
        (
            WindowFrame::new(Rows, UnboundedPreceding, UnboundedPreceding),
            1,
            OverError::EndsUnboundedPreceding,
        ),
        (
            WindowFrame::new(Rows, CurrentRow, Preceding(1)),
            1,
            OverError::EndsBeforeStart,
        ),
        (
            WindowFrame::new(Rows, Following(1), Preceding(1)),
            1,
            OverError::EndsBeforeStart,
        ),
        (
            WindowFrame::new(Rows, Following(1), CurrentRow),
            1,
            OverError::EndsBeforeStart,
        ),
        (
            WindowFrame::new(Range, Preceding(5), CurrentRow),
            0,
            OverError::RangeOffsetKeys { keys: 0 },
        ),
        (
            WindowFrame::new(Range, CurrentRow, Following(5)),
            2,
            OverError::RangeOffsetKeys { keys: 2 },
        ),
        (
            WindowFrame::new(Groups, UnboundedPreceding, UnboundedFollowing),
            0,
            OverError::GroupsWithoutOrderBy,
        ),
        (
            WindowFrame::new(Groups, Preceding(1), Following(1)),
            0,
            OverError::GroupsWithoutOrderBy,
        ),
    ]
}

fn built() -> [(WindowFrame, usize); 8] {
    [
        (WindowFrame::new(Rows, Following(5), Following(2)), 1),
        (WindowFrame::new(Rows, Preceding(7), Preceding(8)), 1),
        (WindowFrame::new(Rows, UnboundedPreceding, CurrentRow), 0),
        (WindowFrame::new(Range, UnboundedPreceding, CurrentRow), 0),
        (WindowFrame::new(Range, CurrentRow, UnboundedFollowing), 2),
        (WindowFrame::new(Range, Preceding(5), Following(5)), 1),
        (WindowFrame::new(Groups, Preceding(1), Following(1)), 1),
        (WindowFrame::new(Groups, CurrentRow, UnboundedFollowing), 2),
    ]
}

fn first_value(
    window_frame: WindowFrame,
    keys: usize,
) -> Result<WindowCall<Id, AggregateAllowed>, CallError> {
    let over = Over {
        partition_by: Vec::new(),
        order_by: (0..keys).map(|_| ascending(day())).collect(),
        window_frame: Some(window_frame),
    };
    WindowCall::builder(window_function("first_value"), vec![day()])
        .over(over)
        .build()
}

fn first_value_form(window_frame: WindowFrame, keys: usize) -> Value {
    let key = json!({"expr": day_form(), "dir": {"asc": {"nulls": "unspecified"}}});
    let window_frame = serde_json::to_value(window_frame).expect("a window frame writes");
    json!({"type": "window", "content": {
        "name": "first_value",
        "args": [day_form()],
        "over": {"order_by": vec![key; keys], "window_frame": window_frame},
    }})
}

fn read(form: Value) -> Result<Expr<Id>, serde_json::Error> {
    JsonReader::new(DocumentDialect::Canonical).deserialize(form)
}

fn over_refused(error: OverError) -> CallError {
    CallError::Over {
        function: "first_value",
        error,
    }
}

#[test]
fn the_window_call_s_constructor_refuses_each_window_frame_that_sql_refuses() {
    for (window_frame, keys, error) in refused() {
        assert_eq!(
            first_value(window_frame, keys),
            Err(over_refused(error)),
            "{window_frame:?} over {keys} keys"
        );
    }
}

#[test]
fn reading_refuses_each_window_frame_that_sql_refuses() {
    for (window_frame, keys, error) in refused() {
        let refusal = read(first_value_form(window_frame, keys)).expect_err("refused");
        assert_eq!(
            refusal.to_string(),
            over_refused(error).to_string(),
            "{window_frame:?} over {keys} keys"
        );
    }
}

#[test]
fn reading_refuses_a_window_frame_with_an_unknown_key() {
    let mut form = first_value_form(WindowFrame::new(Rows, UnboundedPreceding, CurrentRow), 1);
    form["content"]["over"]["window_frame"]["exclude"] = json!("ties");
    let refusal = read(form).expect_err("refused").to_string();
    assert!(refusal.starts_with("unknown field `exclude`"), "{refusal}");
}

#[test]
fn the_window_call_s_constructor_builds_each_window_frame_that_sql_allows() {
    for (window_frame, keys) in built() {
        let call = first_value(window_frame, keys);
        assert!(call.is_ok(), "{window_frame:?} over {keys} keys: {call:?}");
        let read = read(first_value_form(window_frame, keys));
        assert!(read.is_ok(), "{window_frame:?} over {keys} keys: {read:?}");
    }
}
