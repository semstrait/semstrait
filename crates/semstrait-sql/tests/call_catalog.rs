use semstrait_sql::expr::{
    AggregateAllowed, AggregateCall, Call, CallClause, CallError, Expr, Literal, NamedReference,
    NoLeaf, NullTreatment, WindowCall,
};
mod common;

use common::{aggregate_function, ascending, function, scalar, window_function};

type Tree = Expr;

fn col(name: &str) -> Tree {
    Expr::Column(NamedReference::new([name]).expect("one part"))
}

fn half() -> Tree {
    Expr::Literal(Literal::Double(0.5))
}

fn aggregate(
    name: &str,
    args: Vec<Tree>,
    distinct: bool,
    order_by: Vec<Tree>,
) -> Result<AggregateCall<NoLeaf, AggregateAllowed>, CallError> {
    AggregateCall::builder(aggregate_function(name), args)
        .distinct(distinct)
        .order_by(order_by.into_iter().map(ascending).collect())
        .build()
}

#[test]
fn count_distinct_is_kept_because_count_s_distinct_is_allowed() {
    let call = aggregate("count", vec![col("x")], true, Vec::new());
    assert!(call.is_ok_and(|call| call.distinct()));
}

#[test]
fn max_distinct_is_refused_by_its_constructor_because_max_s_distinct_has_no_effect() {
    assert_eq!(
        aggregate("max", vec![col("x")], true, Vec::new()),
        Err(CallError::ClauseRefused {
            function: "max",
            clause: CallClause::Distinct,
        })
    );
}

#[test]
fn percentile_cont_is_kept_with_its_required_within_group_and_refused_without_it() {
    let kept = aggregate("percentile_cont", vec![half()], false, vec![col("price")]);
    assert!(kept.is_ok_and(|call| call.order_by().len() == 1));
    assert_eq!(
        Call::new(function("percentile_cont"), vec![half()]),
        Err(CallError::ClauseRequired {
            function: "percentile_cont",
            clause: CallClause::WithinGroup,
        })
    );
}

#[test]
fn a_refused_clause_is_refused_corr_distinct_and_row_number_ignore_nulls() {
    assert_eq!(
        aggregate("corr", vec![col("y"), col("x")], true, Vec::new()),
        Err(CallError::ClauseRefused {
            function: "corr",
            clause: CallClause::Distinct,
        })
    );
    let row_number =
        WindowCall::<NoLeaf, AggregateAllowed>::builder(window_function("row_number"), Vec::new())
            .nulls(NullTreatment::IgnoreNulls)
            .build();
    assert_eq!(
        row_number,
        Err(CallError::ClauseRefused {
            function: "row_number",
            clause: CallClause::Nulls,
        })
    );
}

#[test]
fn date_diff_takes_second_and_the_parts_below_it() {
    for part in ["second", "millisecond", "microsecond", "nanosecond"] {
        let part_arg = Expr::Literal(Literal::String(part.to_owned()));
        let args = vec![part_arg, col("started_at"), col("ended_at")];
        assert_eq!(Call::new(function("date_diff"), args).err(), None, "{part}");
    }
}

#[test]
fn greatest_and_least_take_one_argument_or_more_and_coalesce_concat_and_concat_ws_two_or_more() {
    for (name, fewest) in [
        ("greatest", 1),
        ("least", 1),
        ("coalesce", 2),
        ("concat", 2),
        ("concat_ws", 2),
    ] {
        let call = |count| {
            Call::new(
                function(name),
                ["a", "b"][..count].iter().copied().map(col).collect(),
            )
        };
        assert_eq!(call(fewest).err(), None, "{name}");
        assert_eq!(
            call(fewest - 1),
            Err(CallError::TooFewArguments {
                function: name,
                min: fewest,
                max: None,
            }),
            "{name}"
        );
    }
}

#[test]
fn call_new_refuses_corr_string_agg_and_percentile_cont_of_one_value() {
    let of_value = |name| Call::new(function(name), vec![col("value")]);
    let cases = [
        (
            "corr",
            CallError::TooFewArguments {
                function: "corr",
                min: 2,
                max: Some(2),
            },
        ),
        (
            "string_agg",
            CallError::TooFewArguments {
                function: "string_agg",
                min: 2,
                max: Some(2),
            },
        ),
        (
            "percentile_cont",
            CallError::NotALiteral {
                function: "percentile_cont",
                parameter: "fraction",
            },
        ),
    ];
    for (name, error) in cases {
        assert_eq!(of_value(name), Err(error), "{name}");
    }
}

fn text(value: &str) -> Tree {
    Expr::Literal(Literal::String(value.to_owned()))
}

#[test]
fn date_trunc_declares_its_timezone_utc_and_its_week_start_monday() {
    let defaults: Vec<_> = (scalar("date_trunc").canonical.signature.parameters.iter())
        .map(|parameter| (parameter.name, parameter.default()))
        .collect();
    let default = |value: &str| Some(Literal::String(value.to_owned()));
    assert_eq!(
        defaults,
        [
            ("granularity", None),
            ("value", None),
            ("timezone", default("UTC")),
            ("week_start", default("monday")),
        ]
    );
}

#[test]
fn a_week_bucket_that_leaves_its_week_start_out_takes_monday() {
    let date_trunc = function("date_trunc");
    let monday = vec![text("week"), col("d"), text("monday")];
    assert_eq!(
        date_trunc.arguments_by_position(vec![text("week"), col("d")]),
        monday
    );
    let named = |args: [(&'static str, Tree); 2]| date_trunc.arguments_by_name(args.into());
    assert_eq!(
        named([("granularity", text("week")), ("value", col("d"))]),
        Ok(monday)
    );
    let month = vec![text("month"), col("d")];
    assert_eq!(date_trunc.arguments_by_position(month.clone()), month);
    assert_eq!(
        date_trunc.arguments_by_name(
            [
                ("granularity", text("month")),
                ("value", col("d")),
                ("week_start", text("monday")),
            ]
            .into()
        ),
        Err(CallError::UnwrittenParameter {
            function: "date_trunc",
            parameter: "week_start",
        })
    );
}
