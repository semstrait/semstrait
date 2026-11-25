use crate::datatype::{CanonicalTypeSystem, DataType, TypeFamily};
use crate::expr::{
    type_tree, AggregateAllowed, AggregateCall, AggregateInRowExpr, Aggregation, Call, CallClause,
    CallError, Expr, ExprOperationExt, FunctionRef, InputTypes, Leaf, Literal, NamedReference,
    NoAggregate, NullOrdering, NullTreatment, OptionSetting, Over, PhysicalLeaf, RowExpr,
    ScalarCall, SemanticLeaf, SortDir, SortKey, TypeError, Typed, UntypedLiteral, WindowCall,
};
use crate::functions::{
    Additivity, Aggregate, AggregateClauses, AggregateFunction, ArgumentForm, Category, ClauseRule,
    Condition, Declaration, ExactType, Function, Nullability, OptionDecl, ParameterDecl,
    ParameterType, Presence, ReturnRule, Scalar, ScalarOption, Signature, Volatility, Window,
    WindowClauses,
};
use crate::DocumentDialect;
use semstrait_common::{Tree, ValidateError};

const fn parameter(name: &'static str, form: ArgumentForm, presence: Presence) -> ParameterDecl {
    ParameterDecl {
        name,
        ty: ParameterType::Any,
        form,
        presence,
    }
}

const fn value(name: &'static str) -> ParameterDecl {
    parameter(name, ArgumentForm::Value, Presence::Required)
}

const fn signature(parameters: &'static [ParameterDecl]) -> Signature {
    Signature {
        parameters,
        variadic: None,
        constraints: &[],
    }
}

const fn scalar(name: &'static str, parameters: &'static [ParameterDecl]) -> Function<Scalar> {
    Function {
        canonical: Declaration {
            name,
            aliases: &[],
            signature: signature(parameters),
            returns: ReturnRule::Fixed(ExactType::Type(&DataType::String)),
            nullability: Nullability::Mirror,
            volatility: Volatility::Immutable,
            specific: Scalar {
                category: Category::String,
                options: &[],
            },
        },
        dialects: &[],
        releases: &[],
    }
}

const NO_AGGREGATE_CLAUSE: AggregateClauses = AggregateClauses {
    distinct: ClauseRule::Refused,
    filter: ClauseRule::Refused,
    order_by: ClauseRule::Refused,
    within_group: ClauseRule::Refused,
    within_group_key: ParameterType::Any,
    nulls: ClauseRule::Refused,
};

const fn aggregate(
    name: &'static str,
    parameters: &'static [ParameterDecl],
    clauses: AggregateClauses,
) -> AggregateFunction {
    AggregateFunction {
        declarations: Function {
            canonical: Declaration {
                name,
                aliases: &[],
                signature: signature(parameters),
                returns: ReturnRule::Fixed(ExactType::Type(&DataType::BigInt)),
                nullability: Nullability::EmptyGroupNull,
                volatility: Volatility::Immutable,
                specific: Aggregate {
                    options: &[],
                    clauses,
                },
            },
            dialects: &[],
            releases: &[],
        },
        additivity: Additivity::NonAdditive,
        distinct_additivity: Additivity::NonAdditive,
    }
}

static UPPER: Function<Scalar> = scalar(
    "upper",
    &[ParameterDecl {
        ty: ParameterType::Family(&TypeFamily::String),
        ..value("value")
    }],
);

static CONCAT: Function<Scalar> = Function {
    canonical: Declaration {
        specific: Scalar {
            category: Category::String,
            options: &[OptionDecl {
                name: ScalarOption::NullHandling,
                values: &["IGNORE_NULLS", "ACCEPT_NULLS"],
                canonical: "ACCEPT_NULLS",
            }],
        },
        ..scalar("concat", &[value("value")]).canonical
    },
    dialects: &[],
    releases: &[],
};

static REGEXP_EXTRACT: Function<Scalar> = scalar(
    "regexp_extract",
    &[
        value("value"),
        parameter("pattern", ArgumentForm::Constant, Presence::Required),
        parameter("group", ArgumentForm::Constant, Presence::Optional),
    ],
);

const COMPONENT: ParameterDecl = parameter(
    "component",
    ArgumentForm::Enum(&["year", "month", "day"]),
    Presence::Required,
);

static EXTRACT: Function<Scalar> = scalar("extract", &[COMPONENT, value("value")]);

static DATE_TRUNC: Function<Scalar> = scalar(
    "date_trunc",
    &[
        parameter(
            "grain",
            ArgumentForm::Enum(&["day", "week"]),
            Presence::Required,
        ),
        value("value"),
        parameter(
            "timezone",
            ArgumentForm::Constant,
            Presence::When {
                condition: Condition::Type {
                    argument: 1,
                    ty: ParameterType::Exact(ExactType::Type(&DataType::Timestamp {
                        precision: 6,
                    })),
                },
                default: None,
            },
        ),
        parameter(
            "week_start",
            ArgumentForm::Enum(&["monday", "sunday"]),
            Presence::When {
                condition: Condition::Value {
                    argument: 0,
                    value: "week",
                },
                default: None,
            },
        ),
    ],
);

static COUNT: AggregateFunction = aggregate(
    "count",
    &[value("value")],
    AggregateClauses {
        distinct: ClauseRule::Allowed,
        filter: ClauseRule::Allowed,
        ..NO_AGGREGATE_CLAUSE
    },
);

static PERCENTILE_CONT: AggregateFunction = aggregate(
    "percentile_cont",
    &[parameter(
        "fraction",
        ArgumentForm::Constant,
        Presence::Required,
    )],
    AggregateClauses {
        within_group: ClauseRule::Required,
        within_group_key: ParameterType::Family(&TypeFamily::Number),
        ..NO_AGGREGATE_CLAUSE
    },
);

static STRING_AGG: AggregateFunction = aggregate(
    "string_agg",
    &[
        value("value"),
        parameter("separator", ArgumentForm::Constant, Presence::Required),
    ],
    AggregateClauses {
        filter: ClauseRule::Allowed,
        order_by: ClauseRule::Allowed,
        ..NO_AGGREGATE_CLAUSE
    },
);

static MAX: AggregateFunction = aggregate(
    "max",
    &[value("value")],
    AggregateClauses {
        distinct: ClauseRule::NoEffect,
        ..NO_AGGREGATE_CLAUSE
    },
);

static CORR: AggregateFunction = aggregate("corr", &[value("y"), value("x")], NO_AGGREGATE_CLAUSE);

static ROW_NUMBER: Function<Window> = Function {
    canonical: Declaration {
        name: "row_number",
        aliases: &[],
        signature: signature(&[]),
        returns: ReturnRule::Fixed(ExactType::Type(&DataType::BigInt)),
        nullability: Nullability::Never,
        volatility: Volatility::Immutable,
        specific: Window {
            options: &[],
            clauses: WindowClauses {
                nulls: ClauseRule::Refused,
                partition_by: ClauseRule::Allowed,
                order_by: ClauseRule::Allowed,
                window_frame: ClauseRule::NoEffect,
            },
        },
    },
    dialects: &[],
    releases: &[],
};

fn over(
    partition_by: Vec<Expr<SemanticLeaf>>,
    order_by: Vec<SortKey<SemanticLeaf, AggregateAllowed>>,
) -> Over<SemanticLeaf, AggregateAllowed> {
    Over {
        partition_by,
        order_by,
        window_frame: None,
    }
}

fn column(name: &str) -> Expr<SemanticLeaf> {
    Expr::Column(NamedReference::new([name]).expect("one part"))
}

#[test]
fn call_new_builds_the_call_of_its_function_s_type_with_no_clause() {
    let call = Call::new(FunctionRef::Scalar(&UPPER), vec![column("name")])
        .expect("upper takes one value");
    assert_eq!(call.name(), "upper");
    assert_eq!(call.args(), [column("name")]);
    assert_eq!(call.options(), []);
    let Call::Scalar(upper) = call else {
        panic!("a scalar function's call is a scalar call: {call:?}");
    };
    assert_eq!(upper.function(), &UPPER);
    let row_number =
        Call::<SemanticLeaf, AggregateAllowed>::new(FunctionRef::Window(&ROW_NUMBER), Vec::new());
    let Ok(Call::Window(row_number)) = row_number else {
        panic!("a window function's call is a window call: {row_number:?}");
    };
    assert_eq!(row_number.nulls(), None);
    assert_eq!(row_number.over(), &over(Vec::new(), Vec::new()));
}

fn half() -> Expr<SemanticLeaf> {
    Expr::Literal(Literal::Double(0.5))
}

fn ascending(name: &str) -> SortKey<SemanticLeaf, AggregateAllowed> {
    SortKey {
        expr: column(name),
        dir: SortDir::Asc {
            nulls: NullOrdering::Last,
        },
    }
}

fn distinct(
    function: &'static AggregateFunction,
    args: Vec<Expr<SemanticLeaf>>,
) -> Result<AggregateCall<SemanticLeaf, AggregateAllowed>, CallError> {
    AggregateCall::builder(function, args)
        .distinct(true)
        .build()
}

fn plain(
    function: &'static AggregateFunction,
    args: Vec<Expr<SemanticLeaf>>,
) -> Result<Call<SemanticLeaf, AggregateAllowed>, CallError> {
    Call::new(FunctionRef::Aggregate(function, AggregateAllowed), args)
}

#[test]
fn an_allowed_clause_is_kept_where_written_and_left_out_where_not() {
    let call = distinct(&COUNT, vec![column("x")]).expect("count takes DISTINCT");
    assert!(call.distinct());
    let Ok(Call::Aggregate(call, AggregateAllowed)) = plain(&COUNT, vec![column("x")]) else {
        panic!("count(x) is an aggregate call");
    };
    assert_eq!(call.function(), &COUNT);
    assert!(!call.distinct());
    assert_eq!(call.filter(), None);
    assert_eq!(call.order_by(), []);
    assert_eq!(call.nulls(), None);
}

#[test]
fn a_required_clause_is_kept_where_written_and_refused_where_left_out() {
    let call = AggregateCall::builder(&PERCENTILE_CONT, vec![half()])
        .order_by(vec![ascending("price")])
        .build()
        .expect("percentile_cont writes WITHIN GROUP");
    assert_eq!(call.order_by(), [ascending("price")]);
    assert_eq!(
        plain(&PERCENTILE_CONT, vec![half()]),
        Err(CallError::ClauseRequired {
            function: "percentile_cont",
            clause: CallClause::WithinGroup,
        })
    );
}

#[test]
fn a_clause_of_no_effect_is_refused_where_written_and_built_where_not() {
    assert_eq!(
        distinct(&MAX, vec![column("x")]),
        Err(CallError::ClauseRefused {
            function: "max",
            clause: CallClause::Distinct,
        })
    );
    assert!(plain(&MAX, vec![column("x")]).is_ok());
}

#[test]
fn a_refused_clause_is_refused_where_written_and_built_where_not() {
    assert_eq!(
        distinct(&CORR, vec![column("y"), column("x")]),
        Err(CallError::ClauseRefused {
            function: "corr",
            clause: CallClause::Distinct,
        })
    );
    assert!(plain(&CORR, vec![column("y"), column("x")]).is_ok());
    let ignore_nulls =
        WindowCall::<SemanticLeaf, AggregateAllowed>::builder(&ROW_NUMBER, Vec::new())
            .nulls(NullTreatment::IgnoreNulls)
            .build();
    assert_eq!(
        ignore_nulls,
        Err(CallError::ClauseRefused {
            function: "row_number",
            clause: CallClause::Nulls,
        })
    );
    assert!(Call::<SemanticLeaf, AggregateAllowed>::new(
        FunctionRef::Window(&ROW_NUMBER),
        Vec::new()
    )
    .is_ok());
}

#[test]
fn a_window_call_keeps_its_over() {
    let over = over(vec![column("channel")], vec![ascending("day")]);
    let call = WindowCall::builder(&ROW_NUMBER, Vec::new())
        .over(over.clone())
        .build()
        .expect("row_number takes PARTITION BY and ORDER BY");
    assert_eq!(call.over(), &over);
}

fn scalar_call(
    function: &'static Function<Scalar>,
    args: Vec<Expr<SemanticLeaf>>,
) -> Result<Call<SemanticLeaf, AggregateAllowed>, CallError> {
    Call::new(FunctionRef::Scalar(function), args)
}

#[test]
fn a_call_with_too_few_or_too_many_arguments_is_refused() {
    assert_eq!(
        scalar_call(&UPPER, Vec::new()),
        Err(CallError::TooFewArguments {
            function: "upper",
            min: 1,
            max: Some(1),
        })
    );
    assert_eq!(
        scalar_call(&UPPER, vec![column("name"), column("region")]),
        Err(CallError::TooManyArguments {
            function: "upper",
            got: 2,
        })
    );
}

fn string(value: &str) -> Expr<SemanticLeaf> {
    Expr::Literal(Literal::String(value.to_owned()))
}

#[test]
fn a_column_at_a_constant_parameter_is_refused() {
    let pattern = string("(\\d+)");
    assert_eq!(
        scalar_call(
            &REGEXP_EXTRACT,
            vec![column("campaign"), pattern, column("group")]
        ),
        Err(CallError::NotALiteral {
            function: "regexp_extract",
            parameter: "group",
        })
    );
}

#[test]
fn an_enum_value_that_the_declaration_does_not_list_is_refused() {
    assert!(scalar_call(&EXTRACT, vec![string("year"), column("day")]).is_ok());
    assert_eq!(
        scalar_call(&EXTRACT, vec![string("yeer"), column("day")]),
        Err(CallError::EnumValue {
            function: "extract",
            parameter: "component",
            value: Literal::String("yeer".to_owned()),
        })
    );
}

#[test]
fn a_dialect_s_enum_takes_the_canonical_value_that_it_names() {
    static DATE_PART: Function<Scalar> = scalar(
        "date_part",
        &[
            parameter(
                "part",
                ArgumentForm::DialectEnum(&[("day_of_week", "isodow")]),
                Presence::Required,
            ),
            value("value"),
        ],
    );
    assert!(scalar_call(&DATE_PART, vec![string("day_of_week"), column("day")]).is_ok());
    assert_eq!(
        scalar_call(&DATE_PART, vec![string("isodow"), column("day")]),
        Err(CallError::EnumValue {
            function: "date_part",
            parameter: "part",
            value: Literal::String("isodow".to_owned()),
        })
    );
    assert_eq!(
        scalar_call(&DATE_PART, vec![column("part"), column("day")]),
        Err(CallError::NotALiteral {
            function: "date_part",
            parameter: "part",
        })
    );
}

fn untyped(literal: Literal) -> Expr<SemanticLeaf> {
    Expr::Leaf(SemanticLeaf::Untyped(UntypedLiteral(literal)))
}

#[test]
fn an_untyped_literal_at_a_constant_or_an_enum_parameter_is_a_literal() {
    let pattern = untyped(Literal::String("(\\d+)".to_owned()));
    let group = untyped(Literal::BigInt(1));
    assert!(scalar_call(&REGEXP_EXTRACT, vec![column("campaign"), pattern, group]).is_ok());
    let year = untyped(Literal::String("year".to_owned()));
    assert!(scalar_call(&EXTRACT, vec![year, column("day")]).is_ok());
    let mapped = PhysicalLeaf::Untyped(UntypedLiteral(Literal::String("year".to_owned())));
    let mapping_call = ScalarCall::<PhysicalLeaf, NoAggregate>::builder(
        &EXTRACT,
        vec![
            Expr::Leaf(mapped),
            Expr::Column(NamedReference::new(["day"]).expect("one part")),
        ],
    )
    .build();
    assert!(mapping_call.is_ok());
    assert_eq!(
        scalar_call(&EXTRACT, vec![column("component"), column("day")]),
        Err(CallError::NotALiteral {
            function: "extract",
            parameter: "component",
        })
    );
}

#[test]
fn a_conditional_parameter_takes_an_argument_where_its_condition_says_so() {
    let call = |args: Vec<Expr<SemanticLeaf>>| scalar_call(&DATE_TRUNC, args);
    let week_of_a_date = call(vec![string("week"), column("day"), string("monday")]);
    assert!(week_of_a_date.is_ok(), "{week_of_a_date:?}");
    let day_of_a_timestamp = call(vec![string("day"), column("at"), string("UTC")]);
    assert!(day_of_a_timestamp.is_ok(), "{day_of_a_timestamp:?}");
    let week_of_a_timestamp = call(vec![
        string("week"),
        column("at"),
        string("UTC"),
        string("monday"),
    ]);
    assert!(week_of_a_timestamp.is_ok(), "{week_of_a_timestamp:?}");
    assert_eq!(
        call(vec![string("week"), column("day")]),
        Err(CallError::MissingArgument {
            function: "date_trunc",
            parameter: "week_start",
        })
    );
    assert_eq!(
        call(vec![string("week"), column("day"), string("mondai")]),
        Err(CallError::EnumValue {
            function: "date_trunc",
            parameter: "week_start",
            value: Literal::String("mondai".to_owned()),
        })
    );
    assert_eq!(
        call(vec![
            string("day"),
            column("at"),
            string("UTC"),
            string("monday"),
        ]),
        Err(CallError::TooManyArguments {
            function: "date_trunc",
            got: 4,
        })
    );
}

struct NoInputs;

impl InputTypes<SemanticLeaf> for NoInputs {
    fn column(&self, _: &NamedReference) -> Typed {
        Typed::new(DataType::Unknown, true)
    }

    fn leaf(&self, _: &SemanticLeaf) -> Typed {
        Typed::new(DataType::Unknown, true)
    }
}

#[test]
fn a_conditioned_parameter_with_no_default_is_required_where_its_condition_holds() {
    let week_of_a_date = vec![string("week"), column("day")];
    assert_eq!(
        FunctionRef::<AggregateAllowed>::Scalar(&DATE_TRUNC)
            .arguments_by_position(week_of_a_date.clone()),
        week_of_a_date
    );
    let at = Expr::Literal(Literal::Timestamp {
        value: "2026-05-19".to_owned(),
        precision: 6,
    });
    let day_of_a_timestamp =
        scalar_call(&DATE_TRUNC, vec![string("day"), at]).expect("only typing reads the timestamp");
    let typing = type_tree(
        Expr::Function(day_of_a_timestamp),
        &NoInputs,
        DocumentDialect::Canonical,
        &CanonicalTypeSystem,
        None,
    );
    assert_eq!(
        typing.err(),
        Some(TypeError::ParameterRequired {
            name: "date_trunc",
            parameter: "timezone",
        })
    );
}

#[test]
fn narrowing_refuses_an_aggregate_s_handle_and_accepts_a_scalar_s() {
    let narrowed =
        |function: FunctionRef<AggregateAllowed>| FunctionRef::<NoAggregate>::try_from(function);
    assert_eq!(
        narrowed(FunctionRef::Aggregate(&COUNT, AggregateAllowed)),
        Err(AggregateInRowExpr)
    );
    assert_eq!(
        narrowed(FunctionRef::Scalar(&UPPER)),
        Ok(FunctionRef::Scalar(&UPPER))
    );
    assert_eq!(
        narrowed(FunctionRef::Window(&ROW_NUMBER)),
        Ok(FunctionRef::Window(&ROW_NUMBER))
    );
}

fn upper_is_eu<A: Aggregation>() -> Expr<SemanticLeaf, A> {
    let name = Expr::Column(NamedReference::new(["name"]).expect("one part"));
    let upper = Call::new(FunctionRef::Scalar(&UPPER), vec![name]).expect("upper takes one value");
    let eu = Expr::Literal(Literal::String("EU".to_owned()));
    ExprOperationExt::eq(Expr::Function(upper), eu)
}

#[test]
fn a_tree_narrows_into_a_row_level_tree_only_where_it_holds_no_aggregate_call() {
    assert_eq!(
        RowExpr::try_from(upper_is_eu::<AggregateAllowed>()),
        Ok(upper_is_eu::<NoAggregate>())
    );
    let count = plain(&COUNT, vec![column("x")]).expect("count takes one value");
    let tree = Expr::IsNull(Box::new(Expr::Function(count)));
    assert_eq!(RowExpr::try_from(tree), Err(AggregateInRowExpr));
}

fn row_number_by_day<A: Aggregation>() -> Expr<SemanticLeaf, A> {
    let column = |name| Expr::Column(NamedReference::new([name]).expect("one part"));
    let over = Over {
        partition_by: vec![column("channel")],
        order_by: vec![SortKey {
            expr: column("day"),
            dir: SortDir::Desc {
                nulls: NullOrdering::First,
            },
        }],
        window_frame: None,
    };
    let call = WindowCall::builder(&ROW_NUMBER, Vec::new())
        .over(over)
        .build()
        .expect("row_number takes OVER");
    Expr::Function(Call::Window(call))
}

#[test]
fn a_window_call_narrows_with_its_over() {
    assert_eq!(
        RowExpr::try_from(row_number_by_day::<AggregateAllowed>()),
        Ok(row_number_by_day::<NoAggregate>())
    );
}

fn calls_with_clauses([value, key, filter, partition]: [&str; 4]) -> [Expr<SemanticLeaf>; 2] {
    let string_agg = AggregateCall::builder(&STRING_AGG, vec![column(value), string(",")])
        .filter(column(filter))
        .order_by(vec![ascending(key)])
        .build()
        .expect("string_agg takes FILTER and ORDER BY");
    let row_number = WindowCall::builder(&ROW_NUMBER, Vec::new())
        .over(over(vec![column(partition)], vec![ascending(key)]))
        .build()
        .expect("row_number takes OVER");
    [
        Expr::Function(Call::Aggregate(string_agg, AggregateAllowed)),
        Expr::Function(Call::Window(row_number)),
    ]
}

#[test]
fn a_call_gives_its_arguments_then_its_clauses_operands_as_its_children() {
    let [string_agg, row_number] = calls_with_clauses(["name", "day", "paid", "channel"]);
    assert_eq!(
        string_agg.children(),
        [
            &column("name"),
            &string(","),
            &column("paid"),
            &column("day")
        ]
    );
    assert_eq!(row_number.children(), [&column("channel"), &column("day")]);
}

#[test]
fn a_call_rebuilds_with_new_children_in_their_place() {
    let [string_agg, row_number] = calls_with_clauses(["name", "day", "paid", "channel"]);
    let [new_string_agg, new_row_number] = calls_with_clauses(["city", "at", "sent", "region"]);
    let new = vec![column("city"), string(","), column("sent"), column("at")];
    assert_eq!(string_agg.with_new_children(new), Ok(new_string_agg));
    let new = vec![column("region"), column("at")];
    assert_eq!(row_number.with_new_children(new), Ok(new_row_number));
}

fn refused<L: Leaf>(error: CallError) -> Result<Expr<L>, ValidateError> {
    Err(ValidateError::CallRefused {
        reason: error.to_string(),
    })
}

#[test]
fn a_rebuilt_call_is_refused_where_its_new_operands_do_not_fit_its_declaration() {
    let call = |function, args| Expr::Function(scalar_call(function, args).expect("it fits"));
    let regexp_extract = call(&REGEXP_EXTRACT, vec![column("campaign"), string("(\\d+)")]);
    assert_eq!(
        regexp_extract.with_new_children(vec![column("campaign"), column("pattern")]),
        refused(CallError::NotALiteral {
            function: "regexp_extract",
            parameter: "pattern",
        })
    );
    let extract = call(&EXTRACT, vec![string("year"), column("day")]);
    let misspelt = extract.transform(&mut |expr| {
        Ok(if expr == string("year") {
            string("yeer")
        } else {
            expr
        })
    });
    assert_eq!(
        misspelt,
        refused(CallError::EnumValue {
            function: "extract",
            parameter: "component",
            value: Literal::String("yeer".to_owned()),
        })
    );
    let date_trunc = |grain| vec![string(grain), column("at"), string("UTC"), string("monday")];
    let week = call(&DATE_TRUNC, date_trunc("week"));
    assert_eq!(
        week.with_new_children(date_trunc("day")),
        refused(CallError::TooManyArguments {
            function: "date_trunc",
            got: 4,
        })
    );
}

fn settings(options: &[(&str, &str)]) -> Vec<OptionSetting> {
    options
        .iter()
        .map(|&(name, value)| OptionSetting {
            name: name.to_owned(),
            value: value.to_owned(),
        })
        .collect()
}

#[test]
fn check_options_refuses_an_option_name_or_a_value_that_the_declaration_does_not_have() {
    let check = |options: &[(&str, &str)]| {
        ScalarCall::builder(&CONCAT, vec![column("name")])
            .options(settings(options))
            .build()
            .map(|call| Call::<SemanticLeaf, AggregateAllowed>::Scalar(call).check_options())
    };
    assert_eq!(check(&[]), Ok(Ok(())));
    assert_eq!(check(&[("null_handling", "IGNORE_NULLS")]), Ok(Ok(())));
    assert_eq!(
        check(&[("null_policy", "IGNORE_NULLS")]),
        Ok(Err(CallError::UnknownOption {
            function: "concat",
            option: "null_policy".to_owned(),
        }))
    );
    assert_eq!(
        check(&[("null_handling", "SKIP_NULLS")]),
        Ok(Err(CallError::OptionValue {
            function: "concat",
            option: "null_handling",
            value: "SKIP_NULLS".to_owned(),
        }))
    );
}

fn null_handling_twice(value: &str) -> Vec<OptionSetting> {
    settings(&[("null_handling", "IGNORE_NULLS"), ("null_handling", value)])
}

fn set_twice(function: &'static str) -> Result<(), CallError> {
    Err(CallError::RepeatedOption {
        function,
        option: "null_handling".to_owned(),
    })
}

#[test]
fn each_call_constructor_refuses_an_option_that_the_call_sets_twice() {
    for value in ["IGNORE_NULLS", "ACCEPT_NULLS"] {
        let call =
            ScalarCall::<SemanticLeaf, AggregateAllowed>::builder(&CONCAT, vec![column("name")])
                .options(null_handling_twice(value))
                .build();
        assert_eq!(call.map(|_| ()), set_twice("concat"), "{value}");
    }
    let max = AggregateCall::builder(&MAX, vec![column("price")])
        .options(null_handling_twice("IGNORE_NULLS"))
        .build();
    assert_eq!(max.map(|_| ()), set_twice("max"));
    let row_number = WindowCall::<SemanticLeaf, AggregateAllowed>::builder(&ROW_NUMBER, Vec::new())
        .options(null_handling_twice("IGNORE_NULLS"))
        .build();
    assert_eq!(row_number.map(|_| ()), set_twice("row_number"));
}

#[test]
fn the_check_of_options_refuses_an_option_that_the_call_sets_twice() {
    let declared = CONCAT.canonical.specific.options;
    let twice = null_handling_twice("IGNORE_NULLS");
    assert_eq!(
        super::check::check_options("concat", declared, &twice),
        set_twice("concat")
    );
}

#[test]
fn a_call_error_writes_the_sql_clause_and_the_enum_value() {
    let enum_value = |value| CallError::EnumValue {
        function: "extract",
        parameter: "component",
        value,
    };
    let cases = [
        (
            CallError::ClauseRefused {
                function: "max",
                clause: CallClause::Distinct,
            },
            "`max` takes no DISTINCT clause",
        ),
        (
            CallError::ClauseRefused {
                function: "row_number",
                clause: CallClause::Nulls,
            },
            "`row_number` takes no IGNORE NULLS or RESPECT NULLS clause",
        ),
        (
            CallError::ClauseRequired {
                function: "percentile_cont",
                clause: CallClause::WithinGroup,
            },
            "`percentile_cont` requires its WITHIN GROUP clause",
        ),
        (
            enum_value(Literal::String("yeer".to_owned())),
            "`extract`'s `component` has no value 'yeer'",
        ),
        (
            enum_value(Literal::BigInt(1)),
            "`extract`'s `component` has no bigint value",
        ),
    ];
    for (error, message) in cases {
        assert_eq!(error.to_string(), message);
    }
}
