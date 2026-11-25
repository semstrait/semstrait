use semstrait_common::Dialect;

use super::{
    constant, fixed, optional, value, DOUBLE, NUMBER, NUMBER_VALUE, STRING, STRING_VALUE, T_VALUE,
    VALUE_OF_ANY_TYPE,
};
use crate::datatype::DataType;
use crate::functions::Additivity::{self, Additive, NonAdditive};
use crate::functions::ClauseRule::{self, Allowed, NoEffect, Refused, Required};
use crate::functions::Nullability::{self, Always, EmptyGroupNull, Never};
use crate::functions::ReturnRule::{self, AvgOf, SumOf};
use crate::functions::{
    Aggregate, AggregateClauses, AggregateFunction, Declaration, ExactType, Function,
    ParameterDecl, ParameterType, Signature, Volatility, BOOLEAN,
};

const Y_X: &[ParameterDecl] = &[value("y", NUMBER), value("x", NUMBER)];
const FRACTION: ParameterDecl = constant(
    "fraction",
    ParameterType::Exact(ExactType::Type(&DataType::Double)),
);
const BIGINT: ReturnRule = fixed(&DataType::BigInt);

const fn clauses(distinct: ClauseRule) -> AggregateClauses {
    AggregateClauses {
        distinct,
        filter: Allowed,
        order_by: NoEffect,
        within_group: Refused,
        within_group_key: ParameterType::Any,
        nulls: Refused,
    }
}

const fn declaration(
    name: &'static str,
    parameters: &'static [ParameterDecl],
    returns: ReturnRule,
    nullability: Nullability,
    clauses: AggregateClauses,
) -> Declaration<Aggregate> {
    Declaration {
        name,
        aliases: &[],
        signature: Signature {
            parameters,
            variadic: None,
            constraints: &[],
        },
        returns,
        nullability,
        volatility: Volatility::Immutable,
        specific: Aggregate {
            options: &[],
            clauses,
        },
    }
}

const fn function(
    canonical: Declaration<Aggregate>,
    dialects: &'static [(Dialect, Option<Declaration<Aggregate>>)],
    additivity: Additivity,
    distinct_additivity: Additivity,
) -> AggregateFunction {
    AggregateFunction {
        declarations: Function {
            canonical,
            dialects,
            releases: &[],
        },
        additivity,
        distinct_additivity,
    }
}

const fn non_additive(canonical: Declaration<Aggregate>) -> AggregateFunction {
    function(canonical, &[], NonAdditive, NonAdditive)
}

const fn same_type(name: &'static str) -> AggregateFunction {
    function(
        declaration(
            name,
            &[T_VALUE],
            ReturnRule::Var,
            EmptyGroupNull,
            clauses(NoEffect),
        ),
        &[],
        Additive,
        Additive,
    )
}

const fn boolean(name: &'static str, aliases: &'static [&'static str]) -> AggregateFunction {
    const VALUE: &[ParameterDecl] = &[value("value", BOOLEAN)];
    function(
        Declaration {
            aliases,
            ..declaration(
                name,
                VALUE,
                fixed(&DataType::Boolean),
                EmptyGroupNull,
                clauses(NoEffect),
            )
        },
        &[],
        Additive,
        Additive,
    )
}

const fn of_number(
    name: &'static str,
    returns: ReturnRule,
    nullability: Nullability,
    distinct: ClauseRule,
) -> Declaration<Aggregate> {
    declaration(
        name,
        &[NUMBER_VALUE],
        returns,
        nullability,
        clauses(distinct),
    )
}

const fn regression(name: &'static str, nullability: Nullability) -> AggregateFunction {
    let canonical = declaration(name, Y_X, DOUBLE, nullability, clauses(Refused));
    non_additive(canonical)
}

const fn percentile_cont(
    parameters: &'static [ParameterDecl],
    order_by: ClauseRule,
    within_group: ClauseRule,
) -> Declaration<Aggregate> {
    declaration(
        "percentile_cont",
        parameters,
        DOUBLE,
        EmptyGroupNull,
        AggregateClauses {
            order_by,
            within_group,
            within_group_key: NUMBER,
            ..clauses(Refused)
        },
    )
}

const fn approx_count_distinct(name: &'static str) -> Declaration<Aggregate> {
    declaration(name, &[VALUE_OF_ANY_TYPE], BIGINT, Never, clauses(Refused))
}

static COUNT: AggregateFunction = function(
    declaration(
        "count",
        &[optional(VALUE_OF_ANY_TYPE)],
        BIGINT,
        Never,
        clauses(Allowed),
    ),
    &[],
    Additive,
    NonAdditive,
);
static SUM: AggregateFunction = function(
    of_number("sum", SumOf, EmptyGroupNull, Allowed),
    &[],
    Additive,
    NonAdditive,
);
static AVG: AggregateFunction = non_additive(of_number("avg", AvgOf, EmptyGroupNull, Allowed));
static MIN: AggregateFunction = same_type("min");
static MAX: AggregateFunction = same_type("max");
static ANY_VALUE: AggregateFunction = same_type("any_value");
static BOOL_AND: AggregateFunction = boolean("bool_and", &["every"]);
static BOOL_OR: AggregateFunction = boolean("bool_or", &["any", "some"]);
static STDDEV_SAMP: AggregateFunction = function(
    of_number("stddev_samp", DOUBLE, Always, Allowed),
    &[(
        Dialect::DataFusion,
        Some(of_number("stddev_samp", DOUBLE, Always, Refused)),
    )],
    NonAdditive,
    NonAdditive,
);
static STDDEV_POP: AggregateFunction = function(
    of_number("stddev_pop", DOUBLE, EmptyGroupNull, Allowed),
    &[(
        Dialect::DataFusion,
        Some(of_number("stddev_pop", DOUBLE, EmptyGroupNull, Refused)),
    )],
    NonAdditive,
    NonAdditive,
);
static VAR_SAMP: AggregateFunction = non_additive(of_number("var_samp", DOUBLE, Always, Allowed));
static VAR_POP: AggregateFunction =
    non_additive(of_number("var_pop", DOUBLE, EmptyGroupNull, Allowed));
static COVAR_SAMP: AggregateFunction = regression("covar_samp", Always);
static COVAR_POP: AggregateFunction = regression("covar_pop", EmptyGroupNull);
static CORR: AggregateFunction = regression("corr", Always);
static REGR_SLOPE: AggregateFunction = regression("regr_slope", Always);
static REGR_INTERCEPT: AggregateFunction = regression("regr_intercept", Always);
static REGR_R2: AggregateFunction = regression("regr_r2", Always);
static REGR_AVGX: AggregateFunction = regression("regr_avgx", EmptyGroupNull);
static REGR_AVGY: AggregateFunction = regression("regr_avgy", EmptyGroupNull);
static REGR_SXX: AggregateFunction = regression("regr_sxx", EmptyGroupNull);
static REGR_SYY: AggregateFunction = regression("regr_syy", EmptyGroupNull);
static REGR_SXY: AggregateFunction = regression("regr_sxy", EmptyGroupNull);
static REGR_COUNT: AggregateFunction = function(
    declaration("regr_count", Y_X, BIGINT, Never, clauses(Refused)),
    &[],
    Additive,
    Additive,
);
static PERCENTILE_CONT: AggregateFunction = function(
    percentile_cont(&[FRACTION], Refused, Required),
    &[(
        Dialect::DataFusion,
        Some(percentile_cont(&[NUMBER_VALUE, FRACTION], Allowed, Refused)),
    )],
    NonAdditive,
    NonAdditive,
);
static MEDIAN: AggregateFunction =
    non_additive(of_number("median", AvgOf, EmptyGroupNull, Allowed));
static STRING_AGG: AggregateFunction = non_additive(declaration(
    "string_agg",
    &[STRING_VALUE, constant("separator", STRING)],
    fixed(&DataType::String),
    EmptyGroupNull,
    AggregateClauses {
        order_by: Allowed,
        ..clauses(Allowed)
    },
));
static APPROX_COUNT_DISTINCT: AggregateFunction = function(
    approx_count_distinct("approx_count_distinct"),
    &[(
        Dialect::DataFusion,
        Some(approx_count_distinct("approx_distinct")),
    )],
    NonAdditive,
    NonAdditive,
);

pub(crate) static FUNCTIONS: &[&AggregateFunction] = &[
    &COUNT,
    &SUM,
    &AVG,
    &MIN,
    &MAX,
    &ANY_VALUE,
    &BOOL_AND,
    &BOOL_OR,
    &STDDEV_SAMP,
    &STDDEV_POP,
    &VAR_SAMP,
    &VAR_POP,
    &COVAR_SAMP,
    &COVAR_POP,
    &CORR,
    &REGR_SLOPE,
    &REGR_INTERCEPT,
    &REGR_R2,
    &REGR_AVGX,
    &REGR_AVGY,
    &REGR_SXX,
    &REGR_SYY,
    &REGR_SXY,
    &REGR_COUNT,
    &PERCENTILE_CONT,
    &MEDIAN,
    &STRING_AGG,
    &APPROX_COUNT_DISTINCT,
];
