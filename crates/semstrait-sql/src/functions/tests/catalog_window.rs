use super::{fixed, window_function, INTEGER};
use crate::datatype::DataType;
use crate::expr::FunctionRef;
use crate::functions::{
    registry, ArgumentForm, ClauseRule, Declaration, Nullability, ParameterDecl, ParameterDefault,
    ParameterType, Presence, ReturnRule, Signature, Volatility, Window, WindowClauses,
};
use semstrait_common::Dialect;

use ArgumentForm::{Constant, Value};
use ClauseRule::{Allowed, NoEffect, Refused, Required};
use Nullability::{Always, Never};
use Presence::{Defaulted, Optional};

const fn parameter(
    name: &'static str,
    ty: ParameterType,
    form: ArgumentForm,
    presence: Presence,
) -> ParameterDecl {
    ParameterDecl {
        name,
        ty,
        form,
        presence,
    }
}

const T: ParameterType = ParameterType::Var(None);
const BIGINT: ReturnRule = fixed(&DataType::BigInt);
const DOUBLE: ReturnRule = fixed(&DataType::Double);
const T_RESULT: ReturnRule = ReturnRule::Var;
const VALUE: ParameterDecl = parameter("value", T, Value, Presence::Required);
const BUCKETS: &[ParameterDecl] = &[parameter("buckets", INTEGER, Constant, Presence::Required)];
const LAG_OR_LEAD: &[ParameterDecl] = &[
    VALUE,
    parameter(
        "offset",
        INTEGER,
        Constant,
        Defaulted(ParameterDefault::Integer(1)),
    ),
    parameter("default", T, Value, Optional),
];
const NTH: &[ParameterDecl] = &[VALUE, parameter("n", INTEGER, Constant, Presence::Required)];

type Row = (
    &'static str,
    &'static [ParameterDecl],
    ReturnRule,
    Nullability,
    [ClauseRule; 3],
);

#[rustfmt::skip]
const RULE_19: [Row; 11] = [
    ("row_number",   &[],         BIGINT,   Never,  [Refused, Allowed,  NoEffect]),
    ("rank",         &[],         BIGINT,   Never,  [Refused, Required, NoEffect]),
    ("dense_rank",   &[],         BIGINT,   Never,  [Refused, Required, NoEffect]),
    ("percent_rank", &[],         DOUBLE,   Never,  [Refused, Required, NoEffect]),
    ("cume_dist",    &[],         DOUBLE,   Never,  [Refused, Required, NoEffect]),
    ("ntile",        BUCKETS,     BIGINT,   Never,  [Refused, Required, NoEffect]),
    ("lag",          LAG_OR_LEAD, T_RESULT, Always, [Allowed, Required, NoEffect]),
    ("lead",         LAG_OR_LEAD, T_RESULT, Always, [Allowed, Required, NoEffect]),
    ("first_value",  &[VALUE],    T_RESULT, Always, [Allowed, Allowed,  Allowed]),
    ("last_value",   &[VALUE],    T_RESULT, Always, [Allowed, Allowed,  Allowed]),
    ("nth_value",    NTH,         T_RESULT, Always, [Allowed, Allowed,  Allowed]),
];

const IGNORE_NULLS_REFUSED_ON_DATAFUSION: &[&str] =
    &["lag", "lead", "first_value", "last_value", "nth_value"];

#[test]
fn the_catalog_declares_the_eleven_window_functions_of_rule_19() {
    let declared = registry()
        .all()
        .filter_map(|function| match function {
            FunctionRef::Window(function) => Some(function.canonical.name),
            _ => None,
        })
        .collect::<Vec<_>>();
    let rule_19 = RULE_19.iter().map(|row| row.0).collect::<Vec<_>>();
    assert_eq!(declared, rule_19);
}

#[test]
fn each_window_function_has_its_shape_its_null_rule_and_its_clauses() {
    for (name, parameters, returns, nullability, [nulls, order_by, window_frame]) in RULE_19 {
        let clauses = WindowClauses {
            nulls,
            partition_by: Allowed,
            order_by,
            window_frame,
        };
        let declaration = Declaration {
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
            specific: Window {
                options: &[],
                clauses,
            },
        };
        assert_eq!(window_function(name).canonical, declaration);
    }
}

#[test]
fn on_datafusion_the_analytical_functions_refuse_ignore_nulls() {
    for name in IGNORE_NULLS_REFUSED_ON_DATAFUSION {
        let function = window_function(name);
        let Some(datafusion) = function.on(Dialect::DataFusion) else {
            panic!("DataFusion has `{name}`");
        };
        let canonical = &function.canonical;
        assert_eq!(datafusion.specific.clauses.nulls, Refused, "{name}");
        assert_eq!(
            without_nulls(datafusion),
            without_nulls(canonical),
            "{name}"
        );
    }
}

fn without_nulls(declaration: &Declaration<Window>) -> impl PartialEq + std::fmt::Debug + '_ {
    let clauses = &declaration.specific.clauses;
    (
        declaration.name,
        &declaration.signature,
        &declaration.returns,
        &declaration.nullability,
        declaration.volatility,
        declaration.specific.options,
        (clauses.partition_by, clauses.order_by, clauses.window_frame),
    )
}
