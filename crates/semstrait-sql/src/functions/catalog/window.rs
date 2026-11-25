use semstrait_common::Dialect;

use super::{constant, defaulted, fixed, optional, value, INTEGER, T_VALUE};
use crate::datatype::DataType;
use crate::functions::{
    ClauseRule, Declaration, Function, Nullability, ParameterDecl, ParameterDefault, ParameterType,
    ReturnRule, Signature, Volatility, Window, WindowClauses,
};
use ClauseRule::{Allowed, NoEffect, Refused, Required};

macro_rules! analytical {
    (
        $name:literal,
        $parameters:expr,
        order_by: $order_by:ident,
        window_frame: $window_frame:ident
    ) => {{
        const CANONICAL: Declaration<Window> = declaration(
            $name,
            $parameters,
            ReturnRule::Var,
            Nullability::Always,
            WindowClauses {
                nulls: Allowed,
                partition_by: Allowed,
                order_by: $order_by,
                window_frame: $window_frame,
            },
        );
        let mut datafusion = CANONICAL;
        datafusion.specific.clauses.nulls = Refused;
        Function {
            canonical: CANONICAL,
            dialects: &[(Dialect::DataFusion, Some(datafusion))],
            releases: &[],
        }
    }};
}

pub(crate) static FUNCTIONS: &[&Function<Window>] = &[
    &ranking("row_number", &[], &DataType::BigInt, Allowed),
    &ranking("rank", &[], &DataType::BigInt, Required),
    &ranking("dense_rank", &[], &DataType::BigInt, Required),
    &ranking("percent_rank", &[], &DataType::Double, Required),
    &ranking("cume_dist", &[], &DataType::Double, Required),
    &ranking(
        "ntile",
        &[constant("buckets", INTEGER)],
        &DataType::BigInt,
        Required,
    ),
    &analytical!("lag", LAG_OR_LEAD, order_by: Required, window_frame: NoEffect),
    &analytical!("lead", LAG_OR_LEAD, order_by: Required, window_frame: NoEffect),
    &analytical!("first_value", &[T_VALUE], order_by: Allowed, window_frame: Allowed),
    &analytical!("last_value", &[T_VALUE], order_by: Allowed, window_frame: Allowed),
    &analytical!(
        "nth_value",
        &[T_VALUE, constant("n", INTEGER)],
        order_by: Allowed,
        window_frame: Allowed
    ),
];

const LAG_OR_LEAD: &[ParameterDecl] = &[
    T_VALUE,
    defaulted(constant("offset", INTEGER), ParameterDefault::Integer(1)),
    optional(value("default", ParameterType::Var(None))),
];

const fn ranking(
    name: &'static str,
    parameters: &'static [ParameterDecl],
    returns: &'static DataType,
    order_by: ClauseRule,
) -> Function<Window> {
    Function {
        canonical: declaration(
            name,
            parameters,
            fixed(returns),
            Nullability::Never,
            WindowClauses {
                nulls: Refused,
                partition_by: Allowed,
                order_by,
                window_frame: NoEffect,
            },
        ),
        dialects: &[],
        releases: &[],
    }
}

const fn declaration(
    name: &'static str,
    parameters: &'static [ParameterDecl],
    returns: ReturnRule,
    nullability: Nullability,
    clauses: WindowClauses,
) -> Declaration<Window> {
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
        specific: Window {
            options: &[],
            clauses,
        },
    }
}
