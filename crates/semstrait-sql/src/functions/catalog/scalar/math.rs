use semstrait_common::Dialect;

use super::super::{
    canonical, constant, fixed, optional, scalar, value, DOUBLE, INTEGER, NUMBER, NUMBER_VALUE,
};
use crate::datatype::{DataType, TypeFamily};
use crate::functions::{
    Category, Declaration, Function, Nullability, ParameterDecl, ParameterType, ReturnRule, Scalar,
    Volatility,
};

const fn number(name: &'static str) -> ParameterDecl {
    value(name, NUMBER)
}

const fn math(
    name: &'static str,
    parameters: &'static [ParameterDecl],
    returns: ReturnRule,
) -> Declaration<Scalar> {
    scalar(
        Category::Math,
        name,
        parameters,
        returns,
        Nullability::Mirror,
    )
}

const SIGN: Declaration<Scalar> = math("sign", &[NUMBER_VALUE], fixed(&DataType::Integer));

pub(crate) static FUNCTIONS: &[&Function<Scalar>] = &[
    &canonical(math(
        "abs",
        &[value(
            "value",
            ParameterType::Var(Some(&TypeFamily::Number)),
        )],
        ReturnRule::Var,
    )),
    &canonical(Declaration {
        aliases: &["ceiling"],
        ..math("ceil", &[NUMBER_VALUE], ReturnRule::DecimalScaleZero)
    }),
    &canonical(math("floor", &[NUMBER_VALUE], ReturnRule::DecimalScaleZero)),
    &canonical(math(
        "round",
        &[NUMBER_VALUE, optional(constant("scale", INTEGER))],
        ReturnRule::RoundOf,
    )),
    &canonical(math("sqrt", &[NUMBER_VALUE], DOUBLE)),
    &canonical(math("exp", &[NUMBER_VALUE], DOUBLE)),
    &canonical(math("ln", &[NUMBER_VALUE], DOUBLE)),
    &canonical(math("log10", &[NUMBER_VALUE], DOUBLE)),
    &canonical(math("log2", &[NUMBER_VALUE], DOUBLE)),
    &canonical(math("cbrt", &[NUMBER_VALUE], DOUBLE)),
    &canonical(math("sin", &[NUMBER_VALUE], DOUBLE)),
    &canonical(math("cos", &[NUMBER_VALUE], DOUBLE)),
    &canonical(math("tan", &[NUMBER_VALUE], DOUBLE)),
    &canonical(math("asin", &[NUMBER_VALUE], DOUBLE)),
    &canonical(math("acos", &[NUMBER_VALUE], DOUBLE)),
    &canonical(math("atan", &[NUMBER_VALUE], DOUBLE)),
    &canonical(math("sinh", &[NUMBER_VALUE], DOUBLE)),
    &canonical(math("cosh", &[NUMBER_VALUE], DOUBLE)),
    &canonical(math("tanh", &[NUMBER_VALUE], DOUBLE)),
    &canonical(math("degrees", &[NUMBER_VALUE], DOUBLE)),
    &canonical(math("radians", &[NUMBER_VALUE], DOUBLE)),
    &canonical(math("power", &[number("base"), number("exponent")], DOUBLE)),
    &canonical(math("log", &[number("base"), NUMBER_VALUE], DOUBLE)),
    &canonical(math("atan2", &[number("y"), number("x")], DOUBLE)),
    &Function {
        canonical: SIGN,
        dialects: &[(
            Dialect::DataFusion,
            Some(Declaration {
                name: "signum",
                ..SIGN
            }),
        )],
        releases: &[],
    },
    &Function {
        canonical: math(
            "width_bucket",
            &[
                NUMBER_VALUE,
                number("low"),
                number("high"),
                value("count", INTEGER),
            ],
            fixed(&DataType::Integer),
        ),
        dialects: &[(Dialect::DataFusion, None)],
        releases: &[],
    },
    &canonical(Declaration {
        nullability: Nullability::Never,
        ..math("pi", &[], DOUBLE)
    }),
    &canonical(Declaration {
        nullability: Nullability::Never,
        volatility: Volatility::Volatile,
        ..math("random", &[], DOUBLE)
    }),
];
