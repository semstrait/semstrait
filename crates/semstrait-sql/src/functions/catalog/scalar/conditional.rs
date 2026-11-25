use semstrait_common::Dialect;

use super::super::{canonical, scalar, value, T_VALUE, VALUE_OF_ANY_TYPE};
use crate::functions::{
    Category, Constraint, Declaration, Function, Nullability, ParameterType, ReturnRule, Scalar,
    Signature,
};

const fn conditional(
    name: &'static str,
    signature: Signature,
    returns: ReturnRule,
    nullability: Nullability,
) -> Declaration<Scalar> {
    Declaration {
        signature,
        ..scalar(Category::Conditional, name, &[], returns, nullability)
    }
}

const fn promote(name: &'static str) -> Declaration<Scalar> {
    let signature = Signature {
        parameters: &[],
        variadic: Some(VALUE_OF_ANY_TYPE),
        constraints: &[Constraint::OneFamily],
    };
    conditional(name, signature, ReturnRule::Promote, Nullability::Mirror)
}

const GREATEST: Declaration<Scalar> = promote("greatest");
const LEAST: Declaration<Scalar> = promote("least");

pub(crate) static FUNCTIONS: &[&Function<Scalar>] = &[
    &canonical(conditional(
        "coalesce",
        Signature {
            parameters: &[T_VALUE],
            variadic: Some(T_VALUE),
            constraints: &[],
        },
        ReturnRule::Var,
        Nullability::AllArgs,
    )),
    &canonical(Declaration {
        aliases: &["null_if"],
        ..conditional(
            "nullif",
            Signature {
                parameters: &[
                    value("left", ParameterType::Any),
                    value("right", ParameterType::Any),
                ],
                variadic: None,
                constraints: &[Constraint::OneFamily],
            },
            ReturnRule::SameAs(0),
            Nullability::Always,
        )
    }),
    &Function {
        canonical: GREATEST,
        dialects: &[(
            Dialect::DataFusion,
            Some(Declaration {
                nullability: Nullability::AllArgs,
                ..GREATEST
            }),
        )],
        releases: &[],
    },
    &Function {
        canonical: LEAST,
        dialects: &[(
            Dialect::DataFusion,
            Some(Declaration {
                nullability: Nullability::AllArgs,
                ..LEAST
            }),
        )],
        releases: &[],
    },
];
