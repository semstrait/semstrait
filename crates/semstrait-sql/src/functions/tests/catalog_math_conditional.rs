use super::{fixed, scalar, value, INTEGER};
use crate::datatype::{DataType, TypeFamily};
use crate::expr::FunctionRef;
use crate::functions::{
    registry, ArgumentForm, Category, Constraint, Declaration, Nullability, ParameterDecl,
    ParameterType, Presence, ReturnRule, Scalar, Signature, Volatility,
};
use crate::DocumentDialect;
use semstrait_common::Dialect;

struct Row {
    functions: &'static [&'static str],
    aliases: &'static [&'static str],
    signature: Signature,
    returns: ReturnRule,
    nullability: Nullability,
    volatility: Volatility,
}

const fn row(
    functions: &'static [&'static str],
    parameters: &'static [ParameterDecl],
    returns: ReturnRule,
) -> Row {
    Row {
        functions,
        aliases: &[],
        signature: Signature {
            parameters,
            variadic: None,
            constraints: &[],
        },
        returns,
        nullability: Nullability::Mirror,
        volatility: Volatility::Immutable,
    }
}

const fn number(name: &'static str) -> ParameterDecl {
    value(name, ParameterType::Family(&TypeFamily::Number))
}

fn declared(category: Category) -> Vec<&'static str> {
    registry()
        .all()
        .filter_map(|function| match function {
            FunctionRef::Scalar(function) if function.canonical.specific.category == category => {
                Some(function.canonical.name)
            }
            _ => None,
        })
        .collect()
}

fn functions(rows: &[Row]) -> Vec<&'static str> {
    rows.iter()
        .flat_map(|row| row.functions.iter().copied())
        .collect()
}

type Unnamed<'a> = (
    &'a [&'a str],
    &'a Signature,
    &'a ReturnRule,
    &'a Nullability,
    &'a Volatility,
    &'a Scalar,
);

fn unnamed(declaration: &Declaration<Scalar>) -> Unnamed<'_> {
    let Declaration {
        name: _,
        aliases,
        signature,
        returns,
        nullability,
        volatility,
        specific,
    } = declaration;
    (
        aliases,
        signature,
        returns,
        nullability,
        volatility,
        specific,
    )
}

fn assert_rows(rows: &[Row], category: Category) {
    let specific = Scalar {
        category,
        options: &[],
    };
    for row in rows {
        for &name in row.functions {
            let declaration = &scalar(name).canonical;
            assert_eq!(
                (declaration.name, unnamed(declaration)),
                (
                    name,
                    (
                        row.aliases,
                        &row.signature,
                        &row.returns,
                        &row.nullability,
                        &row.volatility,
                        &specific
                    )
                ),
            );
        }
    }
}

const DOUBLE: ReturnRule = fixed(&DataType::Double);

const MATH: &[Row] = &[
    row(
        &["abs"],
        &[value(
            "value",
            ParameterType::Var(Some(&TypeFamily::Number)),
        )],
        ReturnRule::Var,
    ),
    Row {
        aliases: &["ceiling"],
        ..row(&["ceil"], &[number("value")], ReturnRule::DecimalScaleZero)
    },
    row(&["floor"], &[number("value")], ReturnRule::DecimalScaleZero),
    row(
        &["round"],
        &[
            number("value"),
            ParameterDecl {
                name: "scale",
                ty: INTEGER,
                form: ArgumentForm::Constant,
                presence: Presence::Optional,
            },
        ],
        ReturnRule::RoundOf,
    ),
    row(
        &[
            "sqrt", "exp", "ln", "log10", "log2", "cbrt", "sin", "cos", "tan", "asin", "acos",
            "atan", "sinh", "cosh", "tanh", "degrees", "radians",
        ],
        &[number("value")],
        DOUBLE,
    ),
    row(&["power"], &[number("base"), number("exponent")], DOUBLE),
    row(&["log"], &[number("base"), number("value")], DOUBLE),
    row(&["atan2"], &[number("y"), number("x")], DOUBLE),
    row(&["sign"], &[number("value")], fixed(&DataType::Integer)),
    row(
        &["width_bucket"],
        &[
            number("value"),
            number("low"),
            number("high"),
            value("count", INTEGER),
        ],
        fixed(&DataType::Integer),
    ),
    Row {
        nullability: Nullability::Never,
        ..row(&["pi"], &[], DOUBLE)
    },
    Row {
        nullability: Nullability::Never,
        volatility: Volatility::Volatile,
        ..row(&["random"], &[], DOUBLE)
    },
];

#[test]
fn each_math_function_has_its_rows_declaration() {
    assert_rows(MATH, Category::Math);
}

#[test]
fn the_math_functions_are_rule_13s_in_its_order() {
    assert_eq!(declared(Category::Math), functions(MATH));
}

#[test]
fn get_finds_ceil_by_its_alias_ceiling() {
    assert_eq!(scalar("ceiling").canonical.name, "ceil");
}

#[test]
fn on_datafusion_sign_is_signum_and_otherwise_the_canonical_declaration() {
    let sign = scalar("sign");
    let signum = sign.on(Dialect::DataFusion).expect("DataFusion has sign");
    assert_eq!(
        (signum.name, unnamed(signum)),
        ("signum", unnamed(&sign.canonical))
    );
}

#[test]
fn the_lookup_of_signum_in_datafusion_finds_sign() {
    let find = |name| registry().find(DocumentDialect::Dialect(Dialect::DataFusion), name);
    assert_eq!(find("signum"), Some(FunctionRef::Scalar(scalar("sign"))));
    assert_eq!(find("sign"), None);
}

#[test]
fn on_datafusion_width_bucket_is_none() {
    assert_eq!(scalar("width_bucket").on(Dialect::DataFusion), None);
}

const ANY: ParameterType = ParameterType::Any;

const CONDITIONAL: &[Row] = &[
    Row {
        signature: Signature {
            parameters: &[value("value", ParameterType::Var(None))],
            variadic: Some(value("value", ParameterType::Var(None))),
            constraints: &[],
        },
        nullability: Nullability::AllArgs,
        ..row(&["coalesce"], &[], ReturnRule::Var)
    },
    Row {
        aliases: &["null_if"],
        signature: Signature {
            parameters: &[value("left", ANY), value("right", ANY)],
            variadic: None,
            constraints: &[Constraint::OneFamily],
        },
        nullability: Nullability::Always,
        ..row(&["nullif"], &[], ReturnRule::SameAs(0))
    },
    Row {
        signature: Signature {
            parameters: &[],
            variadic: Some(value("value", ANY)),
            constraints: &[Constraint::OneFamily],
        },
        ..row(&["greatest", "least"], &[], ReturnRule::Promote)
    },
];

#[test]
fn each_conditional_function_has_its_declaration() {
    assert_rows(CONDITIONAL, Category::Conditional);
}

#[test]
fn the_conditional_functions_are_rule_15s_in_its_order() {
    assert_eq!(declared(Category::Conditional), functions(CONDITIONAL));
}

#[test]
fn get_finds_nullif_by_its_alias_null_if() {
    assert_eq!(scalar("null_if").canonical.name, "nullif");
}

#[test]
fn greatest_and_least_are_the_only_functions_that_take_promote() {
    let promoting = registry()
        .all()
        .filter_map(|function| {
            let (name, returns) = match function {
                FunctionRef::Scalar(function) => {
                    (function.canonical.name, &function.canonical.returns)
                }
                FunctionRef::Aggregate(function, _) => (
                    function.declarations.canonical.name,
                    &function.declarations.canonical.returns,
                ),
                FunctionRef::Window(function) => {
                    (function.canonical.name, &function.canonical.returns)
                }
            };
            (*returns == ReturnRule::Promote).then_some(name)
        })
        .collect::<Vec<_>>();
    assert_eq!(promoting, ["greatest", "least"]);
}

#[test]
fn on_datafusion_greatest_and_least_are_null_only_when_every_argument_is_null() {
    for name in ["greatest", "least"] {
        let function = scalar(name);
        let datafusion = function.on(Dialect::DataFusion).expect("DataFusion has it");
        let (aliases, signature, returns, _, volatility, specific) = unnamed(&function.canonical);
        assert_eq!(
            (datafusion.name, unnamed(datafusion)),
            (
                name,
                (
                    aliases,
                    signature,
                    returns,
                    &Nullability::AllArgs,
                    volatility,
                    specific
                )
            ),
        );
    }
}
