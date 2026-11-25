use super::{fixed, optional, scalar, value, INTEGER, STRING};
use crate::datatype::DataType;
use crate::expr::FunctionRef;
use crate::functions::{
    registry, Category, Declaration, Nullability, OptionDecl, ParameterDecl, ReturnRule, Scalar,
    ScalarOption, Signature, Volatility,
};
use crate::DocumentDialect;
use semstrait_common::Dialect;

struct Row {
    functions: &'static [&'static str],
    parameters: &'static [ParameterDecl],
    variadic: Option<ParameterDecl>,
    returns: ReturnRule,
    nullability: Nullability,
    options: &'static [OptionDecl<ScalarOption>],
}

const fn mirror(
    functions: &'static [&'static str],
    parameters: &'static [ParameterDecl],
    returns: ReturnRule,
) -> Row {
    Row {
        functions,
        parameters,
        variadic: None,
        returns,
        nullability: Nullability::Mirror,
        options: &[],
    }
}

const VALUE: ParameterDecl = value("value", STRING);
const SEARCH: ParameterDecl = value("search", STRING);

const RULE_11: &[Row] = &[
    mirror(
        &["upper", "lower", "reverse", "initcap"],
        &[VALUE],
        fixed(&DataType::String),
    ),
    mirror(&["length"], &[VALUE], fixed(&DataType::Integer)),
    mirror(
        &["octet_length", "bit_length", "ascii"],
        &[VALUE],
        fixed(&DataType::Integer),
    ),
    mirror(
        &["chr"],
        &[value("value", INTEGER)],
        fixed(&DataType::String),
    ),
    mirror(
        &["substring"],
        &[
            VALUE,
            value("start", INTEGER),
            optional(value("length", INTEGER)),
        ],
        fixed(&DataType::String),
    ),
    mirror(
        &["overlay"],
        &[
            VALUE,
            value("placing", STRING),
            value("start", INTEGER),
            optional(value("length", INTEGER)),
        ],
        fixed(&DataType::String),
    ),
    mirror(&["position"], &[SEARCH, VALUE], fixed(&DataType::Integer)),
    mirror(
        &["trim", "ltrim", "rtrim"],
        &[VALUE, optional(value("characters", STRING))],
        fixed(&DataType::String),
    ),
    Row {
        variadic: Some(VALUE),
        nullability: Nullability::ByOption(&[
            ("IGNORE_NULLS", Nullability::Never),
            ("ACCEPT_NULLS", Nullability::Mirror),
        ]),
        options: &[OptionDecl {
            name: ScalarOption::NullHandling,
            values: &["IGNORE_NULLS", "ACCEPT_NULLS"],
            canonical: "ACCEPT_NULLS",
        }],
        ..mirror(&["concat"], &[VALUE], fixed(&DataType::String))
    },
    Row {
        variadic: Some(VALUE),
        nullability: Nullability::MirrorArgs(&[0]),
        ..mirror(
            &["concat_ws"],
            &[value("separator", STRING)],
            fixed(&DataType::String),
        )
    },
    mirror(
        &["replace", "translate"],
        &[VALUE, value("from", STRING), value("to", STRING)],
        fixed(&DataType::String),
    ),
    mirror(
        &["lpad", "rpad"],
        &[
            VALUE,
            value("length", INTEGER),
            optional(value("fill", STRING)),
        ],
        fixed(&DataType::String),
    ),
    mirror(
        &["repeat", "left", "right"],
        &[VALUE, value("count", INTEGER)],
        fixed(&DataType::String),
    ),
    mirror(
        &["split_part"],
        &[VALUE, value("delimiter", STRING), value("part", INTEGER)],
        fixed(&DataType::String),
    ),
    mirror(
        &["starts_with", "ends_with", "contains"],
        &[VALUE, SEARCH],
        fixed(&DataType::Boolean),
    ),
    mirror(
        &["levenshtein"],
        &[VALUE, value("other", STRING)],
        fixed(&DataType::Integer),
    ),
    mirror(&["md5"], &[VALUE], fixed(&DataType::String)),
];

const ALIASES: &[(&str, &[&str])] = &[
    ("length", &["char_length", "character_length"]),
    ("trim", &["btrim"]),
];

#[test]
fn the_string_functions_are_the_thirty_of_rule_11() {
    let mut declared = registry()
        .all()
        .filter_map(|function| match function {
            FunctionRef::Scalar(function)
                if function.canonical.specific.category == Category::String =>
            {
                Some(function.canonical.name)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut expected = RULE_11
        .iter()
        .flat_map(|row| row.functions)
        .copied()
        .collect::<Vec<_>>();
    declared.sort_unstable();
    expected.sort_unstable();
    assert_eq!(expected.len(), 30);
    assert_eq!(declared, expected);
}

#[test]
fn each_string_function_has_its_row_of_rule_11() {
    for row in RULE_11 {
        for &name in row.functions {
            let declaration = &scalar(name).canonical;
            let aliases = ALIASES
                .iter()
                .find(|(function, _)| *function == name)
                .map_or(&[][..], |(_, aliases)| aliases);
            assert_eq!(declaration.name, name);
            assert_eq!(declaration.aliases, aliases, "{name}");
            assert_eq!(declaration.signature.parameters, row.parameters, "{name}");
            assert_eq!(declaration.signature.variadic, row.variadic, "{name}");
            assert_eq!(declaration.signature.constraints, [], "{name}");
            assert_eq!(declaration.returns, row.returns, "{name}");
            assert_eq!(declaration.nullability, row.nullability, "{name}");
            assert_eq!(declaration.volatility, Volatility::Immutable, "{name}");
            assert_eq!(
                declaration.specific,
                Scalar {
                    category: Category::String,
                    options: row.options
                },
                "{name}"
            );
        }
    }
}

#[test]
fn get_finds_a_string_function_by_each_alias() {
    for (name, aliases) in ALIASES {
        for alias in *aliases {
            assert_eq!(scalar(alias), scalar(name), "{alias}");
        }
    }
}

#[test]
fn position_on_datafusion_is_strpos_with_the_value_first() {
    assert_eq!(
        scalar("position").on(Dialect::DataFusion),
        Some(&Declaration {
            name: "strpos",
            aliases: &[],
            signature: Signature {
                parameters: &[VALUE, SEARCH],
                variadic: None,
                constraints: &[],
            },
            returns: fixed(&DataType::Integer),
            nullability: Nullability::Mirror,
            volatility: Volatility::Immutable,
            specific: Scalar {
                category: Category::String,
                options: &[],
            },
        })
    );
}

#[test]
fn concat_on_datafusion_takes_null_handling_with_ignore_nulls_only() {
    assert_eq!(
        scalar("concat").on(Dialect::DataFusion),
        Some(&Declaration {
            name: "concat",
            aliases: &[],
            signature: Signature {
                parameters: &[VALUE],
                variadic: Some(VALUE),
                constraints: &[],
            },
            returns: fixed(&DataType::String),
            nullability: Nullability::Never,
            volatility: Volatility::Immutable,
            specific: Scalar {
                category: Category::String,
                options: &[OptionDecl {
                    name: ScalarOption::NullHandling,
                    values: &["IGNORE_NULLS"],
                    canonical: "IGNORE_NULLS",
                }],
            },
        })
    );
}

#[test]
fn the_lookup_in_datafusion_finds_position_by_strpos() {
    let find = |name| registry().find(DocumentDialect::Dialect(Dialect::DataFusion), name);
    let found = |name| Some(FunctionRef::Scalar(scalar(name)));
    assert_eq!(find("strpos"), found("position"));
    assert_eq!(find("position"), None);
    assert_eq!(find("concat"), found("concat"));
}
