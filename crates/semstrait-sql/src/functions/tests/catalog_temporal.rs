use super::{constant, fixed, row, scalar, value, INTEGER, STRING};
use crate::datatype::{DataType, TypeFamily};
use crate::expr::FunctionRef;
use crate::functions::{
    registry, ArgumentForm, Category, Condition, Constraint, Declaration, ExactType, Nullability,
    ParameterDecl, ParameterDefault, ParameterType, Presence, ReturnRule, Scalar, Signature,
    Volatility,
};
use crate::DocumentDialect;
use semstrait_common::{Dialect, Release};

const COMPONENTS: &[&str] = &[
    "year",
    "quarter",
    "month",
    "week",
    "day",
    "day_of_week",
    "day_of_year",
    "hour",
    "minute",
    "second",
];

const GRAINS: &[&str] = &["minute", "hour", "day", "week", "month", "quarter", "year"];
const PARTS: &[&str] = &[
    "nanosecond",
    "microsecond",
    "millisecond",
    "second",
    "minute",
    "hour",
    "day",
    "week",
    "month",
    "quarter",
    "year",
];

const DATE: ParameterType = ParameterType::Exact(ExactType::Type(&DataType::Date));
const DATE_OR_TIMESTAMP: ParameterType = ParameterType::Family(&TypeFamily::DateAndTimestamp);
const T_DATE_OR_TIMESTAMP: ParameterType = ParameterType::Var(Some(&TypeFamily::DateAndTimestamp));

const fn enumeration(name: &'static str, values: &'static [&'static str]) -> ParameterDecl {
    ParameterDecl {
        form: ArgumentForm::Enum(values),
        ..value(name, STRING)
    }
}

const fn temporal(
    name: &'static str,
    parameters: &'static [ParameterDecl],
    constraints: &'static [Constraint],
    returns: ReturnRule,
) -> Declaration<Scalar> {
    let row = row(
        Category::Temporal,
        name,
        parameters,
        returns,
        Nullability::Mirror,
    );
    Declaration {
        signature: Signature {
            constraints,
            ..row.signature
        },
        ..row
    }
}

const fn current(name: &'static str, returns: &'static DataType) -> Declaration<Scalar> {
    Declaration {
        nullability: Nullability::Never,
        volatility: Volatility::Stable,
        ..temporal(name, &[], &[], fixed(returns))
    }
}

const fn refused_on_a_date(values: &'static [&'static str]) -> Constraint {
    Constraint::EnumRefused {
        enumeration: 0,
        values,
        argument: 1,
        ty: DATE,
    }
}

static RULE_14: &[Declaration<Scalar>] = &[
    temporal(
        "extract",
        &[
            enumeration("component", COMPONENTS),
            value("value", DATE_OR_TIMESTAMP),
        ],
        &[refused_on_a_date(&["hour", "minute", "second"])],
        fixed(&DataType::BigInt),
    ),
    temporal(
        "date_trunc",
        &[
            enumeration("granularity", GRAINS),
            value("value", T_DATE_OR_TIMESTAMP),
            ParameterDecl {
                presence: Presence::When {
                    condition: Condition::Type {
                        argument: 1,
                        ty: ParameterType::AnyPrecision(&DataType::Timestamp { precision: 6 }),
                    },
                    default: Some(ParameterDefault::String("UTC")),
                },
                ..constant("timezone", STRING)
            },
            ParameterDecl {
                presence: Presence::When {
                    condition: Condition::Value {
                        argument: 0,
                        value: "week",
                    },
                    default: Some(ParameterDefault::String("monday")),
                },
                ..enumeration("week_start", &["monday", "sunday"])
            },
        ],
        &[refused_on_a_date(&["minute", "hour"])],
        ReturnRule::Var,
    ),
    temporal(
        "date_diff",
        &[
            enumeration("part", PARTS),
            value("start", T_DATE_OR_TIMESTAMP),
            value("end", T_DATE_OR_TIMESTAMP),
        ],
        &[],
        fixed(&DataType::BigInt),
    ),
    current("current_date", &DataType::Date),
    current("current_time", &DataType::Time { precision: 9 }),
    current("current_timestamp", &DataType::Timestamp { precision: 9 }),
    temporal(
        "make_date",
        &[
            value("year", INTEGER),
            value("month", INTEGER),
            value("day", INTEGER),
        ],
        &[],
        fixed(&DataType::Date),
    ),
    temporal(
        "to_date",
        &[value("value", STRING)],
        &[],
        fixed(&DataType::Date),
    ),
    temporal(
        "to_timestamp",
        &[value("value", STRING)],
        &[],
        fixed(&DataType::Timestamp { precision: 9 }),
    ),
];

#[test]
fn each_temporal_function_is_declared_as_rule_14_gives_it() {
    for expected in RULE_14 {
        assert_eq!(scalar(expected.name).canonical, *expected);
    }
    let temporal = registry()
        .all()
        .filter_map(|function| match function {
            FunctionRef::Scalar(function)
                if function.canonical.specific.category == Category::Temporal =>
            {
                Some(function.canonical.name)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let names = RULE_14.iter().map(|declaration| declaration.name);
    assert_eq!(temporal, names.collect::<Vec<_>>());
}

#[test]
fn get_finds_no_retired_name() {
    let retired = [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "date_part",
        "date_add",
        "date_sub",
    ];
    for name in retired {
        assert_eq!(registry().get(name), None, "`{name}` retires");
    }
}

const PART_NAMES: &[(&str, &str)] = &[
    ("year", "year"),
    ("quarter", "quarter"),
    ("month", "month"),
    ("week", "week"),
    ("day", "day"),
    ("day_of_week", "isodow"),
    ("day_of_year", "doy"),
    ("hour", "hour"),
    ("minute", "minute"),
    ("second", "second"),
];

static DATE_PART: Declaration<Scalar> = temporal(
    "date_part",
    &[
        ParameterDecl {
            form: ArgumentForm::DialectEnum(PART_NAMES),
            ..value("part", STRING)
        },
        value("value", DATE_OR_TIMESTAMP),
    ],
    &[],
    fixed(&DataType::BigInt),
);

#[test]
fn data_fusion_s_part_enum_names_each_canonical_component_of_extract() {
    let extract = scalar("extract");
    let part = |declaration: &Declaration<Scalar>| &declaration.signature.parameters[0].form;
    let ArgumentForm::Enum(components) = part(&extract.canonical) else {
        panic!("`extract`'s component is an enum");
    };
    let Some(ArgumentForm::DialectEnum(names)) = extract.on(Dialect::DataFusion).map(part) else {
        panic!("DataFusion's part is a dialect's enum");
    };
    let named = names.iter().map(|(component, _)| component);
    assert!(named.eq(components.iter()), "{names:?}");
}

static DATA_FUSION_DATE_TRUNC: Declaration<Scalar> = temporal(
    "date_trunc",
    &[
        constant("granularity", STRING),
        value("value", T_DATE_OR_TIMESTAMP),
    ],
    &[],
    ReturnRule::Var,
);

#[test]
fn on_data_fusion_gives_rule_23s_declaration_or_none_for_date_diff() {
    let on = |name| scalar(name).on(Dialect::DataFusion);
    assert_eq!(on("extract"), Some(&DATE_PART));
    assert_eq!(on("date_trunc"), Some(&DATA_FUSION_DATE_TRUNC));
    assert_eq!(on("date_diff"), None);
}

#[test]
fn the_lookup_of_date_part_finds_extract_in_data_fusion_and_nothing_in_canonical() {
    let find = |dialect| registry().find(dialect, "date_part");
    assert_eq!(
        find(DocumentDialect::Dialect(Dialect::DataFusion)),
        Some(FunctionRef::Scalar(scalar("extract")))
    );
    assert_eq!(find(DocumentDialect::Canonical), None);
}

const PART_NAMES_53: &[(&str, &str)] = &[
    ("year", "year"),
    ("quarter", "quarter"),
    ("month", "month"),
    ("week", "week"),
    ("day", "day"),
    ("day_of_year", "doy"),
    ("hour", "hour"),
    ("minute", "minute"),
    ("second", "second"),
];

static DATE_PART_53: Declaration<Scalar> = temporal(
    "date_part",
    &[
        ParameterDecl {
            form: ArgumentForm::DialectEnum(PART_NAMES_53),
            ..value("part", STRING)
        },
        value("value", DATE_OR_TIMESTAMP),
    ],
    &[],
    fixed(&DataType::BigInt),
);

#[test]
fn on_release_gives_the_53_declaration_with_no_day_of_week_part() {
    let on_release = |release| scalar("extract").on_release(Dialect::DataFusion, Release(release));
    assert_eq!(on_release(53), Some(&DATE_PART_53));
}

#[test]
fn on_release_gives_on_data_fusion_above_53() {
    for release in [54, 55, 56] {
        assert_eq!(
            scalar("extract").on_release(Dialect::DataFusion, Release(release)),
            scalar("extract").on(Dialect::DataFusion),
            "{release}"
        );
    }
}

#[test]
fn on_release_gives_on_for_every_release_where_the_function_has_no_releases_entry() {
    for release in [53, 54, 55, 56] {
        assert_eq!(
            scalar("date_trunc").on_release(Dialect::DataFusion, Release(release)),
            scalar("date_trunc").on(Dialect::DataFusion),
            "{release}"
        );
    }
}
