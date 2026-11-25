use semstrait_common::{Dialect, Release};

use super::super::{canonical, constant, fixed, scalar, value, INTEGER, STRING, STRING_VALUE};
use crate::datatype::{DataType, TypeFamily};
use crate::functions::{
    ArgumentForm, Category, Condition, Constraint, Declaration, ExactType, Function, Nullability,
    ParameterDecl, ParameterDefault, ParameterType, Presence, ReturnRule, Scalar, Signature,
    Volatility,
};

pub(crate) static FUNCTIONS: &[&Function<Scalar>] = &[
    &EXTRACT,
    &DATE_TRUNC,
    &DATE_DIFF,
    &CURRENT_DATE,
    &CURRENT_TIME,
    &CURRENT_TIMESTAMP,
    &MAKE_DATE,
    &TO_DATE,
    &TO_TIMESTAMP,
];

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
const GRAINS: &[&str] = PARTS.split_at(4).1;

const DATE_OR_TIMESTAMP: ParameterType = ParameterType::Family(&TypeFamily::DateAndTimestamp);
const T_DATE_OR_TIMESTAMP: ParameterType = ParameterType::Var(Some(&TypeFamily::DateAndTimestamp));

static TIMESTAMP: DataType = DataType::timestamp();

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
    let declaration = scalar(
        Category::Temporal,
        name,
        parameters,
        returns,
        Nullability::Mirror,
    );
    Declaration {
        signature: Signature {
            constraints,
            ..declaration.signature
        },
        ..declaration
    }
}

const fn refused_on_a_date(values: &'static [&'static str]) -> Constraint {
    Constraint::EnumRefused {
        enumeration: 0,
        values,
        argument: 1,
        ty: ParameterType::Exact(ExactType::Type(&DataType::Date)),
    }
}

const fn date_part_component(parts: &'static [(&'static str, &'static str)]) -> ParameterDecl {
    ParameterDecl {
        form: ArgumentForm::DialectEnum(parts),
        ..value("part", STRING)
    }
}

const DATE_PART_NAMES: &[(&str, &str)] = &[
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

const DATE_PART_NAMES_53: &[(&str, &str)] = &[
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

static EXTRACT: Function<Scalar> = Function {
    canonical: temporal(
        "extract",
        &[
            enumeration(
                "component",
                &[
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
                ],
            ),
            value("value", DATE_OR_TIMESTAMP),
        ],
        &[refused_on_a_date(&["hour", "minute", "second"])],
        fixed(&DataType::BigInt),
    ),
    dialects: &[(
        Dialect::DataFusion,
        Some(temporal(
            "date_part",
            &[
                date_part_component(DATE_PART_NAMES),
                value("value", DATE_OR_TIMESTAMP),
            ],
            &[],
            fixed(&DataType::BigInt),
        )),
    )],
    releases: &[(
        Dialect::DataFusion,
        Release::DATAFUSION_OLDEST..=Release(53),
        Some(temporal(
            "date_part",
            &[
                date_part_component(DATE_PART_NAMES_53),
                value("value", DATE_OR_TIMESTAMP),
            ],
            &[],
            fixed(&DataType::BigInt),
        )),
    )],
};

static DATE_TRUNC: Function<Scalar> = Function {
    canonical: temporal(
        "date_trunc",
        &[
            enumeration("granularity", GRAINS),
            value("value", T_DATE_OR_TIMESTAMP),
            ParameterDecl {
                presence: Presence::When {
                    condition: Condition::Type {
                        argument: 1,
                        ty: ParameterType::AnyPrecision(&TIMESTAMP),
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
    dialects: &[(
        Dialect::DataFusion,
        Some(temporal(
            "date_trunc",
            &[
                constant("granularity", STRING),
                value("value", T_DATE_OR_TIMESTAMP),
            ],
            &[],
            ReturnRule::Var,
        )),
    )],
    releases: &[],
};

static DATE_DIFF: Function<Scalar> = Function {
    canonical: temporal(
        "date_diff",
        &[
            enumeration("part", PARTS),
            value("start", T_DATE_OR_TIMESTAMP),
            value("end", T_DATE_OR_TIMESTAMP),
        ],
        &[],
        fixed(&DataType::BigInt),
    ),
    dialects: &[(Dialect::DataFusion, None)],
    releases: &[],
};

const fn current(name: &'static str, returns: &'static DataType) -> Function<Scalar> {
    canonical(Declaration {
        nullability: Nullability::Never,
        volatility: Volatility::Stable,
        ..temporal(name, &[], &[], fixed(returns))
    })
}

static CURRENT_DATE: Function<Scalar> = current("current_date", &DataType::Date);
static CURRENT_TIME: Function<Scalar> = current("current_time", &DataType::Time { precision: 9 });
static CURRENT_TIMESTAMP: Function<Scalar> =
    current("current_timestamp", &DataType::Timestamp { precision: 9 });

static MAKE_DATE: Function<Scalar> = canonical(temporal(
    "make_date",
    &[
        value("year", INTEGER),
        value("month", INTEGER),
        value("day", INTEGER),
    ],
    &[],
    fixed(&DataType::Date),
));

const fn from_text(name: &'static str, returns: &'static DataType) -> Function<Scalar> {
    const TEXT: &[ParameterDecl] = &[STRING_VALUE];
    canonical(temporal(name, TEXT, &[], fixed(returns)))
}

static TO_DATE: Function<Scalar> = from_text("to_date", &DataType::Date);
static TO_TIMESTAMP: Function<Scalar> =
    from_text("to_timestamp", &DataType::Timestamp { precision: 9 });
