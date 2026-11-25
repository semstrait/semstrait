use semstrait_common::Dialect;

use super::super::{
    canonical, constant, defaulted, fixed, optional, scalar, value, ARRAY_OF_STRING, INTEGER,
    STRING, STRING_VALUE,
};
use crate::datatype::DataType;
use crate::functions::{
    Category, Function, Nullability, ParameterDecl, ParameterDefault, ReturnRule, Scalar,
};

pub(crate) static FUNCTIONS: &[&Function<Scalar>] = &[
    &REGEXP_LIKE,
    &REGEXP_MATCH,
    &REGEXP_EXTRACT,
    &REGEXP_REPLACE,
    &REGEXP_COUNT,
    &REGEXP_INSTR,
];

const PATTERN: ParameterDecl = value("pattern", STRING);
const REPLACEMENT: ParameterDecl = value("replacement", STRING);
const FLAGS: ParameterDecl = optional(constant("flags", STRING));

static REGEXP_LIKE: Function<Scalar> = canonical(scalar(
    Category::Pattern,
    "regexp_like",
    &[STRING_VALUE, PATTERN],
    fixed(&DataType::Boolean),
    Nullability::Mirror,
));

static REGEXP_MATCH: Function<Scalar> = canonical(scalar(
    Category::Pattern,
    "regexp_match",
    &[STRING_VALUE, PATTERN],
    ReturnRule::Fixed(ARRAY_OF_STRING),
    Nullability::Always,
));

static REGEXP_EXTRACT: Function<Scalar> = Function {
    canonical: scalar(
        Category::Pattern,
        "regexp_extract",
        &[STRING_VALUE, PATTERN, constant("group", INTEGER)],
        fixed(&DataType::String),
        Nullability::Always,
    ),
    dialects: &[(Dialect::DataFusion, None)],
    releases: &[],
};

static REGEXP_REPLACE: Function<Scalar> = Function {
    canonical: scalar(
        Category::Pattern,
        "regexp_replace",
        &[STRING_VALUE, PATTERN, REPLACEMENT],
        fixed(&DataType::String),
        Nullability::Mirror,
    ),
    dialects: &[(
        Dialect::DataFusion,
        Some(scalar(
            Category::Pattern,
            "regexp_replace",
            &[STRING_VALUE, PATTERN, REPLACEMENT, FLAGS],
            fixed(&DataType::String),
            Nullability::Mirror,
        )),
    )],
    releases: &[],
};

static REGEXP_COUNT: Function<Scalar> = canonical(scalar(
    Category::Pattern,
    "regexp_count",
    &[STRING_VALUE, PATTERN],
    fixed(&DataType::BigInt),
    Nullability::Mirror,
));

static REGEXP_INSTR: Function<Scalar> = canonical(scalar(
    Category::Pattern,
    "regexp_instr",
    &[
        STRING_VALUE,
        PATTERN,
        defaulted(value("start", INTEGER), ParameterDefault::Integer(1)),
        defaulted(value("occurrence", INTEGER), ParameterDefault::Integer(1)),
        defaulted(FLAGS, ParameterDefault::String("")),
        optional(constant("group", INTEGER)),
    ],
    fixed(&DataType::BigInt),
    Nullability::Mirror,
));
