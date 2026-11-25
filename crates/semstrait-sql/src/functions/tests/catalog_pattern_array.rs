use super::{constant, defaulted, fixed, optional, row, scalar, value, INTEGER, STRING};
use crate::datatype::DataType;
use crate::expr::FunctionRef;
use crate::functions::{
    registry, Category, Declaration, ExactType, Nullability, ParameterDecl, ParameterDefault,
    ParameterType, ReturnRule, Scalar,
};
use crate::DocumentDialect;
use semstrait_common::Dialect;

const ARRAY_OF_STRING: ExactType = ExactType::Array(&DataType::String);
const VALUE: ParameterDecl = value("value", STRING);
const PATTERN: ParameterDecl = value("pattern", STRING);

static PATTERN_FUNCTIONS: [Declaration<Scalar>; 6] = [
    row(
        Category::Pattern,
        "regexp_like",
        &[VALUE, PATTERN],
        fixed(&DataType::Boolean),
        Nullability::Mirror,
    ),
    row(
        Category::Pattern,
        "regexp_match",
        &[VALUE, PATTERN],
        ReturnRule::Fixed(ARRAY_OF_STRING),
        Nullability::Always,
    ),
    row(
        Category::Pattern,
        "regexp_extract",
        &[VALUE, PATTERN, constant("group", INTEGER)],
        fixed(&DataType::String),
        Nullability::Always,
    ),
    row(
        Category::Pattern,
        "regexp_replace",
        &[VALUE, PATTERN, value("replacement", STRING)],
        fixed(&DataType::String),
        Nullability::Mirror,
    ),
    row(
        Category::Pattern,
        "regexp_count",
        &[VALUE, PATTERN],
        fixed(&DataType::BigInt),
        Nullability::Mirror,
    ),
    row(
        Category::Pattern,
        "regexp_instr",
        &[
            VALUE,
            PATTERN,
            defaulted(value("start", INTEGER), ParameterDefault::Integer(1)),
            defaulted(value("occurrence", INTEGER), ParameterDefault::Integer(1)),
            defaulted(constant("flags", STRING), ParameterDefault::String("")),
            optional(constant("group", INTEGER)),
        ],
        fixed(&DataType::BigInt),
        Nullability::Mirror,
    ),
];

#[test]
fn the_pattern_functions_are_rule_12s_table() {
    for expected in &PATTERN_FUNCTIONS {
        assert_eq!(&scalar(expected.name).canonical, expected);
    }
}

#[test]
fn array_element_is_rule_16s_row() {
    static ARRAY_ELEMENT: Declaration<Scalar> = row(
        Category::Array,
        "array_element",
        &[
            value("array", ParameterType::Exact(ARRAY_OF_STRING)),
            value("index", INTEGER),
        ],
        fixed(&DataType::String),
        Nullability::Always,
    );
    assert_eq!(scalar("array_element").canonical, ARRAY_ELEMENT);
}

#[test]
fn datafusion_declares_regexp_replace_with_its_flags() {
    static DATAFUSION: Declaration<Scalar> = row(
        Category::Pattern,
        "regexp_replace",
        &[
            VALUE,
            PATTERN,
            value("replacement", STRING),
            optional(constant("flags", STRING)),
        ],
        fixed(&DataType::String),
        Nullability::Mirror,
    );
    assert_eq!(
        scalar("regexp_replace").on(Dialect::DataFusion),
        Some(&DATAFUSION)
    );
}

#[test]
fn datafusion_has_no_regexp_extract() {
    let regexp_extract = scalar("regexp_extract");
    assert_eq!(regexp_extract.on(Dialect::DataFusion), None);
    let find = |dialect| registry().find(dialect, "regexp_extract");
    assert_eq!(find(DocumentDialect::Dialect(Dialect::DataFusion)), None);
    assert_eq!(
        find(DocumentDialect::Canonical),
        Some(FunctionRef::Scalar(regexp_extract))
    );
}
