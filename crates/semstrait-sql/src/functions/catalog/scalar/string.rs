use semstrait_common::Dialect;

use super::super::{canonical, fixed, optional, scalar, value, INTEGER, STRING, STRING_VALUE};
use crate::datatype::DataType;
use crate::functions::{
    Category, Declaration, Function, Nullability, OptionDecl, ParameterDecl, Scalar, ScalarOption,
    Signature,
};

const SEARCH: ParameterDecl = value("search", STRING);

const ONE_STRING: &[ParameterDecl] = &[STRING_VALUE];
const TRIM: &[ParameterDecl] = &[STRING_VALUE, optional(value("characters", STRING))];
const REPLACE: &[ParameterDecl] = &[STRING_VALUE, value("from", STRING), value("to", STRING)];
const PAD: &[ParameterDecl] = &[
    STRING_VALUE,
    value("length", INTEGER),
    optional(value("fill", STRING)),
];
const COUNT: &[ParameterDecl] = &[STRING_VALUE, value("count", INTEGER)];
const FIND: &[ParameterDecl] = &[STRING_VALUE, SEARCH];

const fn declaration(
    name: &'static str,
    parameters: &'static [ParameterDecl],
    result: &'static DataType,
) -> Declaration<Scalar> {
    scalar(
        Category::String,
        name,
        parameters,
        fixed(result),
        Nullability::Mirror,
    )
}

const fn variadic(
    name: &'static str,
    parameters: &'static [ParameterDecl],
    tail: ParameterDecl,
) -> Declaration<Scalar> {
    let declaration = declaration(name, parameters, &DataType::String);
    Declaration {
        signature: Signature {
            variadic: Some(tail),
            ..declaration.signature
        },
        ..declaration
    }
}

const fn function(
    name: &'static str,
    parameters: &'static [ParameterDecl],
    result: &'static DataType,
) -> Function<Scalar> {
    canonical(declaration(name, parameters, result))
}

pub(crate) static FUNCTIONS: &[&Function<Scalar>] = &[
    &function("upper", ONE_STRING, &DataType::String),
    &function("lower", ONE_STRING, &DataType::String),
    &function("reverse", ONE_STRING, &DataType::String),
    &function("initcap", ONE_STRING, &DataType::String),
    &canonical(Declaration {
        aliases: &["char_length", "character_length"],
        ..declaration("length", ONE_STRING, &DataType::Integer)
    }),
    &function("octet_length", ONE_STRING, &DataType::Integer),
    &function("bit_length", ONE_STRING, &DataType::Integer),
    &function("ascii", ONE_STRING, &DataType::Integer),
    &function("chr", &[value("value", INTEGER)], &DataType::String),
    &function(
        "substring",
        &[
            STRING_VALUE,
            value("start", INTEGER),
            optional(value("length", INTEGER)),
        ],
        &DataType::String,
    ),
    &function(
        "overlay",
        &[
            STRING_VALUE,
            value("placing", STRING),
            value("start", INTEGER),
            optional(value("length", INTEGER)),
        ],
        &DataType::String,
    ),
    &Function {
        canonical: declaration("position", &[SEARCH, STRING_VALUE], &DataType::Integer),
        dialects: &[(
            Dialect::DataFusion,
            Some(declaration("strpos", FIND, &DataType::Integer)),
        )],
        releases: &[],
    },
    &canonical(Declaration {
        aliases: &["btrim"],
        ..declaration("trim", TRIM, &DataType::String)
    }),
    &function("ltrim", TRIM, &DataType::String),
    &function("rtrim", TRIM, &DataType::String),
    &Function {
        canonical: Declaration {
            nullability: Nullability::ByOption(&[
                ("IGNORE_NULLS", Nullability::Never),
                ("ACCEPT_NULLS", Nullability::Mirror),
            ]),
            specific: Scalar {
                category: Category::String,
                options: &[OptionDecl {
                    name: ScalarOption::NullHandling,
                    values: &["IGNORE_NULLS", "ACCEPT_NULLS"],
                    canonical: "ACCEPT_NULLS",
                }],
            },
            ..variadic("concat", ONE_STRING, STRING_VALUE)
        },
        dialects: &[(
            Dialect::DataFusion,
            Some(Declaration {
                nullability: Nullability::Never,
                specific: Scalar {
                    category: Category::String,
                    options: &[OptionDecl {
                        name: ScalarOption::NullHandling,
                        values: &["IGNORE_NULLS"],
                        canonical: "IGNORE_NULLS",
                    }],
                },
                ..variadic("concat", ONE_STRING, STRING_VALUE)
            }),
        )],
        releases: &[],
    },
    &canonical(Declaration {
        nullability: Nullability::MirrorArgs(&[0]),
        ..variadic("concat_ws", &[value("separator", STRING)], STRING_VALUE)
    }),
    &function("replace", REPLACE, &DataType::String),
    &function("translate", REPLACE, &DataType::String),
    &function("lpad", PAD, &DataType::String),
    &function("rpad", PAD, &DataType::String),
    &function("repeat", COUNT, &DataType::String),
    &function("left", COUNT, &DataType::String),
    &function("right", COUNT, &DataType::String),
    &function(
        "split_part",
        &[
            STRING_VALUE,
            value("delimiter", STRING),
            value("part", INTEGER),
        ],
        &DataType::String,
    ),
    &function("starts_with", FIND, &DataType::Boolean),
    &function("ends_with", FIND, &DataType::Boolean),
    &function("contains", FIND, &DataType::Boolean),
    &function(
        "levenshtein",
        &[STRING_VALUE, value("other", STRING)],
        &DataType::Integer,
    ),
    &function("md5", ONE_STRING, &DataType::String),
];
