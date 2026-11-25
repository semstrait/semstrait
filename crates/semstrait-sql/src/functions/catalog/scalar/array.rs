use super::super::{canonical, fixed, scalar, value, ARRAY_OF_STRING, INTEGER};
use crate::datatype::DataType;
use crate::functions::{Category, Function, Nullability, ParameterType, Scalar};

pub(crate) static FUNCTIONS: &[&Function<Scalar>] = &[&ARRAY_ELEMENT];

static ARRAY_ELEMENT: Function<Scalar> = canonical(scalar(
    Category::Array,
    "array_element",
    &[
        value("array", ParameterType::Exact(ARRAY_OF_STRING)),
        value("index", INTEGER),
    ],
    fixed(&DataType::String),
    Nullability::Always,
));
