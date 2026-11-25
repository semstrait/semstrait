mod aggregate;
mod window;

mod scalar {
    pub(super) mod array;
    pub(super) mod conditional;
    pub(super) mod math;
    pub(super) mod pattern;
    pub(super) mod string;
    pub(super) mod temporal;
}

use super::{
    value, ArgumentForm, Category, Declaration, ExactType, Function, FunctionCatalog, Nullability,
    ParameterDecl, ParameterDefault, ParameterType, Presence, ReturnRule, Scalar, Signature,
    Volatility, NUMBER, STRING,
};
use crate::datatype::DataType;

pub(super) static CATALOG: FunctionCatalog = FunctionCatalog {
    scalar: &[
        (Category::String, scalar::string::FUNCTIONS),
        (Category::Pattern, scalar::pattern::FUNCTIONS),
        (Category::Math, scalar::math::FUNCTIONS),
        (Category::Temporal, scalar::temporal::FUNCTIONS),
        (Category::Conditional, scalar::conditional::FUNCTIONS),
        (Category::Array, scalar::array::FUNCTIONS),
    ],
    aggregate: aggregate::FUNCTIONS,
    window: window::FUNCTIONS,
};

const INTEGER: ParameterType = ParameterType::Exact(ExactType::Type(&DataType::Integer));
const ARRAY_OF_STRING: ExactType = ExactType::Array(&DataType::String);

const STRING_VALUE: ParameterDecl = value("value", STRING);
const NUMBER_VALUE: ParameterDecl = value("value", NUMBER);
const T_VALUE: ParameterDecl = value("value", ParameterType::Var(None));
const VALUE_OF_ANY_TYPE: ParameterDecl = value("value", ParameterType::Any);

const fn constant(name: &'static str, ty: ParameterType) -> ParameterDecl {
    ParameterDecl {
        form: ArgumentForm::Constant,
        ..value(name, ty)
    }
}

const fn optional(parameter: ParameterDecl) -> ParameterDecl {
    ParameterDecl {
        presence: Presence::Optional,
        ..parameter
    }
}

const fn defaulted(parameter: ParameterDecl, default: ParameterDefault) -> ParameterDecl {
    ParameterDecl {
        presence: Presence::Defaulted(default),
        ..parameter
    }
}

const fn fixed(ty: &'static DataType) -> ReturnRule {
    ReturnRule::Fixed(ExactType::Type(ty))
}

const DOUBLE: ReturnRule = fixed(&DataType::Double);

const fn scalar(
    category: Category,
    name: &'static str,
    parameters: &'static [ParameterDecl],
    returns: ReturnRule,
    nullability: Nullability,
) -> Declaration<Scalar> {
    Declaration {
        name,
        aliases: &[],
        signature: Signature {
            parameters,
            variadic: None,
            constraints: &[],
        },
        returns,
        nullability,
        volatility: Volatility::Immutable,
        specific: Scalar {
            category,
            options: &[],
        },
    }
}

const fn canonical(canonical: Declaration<Scalar>) -> Function<Scalar> {
    Function {
        canonical,
        dialects: &[],
        releases: &[],
    }
}
