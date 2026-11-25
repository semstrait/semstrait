mod catalog_math_conditional;
mod catalog_pattern_array;
mod catalog_temporal;
mod catalog_window;
mod functions;
mod string_functions;

use crate::datatype::{DataType, TypeFamily};
use crate::expr::{AggregateAllowed, FunctionRef};
use crate::functions::{
    registry, AggregateFunction, ArgumentForm, Category, Declaration, ExactType, Function,
    Nullability, ParameterDecl, ParameterDefault, ParameterType, Presence, ReturnRule, Scalar,
    Signature, Volatility, Window,
};

const STRING: ParameterType = ParameterType::Family(&TypeFamily::String);
const INTEGER: ParameterType = ParameterType::Exact(ExactType::Type(&DataType::Integer));

fn function(name: &str) -> FunctionRef<AggregateAllowed> {
    registry()
        .get(name)
        .unwrap_or_else(|| panic!("the catalog declares `{name}`"))
}

fn scalar(name: &str) -> &'static Function<Scalar> {
    match function(name) {
        FunctionRef::Scalar(function) => function,
        other => panic!("{name} is no scalar function of the catalog: {other:?}"),
    }
}

fn aggregate_function(name: &str) -> &'static AggregateFunction {
    match function(name) {
        FunctionRef::Aggregate(function, AggregateAllowed) => function,
        other => panic!("{name} is no aggregate function of the catalog: {other:?}"),
    }
}

fn window_function(name: &str) -> &'static Function<Window> {
    match function(name) {
        FunctionRef::Window(function) => function,
        other => panic!("{name} is no window function of the catalog: {other:?}"),
    }
}

const fn value(name: &'static str, ty: ParameterType) -> ParameterDecl {
    ParameterDecl {
        name,
        ty,
        form: ArgumentForm::Value,
        presence: Presence::Required,
    }
}

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

const fn row(
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
