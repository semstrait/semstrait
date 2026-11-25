use semstrait_common::diagnostic::{Diagnose, Severity};
use semstrait_common::Dialect;
use thiserror::Error;

use super::{Category, ParameterType};
use crate::datatype::TypeOrFamily;

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SealFault {
    #[error("two functions share the name `{name}`")]
    SharedName { name: &'static str },
    #[error("two functions share the name `{name}` in {dialect:?}")]
    SharedDialectName {
        dialect: Dialect,
        name: &'static str,
    },
    #[error("the function name `{name}` is an operation's name")]
    OperationName { name: &'static str },
    #[error("`{function}`: the category {category:?} is not its file's, {file:?}")]
    CategoryNotItsFile {
        function: &'static str,
        category: Category,
        file: Category,
    },
    #[error("`{function}`: the option `{option}` has no value `{value}`")]
    OptionValueOutsideOption {
        function: &'static str,
        option: &'static str,
        value: &'static str,
    },
    #[error("`{function}`: the canonical value of `{option}` is not among its values")]
    CanonicalValueNotAmongValues {
        function: &'static str,
        option: &'static str,
    },
    #[error("`{function}`: the required `{parameter}` follows an optional parameter")]
    OptionalBeforeRequired {
        function: &'static str,
        parameter: &'static str,
    },
    #[error("`{function}`: the optional `{parameter}`, which a later parameter follows, declares no default")]
    OptionalWithNoDefault {
        function: &'static str,
        parameter: &'static str,
    },
    #[error("`{function}`: the default of `{parameter}` is not a literal of its type")]
    DefaultNotOfItsType {
        function: &'static str,
        parameter: &'static str,
    },
    #[error("`{function}`: the condition of `{parameter}` names a later argument")]
    ConditionNamesLaterArgument {
        function: &'static str,
        parameter: &'static str,
    },
    #[error("`{function}`: the argument index {index} is out of range")]
    ArgumentIndexOutOfRange {
        function: &'static str,
        index: usize,
    },
    #[error("`{function}`: no parameter binds the type variable `T`")]
    UnboundTypeVariable { function: &'static str },
    #[error("`{function}`: a scalar function has an aggregate's nullability")]
    AggregateNullability { function: &'static str },
    #[error("`{function}`: two declarations for {dialect:?}")]
    TwoDialectDeclarations {
        function: &'static str,
        dialect: Dialect,
    },
    #[error("`{function}`: the {dialect:?} declaration has aliases")]
    DialectAliases {
        function: &'static str,
        dialect: Dialect,
    },
    #[error("`{function}`: the {dialect:?} declaration equals the canonical one")]
    DialectEqualsCanonical {
        function: &'static str,
        dialect: Dialect,
    },
    #[error("`{function}`: two `releases` entries for {dialect:?} overlap")]
    OverlappingReleases {
        function: &'static str,
        dialect: Dialect,
    },
    #[error("`{function}`: a `releases` entry for {dialect:?} holds the reference release")]
    ReleaseHoldsReference {
        function: &'static str,
        dialect: Dialect,
    },
}

impl Diagnose for SealFault {
    fn message(&self) -> String {
        self.to_string()
    }

    fn severity_default(&self) -> Severity {
        Severity::Error
    }
}

#[derive(Debug)]
pub(crate) enum ArgumentRefusal {
    Argument {
        parameter: &'static str,
        takes: &'static ParameterType,
        got: TypeOrFamily,
    },
    Arguments {
        types: Vec<TypeOrFamily>,
    },
    TypeVariable {
        bound: TypeOrFamily,
        got: TypeOrFamily,
    },
    EnumRefused {
        parameter: &'static str,
        value: String,
        got: TypeOrFamily,
    },
    ParameterRefused {
        parameter: &'static str,
    },
    ParameterRequired {
        parameter: &'static str,
    },
}
