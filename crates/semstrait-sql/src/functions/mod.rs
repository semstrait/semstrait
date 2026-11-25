mod catalog;
mod error;
pub(crate) mod handle;
pub(crate) mod operation;
mod registry;
pub(crate) mod signature;

use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem;
use std::ops::RangeInclusive;

use semstrait_common::{Dialect, Release};

use crate::datatype::{DataType, TypeFamily};
use crate::literal::{IntoLiteral, Literal};
use crate::DocumentDialect;

pub(crate) use error::ArgumentRefusal;
pub use error::SealFault;
pub use registry::{registry, seal, FunctionCatalog, Registry};
pub use signature::Signature;

#[derive(Debug)]
pub struct Function<F: FunctionType> {
    pub canonical: Declaration<F>,
    pub dialects: &'static [(Dialect, Option<Declaration<F>>)],
    #[allow(clippy::type_complexity)]
    pub releases: &'static [(Dialect, RangeInclusive<Release>, Option<Declaration<F>>)],
}

impl<F: FunctionType> Function<F> {
    pub fn on(&'static self, dialect: Dialect) -> Option<&'static Declaration<F>> {
        crate::on_dialect(&self.canonical, self.dialects, dialect)
    }

    pub fn on_release(
        &'static self,
        dialect: Dialect,
        release: Release,
    ) -> Option<&'static Declaration<F>> {
        self.releases
            .iter()
            .find(|(entry, range, _)| *entry == dialect && range.contains(&release))
            .map_or_else(
                || self.on(dialect),
                |(_, _, declaration)| declaration.as_ref(),
            )
    }

    fn name_in(&'static self, dialect: DocumentDialect) -> Option<&'static str> {
        match dialect {
            DocumentDialect::Canonical => Some(self.canonical.name),
            DocumentDialect::Dialect(dialect) => {
                self.on(dialect).map(|declaration| declaration.name)
            }
        }
    }
}

impl<F: FunctionType> PartialEq for Function<F> {
    fn eq(&self, other: &Self) -> bool {
        self.canonical.name == other.canonical.name
    }
}

impl<F: FunctionType> Eq for Function<F> {}

impl<F: FunctionType> Hash for Function<F> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.canonical.name.hash(state);
    }
}

#[derive(Debug)]
pub struct AggregateFunction {
    pub declarations: Function<Aggregate>,
    pub additivity: Additivity,
    pub distinct_additivity: Additivity,
}

impl PartialEq for AggregateFunction {
    fn eq(&self, other: &Self) -> bool {
        self.declarations == other.declarations
    }
}

impl Eq for AggregateFunction {}

impl Hash for AggregateFunction {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.declarations.hash(state);
    }
}

#[non_exhaustive]
#[derive(Debug, PartialEq)]
pub struct Declaration<F: FunctionType> {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub signature: Signature,
    pub returns: ReturnRule,
    pub nullability: Nullability,
    pub volatility: Volatility,
    pub specific: F,
}

pub trait FunctionType: sealed::Sealed + PartialEq + 'static {}

#[non_exhaustive]
#[derive(Debug, PartialEq)]
pub struct Scalar {
    pub category: Category,
    pub options: &'static [OptionDecl<ScalarOption>],
}

#[non_exhaustive]
#[derive(Debug, PartialEq)]
pub struct Aggregate {
    pub options: &'static [OptionDecl<AggregateOption>],
    pub clauses: AggregateClauses,
}

#[non_exhaustive]
#[derive(Debug, PartialEq)]
pub struct Window {
    pub options: &'static [OptionDecl<WindowOption>],
    pub clauses: WindowClauses,
}

impl FunctionType for Scalar {}
impl FunctionType for Aggregate {}
impl FunctionType for Window {}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Category {
    String,
    Math,
    Temporal,
    Pattern,
    Conditional,
    Array,
}

#[non_exhaustive]
#[derive(Debug, PartialEq)]
pub struct ParameterDecl {
    pub name: &'static str,
    pub ty: ParameterType,
    pub form: ArgumentForm,
    pub presence: Presence,
}

impl ParameterDecl {
    pub fn default(&self) -> Option<Literal> {
        let (Presence::Defaulted(default)
        | Presence::When {
            default: Some(default),
            ..
        }) = self.presence
        else {
            return None;
        };
        Some(match default {
            ParameterDefault::Integer(value) => value.into_literal(),
            ParameterDefault::String(value) => value.into_literal(),
        })
    }
}

pub(crate) const fn value(name: &'static str, ty: ParameterType) -> ParameterDecl {
    ParameterDecl {
        name,
        ty,
        form: ArgumentForm::Value,
        presence: Presence::Required,
    }
}

pub(crate) const STRING: ParameterType = ParameterType::Family(&TypeFamily::String);
pub(crate) const NUMBER: ParameterType = ParameterType::Family(&TypeFamily::Number);
pub(crate) const BOOLEAN: ParameterType = ParameterType::Family(&TypeFamily::Boolean);

#[non_exhaustive]
#[derive(Debug, PartialEq)]
pub enum ParameterType {
    Exact(ExactType),
    Family(&'static TypeFamily),
    AnyPrecision(&'static DataType),
    Any,
    Var(Option<&'static TypeFamily>),
}

impl ParameterType {
    pub(crate) fn takes(&self, data_type: &DataType) -> bool {
        let in_family = |family: &TypeFamily| data_type.family().as_ref() == Some(family);
        match self {
            Self::Exact(ExactType::Type(named)) => *named == data_type,
            Self::Exact(ExactType::Array(element)) => {
                matches!(data_type, DataType::Array(inner) if **inner == **element)
            }
            Self::Family(family) | Self::Var(Some(family)) => in_family(family),
            Self::AnyPrecision(of) => mem::discriminant(*of) == mem::discriminant(data_type),
            Self::Any | Self::Var(None) => true,
        }
    }

    pub(crate) fn takes_value(&self, data_type: &DataType, family: Option<&TypeFamily>) -> bool {
        match (data_type, family) {
            (DataType::Unknown, None) => true,
            (DataType::Unknown, Some(family)) => match self {
                Self::Exact(exact) => exact.data_type().family().as_ref() == Some(family),
                Self::Family(of) | Self::Var(Some(of)) => *of == family,
                Self::AnyPrecision(of) => of.family().as_ref() == Some(family),
                Self::Any | Self::Var(None) => true,
            },
            (data_type, _) => self.takes(data_type),
        }
    }
}

impl fmt::Display for ParameterType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact(ExactType::Type(data_type)) => write!(formatter, "{data_type}"),
            Self::Exact(exact) => write!(formatter, "{}", exact.data_type()),
            Self::Family(family) => write!(formatter, "the {family} family"),
            Self::AnyPrecision(of) => {
                let name = of
                    .declarations()
                    .map_or("unknown", |declarations| declarations.canonical.name);
                write!(formatter, "{name} at any precision")
            }
            Self::Any => formatter.write_str("any type"),
            Self::Var(None) => formatter.write_str("T"),
            Self::Var(Some(family)) => write!(formatter, "T of the {family} family"),
        }
    }
}

#[non_exhaustive]
#[derive(Debug, PartialEq)]
pub enum ExactType {
    Type(&'static DataType),
    Array(&'static DataType),
}

impl ExactType {
    pub(crate) fn data_type(&self) -> DataType {
        match self {
            Self::Type(data_type) => (*data_type).clone(),
            Self::Array(element) => DataType::Array(Box::new((*element).clone())),
        }
    }
}

#[non_exhaustive]
#[derive(Debug, PartialEq)]
pub enum ArgumentForm {
    Value,
    Constant,
    Enum(&'static [&'static str]),
    DialectEnum(&'static [(&'static str, &'static str)]),
}

#[non_exhaustive]
#[derive(Debug, PartialEq)]
pub enum Presence {
    Required,
    Optional,
    Defaulted(ParameterDefault),
    When {
        condition: Condition,
        default: Option<ParameterDefault>,
    },
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParameterDefault {
    Integer(i32),
    String(&'static str),
}

#[non_exhaustive]
#[derive(Debug, PartialEq)]
pub enum Condition {
    Type {
        argument: usize,
        ty: ParameterType,
    },
    Value {
        argument: usize,
        value: &'static str,
    },
}

#[non_exhaustive]
#[derive(Debug, PartialEq)]
pub enum Constraint {
    EnumRefused {
        enumeration: usize,
        values: &'static [&'static str],
        argument: usize,
        ty: ParameterType,
    },
    OneFamily,
    Families(&'static [&'static [TypeFamily]]),
}

#[non_exhaustive]
#[derive(Debug, PartialEq)]
pub enum ReturnRule {
    Fixed(ExactType),
    Var,
    SameAs(usize),
    Promote,
    Sum,
    Product,
    Quotient,
    Mod,
    SumOf,
    AvgOf,
    DecimalScaleZero,
    RoundOf,
}

#[non_exhaustive]
#[derive(Debug, PartialEq)]
pub enum Nullability {
    Mirror,
    MirrorArgs(&'static [usize]),
    AllArgs,
    Always,
    Never,
    ByOption(&'static [(&'static str, Nullability)]),
    EmptyGroupNull,
    Kleene,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Volatility {
    Immutable,
    Stable,
    Volatile,
}

#[non_exhaustive]
#[derive(Debug, PartialEq)]
pub struct OptionDecl<N: OptionName> {
    pub name: N,
    pub values: &'static [&'static str],
    pub canonical: &'static str,
}

pub trait OptionName: Copy {
    fn name(self) -> &'static str;
    fn values(self) -> &'static [&'static str];
}

macro_rules! options {
    ($(#[$doc:meta])* $options:ident {
        $($(#[$option_doc:meta])* $option:ident: $name:literal [$($value:literal),*]),* $(,)?
    }) => {
        $(#[$doc])*
        #[non_exhaustive]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum $options {
            $($(#[$option_doc])* $option),*
        }

        impl $crate::functions::OptionName for $options {
            fn name(self) -> &'static str {
                match self {
                    $(Self::$option => $name),*
                }
            }

            fn values(self) -> &'static [&'static str] {
                match self {
                    $(Self::$option => &[$($value),*]),*
                }
            }
        }
    };
}

pub(crate) use options;

options!(
    ScalarOption {
        NullHandling: "null_handling" ["IGNORE_NULLS", "ACCEPT_NULLS"],
    }
);

options!(AggregateOption {});

options!(WindowOption {});

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClauseRule {
    Refused,
    Allowed,
    NoEffect,
    Required,
}

#[non_exhaustive]
#[derive(Debug, PartialEq)]
pub struct AggregateClauses {
    pub distinct: ClauseRule,
    pub filter: ClauseRule,
    pub order_by: ClauseRule,
    pub within_group: ClauseRule,
    pub within_group_key: ParameterType,
    pub nulls: ClauseRule,
}

#[non_exhaustive]
#[derive(Debug, PartialEq)]
pub struct WindowClauses {
    pub nulls: ClauseRule,
    pub partition_by: ClauseRule,
    pub order_by: ClauseRule,
    pub window_frame: ClauseRule,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Additivity {
    Additive,
    SemiAdditive,
    NonAdditive,
}

#[cfg(doctest)]
pub struct DeclarationParts;

mod sealed {
    use super::registry::option_faults;
    use super::{Scalar, SealFault};

    pub trait Sealed {
        fn scalar(&self) -> Option<&Scalar> {
            None
        }

        fn option_faults(&self, function: &'static str) -> Vec<SealFault>;
    }

    impl Sealed for Scalar {
        fn scalar(&self) -> Option<&Scalar> {
            Some(self)
        }

        fn option_faults(&self, function: &'static str) -> Vec<SealFault> {
            option_faults(function, self.options)
        }
    }

    impl Sealed for super::Aggregate {
        fn option_faults(&self, function: &'static str) -> Vec<SealFault> {
            option_faults(function, self.options)
        }
    }

    impl Sealed for super::Window {
        fn option_faults(&self, function: &'static str) -> Vec<SealFault> {
            option_faults(function, self.options)
        }
    }
}

#[cfg(test)]
mod tests;
