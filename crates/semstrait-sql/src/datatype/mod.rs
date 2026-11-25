mod declaration;
pub(crate) mod family;
mod name;
#[cfg(feature = "schemars")]
mod schema;
pub(crate) mod type_system;
mod written;

use semstrait_common::Dialect;

use crate::DocumentDialect;

pub use declaration::{Limit, TypeDeclaration, TypeDeclarations, Values};
pub use family::{TypeFamily, TypeOrFamily};
#[cfg(feature = "schemars")]
pub use schema::any_case;
pub use type_system::{promote, CanonicalTypeSystem, Decimal, SubstraitTypeSystem, TypeSystem};
pub use written::WrittenDataType;

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DataType {
    Boolean,
    TinyInt,
    SmallInt,
    Integer,
    BigInt,
    Real,
    Double,
    Decimal { precision: u8, scale: i8 },
    String,
    VarChar { length: u32 },
    FixedChar { length: u32 },
    Binary,
    FixedBinary { length: u32 },
    Date,
    Time { precision: u8 },
    Timestamp { precision: u8 },
    Interval,
    IntervalYearMonth,
    IntervalDaySecond { precision: u8 },
    Uuid,
    Array(Box<DataType>),
    Unknown,
}

impl DataType {
    pub const fn decimal() -> Self {
        Self::Decimal {
            precision: 38,
            scale: 9,
        }
    }

    pub const fn time() -> Self {
        Self::Time { precision: 9 }
    }

    pub const fn timestamp() -> Self {
        Self::Timestamp { precision: 6 }
    }

    pub const fn interval_day_second() -> Self {
        Self::IntervalDaySecond { precision: 9 }
    }

    pub fn is_valid(&self) -> bool {
        let fits = self
            .declarations()
            .is_some_and(|declarations| self.fits(&declarations.canonical));
        match self {
            Self::Array(element) => fits && element.is_valid(),
            _ => fits,
        }
    }

    pub fn is_valid_on(&self, dialect: Dialect) -> bool {
        self.is_valid_in(DocumentDialect::Dialect(dialect))
    }

    fn is_valid_in(&self, dialect: DocumentDialect) -> bool {
        let fits = self.is_valid()
            && self
                .declarations()
                .and_then(|declarations| declarations.in_document(dialect))
                .is_some_and(|declaration| self.fits(declaration));
        match self {
            Self::Array(element) => fits && element.is_valid_in(dialect),
            _ => fits,
        }
    }

    fn fits(&self, declaration: &TypeDeclaration) -> bool {
        let taken = self
            .values()
            .zip(declaration.limits())
            .all(|(value, (.., limit))| {
                u32::try_from(value).is_ok_and(|value| limit.values.contains(value))
            });
        taken
            && !matches!(*self, Self::Decimal { precision, scale } if i16::from(scale) > i16::from(precision))
    }

    fn values(&self) -> impl Iterator<Item = i64> {
        let (first, second) = match *self {
            Self::Decimal { precision, scale } => {
                (Some(i64::from(precision)), Some(i64::from(scale)))
            }
            Self::Time { precision }
            | Self::Timestamp { precision }
            | Self::IntervalDaySecond { precision } => (Some(i64::from(precision)), None),
            Self::VarChar { length }
            | Self::FixedChar { length }
            | Self::FixedBinary { length } => (Some(i64::from(length)), None),
            Self::Boolean
            | Self::TinyInt
            | Self::SmallInt
            | Self::Integer
            | Self::BigInt
            | Self::Real
            | Self::Double
            | Self::String
            | Self::Binary
            | Self::Date
            | Self::Interval
            | Self::IntervalYearMonth
            | Self::Uuid
            | Self::Array(_)
            | Self::Unknown => (None, None),
        };
        first.into_iter().chain(second)
    }
}
