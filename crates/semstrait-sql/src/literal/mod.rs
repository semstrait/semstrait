mod cast;
mod check;
mod error;
mod form;
mod parse;

use crate::datatype::DataType;

pub use error::LiteralError;
pub use parse::HeldValue;

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Boolean(bool),
    TinyInt(i8),
    SmallInt(i16),
    Integer(i32),
    BigInt(i64),
    Real(f32),
    Double(f64),
    Decimal {
        value: String,
        precision: u8,
        scale: i8,
    },
    String(String),
    VarChar {
        value: String,
        length: u32,
    },
    FixedChar {
        value: String,
        length: u32,
    },
    Binary(Vec<u8>),
    FixedBinary {
        value: Vec<u8>,
        length: u32,
    },
    Date(String),
    Time {
        value: String,
        precision: u8,
    },
    Timestamp {
        value: String,
        precision: u8,
    },
    Interval(String),
    IntervalYearMonth(String),
    IntervalDaySecond {
        value: String,
        precision: u8,
    },
    Uuid(String),
    Array {
        element: DataType,
        values: Vec<Literal>,
    },
    Null(DataType),
}

impl Literal {
    pub fn data_type(&self) -> DataType {
        match *self {
            Self::Boolean(_) => DataType::Boolean,
            Self::TinyInt(_) => DataType::TinyInt,
            Self::SmallInt(_) => DataType::SmallInt,
            Self::Integer(_) => DataType::Integer,
            Self::BigInt(_) => DataType::BigInt,
            Self::Real(_) => DataType::Real,
            Self::Double(_) => DataType::Double,
            Self::Decimal {
                precision, scale, ..
            } => DataType::Decimal { precision, scale },
            Self::String(_) => DataType::String,
            Self::VarChar { length, .. } => DataType::VarChar { length },
            Self::FixedChar { length, .. } => DataType::FixedChar { length },
            Self::Binary(_) => DataType::Binary,
            Self::FixedBinary { length, .. } => DataType::FixedBinary { length },
            Self::Date(_) => DataType::Date,
            Self::Time { precision, .. } => DataType::Time { precision },
            Self::Timestamp { precision, .. } => DataType::Timestamp { precision },
            Self::Interval(_) => DataType::Interval,
            Self::IntervalYearMonth(_) => DataType::IntervalYearMonth,
            Self::IntervalDaySecond { precision, .. } => DataType::IntervalDaySecond { precision },
            Self::Uuid(_) => DataType::Uuid,
            Self::Array { ref element, .. } => DataType::Array(Box::new(element.clone())),
            Self::Null(ref data_type) => data_type.clone(),
        }
    }
}

pub trait IntoLiteral {
    fn into_literal(self) -> Literal;
}

impl IntoLiteral for Literal {
    fn into_literal(self) -> Literal {
        self
    }
}

macro_rules! into_literal {
    ($($rust:ty => $variant:ident),*) => {
        $(impl IntoLiteral for $rust {
            fn into_literal(self) -> Literal {
                Literal::$variant(self.into())
            }
        })*
    };
}

into_literal!(bool => Boolean, i32 => Integer, i64 => BigInt, f64 => Double, &str => String, String => String);
