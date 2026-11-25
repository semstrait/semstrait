use super::parse::{self, ensure, Number};
use super::{Literal, LiteralError};
use crate::datatype::{DataType, TypeFamily};

impl Literal {
    pub fn check_fit(&self) -> Result<(), LiteralError> {
        let data_type = self.data_type();
        let valid = data_type.is_valid();
        match self {
            Self::Decimal {
                value,
                precision,
                scale,
            } => {
                let number = Number::read(value)?;
                ensure(number.data_type().is_valid(), LiteralError::TooManyDigits)?;
                ensure(valid, LiteralError::InvalidType)?;
                number.fits(*precision, *scale)
            }
            _ if !valid => Err(LiteralError::InvalidType),
            Self::VarChar { value, length } | Self::FixedChar { value, length } => {
                within(value.chars().count(), *length)
            }
            Self::FixedBinary { value, length } => within(value.len(), *length),
            Self::Date(value) => parse::date(value).map(drop),
            Self::Time { value, precision } => {
                parse::fraction_fits(parse::time(value)?.1, *precision)
            }
            Self::Timestamp { value, precision } => parse::timestamp(value, *precision).map(drop),
            Self::Interval(value)
            | Self::IntervalYearMonth(value)
            | Self::IntervalDaySecond { value, .. } => parse::interval(value, &data_type).map(drop),
            Self::Uuid(value) => parse::uuid(value).map(drop),
            Self::Array { element, values } => values.iter().try_for_each(|value| {
                ensure(value.data_type() == *element, LiteralError::ElementType)?;
                value.check_fit()
            }),
            Self::Boolean(_)
            | Self::TinyInt(_)
            | Self::SmallInt(_)
            | Self::Integer(_)
            | Self::BigInt(_)
            | Self::Real(_)
            | Self::Double(_)
            | Self::String(_)
            | Self::Binary(_)
            | Self::Null(_) => Ok(()),
        }
    }

    pub fn read_as(&self, to: &DataType) -> Result<Literal, LiteralError> {
        let read = match self {
            Self::Null(_) => Self::Null(to.clone()),
            Self::String(value) => return Self::parse(value, to),
            _ if self.data_type() == *to => self.clone(),
            _ => match (self.number_text(), to.family()) {
                (Some(value), Some(TypeFamily::Number)) => return Self::parse(&value, to),
                _ => return Err(LiteralError::NoReading),
            },
        };
        read.check_fit()?;
        Ok(read)
    }

    fn number_text(&self) -> Option<String> {
        Some(match self {
            Self::TinyInt(value) => value.to_string(),
            Self::SmallInt(value) => value.to_string(),
            Self::Integer(value) => value.to_string(),
            Self::BigInt(value) => value.to_string(),
            Self::Real(value) => value.to_string(),
            Self::Double(value) => value.to_string(),
            Self::Decimal { value, .. } => value.clone(),
            Self::Boolean(_)
            | Self::String(_)
            | Self::VarChar { .. }
            | Self::FixedChar { .. }
            | Self::Binary(_)
            | Self::FixedBinary { .. }
            | Self::Date(_)
            | Self::Time { .. }
            | Self::Timestamp { .. }
            | Self::Interval(_)
            | Self::IntervalYearMonth(_)
            | Self::IntervalDaySecond { .. }
            | Self::Uuid(_)
            | Self::Array { .. }
            | Self::Null(_) => return None,
        })
    }
}

fn within(count: usize, length: u32) -> Result<(), LiteralError> {
    ensure(
        u32::try_from(count).is_ok_and(|count| count <= length),
        LiteralError::TooLong,
    )
}
