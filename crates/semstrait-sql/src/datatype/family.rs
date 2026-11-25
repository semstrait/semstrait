use std::fmt;

use super::DataType;

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeFamily {
    Number,
    String,
    Boolean,
    Binary,
    DateAndTimestamp,
    Time,
    Interval,
    YearMonthInterval,
    DayTimeInterval,
    Uuid,
    Array(Box<TypeFamily>),
}

impl fmt::Display for TypeFamily {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Number => "number",
            Self::String => "string",
            Self::Boolean => "boolean",
            Self::Binary => "binary",
            Self::DateAndTimestamp => "date and timestamp",
            Self::Time => "time",
            Self::Interval => "interval",
            Self::YearMonthInterval => "year-month interval",
            Self::DayTimeInterval => "day-time interval",
            Self::Uuid => "uuid",
            Self::Array(element) => return write!(formatter, "array<{element}>"),
        })
    }
}

impl DataType {
    pub fn family(&self) -> Option<TypeFamily> {
        Some(match self {
            Self::TinyInt
            | Self::SmallInt
            | Self::Integer
            | Self::BigInt
            | Self::Real
            | Self::Double
            | Self::Decimal { .. } => TypeFamily::Number,
            Self::String | Self::VarChar { .. } | Self::FixedChar { .. } => TypeFamily::String,
            Self::Boolean => TypeFamily::Boolean,
            Self::Binary | Self::FixedBinary { .. } => TypeFamily::Binary,
            Self::Date | Self::Timestamp { .. } => TypeFamily::DateAndTimestamp,
            Self::Time { .. } => TypeFamily::Time,
            Self::Interval => TypeFamily::Interval,
            Self::IntervalYearMonth => TypeFamily::YearMonthInterval,
            Self::IntervalDaySecond { .. } => TypeFamily::DayTimeInterval,
            Self::Uuid => TypeFamily::Uuid,
            Self::Array(element) => TypeFamily::Array(Box::new(element.family()?)),
            Self::Unknown => return None,
        })
    }

    pub fn widens_to(&self, to: &Self) -> bool {
        self != to
            && match (self, to) {
                (Self::SmallInt, Self::Integer | Self::BigInt)
                | (Self::Integer, Self::BigInt | Self::Double | Self::Decimal { .. })
                | (Self::BigInt, Self::Double | Self::Decimal { .. })
                | (Self::Real, Self::Double)
                | (Self::VarChar { .. } | Self::FixedChar { .. }, Self::String)
                | (Self::FixedBinary { .. }, Self::Binary) => true,
                (
                    Self::Decimal { precision, scale },
                    Self::Decimal {
                        precision: to_precision,
                        scale: to_scale,
                    },
                ) => to_precision > precision && to_scale == scale,
                (
                    Self::VarChar { length } | Self::FixedChar { length },
                    Self::VarChar { length: to_length },
                ) => to_length >= length,
                (
                    Self::IntervalDaySecond { precision },
                    Self::IntervalDaySecond {
                        precision: to_precision,
                    },
                ) => to_precision > precision,
                _ => false,
            }
    }

    pub(crate) fn wider(self, other: Self) -> Option<Self> {
        if self == other || other.widens_to(&self) {
            Some(self)
        } else {
            self.widens_to(&other).then_some(other)
        }
    }

    pub fn casts_to(&self, to: &Self) -> bool {
        match (self, to) {
            (Self::Array(element), Self::Array(to_element)) => element.casts_to(to_element),
            (Self::Date, Self::Time { .. }) => false,
            _ => self.family().is_some_and(|family| family.casts_to(to)),
        }
    }
}

impl TypeFamily {
    pub(crate) fn casts_to(&self, to: &DataType) -> bool {
        match (self, to) {
            (Self::Array(element), DataType::Array(to_element)) => element.casts_to(to_element),
            (Self::Array(_), _) | (_, DataType::Array(_)) => false,
            (Self::DateAndTimestamp, DataType::Time { .. }) => true,
            _ => to.family().is_some_and(|to| {
                *self == to
                    || *self == Self::String
                    || to == Self::String
                    || (to == Self::Interval
                        && matches!(self, Self::YearMonthInterval | Self::DayTimeInterval))
            }),
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeOrFamily {
    Type(DataType),
    Family(TypeFamily),
}

impl TypeOrFamily {
    pub fn of(data_type: &DataType, family: Option<&TypeFamily>) -> Self {
        match (data_type, family) {
            (DataType::Unknown, Some(family)) => Self::Family(family.clone()),
            _ => Self::Type(data_type.clone()),
        }
    }
}

impl From<DataType> for TypeOrFamily {
    fn from(data_type: DataType) -> Self {
        Self::Type(data_type)
    }
}

impl From<TypeFamily> for TypeOrFamily {
    fn from(family: TypeFamily) -> Self {
        Self::Family(family)
    }
}

impl fmt::Display for TypeOrFamily {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Type(data_type) => write!(formatter, "{data_type}"),
            Self::Family(family) => {
                write!(formatter, "a value of the {family} family with no type")
            }
        }
    }
}

pub(crate) fn one_family<'f>(
    families: impl IntoIterator<Item = Option<&'f TypeFamily>>,
) -> Option<TypeFamily> {
    let mut families = families.into_iter();
    let first = families.next()??;
    families
        .all(|family| family == Some(first))
        .then(|| first.clone())
}

pub(crate) fn value_casts_to(from: &DataType, family: Option<&TypeFamily>, to: &DataType) -> bool {
    match from {
        DataType::Unknown => family.map_or(true, |family| family.casts_to(to)),
        from => from.casts_to(to),
    }
}
