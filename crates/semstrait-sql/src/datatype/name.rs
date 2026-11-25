use std::{fmt, iter};

use semstrait_common::Dialect;
use serde::{de, ser, Deserialize, Deserializer, Serialize, Serializer};

use super::{DataType, TypeDeclaration};
use crate::DocumentDialect;

macro_rules! roster {
    ($($template:expr => $variant:pat,)*) => {
        static ROSTER: [DataType; [$(stringify!($variant)),*].len()] = [$($template),*];
        const _: fn(&DataType) = |data_type| match data_type {
            $($variant)|* | DataType::Array(_) | DataType::Unknown => {}
        };
    };
}

roster! {
    DataType::Boolean => DataType::Boolean,
    DataType::TinyInt => DataType::TinyInt,
    DataType::SmallInt => DataType::SmallInt,
    DataType::Integer => DataType::Integer,
    DataType::BigInt => DataType::BigInt,
    DataType::Real => DataType::Real,
    DataType::Double => DataType::Double,
    DataType::decimal() => DataType::Decimal { .. },
    DataType::String => DataType::String,
    DataType::VarChar { length: 1 } => DataType::VarChar { .. },
    DataType::FixedChar { length: 1 } => DataType::FixedChar { .. },
    DataType::Binary => DataType::Binary,
    DataType::FixedBinary { length: 1 } => DataType::FixedBinary { .. },
    DataType::Date => DataType::Date,
    DataType::time() => DataType::Time { .. },
    DataType::timestamp() => DataType::Timestamp { .. },
    DataType::Interval => DataType::Interval,
    DataType::IntervalYearMonth => DataType::IntervalYearMonth,
    DataType::interval_day_second() => DataType::IntervalDaySecond { .. },
    DataType::Uuid => DataType::Uuid,
}

pub(crate) const MAX_ARRAY_LAYERS: usize = 128;

#[derive(Clone, Copy)]
pub(crate) enum Spelling {
    Name,
    NameOrAlias,
}

impl Spelling {
    pub(super) fn names(self, declaration: &TypeDeclaration) -> impl Iterator<Item = &'static str> {
        let aliases: &'static [&'static str] = match self {
            Self::Name => &[],
            Self::NameOrAlias => declaration.aliases,
        };
        iter::once(declaration.name).chain(aliases.iter().copied())
    }
}

impl DataType {
    pub fn roster() -> &'static [Self] {
        &ROSTER
    }

    pub fn of_canonical_name(name: &str) -> Option<Self> {
        let array = Self::Array(Box::new(Self::Unknown));
        let named = ROSTER.iter().chain([&array]).find(|template| {
            template
                .declarations()
                .is_some_and(|declarations| declarations.canonical.name == name)
        });
        named.cloned()
    }

    pub fn from_name(spelling: &str) -> Option<Self> {
        Self::named(spelling, DocumentDialect::Canonical, Spelling::NameOrAlias)
    }

    pub fn resolve(spelling: &str, dialect: DocumentDialect) -> Option<Self> {
        Self::named(
            &spelling.to_ascii_lowercase(),
            dialect,
            Spelling::NameOrAlias,
        )
        .filter(|data_type| data_type.is_valid_in(dialect))
    }

    fn named(spelling: &str, dialect: DocumentDialect, rule: Spelling) -> Option<Self> {
        let array = Self::array_declaration(dialect).map(|declaration| declaration.name);
        let (mut element, mut layers) = (spelling, 0);
        while let Some(inner) = array
            .and_then(|array| element.strip_prefix(array))
            .and_then(|rest| rest.strip_prefix('<'))
            .and_then(|rest| rest.strip_suffix('>'))
        {
            layers += 1;
            if layers > MAX_ARRAY_LAYERS {
                return None;
            }
            element = inner.trim();
        }
        let (name, spelled) = match element
            .strip_suffix(')')
            .and_then(|rest| rest.split_once('('))
        {
            Some((name, values)) => {
                let values = values.split(',').map(|value| value.trim().parse().ok());
                (name, Some(values.collect::<Option<Vec<u32>>>()?))
            }
            None => (element, None),
        };
        let element = Self::with_name(name, dialect, rule, |declaration| match &spelled {
            Some(values) => Some(values.clone()),
            None => declaration
                .limits()
                .map(|(.., limit)| limit.default)
                .collect(),
        })?;
        Some((0..layers).fold(element, |element, _| Self::Array(Box::new(element))))
    }

    pub(crate) fn with_name(
        name: &str,
        dialect: DocumentDialect,
        rule: Spelling,
        values: impl Fn(&TypeDeclaration) -> Option<Vec<u32>>,
    ) -> Option<Self> {
        ROSTER.iter().find_map(|template| {
            let declaration = template.declarations()?.in_document(dialect)?;
            if !rule.names(declaration).any(|known| known == name) {
                return None;
            }
            template.with_values(&values(declaration)?)
        })
    }

    pub fn from_native_name(spelling: &str, dialect: Option<Dialect>) -> Option<Self> {
        let canonical = Self::from_name(spelling).filter(|data_type| {
            data_type.is_valid() && (spelling.contains('(') || !data_type.to_string().contains('('))
        });
        canonical.or_else(|| {
            let dialect = dialect?;
            Self::from_dialect_name(spelling, dialect)
                .filter(|data_type| data_type.is_valid_on(dialect))
        })
    }

    fn from_dialect_name(spelling: &str, dialect: Dialect) -> Option<Self> {
        match dialect {
            Dialect::Iceberg => Self::from_iceberg_timestamp(spelling)
                .or_else(|| Self::from_iceberg_time(spelling))
                .or_else(|| Self::from_iceberg_fixed_binary(spelling)),
            _ => None,
        }
    }

    fn from_iceberg_timestamp(spelling: &str) -> Option<Self> {
        let declaration = Self::timestamp().declarations()?.on(Dialect::Iceberg)?;
        let marks = spelling.strip_prefix(declaration.name)?;
        let precision = match marks.strip_prefix("tz").unwrap_or(marks) {
            "" => declaration.precision?.default?,
            "_ns" => 9,
            _ => return None,
        };
        Some(Self::Timestamp {
            precision: precision.try_into().ok()?,
        })
    }

    fn from_iceberg_time(spelling: &str) -> Option<Self> {
        let declaration = Self::time().declarations()?.on(Dialect::Iceberg)?;
        if spelling != declaration.name {
            return None;
        }
        Some(Self::Time {
            precision: declaration.precision?.default?.try_into().ok()?,
        })
    }

    fn from_iceberg_fixed_binary(spelling: &str) -> Option<Self> {
        let declaration = Self::FixedBinary { length: 1 }
            .declarations()?
            .on(Dialect::Iceberg)?;
        let rest = spelling.strip_prefix(declaration.name)?.trim_start();
        let length = rest
            .strip_prefix('[')?
            .strip_suffix(']')?
            .trim()
            .parse()
            .ok()?;
        Some(Self::FixedBinary { length })
    }

    #[deny(clippy::wildcard_enum_match_arm)]
    fn with_values(&self, values: &[u32]) -> Option<Self> {
        let one = || match *values {
            [value] => Some(value),
            _ => None,
        };
        let precision = || u8::try_from(one()?).ok();
        Some(match self {
            Self::Decimal { .. } => match *values {
                [precision, scale] => Self::Decimal {
                    precision: precision.try_into().ok()?,
                    scale: scale.try_into().ok()?,
                },
                _ => return None,
            },
            Self::Time { .. } => Self::Time {
                precision: precision()?,
            },
            Self::Timestamp { .. } => Self::Timestamp {
                precision: precision()?,
            },
            Self::IntervalDaySecond { .. } => Self::IntervalDaySecond {
                precision: precision()?,
            },
            Self::VarChar { .. } => Self::VarChar { length: one()? },
            Self::FixedChar { .. } => Self::FixedChar { length: one()? },
            Self::FixedBinary { .. } => Self::FixedBinary { length: one()? },
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
            | Self::Uuid => return values.is_empty().then(|| self.clone()),
            Self::Array(_) | Self::Unknown => return None,
        })
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Some(declarations) = self.declarations() else {
            return f.write_str("unknown");
        };
        f.write_str(declarations.canonical.name)?;
        if let Self::Array(element) = self {
            return write!(f, "<{element}>");
        }
        let values: Vec<String> = self.values().map(|value| value.to_string()).collect();
        if values.is_empty() {
            Ok(())
        } else {
            write!(f, "({})", values.join(","))
        }
    }
}

impl Serialize for DataType {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if !self.is_valid() {
            return Err(ser::Error::custom(format_args!(
                "`{self}` is not a valid data type"
            )));
        }
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for DataType {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        Self::named(&name, DocumentDialect::Canonical, Spelling::Name)
            .filter(Self::is_valid)
            .ok_or_else(|| de::Error::custom(format_args!("`{name}` names no valid data type")))
    }
}
