use std::fmt::{self, Display};
use std::str::FromStr;

use serde::de::{self, IgnoredAny, MapAccess, Visitor};
use serde::ser::{self, SerializeMap};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{Literal, LiteralError};
use crate::datatype::DataType;

impl Serialize for Literal {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.check_fit().map_err(ser::Error::custom)?;
        let tag = match (self, self.data_type().declarations()) {
            (Self::Null(_), _) => NULL,
            (_, Some(declarations)) => declarations.canonical.name,
            (_, None) => return Err(ser::Error::custom(LiteralError::InvalidType)),
        };
        let mut map = serializer.serialize_map(Some(1))?;
        match *self {
            Self::Boolean(value) => map.serialize_entry(tag, &value),
            Self::TinyInt(value) => map.serialize_entry(tag, &value),
            Self::SmallInt(value) => map.serialize_entry(tag, &value),
            Self::Integer(value) => map.serialize_entry(tag, &value),
            Self::BigInt(value) => map.serialize_entry(tag, &value),
            Self::Real(value) => map.serialize_entry(tag, &value.to_string()),
            Self::Double(value) => map.serialize_entry(tag, &value.to_string()),
            Self::Decimal {
                ref value,
                precision,
                scale,
            } => map.serialize_entry(
                tag,
                &Decimal {
                    value,
                    precision,
                    scale,
                },
            ),
            Self::String(ref value)
            | Self::Date(ref value)
            | Self::Interval(ref value)
            | Self::IntervalYearMonth(ref value)
            | Self::Uuid(ref value) => map.serialize_entry(tag, value),
            Self::VarChar { ref value, length } | Self::FixedChar { ref value, length } => {
                map.serialize_entry(tag, &WithLength { value, length })
            }
            Self::Binary(ref value) => map.serialize_entry(tag, value),
            Self::FixedBinary { ref value, length } => {
                map.serialize_entry(tag, &WithLength { value, length })
            }
            Self::Time {
                ref value,
                precision,
            }
            | Self::Timestamp {
                ref value,
                precision,
            }
            | Self::IntervalDaySecond {
                ref value,
                precision,
            } => map.serialize_entry(tag, &WithPrecision { value, precision }),
            Self::Array {
                ref element,
                ref values,
            } => map.serialize_entry(tag, &Array { element, values }),
            Self::Null(ref data_type) => map.serialize_entry(tag, data_type),
        }?;
        map.end()
    }
}

impl<'de> Deserialize<'de> for Literal {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_map(LiteralVisitor)
    }
}

struct LiteralVisitor;

impl<'de> Visitor<'de> for LiteralVisitor {
    type Value = Literal;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a literal: one canonical type name, with its value")
    }

    fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Literal, M::Error> {
        let Some(tag) = map.next_key::<String>()? else {
            return Err(de::Error::invalid_length(0, &self));
        };
        let literal = match DataType::of_canonical_name(&tag) {
            None if tag == NULL => Literal::Null(map.next_value()?),
            None => {
                return Err(de::Error::custom(format_args!(
                    "`{tag}` is no literal's tag: a tag is a canonical type name"
                )))
            }
            Some(data_type) => match data_type {
                DataType::Boolean => Literal::Boolean(map.next_value()?),
                DataType::TinyInt => Literal::TinyInt(map.next_value()?),
                DataType::SmallInt => Literal::SmallInt(map.next_value()?),
                DataType::Integer => Literal::Integer(map.next_value()?),
                DataType::BigInt => Literal::BigInt(map.next_value()?),
                DataType::Real => Literal::Real(parse(map.next_value()?)?),
                DataType::Double => Literal::Double(parse(map.next_value()?)?),
                DataType::Decimal { .. } => {
                    let Decimal {
                        value,
                        precision,
                        scale,
                    } = map.next_value()?;
                    Literal::Decimal {
                        value,
                        precision,
                        scale,
                    }
                }
                DataType::String => Literal::String(map.next_value()?),
                DataType::VarChar { .. } => {
                    let WithLength { value, length } = map.next_value()?;
                    Literal::VarChar { value, length }
                }
                DataType::FixedChar { .. } => {
                    let WithLength { value, length } = map.next_value()?;
                    Literal::FixedChar { value, length }
                }
                DataType::Binary | DataType::FixedBinary { .. } => match map.next_value()? {
                    Bytes::Bare(value) => Literal::Binary(value),
                    Bytes::WithLength(WithLength { value, length }) => {
                        Literal::FixedBinary { value, length }
                    }
                },
                DataType::Date => Literal::Date(map.next_value()?),
                DataType::Time { .. } => {
                    let WithPrecision { value, precision } = map.next_value()?;
                    Literal::Time { value, precision }
                }
                DataType::Timestamp { .. } => {
                    let WithPrecision { value, precision } = map.next_value()?;
                    Literal::Timestamp { value, precision }
                }
                DataType::Interval => Literal::Interval(map.next_value()?),
                DataType::IntervalYearMonth => Literal::IntervalYearMonth(map.next_value()?),
                DataType::IntervalDaySecond { .. } => {
                    let WithPrecision { value, precision } = map.next_value()?;
                    Literal::IntervalDaySecond { value, precision }
                }
                DataType::Uuid => Literal::Uuid(map.next_value()?),
                DataType::Array(_) => {
                    let Array { element, values } = map.next_value()?;
                    Literal::Array { element, values }
                }
                DataType::Unknown => return Err(de::Error::custom(LiteralError::InvalidType)),
            },
        };
        if map.next_key::<IgnoredAny>()?.is_some() {
            return Err(de::Error::invalid_length(2, &self));
        }
        literal.check_fit().map_err(de::Error::custom)?;
        Ok(literal)
    }
}

const NULL: &str = "null";

fn parse<Float: FromStr<Err: Display>, E: de::Error>(text: String) -> Result<Float, E> {
    text.parse()
        .map_err(|error| E::custom(format_args!("`{text}` is no number: {error}")))
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Decimal<Value> {
    value: Value,
    precision: u8,
    scale: i8,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WithLength<Value> {
    value: Value,
    length: u32,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WithPrecision<Value> {
    value: Value,
    precision: u8,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Array<Element, Values> {
    element: Element,
    values: Values,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Bytes {
    Bare(Vec<u8>),
    WithLength(WithLength<Vec<u8>>),
}
