use std::borrow::Cow;

use semstrait_sql::datatype::{DataType, TypeFamily};
use semstrait_sql::expr::{Literal, LiteralError};
use serde_yaml::{Mapping, Value};

use super::data_type::resolved;
use super::digits::integer;
use super::error::ParseError;
use super::ser::data_type_to_value;
use super::value::{deny_unknown_fields, lookup, map, seq, tag};

pub(crate) const LIT: &str = "lit";

pub(crate) const VALUE: &str = "value";

pub(crate) const VALUES: &str = "values";

pub(crate) fn unknown_tag(tag: &str) -> ParseError {
    ParseError::UnknownTag(format!("{LIT}.{tag}"))
}

pub(crate) fn lit<T>(
    body: &Value,
    tagged: impl FnOnce(Literal) -> T,
    untyped: impl FnOnce(Literal) -> T,
) -> Result<T, ParseError> {
    match body {
        Value::Mapping(body) => match integer(body) {
            Some(digits) => number(digits).map(untyped),
            None => tagged_literal(body).map(tagged),
        },
        _ => untyped_literal(body).map(untyped),
    }
}

pub(crate) fn untyped_literal(value: &Value) -> Result<Literal, ParseError> {
    match value {
        Value::Null => Ok(Literal::Null(DataType::String)),
        Value::Bool(value) => Ok(Literal::Boolean(*value)),
        Value::Number(float) if float.is_f64() => untyped_double(value),
        Value::Number(integer) => number(&integer.to_string()),
        Value::String(value) => Ok(Literal::String(value.clone())),
        Value::Tagged(tagged) => untyped_literal(&tagged.value),
        Value::Sequence(_) | Value::Mapping(_) => Err(ParseError::InvalidValue {
            field: LIT,
            reason: "an untyped list has no default type; write `{array: {element, values}}`"
                .into(),
        }),
    }
}

fn past_yamls_integers(float: f64) -> bool {
    float.is_finite() && (float >= u128::MAX as f64 || float <= i128::MIN as f64)
}

fn untyped_double(value: &Value) -> Result<Literal, ParseError> {
    match read(value, &DataType::Double)? {
        Literal::Double(float) if past_yamls_integers(float) => Err(ParseError::InvalidValue {
            field: LIT,
            reason: format!(
                "`{float:e}` is at or past 2^128, or at or below -2^127, where the YAML reader \
                 reads a whole number of 39 digits or more as the same float, so parse cannot \
                 tell the two apart: no decimal holds such a whole number, and a float that \
                 large is written `{{double: {float:e}}}`"
            ),
        }),
        double => Ok(double),
    }
}

pub(crate) fn number(text: &str) -> Result<Literal, ParseError> {
    if let Ok(value) = text.parse() {
        return Ok(Literal::BigInt(value));
    }
    let token = Value::String(text.to_owned());
    let digits = text.strip_prefix(['-', '+']).unwrap_or(text);
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return untyped_double(&token);
    }
    let advised = Written::of(&token)?.data_type(canonical_name(&DataType::decimal()))?;
    let reason = match read(&token, &advised) {
        Ok(_) => format!(
            "`{text}` is past `bigint`, the type of an untyped whole number; declare its \
             type with a tag, such as `{{decimal: \"{text}\"}}`"
        ),
        Err(_) => format!(
            "`{text}` has more than 38 digits, so no decimal holds it; a float that large is \
             written `{{double: {:e}}}`",
            text.parse::<f64>().unwrap_or(f64::NAN)
        ),
    };
    Err(ParseError::InvalidValue { field: LIT, reason })
}

pub(crate) mod bound {
    use semstrait_sql::expr::Literal;
    use serde::{de, ser, Deserialize, Deserializer, Serialize, Serializer};
    use serde_yaml::Value;

    pub(crate) fn serialize<S: Serializer>(
        bound: &Option<Literal>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match bound {
            Some(literal) => super::tagged(literal)
                .map_err(ser::Error::custom)?
                .serialize(serializer),
            None => serializer.serialize_none(),
        }
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Literal>, D::Error> {
        match Value::deserialize(deserializer)? {
            Value::Null => Ok(None),
            Value::Mapping(body) => super::tagged_literal(&body)
                .map(Some)
                .map_err(de::Error::custom),
            other => Err(de::Error::custom(format!(
                "a bucket's bound is a tagged literal, such as `{{bigint: 100}}`; got {other:?}"
            ))),
        }
    }

    #[cfg(feature = "schemars")]
    pub(crate) fn schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "oneOf": [super::tagged_schema(generator), { "type": "null" }]
        })
    }
}

#[cfg(feature = "schemars")]
pub(crate) fn schema(generator: &mut schemars::SchemaGenerator) -> serde_json::Value {
    serde_json::json!({
        "anyOf": [{ "type": ["string", "number", "boolean", "null"] }, tagged_schema(generator)]
    })
}

#[cfg(feature = "schemars")]
pub(crate) fn tagged_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
    use super::data_type::UnresolvedDataType;
    use super::unresolved::object;
    use serde_json::{json, Map};
    use std::collections::BTreeMap;
    let data_type = generator.subschema_for::<UnresolvedDataType>().to_value();
    let written = json!({ "type": ["string", "number", "boolean", "null", "array"] });
    let mut limits = BTreeMap::<_, BTreeMap<_, _>>::new();
    for template in DataType::roster() {
        let taken = template
            .declarations()
            .into_iter()
            .flat_map(|d| d.canonical.limits());
        limits
            .entry(canonical_name(template))
            .or_default()
            .extend(taken.map(|(_, key, limit)| (key, limit.values.schema().to_value())));
    }
    let mut tags: Map<_, _> = limits
        .into_iter()
        .map(|(name, taken)| {
            let taken: Vec<_> = taken.iter().map(|(&key, schema)| (key, schema)).collect();
            let long = object(&[(VALUE, &written)], &taken);
            (name.to_owned(), json!({ "anyOf": [written, long] }))
        })
        .collect();
    tags.insert("null".to_owned(), data_type.clone());
    tags.insert(
        canonical_name(&DataType::Array(Box::new(DataType::Unknown))).to_owned(),
        object(&[("element", &data_type), (VALUES, &written)], &[]),
    );
    schemars::json_schema!({
        "type": "object",
        "minProperties": 1,
        "maxProperties": 1,
        "properties": tags,
        "additionalProperties": false
    })
}

fn tagged_literal(body: &Mapping) -> Result<Literal, ParseError> {
    let mut entries = body.iter();
    let (Some((key @ (Value::String(_) | Value::Null), value)), None) =
        (entries.next(), entries.next())
    else {
        return Err(ParseError::InvalidValue {
            field: LIT,
            reason: "a tagged literal is one key, a canonical type name, with its value".into(),
        });
    };
    let tag = key.as_str().unwrap_or("null");
    let names_array = matches!(DataType::of_canonical_name(tag), Some(DataType::Array(_)));
    match (tag, value) {
        ("null", data_type) => Ok(Literal::Null(resolved(data_type)?)),
        (_, Value::Mapping(array)) if names_array && integer(array).is_none() => {
            deny_unknown_fields(array, "lit.array", &["element", VALUES])?;
            let element =
                lookup(array, "element").ok_or(ParseError::MissingField("lit.array.element"))?;
            let values =
                lookup(array, VALUES).ok_or(ParseError::MissingField("lit.array.values"))?;
            read(values, &DataType::Array(Box::new(resolved(element)?)))
        }
        _ if names_array => Err(ParseError::InvalidValue {
            field: LIT,
            reason: format!("`{tag}` takes its element and its values"),
        }),
        _ => {
            let written = Written::of(value)?;
            read(written.value, &written.data_type(tag)?)
        }
    }
}

fn read(value: &Value, data_type: &DataType) -> Result<Literal, ParseError> {
    let invalid = |reason| ParseError::InvalidLiteral {
        data_type: data_type.clone(),
        reason,
    };
    if let Some(text) = number_text(value) {
        return match data_type.family() {
            Some(TypeFamily::Number) => Literal::parse(&text, data_type),
            _ => Err(LiteralError::NoReading),
        }
        .map_err(invalid);
    }
    let literal = match (value, data_type) {
        (Value::Sequence(values), DataType::Array(element)) => Literal::Array {
            element: (**element).clone(),
            values: values
                .iter()
                .map(|value| read(value, element))
                .collect::<Result<_, _>>()?,
        },
        (Value::Sequence(bytes), DataType::Binary) => Literal::Binary(read_bytes(bytes)?),
        (Value::Sequence(bytes), DataType::FixedBinary { length }) => Literal::FixedBinary {
            value: read_bytes(bytes)?,
            length: *length,
        },
        _ => return untyped_literal(value)?.read_as(data_type).map_err(invalid),
    };
    literal.check_fit().map(|()| literal).map_err(invalid)
}

fn read_bytes(bytes: &[Value]) -> Result<Vec<u8>, ParseError> {
    bytes
        .iter()
        .map(|byte| {
            byte.as_u64()
                .and_then(|byte| u8::try_from(byte).ok())
                .ok_or_else(|| ParseError::InvalidValue {
                    field: "lit.binary",
                    reason: format!("{byte:?} is no byte from 0 to 255"),
                })
        })
        .collect()
}

fn canonical_name(data_type: &DataType) -> &'static str {
    data_type
        .declarations()
        .map_or("", |declarations| declarations.canonical.name)
}

struct Written<'v> {
    value: &'v Value,
    precision: Option<u8>,
    scale: Option<i8>,
    length: Option<u32>,
}

impl<'v> Written<'v> {
    fn of(body: &'v Value) -> Result<Self, ParseError> {
        let long = match body {
            Value::Mapping(long) if integer(long).is_none() => long,
            _ => {
                return Ok(Self {
                    value: body,
                    precision: None,
                    scale: None,
                    length: None,
                })
            }
        };
        deny_unknown_fields(long, LIT, &[VALUE, "precision", "scale", "length"])?;
        Ok(Self {
            value: lookup(long, VALUE).ok_or(ParseError::MissingField("lit.value"))?,
            precision: written_limit(long, "precision")?,
            scale: written_limit(long, "scale")?,
            length: written_limit(long, "length")?,
        })
    }

    fn data_type(&self, tag: &str) -> Result<DataType, ParseError> {
        let unknown = || unknown_tag(tag);
        let text = match self.value {
            Value::String(text) => Cow::Borrowed(text.as_str()),
            value => number_text(value).unwrap_or_default(),
        };
        let after_point = text.split_once('.').map_or(0, |(_, digits)| digits.len());
        let length = || {
            self.length
                .unwrap_or(u32::try_from(text.chars().count()).unwrap_or(u32::MAX))
        };
        let precision = |default: u8| self.precision.unwrap_or(default);
        let data_type = match DataType::of_canonical_name(tag).ok_or_else(unknown)? {
            DataType::Decimal {
                precision: default, ..
            } => DataType::Decimal {
                precision: precision(default),
                scale: self
                    .scale
                    .unwrap_or(i8::try_from(after_point).unwrap_or(i8::MAX)),
            },
            DataType::VarChar { .. } => DataType::VarChar { length: length() },
            DataType::FixedChar { .. } => DataType::FixedChar { length: length() },
            DataType::Binary => self
                .length
                .map_or(DataType::Binary, |length| DataType::FixedBinary { length }),
            DataType::Timestamp { precision: default } => DataType::Timestamp {
                precision: precision(if after_point > usize::from(default) {
                    9
                } else {
                    default
                }),
            },
            DataType::Time { precision: default } => DataType::Time {
                precision: precision(default),
            },
            DataType::IntervalDaySecond { precision: default } => DataType::IntervalDaySecond {
                precision: precision(default),
            },
            other => other,
        };
        let declaration = &data_type.declarations().ok_or_else(unknown)?.canonical;
        for (field, written, limit) in [
            ("precision", self.precision.is_some(), declaration.precision),
            ("scale", self.scale.is_some(), declaration.scale),
            ("length", self.length.is_some(), declaration.length),
        ] {
            if written && limit.is_none() {
                return Err(ParseError::InvalidValue {
                    field,
                    reason: format!("a `{tag}` literal takes no {field}"),
                });
            }
        }
        Ok(data_type)
    }
}

fn number_text(value: &Value) -> Option<Cow<'_, str>> {
    match value {
        Value::Number(number) => Some(Cow::Owned(match number.as_f64() {
            Some(float) if number.is_f64() => float.to_string(),
            _ => number.to_string(),
        })),
        Value::Mapping(body) => integer(body).map(Cow::Borrowed),
        _ => None,
    }
}

fn written_limit<Value: TryFrom<u64>>(
    long: &Mapping,
    key: &'static str,
) -> Result<Option<Value>, ParseError> {
    lookup(long, key)
        .map(|written| {
            written
                .as_u64()
                .and_then(|written| Value::try_from(written).ok())
                .ok_or_else(|| ParseError::InvalidValue {
                    field: key,
                    reason: format!("{written:?} is no {key}"),
                })
        })
        .transpose()
}

pub(crate) fn untyped_to_value(literal: &Literal) -> Result<Value, ParseError> {
    match literal {
        Literal::Double(float) if past_yamls_integers(*float) => {
            Err(ParseError::NotAuthorRepresentable(
                "an untyped `double` at or past 2^128, or at or below -2^127, which parse refuses",
            ))
        }
        Literal::Null(DataType::String)
        | Literal::Boolean(_)
        | Literal::BigInt(_)
        | Literal::Double(_)
        | Literal::String(_) => Ok(tag(LIT, value_of(literal)?)),
        _ => Err(ParseError::NotAuthorRepresentable(
            "an untyped literal whose type is no parse default",
        )),
    }
}

pub(crate) fn tagged_to_value(literal: &Literal) -> Result<Value, ParseError> {
    tagged(literal).map(|tagged| tag(LIT, tagged))
}

fn tagged(literal: &Literal) -> Result<Value, ParseError> {
    let name = literal
        .data_type()
        .declarations()
        .filter(|_| literal.check_fit().is_ok())
        .map(|declarations| declarations.canonical.name)
        .ok_or(ParseError::NotAuthorRepresentable(
            "a literal that does not fit its type",
        ))?;
    let with = |values: Vec<(&str, Value)>| -> Result<Value, ParseError> {
        let mut body = vec![(VALUE, value_of(literal)?)];
        body.extend(values);
        Ok(map(body))
    };
    let body = match literal {
        Literal::Null(data_type) => return Ok(tag("null", data_type_to_value(data_type)?)),
        Literal::Array { element, .. } => map(vec![
            ("element", data_type_to_value(element)?),
            (VALUES, value_of(literal)?),
        ]),
        Literal::Decimal {
            precision, scale, ..
        } => with(vec![
            ("precision", Value::from(*precision)),
            ("scale", Value::from(*scale)),
        ])?,
        Literal::VarChar { length, .. }
        | Literal::FixedChar { length, .. }
        | Literal::FixedBinary { length, .. } => with(vec![("length", Value::from(*length))])?,
        Literal::Time { precision, .. }
        | Literal::Timestamp { precision, .. }
        | Literal::IntervalDaySecond { precision, .. } => {
            with(vec![("precision", Value::from(*precision))])?
        }
        _ => value_of(literal)?,
    };
    Ok(tag(name, body))
}

fn value_of(literal: &Literal) -> Result<Value, ParseError> {
    Ok(match literal {
        Literal::Boolean(value) => Value::Bool(*value),
        Literal::TinyInt(value) => Value::from(*value),
        Literal::SmallInt(value) => Value::from(*value),
        Literal::Integer(value) => Value::from(*value),
        Literal::BigInt(value) => Value::from(*value),
        Literal::Real(value) if value.is_finite() => {
            Value::from(value.to_string().parse().unwrap_or(f64::from(*value)))
        }
        Literal::Double(value) if value.is_finite() => Value::from(*value),
        Literal::Decimal { value, .. }
        | Literal::String(value)
        | Literal::VarChar { value, .. }
        | Literal::FixedChar { value, .. }
        | Literal::Date(value)
        | Literal::Time { value, .. }
        | Literal::Timestamp { value, .. }
        | Literal::Interval(value)
        | Literal::IntervalYearMonth(value)
        | Literal::IntervalDaySecond { value, .. }
        | Literal::Uuid(value) => Value::String(value.clone()),
        Literal::Binary(bytes) | Literal::FixedBinary { value: bytes, .. } => {
            seq(bytes.iter().map(|byte| Value::from(*byte)).collect())
        }
        Literal::Array { values, .. } => {
            seq(values.iter().map(value_of).collect::<Result<_, _>>()?)
        }
        Literal::Null(_) => Value::Null,
        _ => {
            return Err(ParseError::NotAuthorRepresentable(
                "a literal with no authored form",
            ))
        }
    })
}
