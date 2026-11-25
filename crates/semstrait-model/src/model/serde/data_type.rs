use semstrait_sql::datatype::{DataType, WrittenDataType};
use semstrait_sql::DocumentDialect;
use serde::{de, ser, Deserialize, Deserializer, Serialize, Serializer};
use serde_yaml::Value;

use super::error::ParseError;

#[derive(Debug, Clone)]
pub struct UnresolvedDataType(WrittenDataType);

impl UnresolvedDataType {
    pub fn spelling(&self) -> String {
        self.0.to_string()
    }

    pub fn resolve(&self) -> Option<DataType> {
        checked(&self.0).ok()
    }
}

pub(crate) fn resolved(value: &Value) -> Result<DataType, ParseError> {
    checked(&written(value))
}

fn checked(written: &WrittenDataType) -> Result<DataType, ParseError> {
    DataType::resolve_written(written, DocumentDialect::Canonical)
        .ok_or_else(|| ParseError::InvalidDataType(written.to_string()))
}

fn written(value: &Value) -> WrittenDataType {
    match value {
        Value::String(spelling) => WrittenDataType::Spelling(spelling.clone()),
        Value::Number(number) => number.as_u64().map_or_else(
            || WrittenDataType::Spelling(number.to_string()),
            WrittenDataType::Number,
        ),
        Value::Mapping(entries) => WrittenDataType::Object(
            entries
                .iter()
                .map(|(key, value)| (written(key).to_string(), written(value)))
                .collect(),
        ),
        other => WrittenDataType::Spelling(
            serde_yaml::to_string(other)
                .unwrap_or_default()
                .trim_end()
                .to_owned(),
        ),
    }
}

impl From<DataType> for UnresolvedDataType {
    fn from(data_type: DataType) -> Self {
        Self(WrittenDataType::Spelling(data_type.to_string()))
    }
}

impl PartialEq for UnresolvedDataType {
    fn eq(&self, other: &Self) -> bool {
        match (self.resolve(), other.resolve()) {
            (Some(left), Some(right)) => left == right,
            _ => self.0 == other.0,
        }
    }
}

impl PartialEq<DataType> for UnresolvedDataType {
    fn eq(&self, other: &DataType) -> bool {
        self.resolve().as_ref() == Some(other)
    }
}

impl Serialize for UnresolvedDataType {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        checked(&self.0)
            .map_err(ser::Error::custom)?
            .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for UnresolvedDataType {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let written = written(&Value::deserialize(deserializer)?);
        checked(&written).map_err(de::Error::custom)?;
        Ok(Self(written))
    }
}

#[cfg(feature = "schemars")]
impl schemars::JsonSchema for UnresolvedDataType {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "UnresolvedDataType".into()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        DataType::resolution_schema(
            DocumentDialect::Canonical,
            generator.subschema_for::<Self>(),
        )
    }
}
