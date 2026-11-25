use std::collections::BTreeMap;

use semstrait_common::EntityId;
use semstrait_common::Granularity;
use semstrait_common::SemanticsName;
use semstrait_ir::{AdditivityType, SemanticRole};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{AggregateAllowed, FunctionRef, Literal};
use semstrait_sql::functions::{registry, AggregateFunction, ParameterType, ReturnRule};
use serde::{Deserialize, Serialize};

#[non_exhaustive]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AiContext {
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub synonyms: BTreeMap<String, Vec<String>>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub examples: Vec<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DimensionType {
    Temporal(TemporalDimensionBody),
    Categorical,
    Binary,
    Geo,
    Bucketed(BucketedDimensionBody),
    Metadata(MetadataDimensionBody),
}

impl DimensionType {
    #[must_use]
    pub fn categorical() -> Self {
        Self::Categorical
    }

    #[must_use]
    pub fn temporal(granularities: impl IntoIterator<Item = Granularity>) -> Self {
        Self::Temporal(TemporalDimensionBody {
            granularities: granularities.into_iter().collect(),
        })
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TemporalDimensionBody {
    #[serde(default)]
    pub granularities: Vec<Granularity>,
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BucketedDimensionBody {
    pub buckets: Vec<BucketSpec>,
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BucketSpec {
    pub name: String,

    #[serde(default, skip_serializing_if = "Option::is_none", with = "bound")]
    pub lower: Option<Literal>,

    #[serde(default, skip_serializing_if = "Option::is_none", with = "bound")]
    pub upper: Option<Literal>,
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetadataDimensionBody {
    pub source: MetadataSource,
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetadataSource {
    Path(PathTokenRef),
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PathTokenRef {
    pub token: u32,
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticDefinition {
    pub id: EntityId,

    pub name: SemanticsName,

    pub data_type: Option<DataType>,

    pub role_definition: SemanticRoleDefinition,

    pub description: Option<String>,

    pub ai_context: Option<AiContext>,

    pub bit_position: u32,
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "content")]
pub enum SemanticRoleDefinition {
    Dimension {
        category: DimensionType,
    },

    Measure {
        #[serde(with = "aggregate_function")]
        aggregation: &'static AggregateFunction,
        #[serde(default, skip_serializing_if = "is_false")]
        distinct: bool,
        additivity: Option<AdditivityType>,
    },

    Metric {
        #[serde(default, with = "optional_aggregate_function")]
        aggregation: Option<&'static AggregateFunction>,
        #[serde(default, skip_serializing_if = "is_false")]
        distinct: bool,
    },

    Filter,
}

impl SemanticRoleDefinition {
    #[must_use]
    pub fn role(&self) -> SemanticRole {
        match self {
            Self::Dimension { .. } => SemanticRole::Dimension,
            Self::Measure { .. } => SemanticRole::Measure,
            Self::Metric { .. } => SemanticRole::Metric,
            Self::Filter => SemanticRole::Filter,
        }
    }

    #[must_use]
    pub fn additivity(&self) -> Option<&AdditivityType> {
        match self {
            Self::Measure { additivity, .. } => additivity.as_ref(),
            _ => None,
        }
    }

    #[must_use]
    pub fn aggregation(&self) -> Option<(&'static AggregateFunction, bool)> {
        match self {
            Self::Measure {
                aggregation,
                distinct,
                ..
            } => Some((*aggregation, *distinct)),
            Self::Metric {
                aggregation,
                distinct,
            } => aggregation.map(|function| (function, *distinct)),
            _ => None,
        }
    }

    #[must_use]
    pub fn category(&self) -> Option<&DimensionType> {
        match self {
            Self::Dimension { category } => Some(category),
            _ => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn plain(role: SemanticRole) -> Self {
        match role {
            SemanticRole::Dimension => Self::Dimension {
                category: DimensionType::Categorical,
            },
            SemanticRole::Measure => Self::Measure {
                aggregation: aggregate_function_of("sum").expect("a catalog aggregate"),
                distinct: false,
                additivity: None,
            },
            SemanticRole::Metric => Self::Metric {
                aggregation: None,
                distinct: false,
            },
            SemanticRole::Filter => Self::Filter,
            other => unreachable!("semstrait-ir added a SemanticRole variant: {other:?}"),
        }
    }
}

fn aggregate_function_of(name: &str) -> Option<&'static AggregateFunction> {
    match registry().get(name)? {
        FunctionRef::Aggregate(function, AggregateAllowed)
            if function.declarations.canonical.name == name =>
        {
            Some(function)
        }
        _ => None,
    }
}

fn is_false(value: &bool) -> bool {
    !*value
}

mod aggregate_function {
    use semstrait_sql::functions::AggregateFunction;
    use serde::{de, Deserialize, Deserializer, Serializer};

    pub(super) fn serialize<S: Serializer>(
        function: &&'static AggregateFunction,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(function.declarations.canonical.name)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<&'static AggregateFunction, D::Error> {
        let name = String::deserialize(deserializer)?;
        super::aggregate_function_of(&name).ok_or_else(|| {
            de::Error::custom(format_args!(
                "`{name}` is not a catalog aggregate's canonical name"
            ))
        })
    }
}

mod optional_aggregate_function {
    use semstrait_sql::functions::AggregateFunction;
    use serde::{de, Deserialize, Deserializer, Serializer};

    pub(super) fn serialize<S: Serializer>(
        function: &Option<&'static AggregateFunction>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match function {
            Some(function) => serializer.serialize_some(function.declarations.canonical.name),
            None => serializer.serialize_none(),
        }
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<&'static AggregateFunction>, D::Error> {
        let name: Option<String> = Option::deserialize(deserializer)?;
        name.map(|name| {
            super::aggregate_function_of(&name).ok_or_else(|| {
                de::Error::custom(format_args!(
                    "`{name}` is not a catalog aggregate's canonical name"
                ))
            })
        })
        .transpose()
    }
}

mod bound {
    use semstrait_sql::datatype::DataType;
    use semstrait_sql::expr::Literal;
    use serde::{de, ser, Deserialize, Deserializer, Serialize, Serializer};
    use serde_json::{Map, Value};

    pub(super) fn serialize<S: Serializer>(
        bound: &Option<Literal>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match bound {
            Some(literal) => tagged(literal)
                .map_err(ser::Error::custom)?
                .serialize(serializer),
            None => serializer.serialize_none(),
        }
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Literal>, D::Error> {
        match Option::<Value>::deserialize(deserializer)? {
            None => Ok(None),
            Some(value) => tagged_literal(&value).map(Some).map_err(de::Error::custom),
        }
    }

    fn tag(name: &str, body: Value) -> Value {
        let mut map = Map::new();
        map.insert(name.to_owned(), body);
        Value::Object(map)
    }

    fn one_entry(value: &Value) -> Result<(&str, &Value), String> {
        value
            .as_object()
            .filter(|object| object.len() == 1)
            .and_then(|object| object.iter().next())
            .map(|(name, body)| (name.as_str(), body))
            .ok_or_else(|| {
                format!(
                    "a bucket's bound is a tagged literal, such as `{{\"bigint\": 100}}`; got {value}"
                )
            })
    }

    fn tagged(literal: &Literal) -> Result<Value, String> {
        literal
            .check_fit()
            .map_err(|_| "a literal that does not fit its type".to_owned())?;
        Ok(match literal {
            Literal::Real(value) if value.is_finite() => tag("real", real_value(*value)),
            Literal::Double(value) if value.is_finite() => tag("double", bare_number(*value)),
            Literal::Array { element, values } => tag("array", array_body(element, values)?),
            Literal::Real(_) | Literal::Double(_) => {
                return Err("a literal with no authored form".to_owned())
            }
            _ => serde_json::to_value(literal).map_err(|error| error.to_string())?,
        })
    }

    fn tagged_literal(value: &Value) -> Result<Literal, String> {
        let (name, body) = one_entry(value)?;
        match name {
            "real" => parse_bare(body, &DataType::Real),
            "double" => parse_bare(body, &DataType::Double),
            "array" => array_literal(body),
            _ => serde_json::from_value(value.clone()).map_err(|error| error.to_string()),
        }
    }

    fn array_body(element: &DataType, values: &[Literal]) -> Result<Value, String> {
        let mut body = Map::new();
        body.insert(
            "element".to_owned(),
            serde_json::to_value(element).map_err(|error| error.to_string())?,
        );
        body.insert(
            "values".to_owned(),
            Value::Array(values.iter().map(bare_value).collect::<Result<_, _>>()?),
        );
        Ok(Value::Object(body))
    }

    fn array_literal(body: &Value) -> Result<Literal, String> {
        let object = body
            .as_object()
            .ok_or_else(|| format!("an array bound names its element and values; got {body}"))?;
        let element: DataType = object
            .get("element")
            .ok_or_else(|| "an array bound is missing `element`".to_string())
            .and_then(|value| {
                serde_json::from_value(value.clone()).map_err(|error| error.to_string())
            })?;
        let values = object
            .get("values")
            .and_then(Value::as_array)
            .ok_or_else(|| "an array bound is missing `values`".to_string())?
            .iter()
            .map(|value| value_as(value, &element))
            .collect::<Result<_, _>>()?;
        Ok(Literal::Array { element, values })
    }

    fn bare_value(literal: &Literal) -> Result<Value, String> {
        Ok(match literal {
            Literal::Boolean(value) => Value::Bool(*value),
            Literal::TinyInt(value) => Value::from(*value),
            Literal::SmallInt(value) => Value::from(*value),
            Literal::Integer(value) => Value::from(*value),
            Literal::BigInt(value) => Value::from(*value),
            Literal::Real(value) if value.is_finite() => real_value(*value),
            Literal::Double(value) if value.is_finite() => bare_number(*value),
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
                Value::Array(bytes.iter().map(|byte| Value::from(*byte)).collect())
            }
            Literal::Array { values, .. } => {
                Value::Array(values.iter().map(bare_value).collect::<Result<_, _>>()?)
            }
            Literal::Null(_) => Value::Null,
            _ => return Err("a literal with no authored form".to_owned()),
        })
    }

    fn value_as(value: &Value, data_type: &DataType) -> Result<Literal, String> {
        match (value, data_type) {
            (Value::Null, _) => Ok(Literal::Null(data_type.clone())),
            (Value::Array(items), DataType::Array(element)) => Ok(Literal::Array {
                element: (**element).clone(),
                values: items
                    .iter()
                    .map(|item| value_as(item, element))
                    .collect::<Result<_, _>>()?,
            }),
            (Value::Array(bytes), DataType::Binary) => Ok(Literal::Binary(read_bytes(bytes)?)),
            (Value::Array(bytes), DataType::FixedBinary { length }) => Ok(Literal::FixedBinary {
                value: read_bytes(bytes)?,
                length: *length,
            }),
            _ => parse_bare(value, data_type),
        }
    }

    fn parse_bare(value: &Value, data_type: &DataType) -> Result<Literal, String> {
        let text = as_text(value)?;
        Literal::parse(&text, data_type)
            .map_err(|error| format!("`{text}` does not fit `{data_type}`: {error:?}"))
    }

    fn as_text(value: &Value) -> Result<String, String> {
        match value {
            Value::Bool(value) => Ok(value.to_string()),
            Value::Number(number) => Ok(number.to_string()),
            Value::String(text) => Ok(text.clone()),
            other => Err(format!("{other} is no literal's bare value")),
        }
    }

    fn read_bytes(bytes: &[Value]) -> Result<Vec<u8>, String> {
        bytes
            .iter()
            .map(|byte| {
                byte.as_u64()
                    .and_then(|byte| u8::try_from(byte).ok())
                    .ok_or_else(|| format!("{byte} is no byte from 0 to 255"))
            })
            .collect()
    }

    fn real_value(value: f32) -> Value {
        let clean = value.to_string().parse::<f64>().unwrap_or(f64::from(value));
        bare_number(clean)
    }

    fn bare_number(value: f64) -> Value {
        serde_json::Number::from_f64(value).map_or(Value::Null, Value::Number)
    }
}

pub(crate) fn is_count(function: &'static AggregateFunction) -> bool {
    let canonical = &function.declarations.canonical;
    matches!(canonical.returns, ReturnRule::Fixed(_))
        && (canonical.signature.parameters.iter())
            .all(|parameter| parameter.ty == ParameterType::Any)
}

pub(crate) fn casts_an_observed_column(function: &'static AggregateFunction) -> bool {
    matches!(
        function.declarations.canonical.name,
        "sum" | "avg" | "min" | "max"
    )
}

impl SemanticDefinition {
    #[must_use]
    pub fn new(
        id: impl Into<EntityId>,
        name: SemanticsName,
        data_type: Option<DataType>,
        role_definition: SemanticRoleDefinition,
    ) -> Self {
        Self {
            id: id.into(),
            name,
            data_type,
            role_definition,
            description: None,
            ai_context: None,
            bit_position: 0,
        }
    }

    #[must_use]
    pub fn with_documentation(
        mut self,
        description: Option<String>,
        ai_context: Option<AiContext>,
    ) -> Self {
        self.description = description;
        self.ai_context = ai_context;
        self
    }

    #[must_use]
    pub fn role(&self) -> SemanticRole {
        self.role_definition.role()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dimension(name: &str) -> SemanticDefinition {
        SemanticDefinition::new(
            format!("sem.{name}"),
            SemanticsName::from(name),
            Some(DataType::String),
            SemanticRoleDefinition::Dimension {
                category: DimensionType::categorical(),
            },
        )
    }

    fn aggregate(name: &str) -> &'static AggregateFunction {
        aggregate_function_of(name).expect("a catalog aggregate")
    }

    #[test]
    fn the_role_label_is_derived_from_the_role_body() {
        assert_eq!(dimension("region").role(), SemanticRole::Dimension);
        assert_eq!(
            SemanticRoleDefinition::Measure {
                aggregation: aggregate("sum"),
                distinct: false,
                additivity: Some(AdditivityType::Full),
            }
            .role(),
            SemanticRole::Measure
        );
        assert_eq!(
            SemanticRoleDefinition::Metric {
                aggregation: None,
                distinct: false,
            }
            .role(),
            SemanticRole::Metric
        );
        assert_eq!(SemanticRoleDefinition::Filter.role(), SemanticRole::Filter);
    }

    #[test]
    fn the_role_body_carries_the_three_families() {
        let measure = SemanticRoleDefinition::Measure {
            aggregation: aggregate("count"),
            distinct: true,
            additivity: Some(AdditivityType::Non),
        };
        assert_eq!(measure.aggregation(), Some((aggregate("count"), true)));
        assert_eq!(measure.additivity(), Some(&AdditivityType::Non));
        assert!(measure.category().is_none());

        let dim = SemanticRoleDefinition::Dimension {
            category: DimensionType::temporal([Granularity::Day]),
        };
        assert!(matches!(dim.category(), Some(DimensionType::Temporal(_))));
        assert_eq!(dim.aggregation(), None);
        assert_eq!(dim.additivity(), None);
    }

    #[test]
    fn a_distinct_aggregation_adds_the_distinct_field() {
        let measure = SemanticRoleDefinition::Measure {
            aggregation: aggregate("count"),
            distinct: true,
            additivity: None,
        };
        let json = serde_json::to_string(&measure).expect("serialize");
        assert_eq!(
            json,
            r#"{"type":"measure","content":{"aggregation":"count","distinct":true,"additivity":null}}"#
        );
        let back: SemanticRoleDefinition = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, measure);
    }

    #[test]
    fn a_named_filter_is_an_ordinary_definition() {
        let filter = SemanticDefinition::new(
            "sem.recent",
            SemanticsName::from("recent"),
            None,
            SemanticRoleDefinition::Filter,
        )
        .with_documentation(Some("last 30 days".to_owned()), None);
        assert_eq!(filter.role(), SemanticRole::Filter);
        assert_eq!(filter.data_type, None);
        assert_eq!(filter.description.as_deref(), Some("last 30 days"));
    }

    #[test]
    fn a_definition_round_trips_through_json() {
        let mut definition = dimension("region");
        definition.bit_position = 7;
        let json = serde_json::to_string(&definition).expect("serialize definition");
        let back: SemanticDefinition = serde_json::from_str(&json).expect("deserialize definition");
        assert_eq!(back, definition);
    }

    #[test]
    fn the_role_body_wire_form_is_adjacently_tagged() {
        let measure = SemanticRoleDefinition::Measure {
            aggregation: aggregate("sum"),
            distinct: false,
            additivity: None,
        };
        let json = serde_json::to_string(&measure).expect("serialize");
        assert!(json.contains(r#""type":"measure""#), "got {json}");
        assert!(json.contains(r#""aggregation":"sum""#), "got {json}");
        assert!(
            !json.contains("distinct"),
            "distinct omitted when false: {json}"
        );

        let json = serde_json::to_string(&SemanticRoleDefinition::Filter).expect("serialize");
        assert_eq!(json, r#"{"type":"filter"}"#);
    }

    #[test]
    fn a_bucket_bound_refuses_a_non_finite_float() {
        let spec = BucketSpec {
            name: "nan".to_owned(),
            lower: Some(Literal::Double(f64::NAN)),
            upper: None,
        };
        assert!(serde_json::to_value(&spec).is_err());
    }

    #[test]
    fn a_bucket_bound_refuses_an_array_value_that_does_not_fit_its_declared_element() {
        let spec = BucketSpec {
            name: "mismatched".to_owned(),
            lower: Some(Literal::Array {
                element: DataType::BigInt,
                values: vec![Literal::Integer(1)],
            }),
            upper: None,
        };
        assert!(serde_json::to_value(&spec).is_err());
    }
}
