use crate::model::serde::literal::{tagged_to_value, untyped_to_value};
use crate::model::serde::mapping::deserialize_mapping_value;
use crate::model::serde::ser::expr_to_value;
use crate::model::serde::value::tag;
use crate::UnresolvedPhysicalExpr;
use indexmap::IndexMap;
use semstrait_common::SemanticsName;
use semstrait_sql::expr::{Literal, UntypedLiteral};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_yaml::Value;

#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub enum SemanticMapping {
    #[default]
    Auto,
    Explicit(IndexMap<SemanticsName, SemanticMappingValue>),
}

impl SemanticMapping {
    pub fn is_auto(&self) -> bool {
        matches!(self, Self::Auto)
    }

    pub fn explicit(entries: IndexMap<SemanticsName, SemanticMappingValue>) -> Self {
        Self::Explicit(entries)
    }

    pub fn builder() -> SemanticMappingBuilder {
        SemanticMappingBuilder::default()
    }
}

#[derive(Debug, Default)]
pub struct SemanticMappingBuilder {
    entries: IndexMap<SemanticsName, SemanticMappingValue>,
}

impl SemanticMappingBuilder {
    pub fn column(mut self, semantic: impl Into<SemanticsName>, column: impl Into<String>) -> Self {
        self.entries
            .insert(semantic.into(), SemanticMappingValue::Column(column.into()));
        self
    }

    pub fn literal(mut self, semantic: impl Into<SemanticsName>, value: Literal) -> Self {
        self.entries
            .insert(semantic.into(), SemanticMappingValue::Literal(value));
        self
    }

    pub fn untyped(mut self, semantic: impl Into<SemanticsName>, value: Literal) -> Self {
        self.entries.insert(
            semantic.into(),
            SemanticMappingValue::Untyped(UntypedLiteral(value)),
        );
        self
    }

    pub fn expr(
        mut self,
        semantic: impl Into<SemanticsName>,
        expr: UnresolvedPhysicalExpr,
    ) -> Self {
        self.entries
            .insert(semantic.into(), SemanticMappingValue::Expr(expr));
        self
    }

    pub fn with_semantic(
        mut self,
        name: impl Into<SemanticsName>,
        value: SemanticMappingValue,
    ) -> Self {
        self.entries.insert(name.into(), value);
        self
    }

    pub fn entries<I, N>(mut self, items: I) -> Self
    where
        I: IntoIterator<Item = (N, SemanticMappingValue)>,
        N: Into<SemanticsName>,
    {
        for (name, value) in items {
            self.entries.insert(name.into(), value);
        }
        self
    }

    pub fn build(self) -> SemanticMapping {
        if self.entries.is_empty() {
            SemanticMapping::Auto
        } else {
            SemanticMapping::Explicit(self.entries)
        }
    }
}

impl<'de> Deserialize<'de> for SemanticMapping {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::String(ref s) if s == "auto" => Ok(SemanticMapping::Auto),
            Value::Mapping(_) => {
                let entries: IndexMap<SemanticsName, SemanticMappingValue> =
                    serde_yaml::from_value(value).map_err(serde::de::Error::custom)?;
                Ok(SemanticMapping::Explicit(entries))
            }
            other => Err(serde::de::Error::custom(format!(
                "semantic_mapping: expected `auto` or a mapping, got {:?}",
                other
            ))),
        }
    }
}

impl Serialize for SemanticMapping {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            SemanticMapping::Auto => serializer.serialize_str("auto"),
            SemanticMapping::Explicit(map) => map.serialize(serializer),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SemanticMappingValue {
    Column(String),
    Literal(Literal),
    Untyped(UntypedLiteral),
    Expr(UnresolvedPhysicalExpr),
    Metadata(MetadataDimensionRecipe),
}

impl<'de> Deserialize<'de> for SemanticMappingValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        deserialize_mapping_value(&value).map_err(serde::de::Error::custom)
    }
}

impl Serialize for SemanticMappingValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let written = match self {
            Self::Column(s) => return serializer.serialize_str(s),
            Self::Literal(literal) => tagged_to_value(literal),
            Self::Untyped(UntypedLiteral(literal)) => untyped_to_value(literal),
            Self::Expr(e) => expr_to_value(e).map(|body| tag("expr", body)),
            Self::Metadata(_) => return Err(serde::ser::Error::custom(
                "SemanticMappingValue::Metadata is compile-synthesized; not serialised at the model author surface",
            )),
        };
        written
            .map_err(serde::ser::Error::custom)?
            .serialize(serializer)
    }
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct MetadataDimensionRecipe {
    pub extraction: MetadataExtraction,
    pub data_type: semstrait_sql::datatype::DataType,
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum MetadataExtraction {
    Path { token: u32 },
}

#[cfg(feature = "schemars")]
impl schemars::JsonSchema for SemanticMappingValue {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "SemanticMappingValue".into()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        use crate::model::serde::{literal, mapping::EXPR_KEY, object, PhysicalExpr, Unresolved};
        let lit = literal::schema(generator);
        let expr = generator
            .subschema_for::<Unresolved<PhysicalExpr>>()
            .to_value();
        schemars::json_schema!({
            "oneOf": [
                { "type": "string" },
                object(&[(literal::LIT, &lit)], &[]),
                object(&[(EXPR_KEY, &expr)], &[])
            ]
        })
    }
}

#[cfg(feature = "schemars")]
impl schemars::JsonSchema for SemanticMapping {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "SemanticMapping".into()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        let value = serde_json::to_value(generator.subschema_for::<SemanticMappingValue>())
            .expect("subschema to value");
        serde_json::from_value(serde_json::json!({
            "oneOf": [
                { "type": "string", "const": "auto" },
                { "type": "object", "additionalProperties": value }
            ]
        }))
        .expect("SemanticMapping JSON Schema is valid JSON")
    }
}
