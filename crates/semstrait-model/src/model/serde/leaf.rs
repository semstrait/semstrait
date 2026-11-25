use super::error::ParseError;
use super::literal::untyped_to_value;
use super::value::{deny_unknown_fields, get_optional_value, get_required_str, map, tag};
use semstrait_sql::expr::{
    DimensionAccessor, Expr, Leaf, Literal, MeasureAccessor, MetricAccessor, NamedReference,
    PhysicalLeaf, RowExpr, SemanticLeaf, UntypedLiteral,
};
use serde_yaml::Value;

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::SemanticLeaf {}
    impl Sealed for super::PhysicalLeaf {}
}

pub trait LeafResolver: Leaf + sealed::Sealed {
    fn bare_name(name: NamedReference) -> RowExpr<Self>;

    fn from_semantic_tag(tag: &str, value: &Value) -> Result<Option<Self>, ParseError>;

    fn from_untyped(lit: Literal) -> Self;
}

impl LeafResolver for SemanticLeaf {
    fn bare_name(name: NamedReference) -> RowExpr<Self> {
        Expr::Leaf(SemanticLeaf::Field(name))
    }

    fn from_semantic_tag(tag: &str, value: &Value) -> Result<Option<Self>, ParseError> {
        match tag {
            "field" => {
                let name = value.as_str().ok_or_else(|| ParseError::InvalidValue {
                    field: "field",
                    reason: "expected string name".into(),
                })?;
                let name = authored_reference(name, "field")?;
                Ok(Some(SemanticLeaf::Field(name)))
            }
            "dim" => {
                let (name, accessor) = parse_named_accessor_body(value, "dim")?;
                Ok(Some(SemanticLeaf::Dimension(name, accessor)))
            }
            "measure" => {
                let (name, accessor) = parse_named_accessor_body(value, "measure")?;
                Ok(Some(SemanticLeaf::Measure(name, accessor)))
            }
            "metric" => {
                let (name, accessor) = parse_named_accessor_body(value, "metric")?;
                Ok(Some(SemanticLeaf::Metric(name, accessor)))
            }
            _ => Ok(None),
        }
    }

    fn from_untyped(lit: Literal) -> Self {
        SemanticLeaf::Untyped(UntypedLiteral(lit))
    }
}

impl LeafResolver for PhysicalLeaf {
    fn bare_name(name: NamedReference) -> RowExpr<Self> {
        Expr::Column(name)
    }

    fn from_semantic_tag(tag: &str, _value: &Value) -> Result<Option<Self>, ParseError> {
        match tag {
            "field" | "dim" | "measure" | "metric" => Err(ParseError::TagNotAllowedAtSite {
                tag: tag.to_owned(),
                site: "physical-mapping",
            }),
            _ => Ok(None),
        }
    }

    fn from_untyped(lit: Literal) -> Self {
        PhysicalLeaf::Untyped(UntypedLiteral(lit))
    }
}

pub trait SerializeLeaf: Leaf + sealed::Sealed {
    fn leaf_to_value(&self) -> Result<Value, ParseError>;
}

impl SerializeLeaf for SemanticLeaf {
    fn leaf_to_value(&self) -> Result<Value, ParseError> {
        match self {
            SemanticLeaf::Untyped(UntypedLiteral(lit)) => untyped_to_value(lit),
            SemanticLeaf::Field(name) => Ok(tag("field", authored_name(name)?)),
            SemanticLeaf::Dimension(name, accessor) => Ok(tag(
                "dim",
                name_accessor_body(name, accessor.map(accessor_to_value))?,
            )),
            SemanticLeaf::Measure(name, accessor) => Ok(tag(
                "measure",
                name_accessor_body(name, accessor.map(accessor_to_value))?,
            )),
            SemanticLeaf::Metric(name, accessor) => Ok(tag(
                "metric",
                name_accessor_body(name, accessor.map(accessor_to_value))?,
            )),
            _ => Err(ParseError::NotAuthorRepresentable(
                "a semantic leaf that the authoring grammar has no form for",
            )),
        }
    }
}

impl SerializeLeaf for PhysicalLeaf {
    fn leaf_to_value(&self) -> Result<Value, ParseError> {
        match self {
            PhysicalLeaf::Untyped(UntypedLiteral(lit)) => untyped_to_value(lit),
            _ => Err(ParseError::NotAuthorRepresentable(
                "a mapping's leaf that the authoring grammar has no form for",
            )),
        }
    }
}

pub(super) fn authored_reference(
    name: impl Into<String>,
    field: &'static str,
) -> Result<NamedReference, ParseError> {
    NamedReference::unqualified(name).ok_or_else(|| ParseError::InvalidValue {
        field,
        reason: "a name cannot be empty".into(),
    })
}

#[cfg(feature = "schemars")]
pub(super) fn reference_schema(tag: &str) -> serde_json::Value {
    let name = serde_json::json!({ "type": "string", "minLength": 1 });
    let named = |accessor| {
        let body = super::unresolved::object(&[("name", &name)], &[(ACCESSOR, &accessor)]);
        serde_json::json!({ "oneOf": [name, body] })
    };
    match tag {
        "dim" => named(accessor_schema::<DimensionAccessor>()),
        "measure" => named(accessor_schema::<MeasureAccessor>()),
        "metric" => named(accessor_schema::<MetricAccessor>()),
        _ => name,
    }
}

#[cfg(feature = "schemars")]
fn accessor_schema<A: Accessor>() -> serde_json::Value {
    let words: Vec<_> = A::WORDS.iter().map(|&(word, _)| word).collect();
    let tags: Vec<_> = [LAG, LEAD].iter().chain(&words).collect();
    serde_json::json!({ "oneOf": [
        { "enum": words },
        {
            "type": "object",
            "minProperties": 1,
            "maxProperties": 1,
            "propertyNames": { "enum": tags },
            "additionalProperties": { "type": "integer", "minimum": 0, "maximum": u32::MAX }
        }
    ] })
}

pub(super) fn authored_name(reference: &NamedReference) -> Result<Value, ParseError> {
    match reference.parts() {
        [name] => Ok(Value::String(name.clone())),
        _ => Err(ParseError::NotAuthorRepresentable("a qualified name path")),
    }
}

fn parse_named_accessor_body<A: Accessor>(
    value: &Value,
    tag: &'static str,
) -> Result<(NamedReference, Option<A>), ParseError> {
    if let Some(s) = value.as_str() {
        return Ok((authored_reference(s, tag)?, None));
    }
    let m = value.as_mapping().ok_or_else(|| ParseError::InvalidValue {
        field: tag,
        reason: "expected string name or mapping with `name` and optional `accessor`".into(),
    })?;
    let name = get_required_str(m, "name", tag)?;
    let accessor = match get_optional_value(m, ACCESSOR) {
        Some(v) => Some(parse_accessor(v)?),
        None => None,
    };
    deny_unknown_fields(m, tag, &["name", ACCESSOR])?;
    Ok((authored_reference(name, tag)?, accessor))
}

pub(crate) const ACCESSOR: &str = "accessor";

const LAG: &str = "lag";
const LEAD: &str = "lead";

trait Accessor: Copy + PartialEq + 'static {
    const KIND: &'static str;
    const WORDS: &'static [(&'static str, Self)];
    const LAG: fn(u32) -> Self;
    const LEAD: fn(u32) -> Self;

    fn count(self) -> Option<(&'static str, u32)>;
}

impl Accessor for DimensionAccessor {
    const KIND: &'static str = "DimensionAccessor";
    const WORDS: &'static [(&'static str, Self)] = &[("first", Self::First), ("last", Self::Last)];
    const LAG: fn(u32) -> Self = Self::Lag;
    const LEAD: fn(u32) -> Self = Self::Lead;

    fn count(self) -> Option<(&'static str, u32)> {
        match self {
            Self::Lag(n) => Some((LAG, n)),
            Self::Lead(n) => Some((LEAD, n)),
            _ => None,
        }
    }
}

impl Accessor for MeasureAccessor {
    const KIND: &'static str = "MeasureAccessor";
    const WORDS: &'static [(&'static str, Self)] = &[
        ("previous", Self::Previous),
        ("next", Self::Next),
        ("delta", Self::Delta),
        ("percent_change", Self::PercentChange),
    ];
    const LAG: fn(u32) -> Self = Self::Lag;
    const LEAD: fn(u32) -> Self = Self::Lead;

    fn count(self) -> Option<(&'static str, u32)> {
        match self {
            Self::Lag(n) => Some((LAG, n)),
            Self::Lead(n) => Some((LEAD, n)),
            _ => None,
        }
    }
}

impl Accessor for MetricAccessor {
    const KIND: &'static str = "MetricAccessor";
    const WORDS: &'static [(&'static str, Self)] = &[
        ("previous", Self::Previous),
        ("next", Self::Next),
        ("delta", Self::Delta),
        ("percent_change", Self::PercentChange),
    ];
    const LAG: fn(u32) -> Self = Self::Lag;
    const LEAD: fn(u32) -> Self = Self::Lead;

    fn count(self) -> Option<(&'static str, u32)> {
        match self {
            Self::Lag(n) => Some((LAG, n)),
            Self::Lead(n) => Some((LEAD, n)),
            _ => None,
        }
    }
}

fn parse_accessor<A: Accessor>(value: &Value) -> Result<A, ParseError> {
    let word = |word: &str, form: &str| {
        A::WORDS
            .iter()
            .find(|(written, _)| *written == word)
            .map(|&(_, accessor)| accessor)
            .ok_or_else(|| ParseError::InvalidValue {
                field: ACCESSOR,
                reason: format!("unknown {} {form} `{word}`", A::KIND),
            })
    };
    if let Some(s) = value.as_str() {
        return word(s, "variant");
    }
    let m = value.as_mapping().ok_or_else(|| ParseError::InvalidValue {
        field: ACCESSOR,
        reason: "expected string variant or single-key tagged map".into(),
    })?;
    let mut entries = m.iter();
    let (Some((k, v)), None) = (entries.next(), entries.next()) else {
        return Err(ParseError::AmbiguousTag(m.len()));
    };
    let tag = k.as_str().ok_or_else(|| ParseError::InvalidValue {
        field: ACCESSOR,
        reason: "tag key must be a string".into(),
    })?;
    let n = parse_u32_body(v, tag)?;
    match tag {
        LAG => Ok((A::LAG)(n)),
        LEAD => Ok((A::LEAD)(n)),
        _ => word(tag, "tag"),
    }
}

fn parse_u32_body(value: &Value, tag: &str) -> Result<u32, ParseError> {
    let n = value.as_u64().ok_or_else(|| ParseError::InvalidValue {
        field: ACCESSOR,
        reason: format!("`{tag}` body must be a non-negative integer"),
    })?;
    u32::try_from(n).map_err(|_| ParseError::InvalidValue {
        field: ACCESSOR,
        reason: format!("`{tag}` body {n} exceeds u32::MAX"),
    })
}

fn accessor_to_value<A: Accessor>(accessor: A) -> Result<Value, ParseError> {
    if let Some((counted, n)) = accessor.count() {
        return Ok(tag(counted, Value::Number(u64::from(n).into())));
    }
    A::WORDS
        .iter()
        .find(|&&(_, written)| written == accessor)
        .map(|&(word, _)| Value::String(word.into()))
        .ok_or(ParseError::NotAuthorRepresentable(
            "an accessor that the authoring grammar has no word for",
        ))
}

fn name_accessor_body(
    name: &NamedReference,
    accessor: Option<Result<Value, ParseError>>,
) -> Result<Value, ParseError> {
    let name = authored_name(name)?;
    match accessor {
        None => Ok(name),
        Some(acc) => Ok(map(vec![("name", name), (ACCESSOR, acc?)])),
    }
}
