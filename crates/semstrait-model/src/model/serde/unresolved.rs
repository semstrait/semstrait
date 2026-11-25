use super::de::parse_expr;
use super::error::ParseError;
use super::leaf::{LeafResolver, SerializeLeaf};
use super::ser::expr_to_value;
use semstrait_sql::expr::{PhysicalLeaf, RowExpr, SemanticLeaf};
use serde::de::value::{EnumAccessDeserializer, MapAccessDeserializer, SeqAccessDeserializer};
use serde::de::{self, EnumAccess, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_yaml::Value;
use std::convert::Infallible;
use std::fmt;
use std::marker::PhantomData;

pub struct Unresolved<E>(Infallible, PhantomData<E>);

pub enum SemanticExpr {}

pub enum PhysicalExpr {}

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::SemanticExpr {}
    impl Sealed for super::PhysicalExpr {}
}

pub trait Expression<L>: sealed::Sealed {}

impl Expression<SemanticLeaf> for SemanticExpr {}

impl Expression<PhysicalLeaf> for PhysicalExpr {}

impl<E> Unresolved<E> {
    pub fn serialize<T: Holds<E>, S: Serializer>(
        tree: &T,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        tree.write()
            .map_err(serde::ser::Error::custom)?
            .serialize(serializer)
    }

    pub fn deserialize<'de, T: Holds<E>, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<T, D::Error> {
        deserializer.deserialize_any(Read(T::read))
    }
}

struct Read<T>(fn(Value) -> Result<T, ParseError>);

impl<T> Read<T> {
    fn read<E: de::Error>(self, value: impl Into<Value>) -> Result<T, E> {
        (self.0)(value.into()).map_err(E::custom)
    }
}

impl<'de, T> Visitor<'de> for Read<T> {
    type Value = T;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("an expression")
    }

    fn visit_bool<E: de::Error>(self, v: bool) -> Result<T, E> {
        self.read(v)
    }

    fn visit_i64<E: de::Error>(self, v: i64) -> Result<T, E> {
        self.read(v)
    }

    fn visit_u64<E: de::Error>(self, v: u64) -> Result<T, E> {
        self.read(v)
    }

    fn visit_f64<E: de::Error>(self, v: f64) -> Result<T, E> {
        self.read(v)
    }

    fn visit_str<E: de::Error>(self, v: &str) -> Result<T, E> {
        self.read(v)
    }

    fn visit_unit<E: de::Error>(self) -> Result<T, E> {
        self.read(Value::Null)
    }

    fn visit_none<E: de::Error>(self) -> Result<T, E> {
        self.visit_unit()
    }

    fn visit_some<D: Deserializer<'de>>(self, d: D) -> Result<T, D::Error> {
        d.deserialize_any(self)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, a: A) -> Result<T, A::Error> {
        self.read(Value::deserialize(SeqAccessDeserializer::new(a))?)
    }

    fn visit_map<A: MapAccess<'de>>(self, a: A) -> Result<T, A::Error> {
        self.read(Value::deserialize(MapAccessDeserializer::new(a))?)
    }

    fn visit_enum<A: EnumAccess<'de>>(self, a: A) -> Result<T, A::Error> {
        self.read(Value::deserialize(EnumAccessDeserializer::new(a))?)
    }
}

pub trait Holds<E>: Sized {
    fn write(&self) -> Result<Value, ParseError>;

    fn read(value: Value) -> Result<Self, ParseError>;
}

impl<L: LeafResolver + SerializeLeaf, E: Expression<L>> Holds<E> for RowExpr<L> {
    fn write(&self) -> Result<Value, ParseError> {
        expr_to_value(self)
    }

    fn read(value: Value) -> Result<Self, ParseError> {
        parse_expr(&value)
    }
}

impl<L: LeafResolver + SerializeLeaf, E: Expression<L>> Holds<E> for Option<RowExpr<L>> {
    fn write(&self) -> Result<Value, ParseError> {
        self.as_ref().map_or(Ok(Value::Null), expr_to_value)
    }

    fn read(value: Value) -> Result<Self, ParseError> {
        (!value.is_null()).then(|| parse_expr(&value)).transpose()
    }
}

#[cfg(feature = "schemars")]
impl schemars::JsonSchema for Unresolved<SemanticExpr> {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "UnresolvedSemanticExpr".into()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schema::<Self>(generator, true)
    }
}

#[cfg(feature = "schemars")]
impl schemars::JsonSchema for Unresolved<PhysicalExpr> {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "UnresolvedPhysicalExpr".into()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schema::<Self>(generator, false)
    }
}

#[cfg(feature = "schemars")]
fn schema<Operand: schemars::JsonSchema>(
    generator: &mut schemars::SchemaGenerator,
    refers_to_semantics: bool,
) -> schemars::Schema {
    use super::data_type::UnresolvedDataType;
    use super::de::{TreeTag, CAST_FAILURES, LIST};
    use super::literal::{LIT, VALUE};
    use super::token::COMPOUND_CHARS;
    use super::{leaf, literal};
    use semstrait_sql::datatype::{any_case, DataType};
    use semstrait_sql::expr::{FunctionRef, OperationKind};
    use semstrait_sql::functions::{
        registry, ArgumentForm, ClauseRule, ParameterDecl, Presence, Signature,
    };
    use serde_json::{json, Value};
    let argument = json!({ "anyOf": [{ "type": "string" }, generator.subschema_for::<Operand>()] });
    let target = generator.subschema_for::<UnresolvedDataType>().to_value();
    let lit = literal::schema(generator);
    let whens = json!({
        "type": "array",
        "items": object(&[("when", &argument), ("then", &argument)], &[]),
        "minItems": 1
    });
    let failures = json!({ "enum": CAST_FAILURES.map(|(word, _)| word) });
    let list = json!({ "type": "array", "items": argument, "minItems": 1 });
    let tree = TreeTag::ALL
        .into_iter()
        .filter(|(_, tag)| refers_to_semantics || !matches!(tag, TreeTag::Semantic))
        .map(|(name, tag)| {
            let body = match tag {
                TreeTag::Lit => lit.clone(),
                TreeTag::Col | TreeTag::Semantic => leaf::reference_schema(name),
                TreeTag::Case => object(&[("whens", &whens)], &[("else", &argument)]),
                TreeTag::If => object(
                    &[("cond", &argument), ("then", &argument)],
                    &[("else", &argument)],
                ),
                TreeTag::Cast => object(
                    &[("input", &argument), ("target", &target)],
                    &[("on_failure", &failures)],
                ),
            };
            (name, body)
        });
    let names = |(name, aliases): (&'static str, &'static [&'static str])| {
        std::iter::once(name).chain(aliases.iter().copied())
    };
    let operations = OperationKind::all().flat_map(|kind| {
        let operation = &kind.operation().canonical;
        let signature = &operation.signature;
        let mut keys: Vec<_> = signature
            .parameters
            .iter()
            .map(|parameter| (parameter.name, &argument))
            .collect();
        keys.extend(signature.variadic.as_ref().map(|_| (LIST, &list)));
        let count = keys.len();
        let keyed = object(&keys, &[]);
        let body = match kind {
            OperationKind::Binary(_) => json!({ "oneOf": [
                { "type": "array", "items": argument, "minItems": count, "maxItems": count },
                keyed
            ] }),
            OperationKind::Unary(_) | OperationKind::IsNull => argument.clone(),
            _ => keyed,
        };
        names((operation.name, operation.aliases)).map(move |name| (name, body.clone()))
    });
    let tag = |tag: &str, body: Value| object(&[(tag, &body)], &[]);
    let scalar = json!({ "type": ["string", "number", "boolean", "null"] });
    let constant = json!({ "anyOf": [scalar, tag(LIT, lit.clone())] });
    let one_of = |values: Vec<&str>| {
        let spellings: Vec<_> = (values.iter())
            .flat_map(|value| [value.to_string(), format!("'{value}'")])
            .collect();
        let string = DataType::String.to_string();
        let value = json!({ "enum": values });
        let typed = tag(&string, value.clone());
        let long = tag(&string, tag(VALUE, value.clone()));
        let lit = tag(LIT, json!({ "anyOf": [value, typed, long] }));
        json!({ "anyOf": [{ "enum": spellings }, lit] })
    };
    let parameter = |declared: &ParameterDecl| match declared.form {
        ArgumentForm::Constant => constant.clone(),
        ArgumentForm::Enum(values) => one_of(values.to_vec()),
        ArgumentForm::DialectEnum(names) => {
            one_of(names.iter().map(|(canonical, _)| *canonical).collect())
        }
        _ => argument.clone(),
    };
    let call = |signature: &Signature| {
        let (fewest, most) = signature.argument_count();
        let parameters: Vec<_> = signature.parameters.iter().map(parameter).collect();
        let mut list = json!({
            "type": "array",
            "prefixItems": (!parameters.is_empty()).then_some(parameters),
            "items": signature.variadic.as_ref().map(parameter),
            "minItems": fewest,
            "maxItems": most
        });
        if let Value::Object(list) = &mut list {
            list.retain(|_, value| !value.is_null());
        }
        let required: Vec<_> = (signature.parameters.iter())
            .filter(|parameter| parameter.presence == Presence::Required)
            .map(|parameter| parameter.name)
            .collect();
        let named = (required.len() >= fewest).then(|| {
            json!({
                "type": "object",
                "minProperties": 1,
                "properties": (signature.parameters.iter())
                    .map(|declared| (declared.name.to_owned(), parameter(declared)))
                    .collect::<serde_json::Map<_, _>>(),
                "required": required,
                "additionalProperties": false
            })
        });
        let bare = (signature.parameters.first().or(signature.variadic.as_ref()))
            .filter(|_| fewest <= 1)
            .map(parameter);
        let forms: Vec<_> = [Some(list), named, bare].into_iter().flatten().collect();
        json!({ "anyOf": forms })
    };
    let functions: Vec<(Vec<&str>, Value)> = registry()
        .all()
        .filter_map(|function| {
            let declared = match function {
                FunctionRef::Scalar(f) => Some((f.canonical.name, f.canonical.aliases)),
                FunctionRef::Window(f) => {
                    let c = &f.canonical.specific.clauses;
                    let rules = [c.nulls, c.partition_by, c.order_by, c.window_frame];
                    let bound = rules.contains(&ClauseRule::Required);
                    (!bound).then_some((f.canonical.name, f.canonical.aliases))
                }
                _ => None,
            }?;
            Some((names(declared).collect(), call(function.canonical().1)))
        })
        .collect();
    let tags = (tree.chain(operations).map(|(name, body)| tag(name, body))).chain(
        (functions.iter())
            .flat_map(|(names, body)| names.iter().map(|name| tag(name, body.clone()))),
    );
    let other_cases = functions.iter().map(|(function, body)| {
        let spellings: Vec<String> = function.iter().map(|name| any_case(name)).collect();
        let key = format!("^(?:{})$", spellings.join("|"));
        json!({
            "type": "object",
            "minProperties": 1,
            "maxProperties": 1,
            "propertyNames": { "not": { "enum": function } },
            "patternProperties": { key: body },
            "additionalProperties": false
        })
    });
    let compound: String = COMPOUND_CHARS
        .bytes()
        .map(|c| format!("\\x{c:02x}"))
        .collect();
    let pattern =
        format!(r"^(?:[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|[^{compound}]+)$");
    let scalars = [
        json!({ "type": "string", "pattern": pattern }),
        json!({ "type": ["number", "boolean", "null"] }),
    ];
    let branches: Vec<_> = scalars.into_iter().chain(tags).chain(other_cases).collect();
    schemars::json_schema!({ "oneOf": branches })
}

#[cfg(feature = "schemars")]
pub(crate) fn object(
    required: &[(&str, &serde_json::Value)],
    optional: &[(&str, &serde_json::Value)],
) -> serde_json::Value {
    let properties: serde_json::Map<_, _> = required
        .iter()
        .chain(optional)
        .map(|&(key, schema)| (key.to_owned(), schema.clone()))
        .collect();
    let required: Vec<_> = required.iter().map(|&(key, _)| key).collect();
    serde_json::json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false
    })
}
