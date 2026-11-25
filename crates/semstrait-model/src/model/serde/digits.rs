use super::leaf::ACCESSOR;
use super::literal::{unknown_tag, LIT, VALUE, VALUES};
use super::mapping::EXPR_KEY;
use serde::de::value::MapDeserializer;
use serde::de::{
    DeserializeSeed, Deserializer, EnumAccess, Error, MapAccess, SeqAccess, VariantAccess, Visitor,
};
use serde::Deserialize;
use serde_yaml::Mapping;
use std::{fmt, iter};

const INTEGER_TAG: &str = "tag:yaml.org,2002:int";

const MODEL_KEYS: [(&str, At); 6] = [
    ("expr", At::Expression),
    ("filter", At::Expression),
    ("semantic_mapping", At::Mapping),
    ("lower", At::Bound),
    ("upper", At::Bound),
    ("ai_context", At::Opaque),
];

pub(crate) fn integer(body: &Mapping) -> Option<&str> {
    body.get(INTEGER_TAG).filter(|_| body.len() == 1)?.as_str()
}

pub(crate) fn deserialize<'de, T: Deserialize<'de>, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<T, D::Error> {
    T::deserialize(Digits::new(deserializer))
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum At {
    Model,
    Opaque,
    Mapping,
    Arm,
    Expression,
    Lit,
    Bound,
    Body,
    Value,
    Values,
}

impl At {
    fn value_of(self, key: &str) -> At {
        match self {
            At::Model => MODEL_KEYS
                .iter()
                .find(|(name, _)| *name == key)
                .map_or(At::Model, |&(_, at)| at),
            At::Mapping => At::Arm,
            At::Arm | At::Expression if key == LIT => At::Lit,
            At::Arm if key == EXPR_KEY => At::Expression,
            At::Expression if key != ACCESSOR => At::Expression,
            At::Lit | At::Bound => At::Body,
            At::Body if key == VALUE => At::Value,
            At::Body if key == VALUES => At::Values,
            At::Arm | At::Expression | At::Body | At::Value | At::Values | At::Opaque => At::Opaque,
        }
    }

    fn element(self) -> At {
        match self {
            At::Model | At::Expression => self,
            At::Values => At::Value,
            _ => At::Opaque,
        }
    }

    fn reads_a_number(self) -> bool {
        matches!(self, At::Lit | At::Body | At::Value)
    }

    fn gives_digits(self) -> bool {
        self.reads_a_number() || self == At::Expression
    }
}

enum Place<'k> {
    Value(At),
    Key { map: At, next: &'k mut At },
}

pub(crate) struct Digits<'k, T> {
    inner: T,
    place: Place<'k>,
}

impl<T> Digits<'static, T> {
    pub(crate) fn new(inner: T) -> Self {
        Self {
            inner,
            place: Place::Value(At::Model),
        }
    }
}

macro_rules! forward {
    ($de:lifetime; $($method:ident($($arg:ident: $type:ty),*);)*) => {$(
        fn $method<V: Visitor<$de>>(
            self,
            $($arg: $type,)*
            visitor: V,
        ) -> Result<V::Value, Self::Error> {
            self.inner.$method($($arg,)* Digits { inner: visitor, place: self.place })
        }
    )*};
}

impl<'de, D: Deserializer<'de>> Deserializer<'de> for Digits<'_, D> {
    type Error = D::Error;

    forward! { 'de;
        deserialize_any(); deserialize_bool(); deserialize_i8(); deserialize_i16();
        deserialize_i32(); deserialize_i64(); deserialize_i128(); deserialize_u8();
        deserialize_u16(); deserialize_u32(); deserialize_u64(); deserialize_u128();
        deserialize_f32(); deserialize_f64(); deserialize_char(); deserialize_str();
        deserialize_string(); deserialize_bytes(); deserialize_byte_buf(); deserialize_option();
        deserialize_unit(); deserialize_unit_struct(name: &'static str);
        deserialize_newtype_struct(name: &'static str); deserialize_seq();
        deserialize_tuple(len: usize); deserialize_tuple_struct(name: &'static str, len: usize);
        deserialize_map(); deserialize_struct(name: &'static str, fields: &'static [&'static str]);
        deserialize_enum(name: &'static str, variants: &'static [&'static str]);
        deserialize_identifier(); deserialize_ignored_any();
    }

    fn is_human_readable(&self) -> bool {
        self.inner.is_human_readable()
    }
}

impl<T> Digits<'_, T> {
    fn at(&self) -> At {
        match self.place {
            Place::Value(at) => at,
            Place::Key { .. } => At::Opaque,
        }
    }
}

impl<'de, V: Visitor<'de>> Visitor<'de> for Digits<'_, V> {
    type Value = V::Value;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        self.inner.expecting(formatter)
    }

    fn visit_bool<E: Error>(self, value: bool) -> Result<V::Value, E> {
        self.inner.visit_bool(value)
    }

    fn visit_i64<E: Error>(self, value: i64) -> Result<V::Value, E> {
        self.inner.visit_i64(value)
    }

    fn visit_u64<E: Error>(self, value: u64) -> Result<V::Value, E> {
        self.inner.visit_u64(value)
    }

    fn visit_i128<E: Error>(self, value: i128) -> Result<V::Value, E> {
        if self.at().gives_digits() {
            self.digits(value.to_string())
        } else {
            self.inner.visit_i128(value)
        }
    }

    fn visit_u128<E: Error>(self, value: u128) -> Result<V::Value, E> {
        if self.at().gives_digits() {
            self.digits(value.to_string())
        } else {
            self.inner.visit_u128(value)
        }
    }

    fn visit_f64<E: Error>(self, value: f64) -> Result<V::Value, E> {
        self.inner.visit_f64(value)
    }

    fn visit_str<E: Error>(self, text: &str) -> Result<V::Value, E> {
        if let Place::Key { map, next } = self.place {
            if map.reads_a_number() && text == INTEGER_TAG {
                return Err(E::custom(unknown_tag(text)));
            }
            *next = map.value_of(text);
        }
        self.inner.visit_str(text)
    }

    fn visit_unit<E: Error>(self) -> Result<V::Value, E> {
        self.inner.visit_unit()
    }

    fn visit_none<E: Error>(self) -> Result<V::Value, E> {
        self.inner.visit_none()
    }

    fn visit_some<D: Deserializer<'de>>(self, inner: D) -> Result<V::Value, D::Error> {
        self.inner.visit_some(Digits {
            inner,
            place: self.place,
        })
    }

    fn visit_newtype_struct<D: Deserializer<'de>>(self, inner: D) -> Result<V::Value, D::Error> {
        self.inner.visit_newtype_struct(Digits {
            inner,
            place: self.place,
        })
    }

    fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<V::Value, A::Error> {
        let at = self.at().element();
        self.inner.visit_seq(Access { access: seq, at })
    }

    fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<V::Value, A::Error> {
        let at = self.at();
        self.inner.visit_map(Map {
            map,
            at,
            next: At::Opaque,
        })
    }

    fn visit_enum<A: EnumAccess<'de>>(self, data: A) -> Result<V::Value, A::Error> {
        let at = self.at();
        self.inner.visit_enum(Access { access: data, at })
    }
}

impl<'de, V: Visitor<'de>> Digits<'_, V> {
    fn digits<E: Error>(self, digits: String) -> Result<V::Value, E> {
        let tagged = MapDeserializer::new(iter::once((INTEGER_TAG, digits)));
        match self.at() {
            At::Expression => self
                .inner
                .visit_map(MapDeserializer::new(iter::once((LIT, tagged)))),
            _ => self.inner.visit_map(tagged),
        }
    }
}

impl<'de, S: DeserializeSeed<'de>> DeserializeSeed<'de> for Digits<'_, S> {
    type Value = S::Value;

    fn deserialize<D: Deserializer<'de>>(self, inner: D) -> Result<S::Value, D::Error> {
        self.inner.deserialize(Digits {
            inner,
            place: self.place,
        })
    }
}

struct Access<A> {
    access: A,
    at: At,
}

impl<A> Access<A> {
    fn reader<T>(&self, inner: T) -> Digits<'static, T> {
        Digits {
            inner,
            place: Place::Value(self.at),
        }
    }
}

impl<'de, A: SeqAccess<'de>> SeqAccess<'de> for Access<A> {
    type Error = A::Error;

    fn next_element_seed<S: DeserializeSeed<'de>>(
        &mut self,
        seed: S,
    ) -> Result<Option<S::Value>, A::Error> {
        let seed = self.reader(seed);
        self.access.next_element_seed(seed)
    }

    fn size_hint(&self) -> Option<usize> {
        self.access.size_hint()
    }
}

impl<'de, A: EnumAccess<'de>> EnumAccess<'de> for Access<A> {
    type Error = A::Error;
    type Variant = Access<A::Variant>;

    fn variant_seed<S: DeserializeSeed<'de>>(
        self,
        seed: S,
    ) -> Result<(S::Value, Self::Variant), A::Error> {
        let seed = self.reader(seed);
        let (value, variant) = self.access.variant_seed(seed)?;
        let at = self.at;
        Ok((
            value,
            Access {
                access: variant,
                at,
            },
        ))
    }
}

impl<'de, A: VariantAccess<'de>> VariantAccess<'de> for Access<A> {
    type Error = A::Error;

    fn unit_variant(self) -> Result<(), A::Error> {
        self.access.unit_variant()
    }

    fn newtype_variant_seed<S: DeserializeSeed<'de>>(self, seed: S) -> Result<S::Value, A::Error> {
        let seed = self.reader(seed);
        self.access.newtype_variant_seed(seed)
    }

    fn tuple_variant<V: Visitor<'de>>(self, len: usize, visitor: V) -> Result<V::Value, A::Error> {
        let visitor = self.reader(visitor);
        self.access.tuple_variant(len, visitor)
    }

    fn struct_variant<V: Visitor<'de>>(
        self,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, A::Error> {
        let visitor = self.reader(visitor);
        self.access.struct_variant(fields, visitor)
    }
}

struct Map<A> {
    map: A,
    at: At,
    next: At,
}

impl<'de, A: MapAccess<'de>> MapAccess<'de> for Map<A> {
    type Error = A::Error;

    fn next_key_seed<S: DeserializeSeed<'de>>(
        &mut self,
        seed: S,
    ) -> Result<Option<S::Value>, A::Error> {
        self.next = At::Opaque;
        let place = Place::Key {
            map: self.at,
            next: &mut self.next,
        };
        self.map.next_key_seed(Digits { inner: seed, place })
    }

    fn next_value_seed<S: DeserializeSeed<'de>>(&mut self, seed: S) -> Result<S::Value, A::Error> {
        let place = Place::Value(self.next);
        self.map.next_value_seed(Digits { inner: seed, place })
    }

    fn size_hint(&self) -> Option<usize> {
        self.map.size_hint()
    }
}

#[cfg(all(test, feature = "schemars"))]
mod tests {
    use super::*;
    use serde_json::Value;

    const DEFINITIONS: [(&str, At); 4] = [
        ("#/$defs/UnresolvedSemanticExpr", At::Expression),
        ("#/$defs/UnresolvedPhysicalExpr", At::Expression),
        ("#/$defs/SemanticMapping", At::Mapping),
        ("#/$defs/AiContext", At::Opaque),
    ];

    fn placed(schema: &Value, bound: &Value, found: &mut Vec<(String, At)>) {
        match schema {
            Value::Object(object) => {
                let properties = object.get("properties").and_then(Value::as_object);
                for (name, property) in properties.into_iter().flatten() {
                    let text = property.to_string();
                    found.extend(
                        (DEFINITIONS.iter())
                            .filter(|(reference, _)| text.contains(&format!("\"{reference}\"")))
                            .map(|&(_, at)| (name.clone(), at)),
                    );
                    if property.get("oneOf") == bound.get("oneOf") {
                        found.push((name.clone(), At::Bound));
                    }
                }
                object
                    .values()
                    .for_each(|value| placed(value, bound, found));
            }
            Value::Array(values) => values.iter().for_each(|value| placed(value, bound, found)),
            _ => {}
        }
    }

    #[test]
    fn the_model_keys_are_the_schemas_expressions_semantic_mappings_bucket_bounds_and_ai_contexts()
    {
        let bound = serde_json::to_value(super::super::literal::bound::schema(
            &mut schemars::SchemaGenerator::default(),
        ))
        .expect("a bound's schema is JSON");
        let mut schema = crate::schema::semantic_model_json_schema();
        let definitions = schema["$defs"].as_object_mut().expect("the definitions");
        for (reference, at) in DEFINITIONS {
            if at == At::Expression {
                definitions.remove(reference.trim_start_matches("#/$defs/"));
            }
        }
        let mut found = Vec::new();
        placed(&schema, &bound, &mut found);
        found.sort_by(|left, right| left.0.cmp(&right.0));
        found.dedup();
        let mut keys: Vec<_> = (MODEL_KEYS.iter())
            .map(|&(name, at)| (name.to_owned(), at))
            .collect();
        keys.sort_by(|left, right| left.0.cmp(&right.0));
        assert_eq!(found, keys);
    }
}
