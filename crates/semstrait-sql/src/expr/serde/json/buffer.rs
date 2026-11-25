use std::borrow::Cow;
use std::fmt;
use std::marker::PhantomData;

use serde::de::value::{MapAccessDeserializer, MapDeserializer, SeqDeserializer};
use serde::de::{self, IntoDeserializer, MapAccess, SeqAccess, Visitor};
use serde::{forward_to_deserialize_any, Deserialize, Deserializer};

pub(super) enum Buffer<'de> {
    Bool(bool),
    U64(u64),
    I64(i64),
    F64(f64),
    Str(Cow<'de, str>),
    Unit,
    Seq(Vec<Buffer<'de>>),
    Map(Vec<(Cow<'de, str>, Buffer<'de>)>),
}

impl Buffer<'_> {
    pub(super) fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(text) => Some(text),
            _ => None,
        }
    }
}

impl<'de> Deserialize<'de> for Buffer<'de> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(BufferVisitor)
    }
}

struct BufferVisitor;

impl<'de> Visitor<'de> for BufferVisitor {
    type Value = Buffer<'de>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value")
    }

    fn visit_bool<E: de::Error>(self, value: bool) -> Result<Buffer<'de>, E> {
        Ok(Buffer::Bool(value))
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> Result<Buffer<'de>, E> {
        Ok(Buffer::U64(value))
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> Result<Buffer<'de>, E> {
        Ok(Buffer::I64(value))
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Buffer<'de>, E> {
        Ok(Buffer::F64(value))
    }

    fn visit_str<E: de::Error>(self, text: &str) -> Result<Buffer<'de>, E> {
        Ok(Buffer::Str(Cow::Owned(text.to_owned())))
    }

    fn visit_borrowed_str<E: de::Error>(self, text: &'de str) -> Result<Buffer<'de>, E> {
        Ok(Buffer::Str(Cow::Borrowed(text)))
    }

    fn visit_string<E: de::Error>(self, text: String) -> Result<Buffer<'de>, E> {
        Ok(Buffer::Str(Cow::Owned(text)))
    }

    fn visit_unit<E: de::Error>(self) -> Result<Buffer<'de>, E> {
        Ok(Buffer::Unit)
    }

    fn visit_seq<S: SeqAccess<'de>>(self, mut seq: S) -> Result<Buffer<'de>, S::Error> {
        let mut items = Vec::new();
        while let Some(item) = seq.next_element()? {
            items.push(item);
        }
        Ok(Buffer::Seq(items))
    }

    fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Buffer<'de>, M::Error> {
        let mut entries = Vec::new();
        while let Some(Text(key)) = map.next_key()? {
            entries.push((key, map.next_value()?));
        }
        Ok(Buffer::Map(entries))
    }
}

#[derive(Deserialize)]
#[serde(transparent)]
pub(super) struct Text<'input>(#[serde(borrow)] pub(super) Cow<'input, str>);

pub(super) struct BufferDeserializer<'de, E> {
    buffer: Buffer<'de>,
    error: PhantomData<E>,
}

impl<'de, E> BufferDeserializer<'de, E> {
    pub(super) fn new(buffer: Buffer<'de>) -> Self {
        Self {
            buffer,
            error: PhantomData,
        }
    }
}

impl<'de, E: de::Error> IntoDeserializer<'de, E> for Buffer<'de> {
    type Deserializer = BufferDeserializer<'de, E>;

    fn into_deserializer(self) -> BufferDeserializer<'de, E> {
        BufferDeserializer::new(self)
    }
}

fn entries<'de, E: de::Error>(
    map: Vec<(Cow<'de, str>, Buffer<'de>)>,
) -> MapDeserializer<'de, impl Iterator<Item = (Buffer<'de>, Buffer<'de>)>, E> {
    MapDeserializer::new((map.into_iter()).map(|(key, value)| (Buffer::Str(key), value)))
}

impl<'de, E: de::Error> Deserializer<'de> for BufferDeserializer<'de, E> {
    type Error = E;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, E> {
        match self.buffer {
            Buffer::Bool(value) => visitor.visit_bool(value),
            Buffer::U64(value) => visitor.visit_u64(value),
            Buffer::I64(value) => visitor.visit_i64(value),
            Buffer::F64(value) => visitor.visit_f64(value),
            Buffer::Str(Cow::Borrowed(text)) => visitor.visit_borrowed_str(text),
            Buffer::Str(Cow::Owned(text)) => visitor.visit_string(text),
            Buffer::Unit => visitor.visit_unit(),
            Buffer::Seq(items) => {
                let mut seq = SeqDeserializer::new(items.into_iter());
                let value = visitor.visit_seq(&mut seq)?;
                seq.end()?;
                Ok(value)
            }
            Buffer::Map(map) => {
                let mut map = entries(map);
                let value = visitor.visit_map(&mut map)?;
                map.end()?;
                Ok(value)
            }
        }
    }

    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, E> {
        match self.buffer {
            Buffer::Unit => visitor.visit_none(),
            _ => visitor.visit_some(self),
        }
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, E> {
        visitor.visit_newtype_struct(self)
    }

    fn deserialize_enum<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, E> {
        match self.buffer {
            Buffer::Str(variant) => visitor.visit_enum(variant.into_deserializer()),
            Buffer::Map(map) if map.len() == 1 => {
                visitor.visit_enum(MapAccessDeserializer::new(entries(map)))
            }
            buffer => Self::new(buffer).deserialize_any(visitor),
        }
    }

    forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string bytes byte_buf
        unit unit_struct seq tuple tuple_struct map struct identifier ignored_any
    }
}
