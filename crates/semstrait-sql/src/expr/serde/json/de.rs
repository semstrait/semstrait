use std::borrow::Cow;
use std::fmt;
use std::marker::PhantomData;

use serde::de::value::{BorrowedStrDeserializer, SeqAccessDeserializer};
use serde::de::{self, DeserializeSeed, IntoDeserializer, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};

use super::buffer::{Buffer, BufferDeserializer, Text};
use super::{keys, Fields, Tag};
use crate::expr::{
    AggregateAllowed, AggregateCall, Call, Expr, FunctionRef, JsonFormError, Leaf, Literal,
    NamedReference, OptionSetting, Over, ScalarCall, SortKey, WindowCall,
};
use crate::functions::registry;
use crate::DocumentDialect;

pub struct JsonReader<L> {
    dialect: DocumentDialect,
    leaf: PhantomData<fn() -> L>,
}

impl<L> JsonReader<L> {
    pub fn new(dialect: DocumentDialect) -> Self {
        Self {
            dialect,
            leaf: PhantomData,
        }
    }
}

impl<L> Clone for JsonReader<L> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<L> Copy for JsonReader<L> {}

impl<'de, L: Leaf + Deserialize<'de>> DeserializeSeed<'de> for JsonReader<L> {
    type Value = Expr<L>;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Expr<L>, D::Error> {
        deserializer.deserialize_any(Node(self))
    }
}

fn leaf<'de, L: Leaf + Deserialize<'de>, D: Deserializer<'de>>(
    value: D,
) -> Result<Expr<L>, D::Error> {
    L::deserialize(value).map(Expr::Leaf)
}

struct Node<L>(JsonReader<L>);

macro_rules! leaves {
    ($($visit:ident: $value:ty),+) => {$(
        fn $visit<E: de::Error>(self, value: $value) -> Result<Expr<L>, E> {
            leaf(value.into_deserializer())
        }
    )+};
}

impl<'de, L: Leaf + Deserialize<'de>> Visitor<'de> for Node<L> {
    type Value = Expr<L>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a node of the tree, or its leaf")
    }

    leaves!(
        visit_bool: bool, visit_i64: i64, visit_u64: u64, visit_f64: f64, visit_str: &str,
        visit_string: String
    );

    fn visit_borrowed_str<E: de::Error>(self, value: &'de str) -> Result<Expr<L>, E> {
        leaf(BorrowedStrDeserializer::new(value))
    }

    fn visit_unit<E: de::Error>(self) -> Result<Expr<L>, E> {
        leaf(().into_deserializer())
    }

    fn visit_seq<S: SeqAccess<'de>>(self, seq: S) -> Result<Expr<L>, S::Error> {
        leaf(SeqAccessDeserializer::new(seq))
    }

    fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Expr<L>, M::Error> {
        let mut held = Vec::new();
        while let Some(Text(key)) = map.next_key()? {
            let value: Buffer<'de> = map.next_value()?;
            if let (keys::r#type, Some(tag)) = (&*key, value.as_str().and_then(Tag::from_name)) {
                return self.node(tag, held, map);
            }
            held.push((key, value));
        }
        leaf(BufferDeserializer::new(Buffer::Map(held)))
    }
}

impl<L: Leaf> Node<L> {
    fn node<'de, M: MapAccess<'de>>(
        self,
        tag: Tag,
        held: Vec<(Cow<'de, str>, Buffer<'de>)>,
        mut map: M,
    ) -> Result<Expr<L>, M::Error>
    where
        L: Deserialize<'de>,
    {
        let body = || Body(self.0, tag);
        let mut node = None;
        for (key, value) in held {
            content_key(&key, node.is_some())?;
            node = Some(body().deserialize(BufferDeserializer::<M::Error>::new(value))?);
        }
        while let Some(Text(key)) = map.next_key()? {
            content_key(&key, node.is_some())?;
            node = Some(map.next_value_seed(body())?);
        }
        let node = node.ok_or_else(|| de::Error::missing_field(keys::content))?;
        node.checked().map_err(de::Error::custom)
    }
}

fn content_key<E: de::Error>(key: &str, read: bool) -> Result<(), E> {
    match key {
        keys::content if !read => Ok(()),
        keys::content => Err(E::duplicate_field(keys::content)),
        keys::r#type => Err(E::duplicate_field(keys::r#type)),
        _ => Err(E::unknown_field(key, &[keys::r#type, keys::content])),
    }
}

struct Body<L>(JsonReader<L>, Tag);

impl<'de, L: Leaf + Deserialize<'de>> DeserializeSeed<'de> for Body<L> {
    type Value = Expr<L>;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Expr<L>, D::Error> {
        let Self(reader, tag) = self;
        Ok(match tag {
            Tag::Literal => Expr::Literal(Literal::deserialize(deserializer)?),
            Tag::Column => Expr::Column(NamedReference::deserialize(deserializer)?),
            Tag::IsNull => Expr::IsNull(Box::new(reader.deserialize(deserializer)?)),
            Tag::Fields(fields) => deserializer.deserialize_map(Object(reader, fields))?,
        })
    }
}

macro_rules! fields {
    ($map:ident, $reader:ident;
        $($field:ident: $kind:ident $(= $default:expr)?),+ => $part:expr) => {{
        $(let mut $field = None;)+
        while let Some(Text(key)) = $map.next_key()? {
            match &*key {
                $(keys::$field if $field.is_none() => {
                    $field = Some(fields!(@$kind $map, $reader));
                })+
                $(keys::$field => return Err(de::Error::duplicate_field(keys::$field)),)+
                _ => return Err(de::Error::unknown_field(&key, &[$(keys::$field),+])),
            }
        }
        $(let $field = match $field {
            Some(value) => value,
            None => fields!(@default keys::$field $(, $default)?),
        };)+
        $part
    }};
    (@value $map:ident, $reader:ident) => { $map.next_value()? };
    (@text $map:ident, $reader:ident) => { $map.next_value::<Text>()?.0 };
    (@node $map:ident, $reader:ident) => { $map.next_value_seed($reader)? };
    (@nodes $map:ident, $reader:ident) => { $map.next_value_seed(List($reader))? };
    (@maybe $map:ident, $reader:ident) => { $map.next_value_seed(Maybe($reader))? };
    (@whens $map:ident, $reader:ident) => { $map.next_value_seed(List(When($reader)))? };
    (@options $map:ident, $reader:ident) => { $map.next_value_seed(Options)? };
    (@keys $map:ident, $reader:ident) => { $map.next_value_seed(List(OrderKey($reader)))? };
    (@over $map:ident, $reader:ident) => { $map.next_value_seed(OverForm($reader))? };
    (@default $key:expr) => { return Err(de::Error::missing_field($key)) };
    (@default $key:expr, $default:expr) => { $default };
}

struct Object<L>(JsonReader<L>, Fields);

impl<'de, L: Leaf + Deserialize<'de>> Visitor<'de> for Object<L> {
    type Value = Expr<L>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "the fields of a `{}`",
            Tag::Fields(self.1).name()
        )
    }

    fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Expr<L>, M::Error> {
        let Self(reader, fields) = self;
        let dialect = reader.dialect;
        Ok(match fields {
            Fields::BinaryOp => fields!(map, reader;
                op: value, left: node, right: node =>
                Expr::BinaryOp { op, left: Box::new(left), right: Box::new(right) }),
            Fields::UnaryOp => fields!(map, reader;
                op: value, operand: node =>
                Expr::UnaryOp { op, operand: Box::new(operand) }),
            Fields::Cast => fields!(map, reader;
                input: node, target: value, on_failure: value =>
                Expr::Cast { input: Box::new(input), target, on_failure }),
            Fields::Case => fields!(map, reader;
                whens: whens, r#else: maybe = None =>
                Expr::Case { whens, else_: r#else.map(Box::new) }),
            Fields::InList => fields!(map, reader;
                value: node, list: nodes, negated: value =>
                Expr::InList { value: Box::new(value), list, negated }),
            Fields::Between => fields!(map, reader;
            value: node, low: node, high: node, negated: value =>
            Expr::Between {
                value: Box::new(value), low: Box::new(low), high: Box::new(high), negated,
            }),
            Fields::Like => fields!(map, reader;
                value: node, pattern: node, kind: value =>
                Expr::Like { value: Box::new(value), pattern: Box::new(pattern), kind }),
            Fields::Scalar => fields!(map, reader;
            name: text, args: nodes = Vec::new(), options: options = Vec::new() =>
            Expr::Function(match function(dialect, &name)? {
                FunctionRef::Scalar(function) => Call::Scalar(
                    ScalarCall::builder(function, args)
                        .options(options)
                        .build()
                        .map_err(de::Error::custom)?,
                ),
                _ => return Err(not_its_type(&name, fields)),
            })),
            Fields::Aggregate => fields!(map, reader;
            name: text, args: nodes = Vec::new(), options: options = Vec::new(),
            distinct: value = false, filter: maybe = None, order_by: keys = Vec::new(),
            nulls: value = None =>
            Expr::Function(match function(dialect, &name)? {
                FunctionRef::Aggregate(function, aggregation) => Call::Aggregate(
                    AggregateCall::builder(function, args)
                        .options(options)
                        .distinct(distinct)
                        .maybe_filter(filter)
                        .order_by(order_by)
                        .maybe_nulls(nulls)
                        .build()
                        .map_err(de::Error::custom)?,
                    aggregation,
                ),
                _ => return Err(not_its_type(&name, fields)),
            })),
            Fields::Window => fields!(map, reader;
            name: text, args: nodes = Vec::new(), options: options = Vec::new(),
            nulls: value = None,
            over: over = Over::default() =>
            Expr::Function(match function(dialect, &name)? {
                FunctionRef::Window(function) => Call::Window(
                    WindowCall::builder(function, args)
                        .options(options)
                        .maybe_nulls(nulls)
                        .over(over)
                        .build()
                        .map_err(de::Error::custom)?,
                ),
                _ => return Err(not_its_type(&name, fields)),
            })),
        })
    }
}

macro_rules! object_seed {
    ($(#[$doc:meta])* $seed:ident, $expecting:literal -> $part:ty { $($fields:tt)+ }) => {
        $(#[$doc])*
        struct $seed<L>(JsonReader<L>);

        impl<L> Clone for $seed<L> {
            fn clone(&self) -> Self {
                *self
            }
        }

        impl<L> Copy for $seed<L> {}

        impl<'de, L: Leaf + Deserialize<'de>> DeserializeSeed<'de> for $seed<L> {
            type Value = $part;

            fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<$part, D::Error> {
                deserializer.deserialize_map(self)
            }
        }

        impl<'de, L: Leaf + Deserialize<'de>> Visitor<'de> for $seed<L> {
            type Value = $part;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str($expecting)
            }

            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<$part, M::Error> {
                let Self(reader) = self;
                Ok(fields!(map, reader; $($fields)+))
            }
        }
    };
}

object_seed! {
    When, "a WHEN with its THEN" -> (Expr<L>, Expr<L>) {
        when: node, then: node => (when, then)
    }
}

object_seed! {
    OrderKey, "an order key" -> SortKey<L, AggregateAllowed> {
        expr: node, dir: value => SortKey { expr, dir }
    }
}

object_seed! {
    OverForm, "a window call's OVER" -> Over<L, AggregateAllowed> {
        partition_by: nodes = Vec::new(), order_by: keys = Vec::new(),
        window_frame: value = None =>
        Over { partition_by, order_by, window_frame }
    }
}

#[derive(Clone, Copy)]
struct List<Item>(Item);

impl<'de, Part, Item: DeserializeSeed<'de, Value = Part> + Copy> DeserializeSeed<'de>
    for List<Item>
{
    type Value = Vec<Part>;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Vec<Part>, D::Error> {
        deserializer.deserialize_seq(self)
    }
}

impl<'de, Part, Item: DeserializeSeed<'de, Value = Part> + Copy> Visitor<'de> for List<Item> {
    type Value = Vec<Part>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a list")
    }

    fn visit_seq<S: SeqAccess<'de>>(self, mut seq: S) -> Result<Vec<Part>, S::Error> {
        let mut items = Vec::new();
        while let Some(item) = seq.next_element_seed(self.0)? {
            items.push(item);
        }
        Ok(items)
    }
}

struct Maybe<L>(JsonReader<L>);

impl<'de, L: Leaf + Deserialize<'de>> DeserializeSeed<'de> for Maybe<L> {
    type Value = Option<Expr<L>>;

    fn deserialize<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> Result<Option<Expr<L>>, D::Error> {
        deserializer.deserialize_option(self)
    }
}

impl<'de, L: Leaf + Deserialize<'de>> Visitor<'de> for Maybe<L> {
    type Value = Option<Expr<L>>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a node, or null")
    }

    fn visit_none<E: de::Error>(self) -> Result<Option<Expr<L>>, E> {
        Ok(None)
    }

    fn visit_some<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> Result<Option<Expr<L>>, D::Error> {
        self.0.deserialize(deserializer).map(Some)
    }
}

struct Options;

impl<'de> DeserializeSeed<'de> for Options {
    type Value = Vec<OptionSetting>;

    fn deserialize<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> Result<Vec<OptionSetting>, D::Error> {
        deserializer.deserialize_map(self)
    }
}

impl<'de> Visitor<'de> for Options {
    type Value = Vec<OptionSetting>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an object of each option's name and its value's name")
    }

    fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Vec<OptionSetting>, M::Error> {
        let mut options: Vec<OptionSetting> = Vec::new();
        while let Some((name, value)) = map.next_entry::<String, String>()? {
            if options.iter().any(|option| option.name == name) {
                return Err(de::Error::custom(JsonFormError::RepeatedOption { name }));
            }
            options.push(OptionSetting { name, value });
        }
        Ok(options)
    }
}

fn function<E: de::Error>(
    dialect: DocumentDialect,
    name: &str,
) -> Result<FunctionRef<AggregateAllowed>, E> {
    registry().find(dialect, name).ok_or_else(|| {
        E::custom(JsonFormError::UnknownFunction {
            name: name.to_owned(),
            dialect,
        })
    })
}

fn not_its_type<E: de::Error>(name: &str, fields: Fields) -> E {
    E::custom(JsonFormError::NotItsFunctionType {
        name: name.to_owned(),
        tag: Tag::Fields(fields).name(),
    })
}
