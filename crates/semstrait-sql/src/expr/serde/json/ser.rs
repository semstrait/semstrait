use serde::ser::{self, SerializeMap};
use serde::{Serialize, Serializer};

use super::{keys, Fields, Tag};
use crate::expr::{
    AggregateAllowed, Call, Expr, JsonFormError, Leaf, OptionSetting, Over, SortKey,
};
use crate::DocumentDialect;

#[derive(Debug)]
pub struct JsonForm<'a, L: Leaf> {
    expr: &'a Expr<L>,
    dialect: DocumentDialect,
}

impl<'a, L: Leaf> JsonForm<'a, L> {
    pub fn new(expr: &'a Expr<L>, dialect: DocumentDialect) -> Self {
        Self { expr, dialect }
    }
}

impl<L: Leaf> Clone for JsonForm<'_, L> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<L: Leaf> Copy for JsonForm<'_, L> {}

impl Tag {
    fn of<L: Leaf>(expr: &Expr<L>) -> Option<Self> {
        Some(match expr {
            Expr::Leaf(_) => return None,
            Expr::Literal(_) => Self::Literal,
            Expr::Column(_) => Self::Column,
            Expr::IsNull(_) => Self::IsNull,
            Expr::BinaryOp { .. } => Self::Fields(Fields::BinaryOp),
            Expr::UnaryOp { .. } => Self::Fields(Fields::UnaryOp),
            Expr::Cast { .. } => Self::Fields(Fields::Cast),
            Expr::Case { .. } => Self::Fields(Fields::Case),
            Expr::InList { .. } => Self::Fields(Fields::InList),
            Expr::Between { .. } => Self::Fields(Fields::Between),
            Expr::Like { .. } => Self::Fields(Fields::Like),
            Expr::Function(Call::Scalar(_)) => Self::Fields(Fields::Scalar),
            Expr::Function(Call::Aggregate(..)) => Self::Fields(Fields::Aggregate),
            Expr::Function(Call::Window(_)) => Self::Fields(Fields::Window),
        })
    }
}

impl<L: Leaf + Serialize> Serialize for JsonForm<'_, L> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let Some(tag) = Tag::of(self.expr) else {
            return Body(*self).serialize(serializer);
        };
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry(keys::r#type, tag.name())?;
        map.serialize_entry(keys::content, &Body(*self))?;
        map.end()
    }
}

macro_rules! object {
    ($serializer:ident, $($key:ident => $value:expr),+) => {{
        let mut map = $serializer.serialize_map(Some([$(keys::$key),+].len()))?;
        $(map.serialize_entry(keys::$key, $value)?;)+
        map.end()
    }};
}

struct Body<'a, L: Leaf>(JsonForm<'a, L>);

impl<'a, L: Leaf + Serialize> Serialize for Body<'a, L> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let JsonForm { expr, dialect } = self.0;
        let node = |expr: &'a Expr<L>| JsonForm::new(expr, dialect);
        let nodes = |exprs: &'a [Expr<L>]| each_node(exprs, dialect);
        let order_keys = |keys: &'a [SortKey<L, AggregateAllowed>]| each_key(keys, dialect);
        match expr {
            Expr::Leaf(leaf) => leaf.serialize(serializer),
            Expr::Literal(literal) => literal.serialize(serializer),
            Expr::Column(column) => column.serialize(serializer),
            Expr::IsNull(operand) => node(operand).serialize(serializer),
            Expr::BinaryOp { op, left, right } => {
                object!(serializer, op => op, left => &node(left), right => &node(right))
            }
            Expr::UnaryOp { op, operand } => {
                object!(serializer, op => op, operand => &node(operand))
            }
            Expr::Cast {
                input,
                target,
                on_failure,
            } => object!(serializer,
                input => &node(input), target => target, on_failure => on_failure),
            Expr::Case { whens, else_ } => {
                let whens: Vec<_> = (whens.iter())
                    .map(|(when, then)| When(node(when), node(then)))
                    .collect();
                let mut map = serializer.serialize_map(None)?;
                map.serialize_entry(keys::whens, &whens)?;
                field(&mut map, keys::r#else, else_.as_deref().map(node))?;
                map.end()
            }
            Expr::InList {
                value,
                list,
                negated,
            } => object!(serializer,
                value => &node(value), list => &nodes(list), negated => negated),
            Expr::Between {
                value,
                low,
                high,
                negated,
            } => object!(serializer,
                value => &node(value), low => &node(low), high => &node(high),
                negated => negated),
            Expr::Like {
                value,
                pattern,
                kind,
            } => object!(serializer,
                value => &node(value), pattern => &node(pattern), kind => kind),
            Expr::Function(call) => {
                let name = call.function().name_in(dialect).ok_or_else(|| {
                    ser::Error::custom(JsonFormError::NoNameInDialect {
                        function: call.name(),
                        dialect,
                    })
                })?;
                let mut map = serializer.serialize_map(None)?;
                map.serialize_entry(keys::name, name)?;
                field(&mut map, keys::args, listed(call.args()).map(nodes))?;
                field(&mut map, keys::options, listed(call.options()).map(Options))?;
                match call {
                    Call::Scalar(_) => {}
                    Call::Aggregate(call, _) => {
                        field(&mut map, keys::distinct, call.distinct().then_some(true))?;
                        field(&mut map, keys::filter, call.filter().map(node))?;
                        let order_by = listed(call.order_by()).map(order_keys);
                        field(&mut map, keys::order_by, order_by)?;
                        field(&mut map, keys::nulls, call.nulls())?;
                    }
                    Call::Window(call) => {
                        let over = call.over();
                        let over = (!over.is_empty()).then_some(OverForm(over, dialect));
                        field(&mut map, keys::nulls, call.nulls())?;
                        field(&mut map, keys::over, over)?;
                    }
                }
                map.end()
            }
        }
    }
}

struct When<'a, L: Leaf>(JsonForm<'a, L>, JsonForm<'a, L>);

impl<L: Leaf + Serialize> Serialize for When<'_, L> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        object!(serializer, when => &self.0, then => &self.1)
    }
}

struct Key<'a, L: Leaf>(&'a SortKey<L, AggregateAllowed>, DocumentDialect);

impl<L: Leaf + Serialize> Serialize for Key<'_, L> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let Self(key, dialect) = *self;
        object!(serializer, expr => &JsonForm::new(&key.expr, dialect), dir => &key.dir)
    }
}

struct OverForm<'a, L: Leaf>(&'a Over<L, AggregateAllowed>, DocumentDialect);

impl<L: Leaf + Serialize> Serialize for OverForm<'_, L> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let Self(over, dialect) = *self;
        let mut map = serializer.serialize_map(None)?;
        let partition_by = listed(&over.partition_by).map(|exprs| each_node(exprs, dialect));
        field(&mut map, keys::partition_by, partition_by)?;
        let order_by = listed(&over.order_by).map(|order_by| each_key(order_by, dialect));
        field(&mut map, keys::order_by, order_by)?;
        field(&mut map, keys::window_frame, over.window_frame)?;
        map.end()
    }
}

struct Options<'a>(&'a [OptionSetting]);

impl Serialize for Options<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map(self.0.iter().map(|setting| (&setting.name, &setting.value)))
    }
}

fn each_node<L: Leaf>(exprs: &[Expr<L>], dialect: DocumentDialect) -> Vec<JsonForm<'_, L>> {
    exprs
        .iter()
        .map(|expr| JsonForm::new(expr, dialect))
        .collect()
}

fn each_key<L: Leaf>(
    keys: &[SortKey<L, AggregateAllowed>],
    dialect: DocumentDialect,
) -> Vec<Key<'_, L>> {
    keys.iter().map(|key| Key(key, dialect)).collect()
}

fn field<M: SerializeMap, V: Serialize>(
    map: &mut M,
    name: &'static str,
    value: Option<V>,
) -> Result<(), M::Error> {
    value.map_or(Ok(()), |value| map.serialize_entry(name, &value))
}

fn listed<T>(items: &[T]) -> Option<&[T]> {
    (!items.is_empty()).then_some(items)
}
