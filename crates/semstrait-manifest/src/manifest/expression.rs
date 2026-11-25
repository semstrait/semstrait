use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::convert::identity;
use std::fmt;

use semstrait_common::EntityId;
use semstrait_common::Tree;
use semstrait_ir::PhysicalExpr;
use semstrait_sql::expr::{
    map_nodes, AggregateCall, Call, Expr, JsonForm, JsonReader, Leaf, NamedReference, RebuildError,
};
use serde::de::{DeserializeSeed, Error as _, Unexpected, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use uuid::Uuid;

use super::expression_layer::ExprLayer;
use super::semantic_signature::SemanticSignature;
use super::top::MANIFEST_DIALECT;
use crate::error::{CompileError, ReferenceSite};
use crate::identity::mint_expr_id;

const RULE_NO_LEAF_STORED: &str = "a stored expression holds no leaf but a link to another \
     entry: no stored tree holds a Parameter (ADR 0088, rule 4)";

const RULE_LEAF_IS_A_LINK: &str = "a stored expression's only leaf is a link, its entry's \
     expression id in UUID text (ADR 0088, rules 1, 4 and 13)";

const RULE_CALL_STORED_AS_BUILT: &str = "a stored expression holds each call of compile's \
     tree, which its constructor's full check builds again (ADR 0088, rules 4 and 9)";

const RULE_LINKED_ENTRY_STORED_FIRST: &str =
    "compile stores an entry after each entry that it links (ADR 0088, rule 5)";

const RULE_ID_MINTED_ONCE: &str = "an expression id must be minted once — a repeated id \
     means one owner emitted two trees under one root marker";

fn pool_defect(rule: &'static str) -> CompileError {
    CompileError::StructuralViolation {
        site: ReferenceSite::ExpressionPool,
        rule,
    }
}

fn stored_as_built(error: RebuildError) -> CompileError {
    pool_defect(match error {
        RebuildError::Leaf => RULE_NO_LEAF_STORED,
        _ => RULE_CALL_STORED_AS_BUILT,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct PhysicalExprId(Uuid);

impl PhysicalExprId {
    #[must_use]
    pub const fn new(uuid: Uuid) -> Self {
        Self(uuid)
    }

    #[must_use]
    pub const fn uuid(self) -> Uuid {
        self.0
    }
}

impl fmt::Display for PhysicalExprId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.hyphenated().fmt(formatter)
    }
}

impl Serialize for PhysicalExprId {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for PhysicalExprId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct UuidText;

        impl Visitor<'_> for UuidText {
            type Value = PhysicalExprId;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("an expression id in lower-case hyphenated UUID text")
            }

            fn visit_str<E: serde::de::Error>(self, text: &str) -> Result<Self::Value, E> {
                Uuid::try_parse(text)
                    .ok()
                    .filter(|uuid| {
                        uuid.hyphenated().encode_lower(&mut Uuid::encode_buffer()) == text
                    })
                    .map(PhysicalExprId)
                    .ok_or_else(|| E::invalid_value(Unexpected::Str(text), &self))
            }
        }

        deserializer.deserialize_str(UuidText)
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExprRoot {
    SignatureExpression,
    SignatureFilter,
    JoinsetOn,
    JoinsetFilter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct Link(pub PhysicalExprId);

impl Leaf for Link {}

impl<'de> Deserialize<'de> for Link {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        PhysicalExprId::deserialize(deserializer)
            .map(Self)
            .map_err(|_| D::Error::custom(pool_defect(RULE_LEAF_IS_A_LINK)))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeptLeaf {
    Link(PhysicalExprId),
    Column(NamedReference),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeptField {
    pub leaf: KeptLeaf,
    pub field: NamedReference,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StoredExpr(Expr<Link>);

impl From<Expr<Link>> for StoredExpr {
    fn from(expr: Expr<Link>) -> Self {
        Self(expr)
    }
}

impl StoredExpr {
    #[must_use]
    pub fn expr(&self) -> &Expr<Link> {
        &self.0
    }

    pub fn links(&self) -> impl Iterator<Item = PhysicalExprId> + '_ {
        self.nodes().filter_map(|node| match node {
            Expr::Leaf(Link(id)) => Some(*id),
            _ => None,
        })
    }

    pub(crate) fn nodes(&self) -> impl Iterator<Item = &Expr<Link>> + '_ {
        self.0.pre_order()
    }
}

impl Serialize for StoredExpr {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        JsonForm::new(&self.0, MANIFEST_DIALECT).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for StoredExpr {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        JsonReader::<Link>::new(MANIFEST_DIALECT)
            .deserialize(deserializer)
            .map(Self)
    }
}

pub(crate) enum LinkStep<'s> {
    Left(PhysicalExprId, &'s ManifestExpression),
    Closes,
    Absent,
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ManifestExpressions {
    pub physical: BTreeMap<PhysicalExprId, ManifestExpression>,
}

impl ManifestExpressions {
    pub fn from_entries(
        entries: impl IntoIterator<Item = ManifestExpression>,
    ) -> Result<Self, CompileError> {
        let mut pool = Self::default();
        for entry in entries {
            pool.insert(entry)?;
        }
        Ok(pool)
    }

    #[must_use]
    pub fn get(&self, id: &PhysicalExprId) -> Option<&ManifestExpression> {
        self.physical.get(id)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.physical.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.physical.is_empty()
    }

    pub fn emit(
        &mut self,
        expr: &PhysicalExpr,
        owner: &str,
        root: ExprRoot,
        physical_sources: impl IntoIterator<Item = EntityId>,
    ) -> Result<PhysicalExprId, CompileError> {
        let expr = link_free(expr).map_err(stored_as_built)?;
        self.emit_linked(&expr, owner, root, physical_sources)
    }

    pub fn emit_linked(
        &mut self,
        expr: &Expr<Link>,
        owner: &str,
        root: ExprRoot,
        physical_sources: impl IntoIterator<Item = EntityId>,
    ) -> Result<PhysicalExprId, CompileError> {
        self.emit_linked_reading(expr, owner, root, physical_sources, &|_| ExprLayer::Scalar)
    }

    pub(crate) fn emit_linked_reading(
        &mut self,
        expr: &Expr<Link>,
        owner: &str,
        root: ExprRoot,
        physical_sources: impl IntoIterator<Item = EntityId>,
        column: &dyn Fn(&NamedReference) -> ExprLayer,
    ) -> Result<PhysicalExprId, CompileError> {
        let mut link = |node: &Expr<Link>| match node {
            Expr::Leaf(link) => Some(Ok(Expr::Leaf(*link))),
            _ => None,
        };
        let expr = StoredExpr(map_nodes(expr, &mut link, &stored_as_built)?);
        let layer = self
            .layer_of_reading(expr.expr(), column)
            .ok_or_else(|| pool_defect(RULE_LINKED_ENTRY_STORED_FIRST))?;
        let id = mint_expr_id(owner, root);
        self.insert(ManifestExpression {
            id,
            expr,
            layer,
            physical_sources: physical_sources.into_iter().collect(),
        })?;
        Ok(id)
    }

    #[must_use]
    pub fn layer_of(&self, expr: &Expr<Link>) -> Option<ExprLayer> {
        self.layer_of_reading(expr, &|_| ExprLayer::Scalar)
    }

    #[must_use]
    pub(crate) fn layer_of_reading(
        &self,
        expr: &Expr<Link>,
        column: &dyn Fn(&NamedReference) -> ExprLayer,
    ) -> Option<ExprLayer> {
        match expr {
            Expr::Leaf(Link(id)) => return self.get(id).map(|entry| entry.layer),
            Expr::Column(reference) => return Some(column(reference)),
            _ => {}
        }
        let child_layers = expr
            .children()
            .into_iter()
            .map(|child| self.layer_of_reading(child, column))
            .collect::<Option<Vec<_>>>()?;
        let is_aggregate = matches!(expr, Expr::Function(Call::Aggregate(..)));
        Some(ExprLayer::of_node(is_aggregate, child_layers))
    }

    #[must_use]
    pub fn linked_root(&self, id: &PhysicalExprId) -> Option<&StoredExpr> {
        let bare = |entry: &ManifestExpression| match entry.expr.expr() {
            Expr::Leaf(Link(next)) => Some(*next),
            _ => None,
        };
        match self.walk_links([*id], bare).next()? {
            LinkStep::Left(_, entry) => Some(&entry.expr),
            LinkStep::Closes | LinkStep::Absent => None,
        }
    }

    #[must_use]
    pub fn reassemble(
        &self,
        root: &PhysicalExprId,
        signatures: &[&SemanticSignature],
        fields: &[KeptField],
    ) -> Option<PhysicalExpr> {
        let filter_of = |link: &PhysicalExprId| {
            signature_of(signatures, link).map(|signature| signature.filter_expr_id)
        };
        let (mut through, mut kept) = (BTreeMap::new(), BTreeMap::new());
        for step in self.walk_links([*root], |entry| links_and_filters(entry, signatures)) {
            let LinkStep::Left(id, entry) = step else {
                return None;
            };
            let tree = entry.expr.expr();
            let read_through = read_links(tree, &through, &through, &[], &filter_of).ok()?;
            through.insert(id, read_through);
            if !fields.is_empty() {
                let read_kept = read_links(tree, &kept, &through, fields, &filter_of).ok()?;
                kept.insert(id, read_kept);
            }
        }
        if fields.is_empty() { through } else { kept }.remove(root)
    }

    #[must_use]
    pub fn linked<'s>(
        &self,
        root: &PhysicalExprId,
        signatures: &[&'s SemanticSignature],
    ) -> Option<Vec<&'s SemanticSignature>> {
        (self.walk_links([*root], links_outside_aggregates))
            .filter_map(|step| match step {
                LinkStep::Left(id, _) if id == *root => None,
                LinkStep::Left(id, _) => Some(signature_of(signatures, &id)),
                LinkStep::Closes | LinkStep::Absent => Some(None),
            })
            .collect()
    }

    #[must_use]
    pub fn reached<'s>(
        &self,
        roots: &[PhysicalExprId],
        signatures: &[&'s SemanticSignature],
    ) -> Option<Vec<&'s SemanticSignature>> {
        let mut left = BTreeSet::new();
        let leads = |entry| links_and_filters(entry, signatures);
        for step in self.walk_links(roots.iter().copied(), leads) {
            let LinkStep::Left(id, entry) = step else {
                return None;
            };
            if (entry.expr.links()).any(|link| signature_of(signatures, &link).is_none()) {
                return None;
            }
            left.insert(id);
        }
        Some(
            (signatures.iter().copied())
                .filter(|signature| left.contains(&signature.expression_id))
                .collect(),
        )
    }

    #[must_use]
    pub fn aggregate_signatures<'s>(
        &self,
        root: &PhysicalExprId,
        signatures: &[&'s SemanticSignature],
        kept: &[PhysicalExprId],
    ) -> Option<Vec<&'s SemanticSignature>> {
        let leads = |entry| {
            let mut links = links_outside_aggregates(entry);
            links.retain(|link| !kept.contains(link));
            links
        };
        let mut held: BTreeMap<PhysicalExprId, Vec<&'s SemanticSignature>> = BTreeMap::new();
        for step in self.walk_links([*root], leads) {
            let LinkStep::Left(id, entry) = step else {
                return None;
            };
            let calls = (outside_aggregates(entry.expr.expr()).into_iter())
                .map(|outside| match outside {
                    Outside::Link(link) if kept.contains(&link) => Some(Vec::new()),
                    Outside::Link(link) => {
                        signature_of(signatures, &link).and_then(|_| held.get(&link).cloned())
                    }
                    Outside::Aggregate => signature_of(signatures, &id).map(|own| vec![own]),
                })
                .collect::<Option<Vec<_>>>()?;
            held.insert(id, calls.concat());
        }
        held.remove(root)
    }

    #[must_use]
    pub fn columns(&self, root: &PhysicalExprId) -> Vec<(PhysicalExprId, &NamedReference)> {
        self.get(root)
            .into_iter()
            .flat_map(|entry| entry.expr.nodes())
            .filter_map(|node| match node {
                Expr::Column(column) => Some((*root, column)),
                _ => None,
            })
            .collect()
    }

    pub(crate) fn links_of(
        &self,
        id: &PhysicalExprId,
    ) -> impl Iterator<Item = PhysicalExprId> + '_ {
        (self.get(id).into_iter()).flat_map(|entry| entry.expr.links())
    }

    pub(crate) fn walk_links<'s, I>(
        &'s self,
        roots: impl IntoIterator<Item = PhysicalExprId> + 's,
        mut next: impl FnMut(&'s ManifestExpression) -> I + 's,
    ) -> impl Iterator<Item = LinkStep<'s>> + 's
    where
        I: IntoIterator<Item = PhysicalExprId>,
        I::IntoIter: 's,
    {
        enum Opened {
            OnPath,
            Left,
        }
        let mut roots = roots.into_iter();
        let mut opened: BTreeMap<PhysicalExprId, Opened> = BTreeMap::new();
        let mut path: Vec<(PhysicalExprId, &'s ManifestExpression, I::IntoIter)> = Vec::new();
        std::iter::from_fn(move || loop {
            let step = match path.last_mut() {
                Some((id, entry, leads)) => match leads.next() {
                    Some(step) => step,
                    None => {
                        let (id, entry) = (*id, *entry);
                        path.pop();
                        opened.insert(id, Opened::Left);
                        return Some(LinkStep::Left(id, entry));
                    }
                },
                None => roots.next()?,
            };
            match opened.get(&step) {
                Some(Opened::OnPath) => return Some(LinkStep::Closes),
                Some(Opened::Left) => {}
                None => {
                    let Some(entry) = self.get(&step) else {
                        return Some(LinkStep::Absent);
                    };
                    opened.insert(step, Opened::OnPath);
                    path.push((step, entry, next(entry).into_iter()));
                }
            }
        })
    }

    fn insert(&mut self, entry: ManifestExpression) -> Result<(), CompileError> {
        match self.physical.entry(entry.id) {
            Entry::Occupied(_) => Err(pool_defect(RULE_ID_MINTED_ONCE)),
            Entry::Vacant(vacant) => {
                vacant.insert(entry);
                Ok(())
            }
        }
    }
}

pub(crate) fn link_free(expr: &PhysicalExpr) -> Result<Expr<Link>, RebuildError> {
    map_nodes(expr, &mut |_| None, &identity)
}

fn signature_of<'s>(
    signatures: &[&'s SemanticSignature],
    id: &PhysicalExprId,
) -> Option<&'s SemanticSignature> {
    (signatures.iter())
        .find(|signature| signature.expression_id == *id)
        .copied()
}

fn links_and_filters(
    entry: &ManifestExpression,
    signatures: &[&SemanticSignature],
) -> Vec<PhysicalExprId> {
    let filters =
        (entry.expr.links()).filter_map(|link| signature_of(signatures, &link)?.filter_expr_id);
    entry.expr.links().chain(filters).collect()
}

enum Outside {
    Link(PhysicalExprId),
    Aggregate,
}

fn outside_aggregates(node: &Expr<Link>) -> Vec<Outside> {
    match node {
        Expr::Leaf(Link(id)) => vec![Outside::Link(*id)],
        Expr::Function(Call::Aggregate(..)) => vec![Outside::Aggregate],
        node => (node.children().into_iter())
            .flat_map(outside_aggregates)
            .collect(),
    }
}

fn links_outside_aggregates(entry: &ManifestExpression) -> Vec<PhysicalExprId> {
    (outside_aggregates(entry.expr.expr()).into_iter())
        .filter_map(|outside| match outside {
            Outside::Link(id) => Some(id),
            Outside::Aggregate => None,
        })
        .collect()
}

fn read_links(
    tree: &Expr<Link>,
    read: &BTreeMap<PhysicalExprId, PhysicalExpr>,
    through: &BTreeMap<PhysicalExprId, PhysicalExpr>,
    fields: &[KeptField],
    filter_of: &impl Fn(&PhysicalExprId) -> Option<Option<PhysicalExprId>>,
) -> Result<PhysicalExpr, RebuildError> {
    let mut leaf = |node: &Expr<Link>| {
        let kept = fields.iter().find(|kept| match (&kept.leaf, node) {
            (KeptLeaf::Link(kept), Expr::Leaf(Link(link))) => kept == link,
            (KeptLeaf::Column(kept), Expr::Column(column)) => kept == column,
            _ => false,
        });
        if let Some(KeptField { field, .. }) = kept {
            return Some(Ok(Expr::Column(field.clone())));
        }
        match node {
            Expr::Leaf(Link(link)) => {
                let tree = read.get(link)?;
                Some(match filter_of(link)? {
                    Some(filter) => join_filter_onto_aggregates(tree, through.get(&filter)?),
                    None => Ok(tree.clone()),
                })
            }
            Expr::Function(Call::Aggregate(..)) if !fields.is_empty() => {
                Some(read_links(node, through, through, &[], filter_of))
            }
            _ => None,
        }
    };
    map_nodes(tree, &mut leaf, &identity)
}

pub fn join_filter_onto_aggregates(
    tree: &PhysicalExpr,
    filter: &PhysicalExpr,
) -> Result<PhysicalExpr, RebuildError> {
    let mut join = |node: &PhysicalExpr| {
        let Expr::Function(Call::Aggregate(call, capability)) = node else {
            return None;
        };
        let joined = match call.filter() {
            Some(existing) => existing.clone() & filter.clone(),
            None => filter.clone(),
        };
        let call = AggregateCall::builder(call.function(), call.args().to_vec())
            .options(call.options().to_vec())
            .distinct(call.distinct())
            .filter(joined)
            .order_by(call.order_by().to_vec())
            .maybe_nulls(call.nulls())
            .build();
        Some(
            call.map(|call| Expr::Function(Call::Aggregate(call, *capability)))
                .map_err(RebuildError::Call),
        )
    };
    map_nodes(tree, &mut join, &identity)
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ManifestExpression {
    pub id: PhysicalExprId,
    pub expr: StoredExpr,
    pub layer: ExprLayer,
    pub physical_sources: BTreeSet<EntityId>,
}

#[cfg(test)]
mod tests {
    use super::*;

    use semstrait_model::Agg;
    use semstrait_sql::datatype::DataType;
    use semstrait_sql::expr::{
        CastFailure, ExprOperationExt, FunctionRef, Literal, NullOrdering, Over, SortDir, SortKey,
        WindowCall,
    };
    use semstrait_sql::functions::registry;

    const EXPR_ID: PhysicalExprId =
        PhysicalExprId::new(Uuid::from_u128(0x26c8a8c0_1f01_5ef4_99a3_87ab96316377));

    #[test]
    fn the_id_is_written_as_uuid_text() {
        let json = serde_json::to_string(&EXPR_ID).expect("an id writes");
        assert_eq!(json, r#""26c8a8c0-1f01-5ef4-99a3-87ab96316377""#);
        let back: PhysicalExprId = serde_json::from_str(&json).expect("an id reads");
        assert_eq!(back, EXPR_ID);
    }

    #[test]
    fn the_id_keys_a_json_map_as_its_text() {
        let map = BTreeMap::from([(EXPR_ID, "a")]);
        let json = serde_json::to_string(&map).expect("a map writes");
        assert_eq!(json, r#"{"26c8a8c0-1f01-5ef4-99a3-87ab96316377":"a"}"#);
        let back: BTreeMap<PhysicalExprId, String> =
            serde_json::from_str(&json).expect("a map reads");
        assert_eq!(back, BTreeMap::from([(EXPR_ID, "a".to_owned())]));
    }

    #[test]
    fn reading_refuses_an_id_that_is_not_uuid_text() {
        for json in [
            r#""26C8A8C0-1F01-5EF4-99A3-87AB96316377""#,
            r#""26c8a8c01f015ef499a387ab96316377""#,
            r#""{26c8a8c0-1f01-5ef4-99a3-87ab96316377}""#,
            r#""urn:uuid:26c8a8c0-1f01-5ef4-99a3-87ab96316377""#,
            "51552545877750077560241905626611016567",
            r#""x""#,
        ] {
            assert!(
                serde_json::from_str::<PhysicalExprId>(json).is_err(),
                "{json} is refused"
            );
        }
    }

    #[test]
    fn the_id_displays_as_its_uuid_text() {
        assert_eq!(EXPR_ID.to_string(), "26c8a8c0-1f01-5ef4-99a3-87ab96316377");
    }

    fn id(raw: u128) -> PhysicalExprId {
        PhysicalExprId::new(Uuid::from_u128(raw))
    }

    #[test]
    fn new_round_trips_through_uuid() {
        for uuid in [Uuid::nil(), Uuid::max(), EXPR_ID.uuid()] {
            assert_eq!(PhysicalExprId::new(uuid).uuid(), uuid);
        }
    }

    #[test]
    fn equal_when_inner_equal() {
        assert_eq!(id(7), id(7));
        assert_ne!(id(7), id(8));
    }

    #[test]
    fn ord_follows_the_uuid() {
        let mut ids = [30, 10, 20].map(id);
        ids.sort();
        assert_eq!(ids, [10, 20, 30].map(id));
    }

    #[test]
    fn usable_as_btreemap_key() {
        let mut m = BTreeMap::new();
        m.insert(id(2), "b");
        m.insert(id(1), "a");
        let ordered: Vec<_> = m.values().copied().collect();
        assert_eq!(ordered, vec!["a", "b"]);
    }

    fn column<L: Leaf>(name: &str) -> Expr<L> {
        Expr::Column(NamedReference::unqualified(name).expect("a name"))
    }

    fn sum<L: Leaf>(inner: Expr<L>) -> Expr<L> {
        let agg = Agg::of_name("sum").expect("a catalog aggregate");
        crate::compile::expressions::lower::wrap_aggregate(
            inner,
            (agg.function(), agg.distinct()),
            "test",
        )
        .expect("sum wraps")
    }

    fn cast<L: Leaf>(input: Expr<L>) -> Expr<L> {
        Expr::Cast {
            input: Box::new(input),
            target: DataType::Decimal {
                precision: 18,
                scale: 2,
            },
            on_failure: CastFailure::Error,
        }
    }

    fn link(raw: u128) -> Expr<Link> {
        Expr::Leaf(Link(id(raw)))
    }

    fn two_sums() -> (ManifestExpressions, PhysicalExprId, PhysicalExprId) {
        let mut pool = ManifestExpressions::default();
        let a = pool
            .emit(
                &sum(column("amount")),
                "sig.a",
                ExprRoot::SignatureExpression,
                [],
            )
            .expect("emit");
        let a2 = pool
            .emit(
                &sum(column("cost")),
                "sig.b",
                ExprRoot::SignatureExpression,
                [],
            )
            .expect("emit");
        (pool, a, a2)
    }

    fn layer(expr: &Expr<Link>) -> Option<ExprLayer> {
        ManifestExpressions::default().layer_of(expr)
    }

    fn int(value: i64) -> Expr<Link> {
        Expr::Literal(Literal::BigInt(value))
    }

    fn lag(arg: Expr<Link>) -> Expr<Link> {
        let Some(FunctionRef::Window(lag)) = registry().get("lag") else {
            panic!("`lag` is a window function");
        };
        let over = Over {
            partition_by: vec![column("region")],
            order_by: vec![SortKey {
                expr: column("month"),
                dir: SortDir::Asc {
                    nulls: NullOrdering::Unspecified,
                },
            }],
            window_frame: None,
        };
        let call = WindowCall::builder(lag, vec![arg])
            .over(over)
            .build()
            .expect("`lag` of one argument, ordered");
        Expr::Function(Call::Window(call))
    }

    #[test]
    fn scalar_when_no_aggregate_anywhere() {
        assert_eq!(layer(&(column("a") + int(1))), Some(ExprLayer::Scalar));
    }

    #[test]
    fn scalar_for_a_bare_column() {
        assert_eq!(layer(&column("a")), Some(ExprLayer::Scalar));
    }

    #[test]
    fn aggregate_when_root_is_aggregate() {
        assert_eq!(layer(&sum(column("a"))), Some(ExprLayer::Aggregate));
    }

    #[test]
    fn post_aggregate_when_aggregate_below_scalar_root() {
        assert_eq!(
            layer(&(sum(column("a")) / sum(column("b")))),
            Some(ExprLayer::PostAggregate)
        );
    }

    #[test]
    fn post_aggregate_when_aggregate_nested_deep() {
        let e = Expr::Case {
            whens: vec![(sum(column("a")).gt(int(1000)), int(1))],
            else_: Some(Box::new(int(0))),
        };
        assert_eq!(layer(&e), Some(ExprLayer::PostAggregate));
    }

    #[test]
    fn window_over_aggregate_is_post_aggregate() {
        assert_eq!(
            layer(&lag(sum(column("a")) + int(0))),
            Some(ExprLayer::PostAggregate)
        );
    }

    #[test]
    fn window_over_rows_only_is_scalar() {
        assert_eq!(layer(&lag(column("a"))), Some(ExprLayer::Scalar));
    }

    #[test]
    fn aggregate_inside_aggregate_arg_keeps_root_classification() {
        assert_eq!(layer(&sum(column("a"))), Some(ExprLayer::Aggregate));
    }

    #[test]
    fn a_link_counts_as_the_layer_of_the_entry_it_links() {
        let (pool, a, a2) = two_sums();
        let b = StoredExpr::from(Expr::Leaf(Link(a)) - Expr::Leaf(Link(a2)));
        assert_eq!(pool.layer_of(b.expr()), Some(ExprLayer::PostAggregate));
        assert_eq!(b.links().collect::<Vec<_>>(), [a, a2]);
        assert_eq!(
            pool.layer_of(&link(0xDEAD)),
            None,
            "a link to no entry has no layer"
        );
    }

    #[test]
    fn a_bare_link_is_a_stored_expression() {
        let (pool, a, _) = two_sums();
        let alias = StoredExpr::from(Expr::Leaf(Link(a)));
        assert_eq!(pool.layer_of(alias.expr()), Some(ExprLayer::Aggregate));
        assert_eq!(alias.links().collect::<Vec<_>>(), [a]);
    }

    #[test]
    fn a_stored_expression_reads_a_nested_tree_and_gives_its_links_in_pre_order() {
        let link_text = r#""26c8a8c0-1f01-5ef4-99a3-87ab96316377""#;
        let nested =
            format!(r#"{{"type":"is_null","content":{{"type":"is_null","content":{link_text}}}}}"#);
        let read: StoredExpr = serde_json::from_str(&nested).expect("a nested tree reads");
        assert_eq!(
            read.expr(),
            &Expr::IsNull(Box::new(Expr::IsNull(Box::new(Expr::Leaf(Link(EXPR_ID))))))
        );
        let tree = StoredExpr::from(link(1) + (link(2) + link(3)));
        assert_eq!(tree.links().collect::<Vec<_>>(), [1, 2, 3].map(id));
    }

    #[test]
    fn emitting_a_tree_stores_one_entry_that_holds_it_whole() {
        for (tree, layer) in [
            (cast(sum(cast(column("amount")))), ExprLayer::PostAggregate),
            (sum(column("amount")), ExprLayer::Aggregate),
            (column("x") + column("x"), ExprLayer::Scalar),
        ] {
            let mut pool = ManifestExpressions::default();
            let root = pool
                .emit(
                    &tree,
                    "sig.a",
                    ExprRoot::SignatureExpression,
                    ["src.orders".to_owned()],
                )
                .expect("emit");
            assert_eq!(pool.len(), 1, "one entry holds {tree:?}");
            let entry = pool.get(&root).expect("the entry");
            assert_eq!(entry.id, root);
            assert_eq!(entry.layer, layer, "the layer of {tree:?}");
            assert_eq!(
                entry.expr.links().count(),
                0,
                "compile's tree holds no link"
            );
            assert_eq!(
                pool.reassemble(&root, &[], &[]),
                Some(tree),
                "the entry holds the whole tree"
            );
        }
    }

    #[test]
    fn emitting_a_column_stores_one_entry_that_owns_its_id() {
        let mut pool = ManifestExpressions::default();
        let id = pool
            .emit(
                &column("region"),
                "sig.a",
                ExprRoot::SignatureExpression,
                [],
            )
            .expect("emit");
        assert_eq!(pool.len(), 1);
        let entry = pool.get(&id).expect("the entry");
        assert_eq!(entry.id, id);
        assert_eq!(
            entry.id,
            mint_expr_id("sig.a", ExprRoot::SignatureExpression)
        );
        assert_eq!(entry.layer, ExprLayer::Scalar);
    }

    #[test]
    fn emitting_a_parameter_is_refused() {
        use semstrait_ir::{Parameter, ParameterKey};

        let parameter = || {
            Expr::Leaf(Parameter {
                key: ParameterKey::RequestTemporalAxis,
                data_type: DataType::Date,
            })
        };
        let mut pool = ManifestExpressions::default();
        for tree in [parameter(), column("x") + parameter()] {
            match pool.emit(&tree, "sig.a", ExprRoot::SignatureExpression, []) {
                Err(CompileError::StructuralViolation { site, rule }) => {
                    assert_eq!(site, ReferenceSite::ExpressionPool);
                    assert_eq!(rule, RULE_NO_LEAF_STORED);
                }
                other => panic!("expected the Parameter to be refused, got {other:?}"),
            }
        }
        assert!(pool.is_empty());
    }

    #[test]
    fn emitting_a_link_to_an_entry_not_yet_stored_is_refused() {
        let mut pool = ManifestExpressions::default();
        let a = mint_expr_id("sig.a", ExprRoot::SignatureExpression);
        let metric = Expr::Leaf(Link(a)) + column("x");
        match pool.emit_linked(&metric, "sig.b", ExprRoot::SignatureExpression, []) {
            Err(CompileError::StructuralViolation { site, rule }) => {
                assert_eq!(site, ReferenceSite::ExpressionPool);
                assert_eq!(rule, RULE_LINKED_ENTRY_STORED_FIRST);
            }
            other => panic!("expected the link to be refused, got {other:?}"),
        }
        assert!(pool.is_empty(), "a refused emission stores nothing");
        pool.emit(
            &sum(column("amount")),
            "sig.a",
            ExprRoot::SignatureExpression,
            [],
        )
        .expect("emit");
        let b = (pool.emit_linked(&metric, "sig.b", ExprRoot::SignatureExpression, []))
            .expect("the linked entry is stored");
        assert_eq!(
            pool.get(&b).expect("the entry").layer,
            ExprLayer::PostAggregate
        );
    }

    #[test]
    fn linked_root_follows_a_chain_of_bare_links() {
        let mut pool = ManifestExpressions::default();
        let a = pool
            .emit(
                &column("region"),
                "sig.a",
                ExprRoot::SignatureExpression,
                [],
            )
            .expect("emit");
        let mut alias = |linked, owner| {
            pool.emit_linked(
                &Expr::Leaf(Link(linked)),
                owner,
                ExprRoot::SignatureExpression,
                [],
            )
            .expect("emit")
        };
        let b = alias(a, "sig.b");
        let c = alias(b, "sig.c");
        for root in [a, b, c] {
            assert_eq!(
                pool.linked_root(&root).map(StoredExpr::expr),
                Some(&column("region")),
                "{root}"
            );
        }
        assert_eq!(pool.linked_root(&id(0xDEAD)), None, "no entry");
        let mut cycle = pool.clone();
        let entry = cycle.physical.get_mut(&a).expect("the entry");
        entry.expr = StoredExpr::from(Expr::Leaf(Link(c)));
        assert_eq!(cycle.linked_root(&c), None, "a cycle of bare links");
    }

    #[test]
    fn reassembling_and_linked_root_follow_a_chain_of_4096_bare_links() {
        let mut pool = ManifestExpressions::default();
        let root = ExprRoot::SignatureExpression;
        let mut top = (pool.emit(&column("region"), "sig.0", root, [])).expect("emit");
        let mut signatures = vec![SemanticSignature::new("sig", "sem", top)];
        for n in 1..=4_096 {
            let alias = Expr::Leaf(Link(top));
            top = (pool.emit_linked(&alias, &format!("sig.{n}"), root, [])).expect("emit");
            signatures.push(SemanticSignature::new("sig", "sem", top));
        }
        assert_eq!(
            pool.linked_root(&top).map(StoredExpr::expr),
            Some(&column("region"))
        );
        let listed: Vec<_> = signatures.iter().collect();
        assert_eq!(pool.reassemble(&top, &listed, &[]), Some(column("region")));
    }

    #[test]
    fn reassembling_an_entry_yields_the_emitted_tree() {
        let expr = sum(column("amount"));
        let mut pool = ManifestExpressions::default();
        let root = pool
            .emit(&expr, "sig.a", ExprRoot::SignatureExpression, [])
            .expect("emit");
        assert_eq!(pool.reassemble(&root, &[], &[]), Some(expr));
    }

    #[test]
    fn reassembling_an_entry_follows_each_link_among_the_signatures() {
        let (mut pool, a, a2) = two_sums();
        let b = id(0xB);
        pool.insert(ManifestExpression {
            id: b,
            expr: StoredExpr::from(Expr::Leaf(Link(a)) - Expr::Leaf(Link(a2))),
            layer: ExprLayer::PostAggregate,
            physical_sources: BTreeSet::new(),
        })
        .expect("a free id");
        let signatures = [
            SemanticSignature::new("sig.a", "sem.a", a),
            SemanticSignature::new("sig.b", "sem.b", a2),
        ];
        let listed: Vec<_> = signatures.iter().collect();
        assert_eq!(
            pool.reassemble(&b, &listed, &[]),
            Some(sum(column("amount")) - sum(column("cost")))
        );
        assert_eq!(
            pool.reassemble(&b, &listed[..1], &[]),
            None,
            "the link to A2 names no signature given"
        );
        assert_eq!(pool.reassemble(&id(0xDEAD), &listed, &[]), None);
    }

    #[test]
    fn linked_gives_the_signature_of_each_entry_linked_outside_an_aggregate_at_any_depth() {
        let (mut pool, a, a2) = two_sums();
        let (b, c) = (id(0xB), id(0xC));
        let mut insert = |id, expr: Expr<Link>| {
            pool.insert(ManifestExpression {
                id,
                expr: StoredExpr::from(expr),
                layer: ExprLayer::PostAggregate,
                physical_sources: BTreeSet::new(),
            })
            .expect("a free id");
        };
        insert(b, Expr::Leaf(Link(a)) - Expr::Leaf(Link(a2)));
        insert(c, Expr::Leaf(Link(b)) * sum(link(0xE)));
        let signatures = [
            SemanticSignature::new("sig.a", "sem.a", a),
            SemanticSignature::new("sig.b", "sem.b", a2),
            SemanticSignature::new("sig.c", "sem.c", b),
        ];
        let listed: Vec<_> = signatures.iter().collect();
        assert_eq!(pool.linked(&c, &listed), Some(listed.clone()));
        assert_eq!(
            pool.linked(&c, &listed[..2]),
            None,
            "the link to B names no signature given"
        );
        assert_eq!(pool.linked(&id(0xDEAD), &listed), None);
        let mut cycle = pool.clone();
        let entry = cycle.physical.get_mut(&a).expect("the entry");
        entry.expr = StoredExpr::from(Expr::Leaf(Link(c)));
        assert_eq!(cycle.linked(&c, &listed), None, "a cycle");
    }

    #[test]
    fn reached_gives_the_signature_of_each_entry_read_through_at_any_depth() {
        let (mut pool, a, a2) = two_sums();
        let (b, f) = (id(0xB), id(0xF));
        let mut insert = |id, expr: Expr<Link>| {
            pool.insert(ManifestExpression {
                id,
                expr: StoredExpr::from(expr),
                layer: ExprLayer::PostAggregate,
                physical_sources: BTreeSet::new(),
            })
            .expect("a free id");
        };
        insert(b, sum(Expr::Leaf(Link(a))));
        insert(f, Expr::Leaf(Link(a2)));
        let signatures = [
            SemanticSignature::new("sig.a", "sem.a", a).with_filter(f),
            SemanticSignature::new("sig.b", "sem.b", a2),
            SemanticSignature::new("sig.c", "sem.c", b),
        ];
        let listed: Vec<_> = signatures.iter().collect();
        assert_eq!(pool.reached(&[b], &listed), Some(listed.clone()));
        assert_eq!(
            pool.reached(&[b], &listed[1..]),
            None,
            "the link to A names no signature given"
        );
        assert_eq!(pool.reached(&[id(0xDEAD)], &listed), None);
        let mut cycle = pool.clone();
        let entry = cycle.physical.get_mut(&a).expect("the entry");
        entry.expr = StoredExpr::from(Expr::Leaf(Link(b)));
        assert_eq!(cycle.reached(&[b], &listed), None, "a cycle");
    }

    #[test]
    fn aggregate_signatures_gives_the_signature_whose_entry_holds_each_aggregate_call_in_pre_order()
    {
        let (mut pool, a, a2) = two_sums();
        let [b, c, d, f] = [0xB, 0xC, 0xD, 0xF].map(id);
        let mut insert = |id, expr: Expr<Link>| {
            pool.insert(ManifestExpression {
                id,
                expr: StoredExpr::from(expr),
                layer: ExprLayer::PostAggregate,
                physical_sources: BTreeSet::new(),
            })
            .expect("a free id");
        };
        insert(b, Expr::Leaf(Link(a2)) - Expr::Leaf(Link(a)));
        insert(c, Expr::Leaf(Link(b)) * Expr::Leaf(Link(a)));
        insert(d, Expr::Leaf(Link(a)) / Expr::Leaf(Link(a)));
        insert(f, sum(link(0xE)) + Expr::Leaf(Link(a)));
        let signatures = [
            SemanticSignature::new("sig.a", "sem.a", a),
            SemanticSignature::new("sig.a2", "sem.a2", a2),
            SemanticSignature::new("sig.b", "sem.b", b),
            SemanticSignature::new("sig.f", "sem.f", f),
        ];
        let listed: Vec<_> = signatures.iter().collect();
        let [sig_a, sig_a2, _, sig_f] = &signatures;
        for (root, held) in [
            (a, vec![sig_a]),
            (b, vec![sig_a2, sig_a]),
            (d, vec![sig_a, sig_a]),
            (c, vec![sig_a2, sig_a, sig_a]),
            (f, vec![sig_f, sig_a]),
        ] {
            assert_eq!(pool.aggregate_signatures(&root, &listed, &[]), Some(held));
        }
        assert_eq!(
            pool.aggregate_signatures(&c, &listed, &[b]),
            Some(vec![sig_a]),
            "the kept link to B holds no call"
        );
        assert_eq!(
            pool.aggregate_signatures(&c, &[sig_a, sig_a2], &[]),
            None,
            "the link to B names no signature given"
        );
        assert_eq!(
            pool.aggregate_signatures(&f, &[sig_a], &[]),
            None,
            "F's own call is held by no signature given"
        );
        assert_eq!(pool.aggregate_signatures(&id(0xDEAD), &listed, &[]), None);
        let mut cycle = pool.clone();
        let entry = cycle.physical.get_mut(&a).expect("the entry");
        entry.expr = StoredExpr::from(Expr::Leaf(Link(c)));
        assert_eq!(
            cycle.aggregate_signatures(&c, &listed, &[]),
            None,
            "a cycle"
        );
    }

    #[test]
    fn one_expression_under_two_owners_is_two_entries() {
        let mut pool = ManifestExpressions::default();
        let first = pool
            .emit(
                &column("region"),
                "sig.a",
                ExprRoot::SignatureExpression,
                [],
            )
            .expect("emit");
        let second = pool
            .emit(
                &column("region"),
                "sig.b",
                ExprRoot::SignatureExpression,
                [],
            )
            .expect("emit");
        assert_ne!(first, second);
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn one_owners_two_markers_are_two_entries() {
        let mut pool = ManifestExpressions::default();
        let expression = pool
            .emit(
                &column("region"),
                "sig.a",
                ExprRoot::SignatureExpression,
                [],
            )
            .expect("emit");
        let filter = pool
            .emit(&column("region"), "sig.a", ExprRoot::SignatureFilter, [])
            .expect("emit");
        assert_ne!(expression, filter);
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn emitting_twice_under_one_owner_and_marker_is_refused() {
        let mut pool = ManifestExpressions::default();
        pool.emit(
            &column("region"),
            "sig.a",
            ExprRoot::SignatureExpression,
            [],
        )
        .expect("emit");
        let before = pool.clone();

        for repeat in [column("region"), column("region") + column("amount")] {
            match pool.emit(&repeat, "sig.a", ExprRoot::SignatureExpression, []) {
                Err(CompileError::StructuralViolation { site, rule }) => {
                    assert_eq!(site, ReferenceSite::ExpressionPool);
                    assert_eq!(rule, RULE_ID_MINTED_ONCE);
                }
                other => panic!("expected the repeated id to be refused, got {other:?}"),
            }
        }
        assert_eq!(
            pool, before,
            "a refused emission leaves the store unchanged"
        );
    }

    #[test]
    fn the_pool_round_trips_through_json() {
        let mut pool = ManifestExpressions::default();
        pool.emit(
            &sum(column("amount")),
            "sig.a",
            ExprRoot::SignatureExpression,
            ["src.a".to_owned()],
        )
        .expect("emit");
        pool.emit(
            &column("region"),
            "sig.b",
            ExprRoot::SignatureExpression,
            [],
        )
        .expect("emit");
        let json = serde_json::to_string(&pool).expect("serialize ManifestExpressions");
        assert!(
            json.contains(
                r#""expr":{"type":"aggregate","content":{"name":"sum","args":[{"type":"column","content":["amount"]}]}}"#
            ),
            "the entry holds the whole aggregate: {json}"
        );
        let back: ManifestExpressions =
            serde_json::from_str(&json).expect("deserialize ManifestExpressions");
        assert_eq!(back, pool);
        assert_eq!(back.len(), 2);
        assert!(!back.is_empty());
    }

    #[test]
    fn emitting_a_deep_chain_stores_one_post_aggregate_entry() {
        const DEPTH: usize = 4_096;
        const STACK_SIZE: usize = 64 * 1024 * 1024;

        std::thread::Builder::new()
            .stack_size(STACK_SIZE)
            .spawn(|| {
                let chain = (0..DEPTH).fold(sum(column("amount")), |inner, _| {
                    Expr::IsNull(Box::new(inner))
                });
                let mut pool = ManifestExpressions::default();
                let root = pool
                    .emit(
                        &chain,
                        "sig.a",
                        ExprRoot::SignatureExpression,
                        ["src.orders".to_owned()],
                    )
                    .expect("emit");
                assert_eq!(pool.len(), 1, "the whole chain is one entry");
                assert_eq!(
                    pool.get(&root).expect("the entry").layer,
                    ExprLayer::PostAggregate
                );
            })
            .expect("spawn the oversized-stack thread")
            .join()
            .expect("the test body panicked");
    }

    #[test]
    fn from_entries_keys_each_entry_by_its_own_id() {
        let mut source = ManifestExpressions::default();
        source
            .emit(&column("a"), "sig.a", ExprRoot::SignatureExpression, [])
            .expect("emit");
        source
            .emit(&column("b"), "sig.b", ExprRoot::SignatureExpression, [])
            .expect("emit");
        let pool =
            ManifestExpressions::from_entries(source.physical.values().cloned()).expect("emit");
        assert_eq!(pool.len(), 2);
        for (key, entry) in &pool.physical {
            assert_eq!(key, &entry.id);
        }
    }
}
