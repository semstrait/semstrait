use std::collections::{BTreeMap, BTreeSet};

use semstrait_common::diagnostic::Location;
use semstrait_common::SemanticsName;
use semstrait_sql::datatype::{DataType, TypeFamily};
use semstrait_sql::expr::Expr;

use crate::manifest::expression::Link;
use crate::manifest::semantic_definition::SemanticDefinition;
use semstrait_common::EntityId;
use semstrait_ir::SchemaColumn;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResolvedExprKey {
    pub semantics_name: SemanticsName,
    pub binding_id: EntityId,
}

impl ResolvedExprKey {
    #[must_use]
    pub fn new(semantics_name: SemanticsName, binding_id: impl Into<EntityId>) -> Self {
        Self {
            semantics_name,
            binding_id: binding_id.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedExprEntry {
    pub expr: Expr<Link>,
    pub inferred_type: DataType,
    pub family: Option<TypeFamily>,
    pub argument_type: Option<DataType>,
    pub referenced_columns: Vec<String>,
    pub provenance: Provenance,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResolvedExprTable {
    entries: BTreeMap<ResolvedExprKey, ResolvedExprEntry>,
}

impl ResolvedExprTable {
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    #[must_use]
    pub fn lookup(
        &self,
        name: &SemanticsName,
        binding_id: &EntityId,
    ) -> Option<&ResolvedExprEntry> {
        let probe = ResolvedExprKey {
            semantics_name: name.clone(),
            binding_id: binding_id.clone(),
        };
        self.entries.get(&probe)
    }

    pub fn lookup_all<'s>(
        &'s self,
        name: &'s SemanticsName,
    ) -> impl Iterator<Item = (&'s EntityId, &'s ResolvedExprEntry)> {
        self.entries
            .iter()
            .filter(move |(k, _)| &k.semantics_name == name)
            .map(|(k, entry)| (&k.binding_id, entry))
    }

    pub fn iter(&self) -> impl Iterator<Item = (&ResolvedExprKey, &ResolvedExprEntry)> {
        self.entries.iter()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub(crate) fn insert(&mut self, key: ResolvedExprKey, entry: ResolvedExprEntry) {
        self.entries.insert(key, entry);
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Provenance {
    // TODO: populate during the Tier-1/Tier-2 occurrence merge in `compile`.
    pub contributions: Vec<Location>,
}

impl Provenance {
    #[must_use]
    pub fn new() -> Self {
        Self {
            contributions: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LeafPlace {
    Root,
    JoinsetConstituent,
    UnionsetBranch,
    GrainsetChild,
}

pub(crate) struct LoweringCtx<'a> {
    pub all_semantics: &'a SemanticsIndex<'a>,
    pub resolved: &'a ResolvedExprTable,
    pub reference: Reference,
    pub binding: &'a EntityId,
    pub owner: &'a SemanticsName,
    pub row_level: bool,
    pub manual_mapping: bool,
    pub composed_filter_unit: Option<&'a str>,
    pub place: Option<LeafPlace>,
    pub recursion: &'a mut RecursionState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Reference {
    Link,
    Inline,
}

#[derive(Debug, Default)]
pub struct SemanticsIndex<'a> {
    by_name: BTreeMap<SemanticsName, &'a SemanticDefinition>,
}

impl<'a> SemanticsIndex<'a> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            by_name: BTreeMap::new(),
        }
    }

    #[must_use]
    pub fn from_map(by_name: BTreeMap<SemanticsName, &'a SemanticDefinition>) -> Self {
        Self { by_name }
    }

    #[must_use]
    pub fn get(&self, name: &SemanticsName) -> Option<&'a SemanticDefinition> {
        self.by_name.get(name).copied()
    }

    pub fn names(&self) -> impl Iterator<Item = &SemanticsName> {
        self.by_name.keys()
    }
}

#[derive(Debug, Default)]
pub struct SchemaIndex {
    by_source: BTreeMap<EntityId, Vec<SchemaColumn>>,
}

impl SchemaIndex {
    #[must_use]
    pub fn new() -> Self {
        Self {
            by_source: BTreeMap::new(),
        }
    }

    #[must_use]
    pub fn from_map(by_source: BTreeMap<EntityId, Vec<SchemaColumn>>) -> Self {
        Self { by_source }
    }

    #[must_use]
    pub fn columns(&self, source_id: &EntityId) -> Option<&[SchemaColumn]> {
        self.by_source.get(source_id).map(Vec::as_slice)
    }
}

#[derive(Debug, Default)]
pub struct ScopeChain {
    frames: Vec<ScopeFrame>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScopeFrame {
    pub scope: ScopeKind,
    pub owner: Option<EntityId>,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScopeKind {
    Binding,
    NestedKind,
    Kind,
    Root,
}

impl ScopeChain {
    #[must_use]
    pub fn new() -> Self {
        Self { frames: Vec::new() }
    }

    #[must_use]
    pub fn from_frames(frames: Vec<ScopeFrame>) -> Self {
        Self { frames }
    }

    #[must_use]
    pub fn frames(&self) -> &[ScopeFrame] {
        &self.frames
    }
}

#[derive(Debug, Default)]
pub struct RecursionState {
    visited: BTreeSet<SemanticsName>,
    stack: Vec<SemanticsName>,
    index: BTreeMap<SemanticsName, usize>,
    low_link: BTreeMap<SemanticsName, usize>,
    on_stack: BTreeSet<SemanticsName>,
    next_index: usize,
}

impl RecursionState {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn is_active(&self, name: &SemanticsName) -> bool {
        self.visited.contains(name)
    }

    pub fn enter(&mut self, name: SemanticsName) -> bool {
        if self.visited.contains(&name) {
            return false;
        }
        self.visited.insert(name.clone());
        self.stack.push(name);
        true
    }

    pub fn exit(&mut self, name: &SemanticsName) {
        if self.stack.last() == Some(name) {
            self.stack.pop();
        }
        self.visited.remove(name);
    }

    #[must_use]
    pub fn stack(&self) -> &[SemanticsName] {
        &self.stack
    }

    pub fn tarjan_visit(&mut self, name: SemanticsName) -> usize {
        let idx = self.next_index;
        self.next_index += 1;
        self.index.insert(name.clone(), idx);
        self.low_link.insert(name.clone(), idx);
        self.on_stack.insert(name);
        idx
    }

    #[must_use]
    pub fn tarjan_index(&self, name: &SemanticsName) -> Option<usize> {
        self.index.get(name).copied()
    }

    #[must_use]
    pub fn tarjan_low_link(&self, name: &SemanticsName) -> Option<usize> {
        self.low_link.get(name).copied()
    }

    pub fn tarjan_relax_low_link(&mut self, name: &SemanticsName, candidate: usize) {
        if let Some(slot) = self.low_link.get_mut(name) {
            if candidate < *slot {
                *slot = candidate;
            }
        }
    }

    #[must_use]
    pub fn tarjan_on_stack(&self, name: &SemanticsName) -> bool {
        self.on_stack.contains(name)
    }

    pub fn tarjan_pop(&mut self, name: &SemanticsName) {
        self.on_stack.remove(name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use semstrait_sql::expr::NamedReference;

    fn entry(col: &str) -> ResolvedExprEntry {
        ResolvedExprEntry {
            expr: Expr::Column(NamedReference::unqualified(col).expect("a name")),
            inferred_type: DataType::Integer,
            family: Some(TypeFamily::Number),
            argument_type: None,
            referenced_columns: vec![col.to_owned()],
            provenance: Provenance::new(),
        }
    }

    fn key(name: &str, binding: &str) -> ResolvedExprKey {
        ResolvedExprKey::new(SemanticsName::from(name), binding)
    }

    #[test]
    fn lookup_hit_and_miss() {
        let mut table = ResolvedExprTable::new();
        table.insert(key("revenue", "bind-a"), entry("amount"));

        let hit = table.lookup(&SemanticsName::from("revenue"), &"bind-a".to_owned());
        assert!(hit.is_some());
        assert_eq!(hit.expect("hit").referenced_columns, vec!["amount"]);

        assert!(table
            .lookup(&SemanticsName::from("revenue"), &"bind-z".to_owned())
            .is_none());
        assert!(table
            .lookup(&SemanticsName::from("missing"), &"bind-a".to_owned())
            .is_none());
        assert_eq!(table.len(), 1);
        assert!(!table.is_empty());
    }

    #[test]
    fn lookup_all_ranges_one_name_across_bindings() {
        let mut table = ResolvedExprTable::new();
        table.insert(key("revenue", "bind-b"), entry("amount_b"));
        table.insert(key("revenue", "bind-a"), entry("amount_a"));
        table.insert(key("region", "bind-a"), entry("region_col"));

        let revenue = SemanticsName::from("revenue");
        let found: Vec<(&EntityId, &ResolvedExprEntry)> = table.lookup_all(&revenue).collect();
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].0, "bind-a");
        assert_eq!(found[1].0, "bind-b");
        assert_eq!(found[0].1.referenced_columns, vec!["amount_a"]);
    }

    #[test]
    fn iter_is_key_ordered() {
        let mut table = ResolvedExprTable::new();
        table.insert(key("revenue", "bind-a"), entry("r"));
        table.insert(key("region", "bind-a"), entry("g"));
        let names: Vec<&str> = table
            .iter()
            .map(|(k, _)| k.semantics_name.as_str())
            .collect();
        assert_eq!(names, vec!["region", "revenue"]);
    }

    #[test]
    fn recursion_state_enter_exit_tracks_cycles() {
        let mut rec = RecursionState::new();
        let a = SemanticsName::from("a");
        let b = SemanticsName::from("b");

        assert!(rec.enter(a.clone()));
        assert!(rec.is_active(&a));
        assert!(rec.enter(b.clone()));
        assert!(!rec.enter(a.clone()));
        assert_eq!(rec.stack(), &[a.clone(), b.clone()]);

        rec.exit(&b);
        assert!(!rec.is_active(&b));
        assert!(!rec.enter(a.clone()));
        rec.exit(&a);
        assert!(rec.enter(a.clone()));
    }

    #[test]
    fn index_constructors_default_empty() {
        assert!(SemanticsIndex::new()
            .get(&SemanticsName::from("x"))
            .is_none());
        assert!(SchemaIndex::new().columns(&"s".to_owned()).is_none());
        assert!(ScopeChain::new().frames().is_empty());
        assert!(Provenance::new().contributions.is_empty());
    }
}
