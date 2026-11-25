use std::collections::{BTreeMap, BTreeSet};

use semstrait_common::SemanticsName;
use semstrait_common::{EntityId, Tree};
use semstrait_ir::SemanticRole;
use semstrait_model::{
    SemanticInterface as ModelInterface, UnresolvedSemanticExpr as SemanticExpr,
};
use semstrait_sql::expr::Expr;

use super::super::definitions;
use super::super::types::{
    LeafPlace, LoweringCtx, RecursionState, Reference, ResolvedExprEntry, ResolvedExprKey,
    ResolvedExprTable, SemanticsIndex,
};
use super::columns::ColumnTypes;
use super::lower;
use crate::error::{CompileError, ReferenceScope};
use crate::identity::{mint_expr_id, mint_signature_id};
use crate::manifest::expression::{ExprRoot, Link, PhysicalExprId};
use crate::manifest::semantic_definition::{SemanticDefinition, SemanticRoleDefinition};

#[derive(Debug, Clone, Default)]
pub(crate) struct SourceBodies<'m>(BTreeMap<SemanticsName, &'m SemanticExpr>);

trait SourceView<'m> {
    fn as_map(&self) -> &BTreeMap<SemanticsName, &'m SemanticExpr>;
}

impl<'m> SourceBodies<'m> {
    fn insert(&mut self, name: SemanticsName, expr: &'m SemanticExpr) {
        self.0.insert(name, expr);
    }

    pub(crate) fn get(&self, name: &SemanticsName) -> Option<&&'m SemanticExpr> {
        self.0.get(name)
    }

    pub(crate) fn contains_key(&self, name: &SemanticsName) -> bool {
        self.0.contains_key(name)
    }

    pub(crate) fn retain(&mut self, mut keep: impl FnMut(&SemanticsName) -> bool) {
        self.0.retain(|name, _| keep(name));
    }
}

impl<'m> SourceView<'m> for SourceBodies<'m> {
    fn as_map(&self) -> &BTreeMap<SemanticsName, &'m SemanticExpr> {
        &self.0
    }
}

impl<'a, 'm> IntoIterator for &'a SourceBodies<'m> {
    type Item = (&'a SemanticsName, &'a &'m SemanticExpr);
    type IntoIter = std::collections::btree_map::Iter<'a, SemanticsName, &'m SemanticExpr>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct SourceFilters<'m>(BTreeMap<SemanticsName, &'m SemanticExpr>);

impl<'m> SourceFilters<'m> {
    fn insert(&mut self, name: SemanticsName, expr: &'m SemanticExpr) {
        self.0.insert(name, expr);
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub(crate) fn retain(&mut self, mut keep: impl FnMut(&SemanticsName) -> bool) {
        self.0.retain(|name, _| keep(name));
    }

    pub(crate) fn singleton(name: SemanticsName, expr: &'m SemanticExpr) -> Self {
        let mut out = Self::default();
        out.insert(name, expr);
        out
    }
}

impl<'m> SourceView<'m> for SourceFilters<'m> {
    fn as_map(&self) -> &BTreeMap<SemanticsName, &'m SemanticExpr> {
        &self.0
    }
}

impl<'a, 'm> IntoIterator for &'a SourceFilters<'m> {
    type Item = (&'a SemanticsName, &'a &'m SemanticExpr);
    type IntoIter = std::collections::btree_map::Iter<'a, SemanticsName, &'m SemanticExpr>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

pub(crate) fn body_view<'m>(
    interface: &'m ModelInterface,
    pool: &'m BTreeMap<SemanticsName, SemanticExpr>,
) -> SourceBodies<'m> {
    let mut out = SourceBodies::default();
    for filter in interface.filters.values() {
        out.insert(filter.name.clone(), &filter.expr);
    }
    let sites = interface
        .dimensions
        .values()
        .map(|site| (site.name(), site.body().expr.as_ref()))
        .chain(
            interface
                .measures
                .values()
                .map(|site| (site.name(), site.body().expr.as_ref())),
        )
        .chain(
            interface
                .metrics
                .values()
                .map(|site| (site.name(), site.body().expr.as_ref())),
        );
    for (name, site_body) in sites {
        if let Some(expr) = site_body.or_else(|| pool.get(name)) {
            out.insert(name.clone(), expr);
        }
    }
    out
}

pub(crate) fn filter_view<'m>(
    interface: &'m ModelInterface,
    pool: &'m BTreeMap<SemanticsName, SemanticExpr>,
) -> SourceFilters<'m> {
    let mut out: SourceFilters<'m> = SourceFilters::default();
    let sites = interface
        .measures
        .values()
        .map(|site| (site.name(), measure_site_filter(site)))
        .chain(
            interface
                .metrics
                .values()
                .map(|site| (site.name(), metric_site_filter(site))),
        );
    for (name, site_filter) in sites {
        if let Some(expr) = site_filter.or_else(|| pool.get(name)) {
            out.insert(name.clone(), expr);
        }
    }
    out
}

pub(crate) fn site_authored_filter_names(interface: &ModelInterface) -> BTreeSet<SemanticsName> {
    let mut out = BTreeSet::new();
    for site in interface.measures.values() {
        if measure_site_filter(site).is_some() {
            out.insert(site.name().clone());
        }
    }
    for site in interface.metrics.values() {
        if metric_site_filter(site).is_some() {
            out.insert(site.name().clone());
        }
    }
    out
}

fn measure_site_filter(site: &semstrait_model::MeasureSite) -> Option<&SemanticExpr> {
    match site {
        semstrait_model::SemanticSite::Inline(measure) => measure.filter.as_ref(),
        semstrait_model::SemanticSite::Reference(reference) => reference.filter.as_ref(),
        _ => None,
    }
}

fn metric_site_filter(site: &semstrait_model::MetricSite) -> Option<&SemanticExpr> {
    match site {
        semstrait_model::SemanticSite::Inline(metric) => metric.filter.as_ref(),
        semstrait_model::SemanticSite::Reference(reference) => reference.filter.as_ref(),
        _ => None,
    }
}

#[derive(Clone, Copy)]
pub(crate) struct SourceScope<'a> {
    pub source_name: &'a str,
    pub source_id: &'a EntityId,
    pub exposed: &'a BTreeSet<SemanticsName>,
    pub index: &'a SemanticsIndex<'a>,
    pub definitions: &'a [SemanticDefinition],
    pub manual_mapping: bool,
    pub composed_filter_unit: Option<&'a str>,
    pub reference: Reference,
    pub columns: &'a ColumnTypes,
    pub place: Option<LeafPlace>,
}

fn registered_names(definitions: &[SemanticDefinition]) -> BTreeSet<SemanticsName> {
    definitions.iter().map(|def| def.name.clone()).collect()
}

fn check_reference_legality<'m>(
    view: &impl SourceView<'m>,
    scope: &SourceScope<'_>,
) -> Result<BTreeMap<SemanticsName, BTreeSet<SemanticsName>>, CompileError> {
    let registered = registered_names(scope.definitions);
    definitions::build_reference_graph(
        view.as_map(),
        scope.exposed,
        &registered,
        scope.source_name,
        scope.source_id,
    )
}

pub(crate) fn seed_table(
    seed: &BTreeMap<SemanticsName, ResolvedExprEntry>,
    source_id: &EntityId,
) -> ResolvedExprTable {
    let mut table = ResolvedExprTable::new();
    for (name, entry) in seed {
        table.insert(
            ResolvedExprKey::new(name.clone(), source_id.clone()),
            entry.clone(),
        );
    }
    table
}

fn is_unmapped_own_reference(err: &CompileError, source_id: &EntityId) -> bool {
    matches!(
        err,
        CompileError::UnknownReference {
            scope: ReferenceScope::Appearance(appearance),
            ..
        } if appearance == source_id
    )
}

#[derive(Clone, Copy)]
pub(crate) struct SourceLowering<'a> {
    pub scope: SourceScope<'a>,
    pub bodies: &'a SourceBodies<'a>,
    pub body_free: &'a BTreeMap<SemanticsName, ResolvedExprEntry>,
    pub filters: &'a SourceFilters<'a>,
}

#[derive(Default)]
pub(crate) struct SourceLowered {
    pub bodies: Vec<(SemanticsName, ResolvedExprEntry)>,
    pub filters: BTreeMap<SemanticsName, ResolvedExprEntry>,
    pub dropped: BTreeSet<SemanticsName>,
    pub row_level_metrics: BTreeSet<SemanticsName>,
}

pub(crate) fn lower_source(input: &SourceLowering<'_>) -> Result<SourceLowered, CompileError> {
    let &SourceLowering {
        scope,
        bodies,
        body_free,
        filters,
    } = input;
    let graph = check_reference_legality(bodies, &scope)?;
    let mut acyclicity = RecursionState::new();
    let mut order = definitions::graph::topological_order(&graph, &mut acyclicity)?;
    order.sort_by_key(|name| {
        (scope.index.get(name)).is_some_and(|definition| definition.role() == SemanticRole::Metric)
    });

    let mut table = seed_table(body_free, scope.source_id);
    let mut lowered = Vec::new();
    let mut recursion = RecursionState::new();
    let mut row_level_metrics: BTreeSet<SemanticsName> = BTreeSet::new();
    for name in order {
        let (Some(body), Some(definition)) = (bodies.get(&name), scope.index.get(&name)) else {
            continue;
        };
        let is_metric = definition.role() == SemanticRole::Metric;
        let body = match lower::resolve(
            body,
            &mut scope.lowering(&table, &name, !is_metric, &mut recursion),
        ) {
            Ok(body) => body,
            Err(err) if is_unmapped_own_reference(&err, scope.source_id) => continue,
            Err(other) => return Err(other),
        };
        let row_level = (graph.get(&name).into_iter().flatten()).all(|referenced| {
            scope
                .index
                .get(referenced)
                .is_some_and(|referenced| match referenced.role() {
                    SemanticRole::Dimension | SemanticRole::Filter => true,
                    SemanticRole::Measure => false,
                    SemanticRole::Metric => row_level_metrics.contains(&referenced.name),
                    other => unreachable!("semstrait-ir added a SemanticRole variant: {other:?}"),
                })
        });
        let (columns, declared) = (scope.columns, definition.data_type.as_ref());
        let aggregation = definition.role_definition.aggregation();
        let entry = match (aggregation, &definition.role_definition) {
            (Some(aggregation), _) if row_level => body
                .aggregated(aggregation, name.as_str())?
                .typed(columns, declared, name.as_str())?,
            (Some(aggregation), _) => {
                body.second_step(aggregation, columns, declared, name.as_str())?
            }
            (_, SemanticRoleDefinition::Filter) => body.typed_condition(columns, name.as_str())?,
            (
                _,
                SemanticRoleDefinition::Dimension { .. }
                | SemanticRoleDefinition::Measure { .. }
                | SemanticRoleDefinition::Metric { .. },
            ) => body.typed(columns, declared, name.as_str())?,
        };
        let entry = lower::reconcile_boundary(entry, declared, name.as_str())?;
        if is_metric && row_level && aggregation.is_none() {
            row_level_metrics.insert(name.clone());
        }
        table.insert(
            ResolvedExprKey::new(name.clone(), scope.source_id.clone()),
            entry.clone(),
        );
        lowered.push((name, entry));
    }
    let FiltersLowered {
        mut filters,
        dropped: dropped_by_filter,
    } = lower_source_filters(&SourceFiltering {
        scope,
        filters,
        resolved: &table,
    })?;
    let dropped = with_linkers(scope.source_id, scope.index, &lowered, dropped_by_filter);
    let bodies = hold_own_filters(
        (lowered.into_iter()).filter(|(name, _)| !dropped.contains(name)),
        &row_level_metrics,
        &mut filters,
    );
    Ok(SourceLowered {
        bodies,
        filters,
        dropped,
        row_level_metrics,
    })
}

pub(crate) fn with_linkers(
    source_id: &EntityId,
    index: &SemanticsIndex<'_>,
    bodies: &[(SemanticsName, ResolvedExprEntry)],
    mut dropped: BTreeSet<SemanticsName>,
) -> BTreeSet<SemanticsName> {
    let link = |name: &SemanticsName| {
        (index.get(name)).map(|definition| {
            let signature = mint_signature_id(source_id, &definition.id);
            mint_expr_id(&signature, ExprRoot::SignatureExpression)
        })
    };
    let mut links: BTreeSet<PhysicalExprId> = dropped.iter().filter_map(link).collect();
    for (name, entry) in bodies {
        let links_dropped = (entry.expr.pre_order())
            .any(|node| matches!(node, Expr::Leaf(Link(id)) if links.contains(id)));
        if links_dropped {
            dropped.insert(name.clone());
            links.extend(link(name));
        }
    }
    dropped
}

pub(crate) fn hold_own_filters(
    bodies: impl IntoIterator<Item = (SemanticsName, ResolvedExprEntry)>,
    row_level_metrics: &BTreeSet<SemanticsName>,
    filters: &mut BTreeMap<SemanticsName, ResolvedExprEntry>,
) -> Vec<(SemanticsName, ResolvedExprEntry)> {
    (bodies.into_iter())
        .map(|(name, mut entry)| {
            let filter = (row_level_metrics.contains(&name))
                .then(|| filters.remove(&name))
                .flatten();
            if let Some(filter) = filter {
                entry.expr = Expr::Case {
                    whens: vec![(filter.expr, entry.expr)],
                    else_: None,
                };
            }
            (name, entry)
        })
        .collect()
}

impl SourceScope<'_> {
    fn lowering<'c>(
        &self,
        resolved: &'c ResolvedExprTable,
        owner: &'c SemanticsName,
        row_level: bool,
        recursion: &'c mut RecursionState,
    ) -> LoweringCtx<'c>
    where
        Self: 'c,
    {
        LoweringCtx {
            all_semantics: self.index,
            resolved,
            reference: self.reference,
            binding: self.source_id,
            owner,
            row_level,
            manual_mapping: self.manual_mapping,
            composed_filter_unit: self.composed_filter_unit,
            place: self.place,
            recursion,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct SourceFiltering<'a> {
    pub scope: SourceScope<'a>,
    pub filters: &'a SourceFilters<'a>,
    pub resolved: &'a ResolvedExprTable,
}

pub(crate) fn lower_source_filters(
    input: &SourceFiltering<'_>,
) -> Result<FiltersLowered, CompileError> {
    let &SourceFiltering {
        scope,
        filters,
        resolved,
    } = input;
    if filters.is_empty() {
        return Ok(FiltersLowered {
            filters: BTreeMap::new(),
            dropped: BTreeSet::new(),
        });
    }

    check_reference_legality(filters, &scope)?;

    let mut recursion = RecursionState::new();
    let mut lowered = BTreeMap::new();
    let mut dropped = BTreeSet::new();
    for (name, filter) in filters {
        let mut ctx = scope.lowering(resolved, name, true, &mut recursion);
        let node = format!("{name} filter");
        match lower::resolve(filter, &mut ctx)
            .and_then(|tree| tree.typed_condition(scope.columns, &node))
        {
            Ok(entry) => {
                lowered.insert(name.clone(), entry);
            }
            Err(err) if is_unmapped_own_reference(&err, scope.source_id) => {
                if scope.place == Some(LeafPlace::GrainsetChild) {
                    dropped.insert(name.clone());
                }
            }
            Err(other) => return Err(other),
        }
    }
    Ok(FiltersLowered {
        filters: lowered,
        dropped,
    })
}

pub(crate) struct FiltersLowered {
    pub filters: BTreeMap<SemanticsName, ResolvedExprEntry>,
    pub dropped: BTreeSet<SemanticsName>,
}
