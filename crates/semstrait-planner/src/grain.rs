use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::iter;

use semstrait_common::Granularity;
use semstrait_ir::{AdditivityType, GranularityGroup, PhysicalExpr};
use semstrait_manifest::{
    SemanticBitmask, SemanticDefinition, SemanticRoleDefinition, SourceAlias,
};
use semstrait_sql::expr::{BinaryOpKind, Expr, NamedReference};
use semstrait_sql::functions::{Additivity, AggregateFunction};

use crate::frame::Frame;
use crate::read_model::{is_dataset, is_grainset, ReadModel, Relationship, TimeAxis};
use crate::strategy::{aggregate_definitions, qualifies, StrategyInput};
use crate::{Advisory, BoundName, FramePath, PlanError, Selection};

#[derive(Clone)]
pub(crate) struct RequestGrain(BTreeMap<u32, Option<Granularity>>);

impl RequestGrain {
    pub(crate) fn new(
        read_model: &(impl ReadModel + ?Sized),
        selected: impl IntoIterator<Item = (u32, Option<Granularity>)>,
    ) -> Self {
        Self(
            (selected.into_iter())
                .filter(|(bit, _)| is_dimension(read_model, *bit))
                .collect(),
        )
    }

    pub(crate) fn of(
        read_model: &impl ReadModel,
        select: &[Selection],
        bindings: &[BoundName],
    ) -> Self {
        let selected = (select.iter().zip(bindings))
            .map(|(selection, binding)| (binding.bit, selection.granularity()));
        Self::new(read_model, selected)
    }

    pub(crate) fn fixed(
        &self,
        source: &SourceGrain,
        granularity: Option<Granularity>,
    ) -> Cow<'_, Self> {
        let Some(granularity) = granularity else {
            return Cow::Borrowed(self);
        };
        Cow::Owned(Self(
            (self.0.iter())
                .map(|(&bit, &asked)| (bit, source.fixed(bit, asked, granularity)))
                .collect(),
        ))
    }

    pub(crate) fn granular(&self) -> Vec<u32> {
        (self.0.iter())
            .filter(|(_, granularity)| granularity.is_some())
            .map(|(bit, _)| *bit)
            .collect()
    }

    pub(crate) fn without_granularity(&self, bits: &[u32]) -> Cow<'_, Self> {
        if !(bits.iter()).any(|bit| self.0.get(bit).is_some_and(Option::is_some)) {
            return Cow::Borrowed(self);
        }
        Cow::Owned(Self(
            (self.0.iter())
                .map(|(&bit, &asked)| (bit, asked.filter(|_| !bits.contains(&bit))))
                .collect(),
        ))
    }

    pub(crate) fn with_dimensions(
        &self,
        read_model: &(impl ReadModel + ?Sized),
        bits: &[u32],
    ) -> Cow<'_, Self> {
        let added = || bits.iter().filter(|bit| is_dimension(read_model, **bit));
        if added().all(|bit| self.0.contains_key(bit)) {
            return Cow::Borrowed(self);
        }
        let mut held = self.0.clone();
        for &bit in added() {
            held.entry(bit).or_insert(None);
        }
        Cow::Owned(Self(held))
    }

    pub(crate) fn selects_without_granularity(&self, source: &SourceGrain) -> bool {
        (self.0.iter())
            .any(|(bit, granularity)| source.axis.contains(*bit) && granularity.is_none())
    }

    fn selected(&self) -> SemanticBitmask {
        SemanticBitmask::from_positions(self.0.keys().copied())
    }

    fn on<'a>(&'a self, source: &'a SourceGrain) -> impl Iterator<Item = Granularity> + 'a {
        (self.0.iter())
            .filter(|(bit, _)| source.axis.contains(**bit))
            .filter_map(|(_, granularity)| *granularity)
    }
}

pub(crate) struct SourceGrain {
    mask: SemanticBitmask,
    xs: Vec<SemanticBitmask>,
    dimensions: SemanticBitmask,
    axis: SemanticBitmask,
    granularity: Option<Granularity>,
    held: Option<Granularity>,
}

fn held(
    read_model: &(impl ReadModel + ?Sized),
    source: &SourceAlias,
    reads: &[Frame],
    axis: &SemanticBitmask,
    granularity: Option<Granularity>,
) -> Option<Granularity> {
    let granularity = granularity?;
    let reads: Vec<&Frame> = reads.iter().collect();
    (axis.set_positions().into_iter())
        .all(|bit| gives_at(read_model, source, &reads, bit, granularity))
        .then_some(granularity)
}

fn gives_at(
    read_model: &(impl ReadModel + ?Sized),
    unit: &SourceAlias,
    reads: &[&Frame],
    bit: u32,
    granularity: Granularity,
) -> bool {
    if !read_model.mask(unit).contains(bit) {
        return true;
    }
    if is_dataset(read_model, unit) {
        let own =
            (read_model.time_axis(unit)).map_or_else(SemanticBitmask::empty, TimeAxis::dimensions);
        return own.contains(bit) && read_model.granularity(unit) == Some(granularity);
    }
    let through = |read: &Frame| {
        let below = read.frames();
        gives_at(read_model, &read.input.alias, &below, bit, granularity)
    };
    let at = |operand: &SourceAlias| {
        let below = (reads.iter())
            .find(|read| &read.input.alias == operand)
            .map_or_else(Vec::new, |read| read.frames());
        gives_at(read_model, operand, &below, bit, granularity)
    };
    if is_grainset(read_model, unit) && !reads.is_empty() {
        return reads.iter().all(|read| through(read));
    }
    let Some(relationship) = read_model.relationship(unit) else {
        return read_model.operands(unit).iter().all(at);
    };
    if let [read] = reads {
        return through(read);
    }
    let equated = (read_model.name_of(bit))
        .is_some_and(|name| equates(read_model, &relationship, name.as_str()));
    match (relationship.ends()).map(|(end, kept)| (equated && kept, at(end))) {
        [(false, from), (false, to)] => from && to,
        [(kept_from, from), (kept_to, to)] => (kept_from && from) || (kept_to && to),
    }
}

fn equates(
    read_model: &(impl ReadModel + ?Sized),
    relationship: &Relationship,
    name: &str,
) -> bool {
    let [(from, _), (to, _)] = relationship.ends();
    let compares = |left: &NamedReference, right: &NamedReference| {
        left.name() == name
            && right.name() == name
            && reads_value(read_model, from, left)
            && reads_value(read_model, to, right)
    };
    (relationship.on.iter())
        .flat_map(conjuncts)
        .any(|conjunct| match conjunct {
            Expr::BinaryOp {
                op: BinaryOpKind::Eq,
                left,
                right,
            } => match (left.as_ref(), right.as_ref()) {
                (Expr::Column(left), Expr::Column(right)) => {
                    compares(left, right) || compares(right, left)
                }
                _ => false,
            },
            _ => false,
        })
}

fn reads_value(
    read_model: &(impl ReadModel + ?Sized),
    unit: &SourceAlias,
    leaf: &NamedReference,
) -> bool {
    if qualifies(unit, leaf) {
        return true;
    }
    let (Some(bit), Some(relationship)) = (
        read_model.bit_of(leaf.name()),
        read_model.relationship(unit),
    ) else {
        return false;
    };
    let reads = |end: &SourceAlias| reads_value(read_model, end, leaf);
    let ends =
        (relationship.ends()).map(|(end, kept)| (end, kept, read_model.mask(end).contains(bit)));
    match ends {
        [(from, _, true), (_, _, false)] => reads(from),
        [(_, _, false), (to, _, true)] => reads(to),
        [(from, kept_from, true), (to, kept_to, true)] => {
            equates(read_model, &relationship, leaf.name())
                && ((kept_from && reads(from)) || (kept_to && reads(to)))
        }
        _ => false,
    }
}

fn conjuncts(tree: &PhysicalExpr) -> Vec<&PhysicalExpr> {
    match tree {
        Expr::BinaryOp {
            op: BinaryOpKind::And,
            left,
            right,
        } => conjuncts(left)
            .into_iter()
            .chain(conjuncts(right))
            .collect(),
        _ => vec![tree],
    }
}

pub(crate) fn coarseness_of(granularity: Option<Granularity>) -> u8 {
    granularity.map_or(u8::MAX, Granularity::coarseness)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GranularityFit {
    pub(crate) group: GranularityGroup,
    pub(crate) granularity: Option<Granularity>,
}

impl GranularityFit {
    fn compare<K: Ord>(&self, other: &Self, key: impl Fn(Option<Granularity>) -> K) -> Ordering {
        let finest_first = key(self.granularity).cmp(&key(other.granularity));
        self.group.cmp(&other.group).then(match self.group {
            GranularityGroup::Selected => finest_first,
            _ => finest_first.reverse(),
        })
    }
}

impl Ord for GranularityFit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.compare(other, |granularity| {
            granularity.map(Granularity::coarseness)
        })
    }
}

impl PartialOrd for GranularityFit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChildFit {
    dimensions: u32,
    fit: GranularityFit,
    fit_first: bool,
}

impl Ord for ChildFit {
    fn cmp(&self, other: &Self) -> Ordering {
        let dimensions = self.dimensions.cmp(&other.dimensions);
        let fit = self.fit.compare(&other.fit, coarseness_of);
        if self.fit_first {
            fit.then(dimensions)
        } else {
            dimensions.then(fit)
        }
    }
}

impl PartialOrd for ChildFit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Served {
    AtItsGrain,
    Reaggregated,
}

impl SourceGrain {
    pub(crate) fn of(
        read_model: &(impl ReadModel + ?Sized),
        source: &SourceAlias,
        mask: &SemanticBitmask,
    ) -> Self {
        Self::with_reads(read_model, source, mask, &[])
    }

    pub(crate) fn of_decided(
        read_model: &(impl ReadModel + ?Sized),
        source: &SourceAlias,
        reads: &[Frame],
    ) -> Self {
        Self::with_reads(read_model, source, &read_model.mask(source), reads)
    }

    fn with_reads(
        read_model: &(impl ReadModel + ?Sized),
        source: &SourceAlias,
        mask: &SemanticBitmask,
        reads: &[Frame],
    ) -> Self {
        let dimensions = SemanticBitmask::from_positions(
            (mask.set_positions().into_iter()).filter(|bit| is_dimension(read_model, *bit)),
        );
        let keysets = read_model.keysets(source);
        let axis = (read_model.time_axis(source))
            .map_or_else(SemanticBitmask::empty, TimeAxis::dimensions);
        let granularity = read_model.granularity(source);
        Self {
            mask: mask.clone(),
            xs: if keysets.is_empty() {
                vec![dimensions.clone()]
            } else {
                keysets
            },
            dimensions,
            held: held(read_model, source, reads, &axis, granularity),
            axis,
            granularity,
        }
    }

    pub(crate) fn of_source(read_model: &(impl ReadModel + ?Sized), source: &SourceAlias) -> Self {
        Self::of(read_model, source, &read_model.mask(source))
    }

    pub(crate) fn of_demand(read_model: &(impl ReadModel + ?Sized), input: &StrategyInput) -> Self {
        let grain = Self::of_source(read_model, &input.alias);
        if input.formula_grain {
            Self {
                granularity: None,
                held: None,
                ..grain
            }
        } else {
            grain
        }
    }

    pub(crate) fn fixed(
        &self,
        bit: u32,
        asked: Option<Granularity>,
        granularity: Granularity,
    ) -> Option<Granularity> {
        asked.or_else(|| self.axis.contains(bit).then_some(granularity))
    }

    pub(crate) fn granularity(&self) -> Option<Granularity> {
        self.granularity
    }

    pub(crate) fn asked(&self, request: &RequestGrain) -> Option<Granularity> {
        request.on(self).next()
    }

    pub(crate) fn unserved(&self, request: &RequestGrain) -> Option<Granularity> {
        let carried = self.granularity?;
        request.on(self).find(|asked| !carried.serves(*asked))
    }

    pub(crate) fn reads_axis(&self, names: &SemanticBitmask) -> bool {
        names.intersection(&self.axis).count_ones() > 0
    }

    pub(crate) fn has_axis(&self) -> bool {
        self.axis.count_ones() > 0
    }

    pub(crate) fn child_fit(&self, request: &RequestGrain, fit_first: bool) -> ChildFit {
        ChildFit {
            dimensions: self.dimensions.count_ones(),
            fit: self.granularity_fit(request),
            fit_first,
        }
    }

    pub(crate) fn granularity_fit(&self, request: &RequestGrain) -> GranularityFit {
        let group = if request.on(self).next().is_some() {
            GranularityGroup::Requested
        } else if self.reads_axis(&request.selected()) {
            GranularityGroup::Selected
        } else {
            GranularityGroup::NotSelected
        };
        GranularityFit {
            group,
            granularity: self.granularity,
        }
    }

    pub(crate) fn grain_test(&self, request: &RequestGrain) -> Option<Served> {
        let selected = request.selected();
        let serves = selected.is_subset_of(&self.dimensions) && self.unserved(request).is_none();
        let at_its_grain = (self.xs.iter()).any(|x| {
            (x.set_positions().into_iter()).all(|bit| self.kept(request.0.get(&bit).copied(), bit))
        });
        serves.then_some(if at_its_grain {
            Served::AtItsGrain
        } else {
            Served::Reaggregated
        })
    }

    pub(crate) fn candidate(
        &self,
        demanded: &SemanticBitmask,
        request: &RequestGrain,
    ) -> Option<Served> {
        demanded
            .is_subset_of(&self.mask)
            .then(|| self.grain_test(request))
            .flatten()
    }

    pub(crate) fn kept(&self, asked: Option<Option<Granularity>>, bit: u32) -> bool {
        match asked {
            Some(None) => true,
            Some(Some(asked)) => {
                self.axis.contains(bit)
                    && (self.held).is_some_and(|held| asked.coarseness() <= held.coarseness())
            }
            None => false,
        }
    }
}

pub(crate) fn lossy_reaggregation<'b>(
    read_model: &impl ReadModel,
    path: &FramePath,
    alias: &SourceAlias,
    served: Served,
    bindings: impl IntoIterator<Item = &'b BoundName>,
) -> Result<Vec<Advisory>, PlanError> {
    if served != Served::Reaggregated {
        return Ok(Vec::new());
    }
    let closure = read_model.operand_closure(alias);
    let aggregated: Vec<&SourceAlias> = (iter::once(alias).chain(&closure))
        .filter(|unit| {
            is_dataset(read_model, unit)
                && matches!(
                    read_model.time_axis(unit),
                    Some(TimeAxis::Timeseries(_) | TimeAxis::Snapshot(_))
                )
        })
        .collect();
    if aggregated.is_empty() {
        return Ok(Vec::new());
    }
    let Some(source) = path.0.last() else {
        return Ok(Vec::new());
    };
    let lossy = |definition: &SemanticDefinition| {
        additivity(&definition.role_definition) == Some(Additivity::NonAdditive)
            && (read_model.bit_of(definition.name.as_str())).is_some_and(|bit| {
                (aggregated.iter()).any(|unit| read_model.signature(unit, bit).is_some())
            })
    };
    let mut non_additive = BTreeSet::new();
    for binding in bindings {
        non_additive.extend(
            (aggregate_definitions(read_model, path, source, alias, binding, &[])?.into_iter())
                .filter(|definition| lossy(definition))
                .map(|definition| definition.name),
        );
    }
    Ok((non_additive.into_iter())
        .map(|semantics| Advisory::LossyReaggregation {
            path: path.clone(),
            semantics,
        })
        .collect())
}

pub(crate) fn additivity(role: &SemanticRoleDefinition) -> Option<Additivity> {
    let (function, distinct) = role.aggregation()?;
    let own = call_additivity(function, distinct);
    Some(match (own, role.additivity()) {
        (Additivity::NonAdditive, _) | (_, Some(AdditivityType::Non)) => Additivity::NonAdditive,
        (_, Some(AdditivityType::Semi(_))) => Additivity::SemiAdditive,
        (own, _) => own,
    })
}

pub(crate) fn call_additivity(function: &AggregateFunction, distinct: bool) -> Additivity {
    if distinct {
        function.distinct_additivity
    } else {
        function.additivity
    }
}

pub(crate) fn is_dimension(read_model: &(impl ReadModel + ?Sized), bit: u32) -> bool {
    (read_model.definition(bit)).is_some_and(|definition| {
        matches!(
            definition.role_definition,
            SemanticRoleDefinition::Dimension { .. }
        )
    })
}
