use std::collections::{BTreeMap, BTreeSet};

use semstrait_common::SemanticsName;
use semstrait_common::{Tree, ValidateError};
use semstrait_ir::{AdditivityType, IrErrorKind, PhysicalExpr, Schema, SemanticRole};
use semstrait_manifest::{
    join_filter_onto_aggregates, KeptField, KeptLeaf, PhysicalExprId, SemanticDefinition,
    SemanticRoleDefinition, SourceAlias,
};
use semstrait_sql::datatype::{DataType, Decimal, TypeFamily};
use semstrait_sql::expr::{
    BinaryOpKind, Call, CastFailure, Expr, ExprOperationExt, Literal, NamedReference, UnaryOpKind,
};
use semstrait_sql::functions::{AggregateFunction, Volatility};

use crate::error::call_refused;
use crate::fan_out::reads_rows_as_they_are;
use crate::frame::Frame;
use crate::parts::{part_key, summed};
use crate::read_model::{is_grainset, is_unionset, ReadModel};
use crate::steps::{
    catalog_call, condition_field, qualified, reference, reserved, unbucketed, Aggregate,
};
use crate::strategy::StrategyInput;
use crate::tree::{aggregate_calls, column_leaves, replace_columns};
use crate::{BoundName, PlanError};

#[derive(Debug, Default)]
pub(crate) struct Grouping {
    kept: Vec<KeptField>,
    grouped: BTreeSet<u32>,
}

impl Grouping {
    pub(crate) fn of(read_model: &dyn ReadModel, input: &StrategyInput) -> Result<Self, PlanError> {
        let mut grouping = Self::default();
        for binding in &input.names {
            let definition = input.definition(read_model, binding)?;
            if input.requested_granularity.contains_key(&binding.name)
                || planned(read_model, input, binding, &definition)? != SemanticRole::Dimension
            {
                continue;
            }
            let root = input.signature(read_model, binding)?.expression_id;
            let field = reference(binding.name.as_str()).map_err(|error| input.refused(error))?;
            grouping.kept.push(KeptField {
                leaf: KeptLeaf::Link(root),
                field,
            });
            grouping.grouped.insert(binding.bit);
        }
        Ok(grouping)
    }

    pub(crate) fn grouped(&self) -> &BTreeSet<u32> {
        &self.grouped
    }

    pub(crate) fn finishes(
        &self,
        read_model: &dyn ReadModel,
        input: &StrategyInput,
        binding: &BoundName,
    ) -> Result<bool, PlanError> {
        let definition = input.definition(read_model, binding)?;
        match (
            planned(read_model, input, binding, &definition)?,
            definition.role_definition.aggregation(),
        ) {
            (SemanticRole::Metric, Some(_)) => Ok(false),
            (SemanticRole::Metric, None) => {
                let linked = input.linked(read_model, binding)?;
                let (_, ungrouped, _) = self.over_final_aggregates(
                    read_model,
                    input,
                    binding,
                    &linked,
                    Naming::default(),
                )?;
                Ok(ungrouped.is_none())
            }
            _ => Ok(true),
        }
    }

    fn over_final_aggregates(
        &self,
        read_model: &dyn ReadModel,
        input: &StrategyInput,
        binding: &BoundName,
        linked: &[SemanticDefinition],
        naming: Naming<'_>,
    ) -> Result<(PhysicalExpr, Option<SemanticsName>, Vec<Linked>), PlanError> {
        let mut kept = self.kept.clone();
        for (_, root, field) in self.ungrouped(read_model, input, binding, linked)? {
            kept.push(KeptField {
                leaf: KeptLeaf::Link(root),
                field,
            });
        }
        let mut links = Vec::new();
        let tree = linked_tree(read_model, input, binding, &kept, naming, &mut links)?;
        let ungrouped = (ungrouped_columns(&tree, &self.kept, &links).first())
            .map(|column| column.name().into());
        Ok((tree, ungrouped, links))
    }

    fn at_formula_grain(
        &self,
        read_model: &dyn ReadModel,
        input: &StrategyInput,
        binding: &BoundName,
        naming: Naming<'_>,
    ) -> Result<(PhysicalExpr, ComputedFields, Vec<Linked>), PlanError> {
        let refused = |error| input.refused(error);
        let linked = input.linked(read_model, binding)?;
        let ungrouped = self.ungrouped(read_model, input, binding, &linked)?;
        let fields = (ungrouped.into_iter())
            .map(|(linked, root, _)| Ok((root, unbucketed(naming.qualifier, &linked.name)?)))
            .collect::<Result<Vec<_>, ValidateError>>()
            .map_err(refused)?;
        let (mut rows, held) = kept_fields(read_model, input, binding, fields)?;
        let mut kept = (self.kept.iter())
            .map(|grouped| {
                Ok(KeptField {
                    field: qualified(naming.qualifier, grouped.field.name())?,
                    ..grouped.clone()
                })
            })
            .collect::<Result<Vec<_>, ValidateError>>()
            .map_err(refused)?;
        kept.extend(held);
        let read = |kept: &[_], links: &mut Vec<_>| {
            linked_tree(read_model, input, binding, kept, naming, links)
        };
        let mut probe = Vec::new();
        let tree = read(&kept, &mut probe)?;
        let columns: Vec<_> = (ungrouped_columns(&tree, &kept, &probe).into_iter())
            .cloned()
            .collect();
        for column in columns {
            let name = reserved("col", column.name());
            let field = qualified(naming.qualifier, name).map_err(refused)?;
            if rows.iter().all(|(held, _)| held != &field) {
                rows.push((field.clone(), Expr::Column(column.clone())));
                kept.push(KeptField {
                    leaf: KeptLeaf::Column(column),
                    field,
                });
            }
        }
        let mut links = Vec::new();
        let tree = read(&kept, &mut links)?;
        let grain = (rows.into_iter())
            .filter(|(field, _)| column_leaves(&tree).any(|column| column == field))
            .collect();
        Ok((tree, grain, links))
    }

    pub(crate) fn steps_at(
        &self,
        read_model: &dyn ReadModel,
        input: &StrategyInput,
        binding: &BoundName,
    ) -> Result<bool, PlanError> {
        let definition = input.definition(read_model, binding)?;
        if definition.role_definition.aggregation().is_none() {
            return Ok(false);
        }
        let linked = input.linked(read_model, binding)?;
        let (_, ungrouped, links) =
            self.over_final_aggregates(read_model, input, binding, &linked, Naming::default())?;
        Ok(ungrouped.is_none() && links.is_empty())
    }

    pub(crate) fn ungrouped<'a>(
        &self,
        read_model: &dyn ReadModel,
        input: &StrategyInput,
        binding: &BoundName,
        linked: &'a [SemanticDefinition],
    ) -> Result<Vec<(&'a SemanticDefinition, PhysicalExprId, NamedReference)>, PlanError> {
        let mut dimensions = dimensions(read_model, input, binding, linked)?;
        dimensions.retain(|(linked, ..)| !self.grouped.contains(&linked.bit_position));
        Ok(dimensions)
    }
}

fn dimensions<'a>(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    binding: &BoundName,
    linked: &'a [SemanticDefinition],
) -> Result<Vec<(&'a SemanticDefinition, PhysicalExprId, NamedReference)>, PlanError> {
    let source = &input.alias;
    (linked.iter())
        .filter(|linked| planned_as(read_model, source, linked) == Some(SemanticRole::Dimension))
        .map(|linked| {
            let root = (read_model.signature(source, linked.bit_position))
                .ok_or_else(|| input.broken_tree(binding))?
                .expression_id;
            let field = reference(linked.name.as_str()).map_err(|error| input.refused(error))?;
            Ok((linked, root, field))
        })
        .collect()
}

fn kept_fields(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    binding: &BoundName,
    fields: impl IntoIterator<Item = (PhysicalExprId, NamedReference)>,
) -> Result<(ComputedFields, Vec<KeptField>), PlanError> {
    let (mut rows, mut kept) = (Vec::new(), Vec::new());
    for (root, field) in fields {
        rows.push((field.clone(), input.tree(read_model, binding, &root, &[])?));
        kept.push(KeptField {
            leaf: KeptLeaf::Link(root),
            field,
        });
    }
    Ok((rows, kept))
}

fn ungrouped_columns<'a>(
    node: &'a PhysicalExpr,
    kept: &[KeptField],
    links: &[Linked],
) -> Vec<&'a NamedReference> {
    match node {
        Expr::Function(Call::Aggregate(..)) => Vec::new(),
        Expr::Column(column) => {
            let held = kept.iter().any(|kept| &kept.field == column) || is_linked(links, column);
            (!held).then_some(column).into_iter().collect()
        }
        node => (node.children().into_iter())
            .flat_map(|child| ungrouped_columns(child, kept, links))
            .collect(),
    }
}

pub(crate) fn is_linked(links: &[Linked], column: &NamedReference) -> bool {
    links.iter().any(|link| &link.field == column)
}

fn linked_tree(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    binding: &BoundName,
    kept: &[KeptField],
    naming: Naming<'_>,
    links: &mut Vec<Linked>,
) -> Result<PhysicalExpr, PlanError> {
    let (source, refused) = (&input.alias, |error| input.refused(error));
    let (mut held, mut keeps) = (kept.to_vec(), Vec::new());
    for linked in input.linked(read_model, binding)? {
        let signature = (read_model.signature(source, linked.bit_position))
            .ok_or_else(|| input.broken_tree(binding))?;
        let filtered = signature.filter_expr_id.is_some()
            && (read_model.linked(source, &signature.expression_id))
                .ok_or_else(|| input.broken_tree(binding))?
                .iter()
                .any(|linked| aggregates_a_formula(read_model, source, linked));
        if filtered || aggregates_a_formula(read_model, source, &linked) {
            let placeholder = reference(linked.name.as_str()).map_err(refused)?;
            held.push(KeptField {
                leaf: KeptLeaf::Link(signature.expression_id),
                field: placeholder.clone(),
            });
            keeps.push((placeholder, linked));
        }
    }
    let root = input.signature(read_model, binding)?.expression_id;
    let tree = input.tree(read_model, binding, &root, &held)?;
    let (start, mut in_place) = (links.len(), Vec::<(NamedReference, PhysicalExpr)>::new());
    for column in column_leaves(&tree) {
        let Some((_, linked)) = (keeps.iter()).find(|(placeholder, _)| placeholder == column)
        else {
            continue;
        };
        if in_place.iter().any(|(read, _)| read == column) {
            continue;
        }
        let bound = BoundName {
            name: linked.name.clone(),
            bit: linked.bit_position,
        };
        let read = if aggregates_a_formula(read_model, source, linked) {
            let n = naming.numbered + links.len() + 1;
            let name = reserved(format_args!("linked{n}"), &linked.name);
            let field = qualified(naming.qualifier, name).map_err(refused)?;
            links.push(Linked {
                binding: bound,
                filters: Vec::new(),
                field: field.clone(),
            });
            Expr::Column(field)
        } else {
            linked_tree(read_model, input, &bound, kept, naming, links)?
        };
        in_place.push((column.clone(), read));
    }
    let tree = replace_columns(tree, &in_place).map_err(refused)?;
    let filters = input.filters_on(read_model, binding)?;
    for link in links.iter_mut().skip(start) {
        link.filters.extend(filters.clone());
    }
    (filters.iter()).try_fold(tree, |tree, filter| {
        join_filter_onto_aggregates(&tree, filter)
            .map_err(|error| input.refused(call_refused(error)))
    })
}

fn aggregates_a_formula(
    read_model: &dyn ReadModel,
    source: &SourceAlias,
    definition: &SemanticDefinition,
) -> bool {
    definition.role_definition.aggregation().is_some()
        && planned_as(read_model, source, definition) == Some(SemanticRole::Metric)
}

#[derive(Clone, Copy, Default)]
pub(crate) struct Naming<'q> {
    pub(crate) qualifier: Option<&'q str>,
    pub(crate) numbered: usize,
}

pub(crate) struct Linked {
    pub(crate) binding: BoundName,
    pub(crate) filters: Vec<PhysicalExpr>,
    pub(crate) field: NamedReference,
}

impl Linked {
    pub(crate) fn stem(
        &self,
        read_model: &dyn ReadModel,
        input: &StrategyInput,
    ) -> Result<Option<String>, PlanError> {
        if self.filters.is_empty() {
            return Ok(None);
        }
        let held = readers_filters(read_model, input)?;
        let mut numbers = (self.filters.iter())
            .map(|filter| held.iter().position(|held| held == filter))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| input.broken_tree(&self.binding))?;
        numbers.sort_unstable();
        numbers.dedup();
        let numbers: Vec<_> = numbers.iter().map(usize::to_string).collect();
        Ok(Some(format!("under{}", numbers.join("_"))))
    }

    pub(crate) fn through(&mut self, reader: &Linked) {
        self.filters.extend(reader.filters.iter().cloned());
    }
}

fn readers_filters(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
) -> Result<Vec<PhysicalExpr>, PlanError> {
    let (source, mut readers) = (&input.alias, Vec::new());
    for binding in input.names.iter().chain(&input.leaves(read_model)?) {
        let definition = input.definition(read_model, binding)?;
        if planned_as(read_model, source, &definition) != Some(SemanticRole::Metric) {
            continue;
        }
        let linked = input.linked(read_model, binding)?;
        for reader in std::iter::once(definition).chain(linked) {
            if planned_as(read_model, source, &reader) == Some(SemanticRole::Metric) {
                readers.push(BoundName::of(reader));
            }
        }
    }
    let mut held = Vec::new();
    for reader in &readers {
        held.extend(input.own_filter(read_model, reader)?);
    }
    for reader in &readers {
        if let Some(condition) = input.conditions.get(&reader.name) {
            held.push(input.lowered(read_model, condition)?);
        }
    }
    for condition in input
        .formula_nodes
        .iter()
        .filter_map(|node| node.condition.as_ref())
    {
        held.push(input.lowered(read_model, condition)?);
    }
    let mut filters = Vec::new();
    for filter in held {
        if !filters.contains(&filter) {
            filters.push(filter);
        }
    }
    Ok(filters)
}

pub(crate) enum Finished {
    Step(PhysicalExpr),
    Joined(PhysicalExpr, Vec<Linked>),
    TwoSteps(TwoSteps),
}

impl Finished {
    pub(crate) fn computed(&mut self) -> &mut PhysicalExpr {
        match self {
            Self::Step(tree) | Self::Joined(tree, _) => tree,
            Self::TwoSteps(two_steps) => &mut two_steps.formula,
        }
    }

    pub(crate) fn linked(&self) -> &[Linked] {
        match self {
            Self::Joined(_, linked) | Self::TwoSteps(TwoSteps { linked, .. }) => linked,
            Self::Step(_) => &[],
        }
    }
}

pub(crate) type ComputedFields = Vec<(NamedReference, PhysicalExpr)>;

pub(crate) struct TwoSteps {
    pub(crate) grain: ComputedFields,
    pub(crate) key: ComputedFields,
    pub(crate) dimensions: ComputedFields,
    pub(crate) formula: PhysicalExpr,
    pub(crate) aggregation: (&'static AggregateFunction, bool),
    pub(crate) value_type: DataType,
    pub(crate) on_failure: CastFailure,
    pub(crate) filters: Vec<PhysicalExpr>,
    pub(crate) stem: Option<String>,
    pub(crate) linked: Vec<Linked>,
}

pub(crate) fn finished(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    operands: &[Frame],
    grouping: &Grouping,
    schema: &Schema,
    binding: &BoundName,
    naming: Naming<'_>,
) -> Result<Finished, PlanError> {
    let definition = input.definition(read_model, binding)?;
    let value_type = || (definition.data_type.clone()).ok_or_else(|| input.broken_tree(binding));
    match (
        planned(read_model, input, binding, &definition)?,
        definition.role_definition.aggregation(),
    ) {
        (SemanticRole::Measure, _) => input
            .own_tree(read_model, binding, &grouping.kept)
            .map(Finished::Step),
        (SemanticRole::Metric, Some(aggregation)) => {
            let (formula, grain, linked) =
                grouping.at_formula_grain(read_model, input, binding, naming)?;
            Ok(Finished::TwoSteps(TwoSteps {
                grain,
                key: Vec::new(),
                dimensions: Vec::new(),
                formula,
                aggregation,
                value_type: value_type()?,
                on_failure: CastFailure::Error,
                filters: input.filters_on(read_model, binding)?,
                stem: None,
                linked,
            }))
        }
        (SemanticRole::Metric, None) => {
            let linked = input.linked(read_model, binding)?;
            let (tree, ungrouped, links) =
                grouping.over_final_aggregates(read_model, input, binding, &linked, naming)?;
            let Some(name) = ungrouped else {
                return Ok(if links.is_empty() {
                    Finished::Step(tree)
                } else {
                    Finished::Joined(tree, links)
                });
            };
            let tree = input.own_tree(read_model, binding, &[])?;
            let end =
                (operands.iter()).find(|end| end.input.pre_aggregated.contains_key(&binding.name));
            if let Some(end) = end {
                let partial = qualified(Some(end.input.alias.as_str()), binding.name.as_str())
                    .map_err(|error| input.refused(error))?;
                if schema.position(&partial).is_some() {
                    let sum = catalog_call("sum", vec![Expr::Column(partial)]);
                    let sum = sum.map_err(|error| input.refused(error))?;
                    let summed = schema.type_of(&sum).map_err(|e| input.node_refused(e))?;
                    let (_, on_failure) = below_root_cast(&tree);
                    let to = value_type()?;
                    let sum = cast_where_differs(sum, &summed.data_type, &to, on_failure);
                    return Ok(Finished::Step(sum));
                }
            }
            if pushes_down(read_model, input, binding)? {
                if let Some(pushed) = push_down(read_model, input, operands, &tree, schema)? {
                    return Ok(Finished::Step(pushed));
                }
            }
            let value_type = value_type()?;
            if value_type.family() != Some(TypeFamily::Number) {
                return Err(PlanError::UngroupedMetricFormula {
                    path: input.path.clone(),
                    metric: binding.name.clone(),
                    name,
                });
            }
            parts(read_model, input, binding, &linked, value_type, naming).map(Finished::TwoSteps)
        }
        _ => Err(input.broken_tree(binding)),
    }
}

fn parts(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    binding: &BoundName,
    linked: &[SemanticDefinition],
    value_type: DataType,
    naming: Naming<'_>,
) -> Result<TwoSteps, PlanError> {
    let fields = dimensions(read_model, input, binding, linked)?.into_iter();
    let fields = fields.map(|(_, root, field)| (root, field));
    let (rows, kept) = kept_fields(read_model, input, binding, fields)?;
    let mut links = Vec::new();
    let tree = linked_tree(read_model, input, binding, &kept, naming, &mut links)?;
    let (body, on_failure) = below_root_cast(&tree);
    let linked = |column: &NamedReference| is_linked(&links, column);
    let key = part_key(body, &linked, &binding.name, naming.qualifier)
        .map_err(|error| input.refused(error))?;
    let aggregation = summed().map_err(|error| input.refused(error))?;
    Ok(TwoSteps {
        grain: Vec::new(),
        key: key.columns,
        dimensions: rows,
        formula: key.per_part,
        aggregation,
        value_type,
        on_failure,
        filters: input.filters_on(read_model, binding)?,
        stem: None,
        linked: links,
    })
}

pub(crate) fn below_root_cast(tree: &PhysicalExpr) -> (&PhysicalExpr, CastFailure) {
    match tree {
        Expr::Cast {
            input, on_failure, ..
        } => (input.as_ref(), *on_failure),
        tree => (tree, CastFailure::Error),
    }
}

pub(crate) fn cast_where_differs(
    value: PhysicalExpr,
    from: &DataType,
    to: &DataType,
    on_failure: CastFailure,
) -> PhysicalExpr {
    if from == to {
        return value;
    }
    Expr::Cast {
        input: Box::new(value),
        target: to.clone(),
        on_failure,
    }
}

pub(crate) fn planned(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    binding: &BoundName,
    definition: &SemanticDefinition,
) -> Result<SemanticRole, PlanError> {
    planned_as(read_model, &input.alias, definition).ok_or_else(|| input.broken_tree(binding))
}

pub(crate) fn planned_as(
    read_model: &dyn ReadModel,
    source: &SourceAlias,
    definition: &SemanticDefinition,
) -> Option<SemanticRole> {
    let SemanticRoleDefinition::Metric { aggregation, .. } = definition.role_definition else {
        return Some(definition.role());
    };
    let root = read_model
        .signature(source, definition.bit_position)?
        .expression_id;
    if is_grainset(read_model, source) {
        return Some(SemanticRole::Metric);
    }
    let row_level = (read_model.linked(source, &root)?)
        .iter()
        .all(|linked| linked.role_definition.aggregation().is_none());
    Some(match (row_level, aggregation) {
        (false, _) => SemanticRole::Metric,
        (true, Some(_)) => SemanticRole::Measure,
        (true, None) => SemanticRole::Dimension,
    })
}

pub(crate) fn links_match(
    read_model: &dyn ReadModel,
    source: &SourceAlias,
    linked: &[SemanticDefinition],
) -> bool {
    (linked.iter()).all(|linked| {
        planned_as(read_model, source, linked).is_some_and(|role| role != SemanticRole::Metric)
            && !matches!(
                linked.role_definition.additivity(),
                Some(AdditivityType::Non | AdditivityType::Semi(_))
            )
    })
}

pub(crate) fn push_down(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    operands: &[Frame],
    tree: &PhysicalExpr,
    schema: &Schema,
) -> Result<Option<PhysicalExpr>, PlanError> {
    let (body, on_failure) = below_root_cast(tree);
    let Some((term, filter)) = pattern(input, body, Some(schema), &gives_field(schema))? else {
        return Ok(None);
    };
    if !reads_joined_rows(read_model, input, operands, body)? {
        return Ok(None);
    }
    let sum = catalog_call("sum", vec![term]).map_err(|error| input.refused(error))?;
    let sum = match filter {
        Some(filter) => join_filter_onto_aggregates(&sum, filter)
            .map_err(|error| input.refused(call_refused(error)))?,
        None => sum,
    };
    let type_of = |node: &PhysicalExpr| schema.type_of(node).map_err(|e| input.node_refused(e));
    let (summed, typed) = (type_of(&sum)?.data_type, type_of(tree)?.data_type);
    Ok(Some(cast_where_differs(sum, &summed, &typed, on_failure)))
}

pub(crate) fn reads_joined_rows(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    operands: &[Frame],
    tree: &PhysicalExpr,
) -> Result<bool, PlanError> {
    let calls: Vec<_> = aggregate_calls(tree).collect();
    for argument in calls.iter().flat_map(|call| call.args()) {
        if !reads_rows_as_they_are(read_model, input, operands, argument)? {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(crate) fn pushes_down(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    binding: &BoundName,
) -> Result<bool, PlanError> {
    let definition = input.definition(read_model, binding)?;
    if planned(read_model, input, binding, &definition)? != SemanticRole::Metric
        || definition.role_definition.aggregation().is_some()
        || !links_match(
            read_model,
            &input.alias,
            &input.linked(read_model, binding)?,
        )
        || conditions_differ(read_model, input, binding)?
    {
        return Ok(false);
    }
    let tree = input.own_tree(read_model, binding, &[])?;
    let (body, _) = below_root_cast(&tree);
    Ok(pattern(input, body, None, &|_| false)?.is_some())
}

fn conditions_differ(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    binding: &BoundName,
) -> Result<bool, PlanError> {
    if !is_unionset(read_model, &input.alias) {
        return Ok(false);
    }
    let mut measures = BTreeSet::new();
    for column in input.columns_of(read_model, binding)? {
        let read = input.bound(read_model, column.name())?;
        let definition = input.definition(read_model, &read)?;
        if planned(read_model, input, &read, &definition)? == SemanticRole::Measure {
            measures.insert(read.name);
        }
    }
    Ok(measures.len() > 1)
}

pub(crate) fn gives_field(schema: &Schema) -> impl Fn(&str) -> bool + '_ {
    |field| (schema.columns.iter()).any(|column| column.name == field)
}

fn pattern<'b>(
    input: &StrategyInput,
    body: &'b PhysicalExpr,
    schema: Option<&Schema>,
    gives: &dyn Fn(&str) -> bool,
) -> Result<Option<(PhysicalExpr, Option<&'b PhysicalExpr>)>, PlanError> {
    let Some(term) = per_row(body, schema).map_err(|error| input.node_refused(error))? else {
        return Ok(None);
    };
    let calls: Vec<_> = aggregate_calls(body).collect();
    let filters = (calls.iter())
        .map(|call| Ok((call.filter(), operand_conditions(call, gives)?)))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| input.refused(error))?;
    let first = filters.first();
    let filter = first.and_then(|(filter, _)| *filter);
    let filtered = first.is_some_and(|(filter, conditions)| {
        filter.is_some()
            || conditions
                .as_ref()
                .is_none_or(|conditions| !conditions.is_empty())
    });
    let counts = (calls.iter()).any(|call| call.function().declarations.canonical.name == "count");
    if filters.iter().any(|each| Some(each) != first) || (filtered && (counts || adds_zero(body))) {
        return Ok(None);
    }
    Ok(Some((term, filter)))
}

pub(crate) fn operand_conditions(
    call: &Aggregate,
    gives: &dyn Fn(&str) -> bool,
) -> Result<Option<Vec<PhysicalExpr>>, ValidateError> {
    let mut operands: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for leaf in call.args().iter().flat_map(column_leaves) {
        if let [operand, name] = leaf.parts() {
            operands
                .entry(name.as_str())
                .or_default()
                .insert(operand.as_str());
        }
    }
    let mut conditions = Vec::new();
    for (name, operands) in operands {
        let mut given = Vec::new();
        for operand in &operands {
            let (field, read) = condition_field(Some(operand), &SemanticsName::new(name))?;
            if gives(field.as_str()) {
                given.push(read);
            }
        }
        match (operands.len(), given.pop()) {
            (_, None) => {}
            (1, Some(condition)) => conditions.push(condition),
            _ => return Ok(None),
        }
    }
    Ok(Some(conditions))
}

fn per_row(
    body: &PhysicalExpr,
    schema: Option<&Schema>,
) -> Result<Option<PhysicalExpr>, IrErrorKind> {
    let Expr::Case { whens, else_ } = body else {
        return term(body, schema);
    };
    let result = |result: &PhysicalExpr| match result {
        Expr::Literal(Literal::Null(_)) => Ok(Some(result.clone())),
        result if zero(result) => Ok(Some(result.clone())),
        result => term(result, schema),
    };
    let whens: Option<Vec<_>> = (whens.iter())
        .map(|(when, then)| {
            if !of_the_row(when) {
                return Ok(None);
            }
            Ok(result(then)?.map(|then| (when.clone(), then)))
        })
        .collect::<Result<_, _>>()?;
    let else_ = match else_.as_deref().map(result).transpose()? {
        Some(None) => return Ok(None),
        else_ => else_.flatten().map(Box::new),
    };
    Ok(whens.map(|whens| Expr::Case { whens, else_ }))
}

fn term(node: &PhysicalExpr, schema: Option<&Schema>) -> Result<Option<PhysicalExpr>, IrErrorKind> {
    let factor = |t: &PhysicalExpr| of_the_row(t) && !volatile(t);
    let per_row = match node {
        Expr::UnaryOp {
            op: UnaryOpKind::Negate,
            operand,
        } => term(operand, schema)?.map(|term| -term),
        Expr::BinaryOp {
            op: BinaryOpKind::Multiply,
            left,
            right,
        } if factor(left) => term(right, schema)?.map(|term| left.as_ref().clone() * term),
        Expr::BinaryOp {
            op: op @ (BinaryOpKind::Multiply | BinaryOpKind::Divide),
            left,
            right,
        } if factor(right) => term(left, schema)?.map(|term| Expr::BinaryOp {
            op: *op,
            left: Box::new(term),
            right: right.clone(),
        }),
        node => return row_argument(node, schema),
    };
    let Some(per_row) = per_row else {
        return Ok(None);
    };
    match schema {
        Some(schema) if !exact(&per_row, schema)? => Ok(None),
        _ => Ok(Some(per_row)),
    }
}

fn row_argument(
    m: &PhysicalExpr,
    schema: Option<&Schema>,
) -> Result<Option<PhysicalExpr>, IrErrorKind> {
    let call = match m {
        Expr::Cast { input, .. } => input.as_ref(),
        m => m,
    };
    let Expr::Function(Call::Aggregate(call, _)) = call else {
        return Ok(None);
    };
    let argument = match (call.function().declarations.canonical.name, call.args()) {
        _ if call.distinct() => return Ok(None),
        ("sum", [x]) => x.clone(),
        ("count", [x]) => Expr::Case {
            whens: vec![(!x.clone().is_null(), Expr::Literal(Literal::BigInt(1)))],
            else_: Some(Box::new(Expr::Literal(Literal::BigInt(0)))),
        },
        _ => return Ok(None),
    };
    let Some(schema) = schema else {
        return Ok(Some(argument));
    };
    let from = schema.type_of(&argument)?.data_type;
    let to = schema.type_of(m)?.data_type;
    if from == to {
        return Ok(Some(argument));
    }
    Ok(exact_cast(&from, &to).then(|| Expr::Cast {
        input: Box::new(argument),
        target: to,
        on_failure: CastFailure::Error,
    }))
}

fn exact(node: &PhysicalExpr, schema: &Schema) -> Result<bool, IrErrorKind> {
    let type_of = |node: &PhysicalExpr| schema.type_of(node).map(|typed| typed.data_type);
    Ok(match (node, type_of(node)?) {
        (_, DataType::Real | DataType::Double) => true,
        (Expr::UnaryOp { .. }, DataType::Decimal { .. }) => true,
        (
            Expr::BinaryOp {
                op: BinaryOpKind::Multiply,
                left,
                right,
            },
            DataType::Decimal { .. },
        ) => matches!(
            (Decimal::of(&type_of(left)?), Decimal::of(&type_of(right)?)),
            (Some(a), Some(b)) if u16::from(a.precision) + u16::from(b.precision) < 38
        ),
        _ => false,
    })
}

fn exact_cast(from: &DataType, to: &DataType) -> bool {
    let integer_digits = |data_type: &DataType| {
        Decimal::of(data_type)
            .map(|decimal| i16::from(decimal.precision) - i16::from(decimal.scale))
    };
    match to {
        DataType::Real | DataType::Double => true,
        DataType::Decimal { .. } => {
            from.widens_to(to)
                && (integer_digits(from).zip(integer_digits(to)))
                    .is_some_and(|(from, to)| to >= from)
        }
        to => from.widens_to(to),
    }
}

fn adds_zero(body: &PhysicalExpr) -> bool {
    matches!(body, Expr::Case { whens, else_ }
        if (whens.iter().map(|(_, then)| then)).chain(else_.as_deref()).any(zero))
}

fn zero(node: &PhysicalExpr) -> bool {
    let Expr::Literal(literal) = node else {
        return false;
    };
    match literal {
        Literal::TinyInt(0) | Literal::SmallInt(0) | Literal::Integer(0) | Literal::BigInt(0) => {
            true
        }
        Literal::Real(value) => *value == 0.0,
        Literal::Double(value) => *value == 0.0,
        Literal::Decimal { value, .. } => value.chars().all(|digit| matches!(digit, '0' | '.')),
        _ => false,
    }
}

pub(crate) fn of_the_row(node: &PhysicalExpr) -> bool {
    !(node.pre_order()).any(|node| matches!(node, Expr::Function(Call::Aggregate(..))))
}

pub(crate) fn volatile(node: &PhysicalExpr) -> bool {
    (node.pre_order()).any(|node| match node {
        Expr::Function(Call::Scalar(call)) => {
            call.function().canonical.volatility == Volatility::Volatile
        }
        Expr::Function(Call::Window(call)) => {
            call.function().canonical.volatility == Volatility::Volatile
        }
        _ => false,
    })
}
