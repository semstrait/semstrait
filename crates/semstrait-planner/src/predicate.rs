use std::collections::BTreeSet;
use std::mem;
use std::ops::BitAnd;

use semstrait_common::SemanticsName;
use semstrait_common::{Tree, ValidateError};
use semstrait_ir::{PhysicalExpr, SemanticRole};
use semstrait_manifest::SourceAlias;
use semstrait_sql::datatype::{CanonicalTypeSystem, DataType, TypeOrFamily};
use semstrait_sql::expr::{
    is_condition, map_nodes, type_tree, BinaryOpKind, Expr, InputTypes, NamedReference, RowExpr,
    SemanticLeaf, Typed,
};
use semstrait_sql::DocumentDialect;

use crate::error::call_refused;
use crate::read_model::ReadModel;
use crate::steps::field;
use crate::strategy::StrategyInput;
use crate::{BoundName, PlanError};

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct Predicate {
    pub tree: RowExpr<SemanticLeaf>,
    pub leaves: Vec<BoundName>,
    pub placement: Placement,
}

impl Predicate {
    pub(crate) fn in_branches(&self) -> bool {
        self.placement == Placement::BeforeAggregation
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Placement {
    BeforeAggregation,
    AfterAggregation,
}

impl Placement {
    pub(crate) fn of<'n>(
        leaves: impl IntoIterator<Item = (&'n SemanticsName, SemanticRole)>,
    ) -> Result<Self, PlanError> {
        let (mut before, mut after) = (None, None);
        for (name, role) in leaves {
            let side = match role {
                SemanticRole::Measure | SemanticRole::Metric => &mut after,
                _ => &mut before,
            };
            side.get_or_insert(name);
        }
        match (before, after) {
            (Some(dimension), Some(aggregated)) => Err(PlanError::MixedPlacement {
                dimension: dimension.clone(),
                aggregated: aggregated.clone(),
            }),
            (None, Some(_)) => Ok(Self::AfterAggregation),
            (_, None) => Ok(Self::BeforeAggregation),
        }
    }
}

pub(crate) enum Read<'p> {
    Column(&'p NamedReference),
    Semantics {
        name: SemanticsName,
        stated: Option<SemanticRole>,
        accessor: bool,
    },
}

impl<'p> Read<'p> {
    pub(crate) fn of(leaf: &'p SemanticLeaf) -> Option<Self> {
        let (name, stated, accessor) = match leaf {
            SemanticLeaf::Field(name) => (name, None, false),
            SemanticLeaf::Dimension(name, accessor) => {
                (name, Some(SemanticRole::Dimension), accessor.is_some())
            }
            SemanticLeaf::Measure(name, accessor) => {
                (name, Some(SemanticRole::Measure), accessor.is_some())
            }
            SemanticLeaf::Metric(name, accessor) => {
                (name, Some(SemanticRole::Metric), accessor.is_some())
            }
            _ => return None,
        };
        Some(Self::Semantics {
            name: SemanticsName::new(name.parts().join(".")),
            stated,
            accessor,
        })
    }
}

struct DefinitionTypes<'r>(&'r dyn ReadModel);

impl InputTypes<SemanticLeaf> for DefinitionTypes<'_> {
    fn column(&self, _: &NamedReference) -> Typed {
        Typed::new(DataType::Unknown, true)
    }

    fn leaf(&self, leaf: &SemanticLeaf) -> Typed {
        let data_type = match Read::of(leaf) {
            Some(Read::Semantics { name, .. }) => (self.0.bit_of(name.as_str()))
                .and_then(|bit| self.0.definition(bit))
                .and_then(|definition| definition.data_type),
            _ => None,
        };
        Typed::new(data_type.unwrap_or(DataType::Unknown), true)
    }
}

pub(crate) fn typed(
    tree: RowExpr<SemanticLeaf>,
    read_model: &dyn ReadModel,
    dialect: DocumentDialect,
) -> Result<RowExpr<SemanticLeaf>, PlanError> {
    let (tree, root) = type_tree(
        tree,
        &DefinitionTypes(read_model),
        dialect,
        &CanonicalTypeSystem,
        Some(&DataType::Boolean),
    )
    .map_err(|error| PlanError::PredicateTypeError { error })?;
    if is_condition(&root.data_type, root.family.as_ref()) {
        Ok(tree)
    } else {
        Err(PlanError::PredicateNotACondition {
            got: TypeOrFamily::of(&root.data_type, root.family.as_ref()),
        })
    }
}

pub(crate) fn leaves_beyond<'b>(
    names: &[BoundName],
    predicates: &'b [Predicate],
) -> Vec<&'b BoundName> {
    beyond(
        names,
        predicates.iter().flat_map(|predicate| &predicate.leaves),
    )
}

pub(crate) fn beyond<'b>(
    names: &[BoundName],
    leaves: impl IntoIterator<Item = &'b BoundName>,
) -> Vec<&'b BoundName> {
    let mut held: BTreeSet<u32> = names.iter().map(|name| name.bit).collect();
    (leaves.into_iter())
        .filter(|leaf| held.insert(leaf.bit))
        .collect()
}

pub(crate) struct JoinEnd<'e> {
    pub(crate) operand: &'e SourceAlias,
    pub(crate) served: &'e [SemanticsName],
    pub(crate) preserved: bool,
}

impl JoinEnd<'_> {
    fn serves(&self, name: &SemanticsName) -> bool {
        self.served.contains(name)
    }
}

pub(crate) enum AtJoin<'e> {
    Into(&'e SourceAlias, Predicate),
    Above(PhysicalExpr),
    Kept(Predicate),
}

pub(crate) fn at_join<'e>(
    input: &StrategyInput,
    predicate: Predicate,
    ends: &[JoinEnd<'e>; 2],
) -> Result<Vec<AtJoin<'e>>, PlanError> {
    let shared = |name: &SemanticsName| ends.iter().all(|end| end.serves(name));
    let leaves = &predicate.leaves;
    if predicate.placement == Placement::AfterAggregation {
        return Ok(vec![AtJoin::Kept(predicate)]);
    }
    if !leaves.iter().any(|leaf| shared(&leaf.name)) {
        let read: Vec<_> = (ends.iter())
            .filter(|end| leaves.iter().any(|leaf| end.serves(&leaf.name)))
            .collect();
        return Ok(vec![match read.as_slice() {
            [end] if end.preserved => AtJoin::Into(end.operand, predicate),
            _ => AtJoin::Kept(predicate),
        }]);
    }
    (ends.iter())
        .map(|copy| {
            let reader = |name: &SemanticsName| {
                if shared(name) {
                    Some(copy)
                } else {
                    ends.iter().find(|end| end.serves(name))
                }
            };
            let read = (leaves.iter())
                .map(|leaf| reader(&leaf.name).ok_or_else(|| input.broken()))
                .collect::<Result<Vec<_>, _>>()?;
            if read.iter().all(|end| end.operand == copy.operand) {
                return Ok(AtJoin::Into(copy.operand, predicate.clone()));
            }
            let qualifier = |name: &SemanticsName| reader(name).map(|end| end.operand.as_str());
            (over_fields(&predicate.tree, qualifier).map(AtJoin::Above))
                .map_err(|error| input.refused(error))
        })
        .collect()
}

pub(crate) fn over_fields<'q>(
    tree: &RowExpr<SemanticLeaf>,
    qualifier: impl Fn(&SemanticsName) -> Option<&'q str>,
) -> Result<PhysicalExpr, ValidateError> {
    let mut leaf = |node: &RowExpr<SemanticLeaf>| {
        let Expr::Leaf(leaf) = node else { return None };
        let Read::Semantics { name, .. } = Read::of(leaf)? else {
            return None;
        };
        Some(field(qualifier(&name), name.as_str()).map(|(_, leaf)| leaf))
    };
    map_nodes(tree, &mut leaf, &call_refused)
}

pub(crate) fn split_at_joins(
    read_model: &dyn ReadModel,
    input: &mut StrategyInput,
) -> Result<(Vec<Predicate>, Vec<Predicate>), PlanError> {
    let predicates = (mem::take(&mut input.predicates).into_iter())
        .map(|predicate| input.with_typed_nulls(read_model, predicate))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((predicates.into_iter()).partition(Predicate::in_branches))
}

pub(crate) fn above_joins<'q>(
    input: &StrategyInput,
    predicates: &[Predicate],
    qualifier: impl Fn(&SemanticsName) -> Option<&'q str>,
) -> Result<Option<PhysicalExpr>, PlanError> {
    let mut above = Vec::new();
    for predicate in predicates {
        let over = over_fields(&predicate.tree, &qualifier);
        above.push(over.map_err(|error| input.refused(error))?);
    }
    Ok(above.into_iter().reduce(BitAnd::bitand))
}

pub(crate) fn read_back(
    stored: &PhysicalExpr,
    placement: Placement,
    bit_of: impl Fn(&str) -> Option<u32>,
    broken: impl Fn() -> PlanError,
) -> Result<Predicate, PlanError> {
    let mut field = |node: &PhysicalExpr| {
        let Expr::Column(column) = node else {
            return None;
        };
        let name = NamedReference::unqualified(column.name()).ok_or_else(&broken);
        Some(name.map(|name| Expr::Leaf(SemanticLeaf::Field(name))))
    };
    let mut read = |copy: &PhysicalExpr| -> Result<Predicate, PlanError> {
        let tree = map_nodes(copy, &mut field, &|_| broken())?;
        let tree = RowExpr::try_from(tree).map_err(|_| broken())?;
        let leaves = (reads(&tree))
            .filter_map(|read| match read {
                Read::Semantics { name, .. } => Some(name),
                Read::Column(_) => None,
            })
            .map(|name| {
                let bit = bit_of(name.as_str()).ok_or_else(&broken)?;
                Ok(BoundName { name, bit })
            })
            .collect::<Result<Vec<_>, PlanError>>()?;
        Ok(Predicate {
            leaves: beyond(&[], &leaves).into_iter().cloned().collect(),
            tree,
            placement,
        })
    };
    if let Expr::BinaryOp {
        op: BinaryOpKind::And,
        left,
        right,
    } = stored
    {
        let first = read(left)?;
        if first == read(right)? {
            return Ok(first);
        }
    }
    read(stored)
}

pub(crate) fn reads(tree: &RowExpr<SemanticLeaf>) -> impl Iterator<Item = Read<'_>> {
    tree.pre_order().filter_map(|node| match node {
        Expr::Column(column) => Some(Read::Column(column)),
        Expr::Leaf(leaf) => Read::of(leaf),
        _ => None,
    })
}
