use semstrait_common::Tree;
use semstrait_sql::expr::{
    AggregateAllowed, AggregateCall, Call, Expr, NullOrdering, SortDir, SortKey,
};

use crate::error::IrErrorKind;
use crate::expr::{Parameter, PhysicalExpr};
use crate::plan::common::{condition, unique_names, NodeCommon};
use crate::plan::node::PlanNode;
use crate::primitives::Name;
use crate::types::{Schema, SchemaColumn};

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct FilterNode {
    pub common: NodeCommon,
    pub input: Box<PlanNode>,
    pub filter: PhysicalExpr,
}

impl FilterNode {
    pub fn new(input: impl Into<PlanNode>, filter: PhysicalExpr) -> Result<Self, IrErrorKind> {
        let input = input.into();
        let fields = &input.common().output_schema;
        condition(fields, &filter)?;
        Ok(Self {
            common: NodeCommon::new(fields.columns.clone()),
            input: Box::new(input),
            filter,
        })
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectNode {
    pub common: NodeCommon,
    pub input: Box<PlanNode>,
    pub expressions: Vec<PhysicalExpr>,
}

impl ProjectNode {
    pub fn new(
        input: impl Into<PlanNode>,
        expressions: Vec<(Name, PhysicalExpr)>,
    ) -> Result<Self, IrErrorKind> {
        let input = input.into();
        unique_names(expressions.iter().map(|(name, _)| name.as_str()))?;
        let fields = &input.common().output_schema;
        let computed = (expressions.iter())
            .map(|(name, expr)| field(name, expr, fields))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            common: NodeCommon::new(fields.columns.iter().cloned().chain(computed).collect()),
            input: Box::new(input),
            expressions: expressions.into_iter().map(|(_, expr)| expr).collect(),
        })
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct AggregateNode {
    pub common: NodeCommon,
    pub input: Box<PlanNode>,
    pub groupings: Vec<PhysicalExpr>,
    pub aggregates: Vec<(Name, AggregateCall<Parameter, AggregateAllowed>)>,
}

impl AggregateNode {
    pub fn new(
        input: impl Into<PlanNode>,
        groupings: Vec<(Name, PhysicalExpr)>,
        aggregates: Vec<(Name, AggregateCall<Parameter, AggregateAllowed>)>,
    ) -> Result<Self, IrErrorKind> {
        let input = input.into();
        let names =
            (groupings.iter().map(|(name, _)| name)).chain(aggregates.iter().map(|(name, _)| name));
        unique_names(names.map(Name::as_str))?;
        let fields = &input.common().output_schema;
        let natural = (groupings.iter())
            .map(|(name, expr)| field(name, expr, fields))
            .chain(aggregates.iter().map(|(name, call)| {
                if let Some(filter) = call.filter() {
                    condition(fields, filter)?;
                }
                let call = Expr::Function(Call::Aggregate(call.clone(), AggregateAllowed));
                field(name, &call, fields)
            }))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            common: NodeCommon::new(natural),
            input: Box::new(input),
            groupings: groupings.into_iter().map(|(_, expr)| expr).collect(),
            aggregates,
        })
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct SortNode {
    pub common: NodeCommon,
    pub input: Box<PlanNode>,
    pub order_by: Vec<SortKey<Parameter, AggregateAllowed>>,
}

impl SortNode {
    pub fn new(
        input: impl Into<PlanNode>,
        order_by: Vec<SortKey<Parameter, AggregateAllowed>>,
    ) -> Result<Self, IrErrorKind> {
        let input = input.into();
        let fields = &input.common().output_schema;
        sort_keys(fields, &order_by)?;
        Ok(Self {
            common: NodeCommon::new(fields.columns.clone()),
            input: Box::new(input),
            order_by,
        })
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct FetchNode {
    pub common: NodeCommon,
    pub input: Box<PlanNode>,
    pub offset: i64,
    pub limit: Option<i64>,
}

impl FetchNode {
    pub fn new(
        input: impl Into<PlanNode>,
        offset: i64,
        limit: Option<i64>,
    ) -> Result<Self, IrErrorKind> {
        let input = input.into();
        bound(offset)?;
        limit.map(bound).transpose()?;
        Ok(Self {
            common: NodeCommon::new(input.common().output_schema.columns.clone()),
            input: Box::new(input),
            offset,
            limit,
        })
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct TopNNode {
    pub common: NodeCommon,
    pub input: Box<PlanNode>,
    pub order_by: Vec<SortKey<Parameter, AggregateAllowed>>,
    pub offset: i64,
    pub limit: i64,
    pub fetch_mode: FetchMode,
}

impl TopNNode {
    pub fn new(
        input: impl Into<PlanNode>,
        order_by: Vec<SortKey<Parameter, AggregateAllowed>>,
        offset: i64,
        limit: i64,
        fetch_mode: FetchMode,
    ) -> Result<Self, IrErrorKind> {
        let input = input.into();
        let fields = &input.common().output_schema;
        sort_keys(fields, &order_by)?;
        bound(offset)?;
        bound(limit)?;
        Ok(Self {
            common: NodeCommon::new(fields.columns.clone()),
            input: Box::new(input),
            order_by,
            offset,
            limit,
            fetch_mode,
        })
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FetchMode {
    RowsOnly,
    WithTies,
}

fn field(name: &Name, expr: &PhysicalExpr, fields: &Schema) -> Result<SchemaColumn, IrErrorKind> {
    order_keys(expr)?;
    let typed = fields.type_of(expr)?;
    let field = SchemaColumn {
        name: name.as_str().to_owned(),
        data_type: typed.data_type,
        nullable: typed.nullable,
    };
    field.check()?;
    Ok(field)
}

fn sort_keys(
    fields: &Schema,
    keys: &[SortKey<Parameter, AggregateAllowed>],
) -> Result<(), IrErrorKind> {
    (keys.iter().enumerate()).try_for_each(|(position, key)| {
        if unspecified(key.dir) {
            return Err(IrErrorKind::UnspecifiedNullOrdering { key: position });
        }
        match fields.type_of(&key.expr)?.data_type {
            data_type if data_type.is_valid() => Ok(()),
            data_type => Err(IrErrorKind::InvalidKeyType {
                key: position,
                data_type,
            }),
        }
    })
}

fn order_keys(expr: &PhysicalExpr) -> Result<(), IrErrorKind> {
    match expr {
        Expr::Function(call @ Call::Window(window)) => {
            if let Some(key) = (window.over().order_by.iter()).position(|key| unspecified(key.dir))
            {
                return Err(IrErrorKind::UnspecifiedWindowNullOrdering {
                    function: call.name(),
                    key,
                });
            }
        }
        Expr::Function(call @ Call::Aggregate(aggregate, _)) => {
            if let Some(key) = (aggregate.order_by().iter()).position(|key| unspecified(key.dir)) {
                return Err(IrErrorKind::UnspecifiedAggregateNullOrdering {
                    function: call.name(),
                    key,
                });
            }
        }
        _ => {}
    }
    expr.children().into_iter().try_for_each(order_keys)
}

fn unspecified(dir: SortDir) -> bool {
    matches!(
        dir,
        SortDir::Asc {
            nulls: NullOrdering::Unspecified
        } | SortDir::Desc {
            nulls: NullOrdering::Unspecified
        }
    )
}

fn bound(bound: i64) -> Result<(), IrErrorKind> {
    if bound < 0 {
        Err(IrErrorKind::NegativeBound { bound })
    } else {
        Ok(())
    }
}
