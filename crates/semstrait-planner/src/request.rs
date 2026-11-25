use semstrait_common::{DataKindName, Granularity, SemanticsName};
use semstrait_ir::SemanticRole;
use semstrait_sql::expr::{RowExpr, SemanticLeaf};

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    pub from: Option<DataKindName>,
    pub select: Vec<Selection>,
    pub predicate: Option<RowExpr<SemanticLeaf>>,
    pub filters: Vec<SemanticsName>,
    pub order_by: Vec<OrderKey>,
    pub limit: Option<u64>,
    pub offset: Option<u64>,
}

impl Request {
    #[must_use]
    pub fn new(select: Vec<Selection>) -> Self {
        Self {
            from: None,
            select,
            predicate: None,
            filters: Vec::new(),
            order_by: Vec::new(),
            limit: None,
            offset: None,
        }
    }

    #[must_use]
    pub fn with_from(self, from: impl Into<DataKindName>) -> Self {
        Self {
            from: Some(from.into()),
            ..self
        }
    }

    #[must_use]
    pub fn with_predicate(self, predicate: RowExpr<SemanticLeaf>) -> Self {
        Self {
            predicate: Some(predicate),
            ..self
        }
    }

    #[must_use]
    pub fn with_filters(self, filters: Vec<SemanticsName>) -> Self {
        Self { filters, ..self }
    }

    #[must_use]
    pub fn with_order_by(self, order_by: Vec<OrderKey>) -> Self {
        Self { order_by, ..self }
    }

    #[must_use]
    pub fn with_limit(self, limit: u64) -> Self {
        Self {
            limit: Some(limit),
            ..self
        }
    }

    #[must_use]
    pub fn with_offset(self, offset: u64) -> Self {
        Self {
            offset: Some(offset),
            ..self
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum Selection {
    Field(SemanticsName),
    Dimension {
        name: SemanticsName,
        modifier: Option<DimensionModifier>,
    },
    Measure {
        name: SemanticsName,
    },
    Metric {
        name: SemanticsName,
    },
}

impl Selection {
    pub(crate) fn name(&self) -> &SemanticsName {
        match self {
            Self::Field(name)
            | Self::Dimension { name, .. }
            | Self::Measure { name }
            | Self::Metric { name } => name,
        }
    }

    pub(crate) fn role(&self) -> Option<SemanticRole> {
        match self {
            Self::Field(_) => None,
            Self::Dimension { .. } => Some(SemanticRole::Dimension),
            Self::Measure { .. } => Some(SemanticRole::Measure),
            Self::Metric { .. } => Some(SemanticRole::Metric),
        }
    }

    pub(crate) fn granularity(&self) -> Option<Granularity> {
        match self {
            Self::Dimension {
                modifier: Some(DimensionModifier::Granularity(granularity)),
                ..
            } => Some(*granularity),
            _ => None,
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DimensionModifier {
    Granularity(Granularity),
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OrderKey {
    pub name: SemanticsName,
    pub direction: Direction,
}

impl OrderKey {
    #[must_use]
    pub fn new(name: impl Into<SemanticsName>, direction: Direction) -> Self {
        let name = name.into();
        Self { name, direction }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    Asc,
    Desc,
}
