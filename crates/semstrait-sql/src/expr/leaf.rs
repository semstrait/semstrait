use super::{Leaf, Literal, NamedReference};

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum SemanticLeaf {
    Untyped(UntypedLiteral),
    Field(NamedReference),
    Dimension(NamedReference, Option<DimensionAccessor>),
    Measure(NamedReference, Option<MeasureAccessor>),
    Metric(NamedReference, Option<MetricAccessor>),
}

impl Leaf for SemanticLeaf {
    fn untyped(&self) -> Option<&UntypedLiteral> {
        match self {
            Self::Untyped(literal) => Some(literal),
            Self::Field(_) | Self::Dimension(..) | Self::Measure(..) | Self::Metric(..) => None,
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum PhysicalLeaf {
    Untyped(UntypedLiteral),
}

impl Leaf for PhysicalLeaf {
    fn untyped(&self) -> Option<&UntypedLiteral> {
        match self {
            Self::Untyped(literal) => Some(literal),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct UntypedLiteral(pub Literal);

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DimensionAccessor {
    First,
    Last,
    Lag(u32),
    Lead(u32),
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MeasureAccessor {
    Previous,
    Next,
    Lag(u32),
    Lead(u32),
    Delta,
    PercentChange,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricAccessor {
    Previous,
    Next,
    Lag(u32),
    Lead(u32),
    Delta,
    PercentChange,
}
