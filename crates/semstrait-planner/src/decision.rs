use std::fmt;

use semstrait_common::{DataKindName, SemanticsName};
use semstrait_ir::{JoinsetJoinType, Reader, Schema};
use semstrait_manifest::SourceAlias;

use crate::read_model::ReadModel;

#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    pub stand_in: Option<DataKindName>,
    pub operands: Vec<Operand>,
    pub combine: Combine,
    pub physical_read: Option<PhysicalRead>,
}

impl Decision {
    pub(crate) fn unread<'s>(
        &self,
        required: impl IntoIterator<Item = &'s SourceAlias>,
    ) -> Option<&'s SourceAlias> {
        (required.into_iter()).find(|&source| {
            self.operands
                .iter()
                .all(|operand| &operand.source != source)
        })
    }

    pub(crate) fn foreign(
        &self,
        read_model: &dyn ReadModel,
        units: &[SourceAlias],
    ) -> Option<DecisionValue> {
        let operand = (self.operands.iter()).find(|operand| !units.contains(&operand.source))?;
        Some(match read_model.name(&operand.source) {
            Some(name) => DecisionValue::NotAnOperand(name),
            None => DecisionValue::UnknownOperand(operand.source.clone()),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Operand {
    pub source: SourceAlias,
    pub names: Vec<SemanticsName>,
    pub null_filled: Vec<SemanticsName>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalRead {
    pub reader: Option<Reader>,
    pub schema: Schema,
}

pub(crate) struct Allowed {
    stand_in: bool,
    operands: bool,
    combines: &'static [Combine],
    physical_read: bool,
}

impl Allowed {
    pub(crate) const DATASET: Self = Self {
        stand_in: false,
        operands: false,
        combines: &[Combine::Leaf],
        physical_read: true,
    };

    pub(crate) const JOINSET: Self = Self {
        stand_in: true,
        operands: true,
        combines: &[Combine::Join, Combine::Cross, Combine::PassThrough],
        physical_read: false,
    };

    pub(crate) const UNIONSET: Self = Self {
        stand_in: false,
        operands: true,
        combines: &[Combine::Stack],
        physical_read: false,
    };

    pub(crate) const GRAINSET: Self = Self {
        stand_in: true,
        operands: true,
        combines: &[Combine::PassThrough, Combine::DrillAcross],
        physical_read: false,
    };

    pub(crate) fn refused(
        &self,
        read_model: &dyn ReadModel,
        decision: &Decision,
    ) -> Option<DecisionValue> {
        let stand_in = (decision.stand_in.as_ref())
            .filter(|_| !self.stand_in)
            .cloned()
            .map(DecisionValue::StandIn);
        let operand = || {
            (!self.operands)
                .then(|| decision.foreign(read_model, &[]))
                .flatten()
        };
        let combine = || {
            (!self.combines.contains(&decision.combine))
                .then_some(DecisionValue::Combine(decision.combine))
        };
        let read = || {
            (self.physical_read && decision.physical_read.is_none())
                .then_some(DecisionValue::NoPhysicalRead)
        };
        stand_in.or_else(operand).or_else(combine).or_else(read)
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DecisionValue {
    StandIn(DataKindName),
    NotRead(DataKindName),
    NotAnOperand(DataKindName),
    UnknownOperand(SourceAlias),
    Combine(Combine),
    NoPhysicalRead,
}

impl fmt::Display for DecisionValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StandIn(source) => write!(f, "the stand-in `{source}`"),
            Self::NotRead(operand) => write!(f, "the operand not read `{operand}`"),
            Self::NotAnOperand(source) => write!(f, "reading `{source}`, which is no operand"),
            Self::UnknownOperand(source) => {
                write!(f, "reading `{source}`, which names no semantic source")
            }
            Self::Combine(combine) => write!(f, "the combine `{combine}`"),
            Self::NoPhysicalRead => f.write_str("a leaf with no physical read"),
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Combine {
    Leaf,
    Join,
    Cross,
    Stack,
    PassThrough,
    DrillAcross,
}

impl From<JoinsetJoinType> for Combine {
    fn from(join_type: JoinsetJoinType) -> Self {
        match join_type {
            JoinsetJoinType::Join(_) => Self::Join,
            JoinsetJoinType::Cross => Self::Cross,
        }
    }
}

impl fmt::Display for Combine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Leaf => "leaf",
            Self::Join => "join",
            Self::Cross => "cross",
            Self::Stack => "stack",
            Self::PassThrough => "pass-through",
            Self::DrillAcross => "drill-across",
        })
    }
}
