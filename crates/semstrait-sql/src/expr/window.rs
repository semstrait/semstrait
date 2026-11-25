use serde::{Deserialize, Serialize};

use super::{Aggregation, Expr, Leaf, OverError, SortKey, TypeError};
use crate::datatype::TypeFamily;

#[derive(Debug, Clone, PartialEq)]
pub struct Over<L: Leaf, A: Aggregation> {
    pub partition_by: Vec<Expr<L, A>>,
    pub order_by: Vec<SortKey<L, A>>,
    pub window_frame: Option<WindowFrame>,
}

impl<L: Leaf, A: Aggregation> Default for Over<L, A> {
    fn default() -> Self {
        Self {
            partition_by: Vec::new(),
            order_by: Vec::new(),
            window_frame: None,
        }
    }
}

impl<L: Leaf, A: Aggregation> Over<L, A> {
    pub fn is_empty(&self) -> bool {
        self.partition_by.is_empty() && self.order_by.is_empty() && self.window_frame.is_none()
    }

    pub(super) fn check(&self) -> Result<(), OverError> {
        use WindowBound::{
            CurrentRow, Following, Preceding, UnboundedFollowing, UnboundedPreceding,
        };
        let Some(window_frame) = self.window_frame else {
            return Ok(());
        };
        let WindowFrame { kind, start, end } = window_frame;
        let keys = self.order_by.len();
        match (start, end) {
            (UnboundedFollowing, _) => Err(OverError::StartsUnboundedFollowing),
            (_, UnboundedPreceding) => Err(OverError::EndsUnboundedPreceding),
            (CurrentRow, Preceding(_)) | (Following(_), Preceding(_) | CurrentRow) => {
                Err(OverError::EndsBeforeStart)
            }
            _ if window_frame.is_range_offset() && keys != 1 => {
                Err(OverError::RangeOffsetKeys { keys })
            }
            _ if kind == WindowFrameKind::Groups && keys == 0 => {
                Err(OverError::GroupsWithoutOrderBy)
            }
            _ => Ok(()),
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WindowFrame {
    pub kind: WindowFrameKind,
    pub start: WindowBound,
    pub end: WindowBound,
}

impl WindowFrame {
    pub fn new(kind: WindowFrameKind, start: WindowBound, end: WindowBound) -> Self {
        Self { kind, start, end }
    }

    fn is_range_offset(&self) -> bool {
        let offset = |bound| matches!(bound, WindowBound::Preceding(_) | WindowBound::Following(_));
        self.kind == WindowFrameKind::Range && (offset(self.start) || offset(self.end))
    }

    pub(crate) fn check_order_key(
        &self,
        name: &'static str,
        family: Option<&TypeFamily>,
    ) -> Result<(), TypeError> {
        match family {
            Some(family @ (TypeFamily::String | TypeFamily::Boolean)) if self.is_range_offset() => {
                Err(TypeError::RangeOrderKey {
                    name,
                    family: family.clone(),
                })
            }
            _ => Ok(()),
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WindowFrameKind {
    Rows,
    Range,
    Groups,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(
    rename_all = "snake_case",
    tag = "type",
    content = "content",
    deny_unknown_fields
)]
pub enum WindowBound {
    UnboundedPreceding,
    Preceding(u64),
    CurrentRow,
    Following(u64),
    UnboundedFollowing,
}
