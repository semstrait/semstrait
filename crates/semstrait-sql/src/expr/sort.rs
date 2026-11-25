use serde::{Deserialize, Serialize};

use super::{Aggregation, Expr, Leaf};

#[derive(Debug, Clone, PartialEq)]
pub struct SortKey<L: Leaf, A: Aggregation> {
    pub expr: Expr<L, A>,
    pub dir: SortDir,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum SortDir {
    Asc { nulls: NullOrdering },
    Desc { nulls: NullOrdering },
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NullOrdering {
    First,
    Last,
    Unspecified,
}
