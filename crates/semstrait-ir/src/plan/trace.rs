use semstrait_common::{DataKindName, Granularity};

use crate::primitives::{Cardinality, Name};

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SemAnnotation {
    Rank {
        scope_root: DataKindName,
        compositions: u32,
        granularity_fit: GranularityGroup,
        granularity: Option<Granularity>,
    },
    SemanticSource {
        semantic_source: DataKindName,
        source_role: SourceRole,
        originators: Vec<DataKindName>,
        physical_source: Option<Name>,
        cardinality: Option<Cardinality>,
        delivery: Option<Delivery>,
        answered_by: Option<DataKindName>,
        pruned: Vec<DataKindName>,
    },
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SourceRole {
    Dataset,
    Joinset,
    Unionset,
    Grainset,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Delivery {
    Rows,
    PartialAggregates,
    FinalAggregates,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GranularityGroup {
    Requested,
    Selected,
    NotSelected,
}
