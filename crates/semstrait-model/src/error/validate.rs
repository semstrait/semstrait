use semstrait_common::diagnostic::{Diagnose, Location, Severity};

use crate::model::types::entities::Optional;
use crate::validation::MAX_COMPOSITION_DEPTH;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ValidateErrorKind {
    ComplexDataKindInsufficientChildren {
        parent: String,
        child_count: usize,
    },
    CompositionNestingTooDeep {
        root: String,
        depth: usize,
    },

    EmptyModel,

    DuplicateDataKindName {
        name: String,
        occurrences: Vec<Location>,
    },
    DuplicateSharedSemanticsName {
        carrier: String,
        name: String,
        occurrences: Vec<Location>,
    },
    DuplicateEntityId {
        id: String,
        occurrences: Vec<Location>,
    },
    MissingEntityId {
        path: String,
    },
    InterfaceMemberPathNotUnique {
        data_kind: String,
        collection: String,
        name: String,
        occurrences: Vec<Location>,
    },

    SemanticsRefMissingExpr {
        carrier: String,
        name: String,
    },
    OrphanSharedSemantics {
        carrier: String,
        name: String,
    },
    RelationshipMissingCardinality {
        relationship: String,
    },
    RelationshipDanglingEndpoint {
        relationship: String,
        side: String,
        endpoint: String,
    },
    TemporalLeafMissingGranularity {
        data_kind: String,
    },
    TemporalGranularityOnComplex {
        data_kind: String,
    },
    GrainsetChildMissingGranularity {
        grainset: String,
        child: String,
    },
    RelationshipSymmetricCardinalityIncomplete {
        relationship: String,
        missing: String,
    },
    RelationshipManyToManyCrossFilterDirectional {
        relationship: String,
    },
    RelationshipCrossWithPreservedSide {
        relationship: String,
        optional: Optional,
    },
    DuplicateRelationshipName {
        relationship: String,
        scope: String,
        occurrences: usize,
    },

    RelationshipJoinBasisEmpty {
        relationship: String,
    },
    RelationshipJoinKeyDuplicated {
        relationship: String,
        key: String,
    },

    DataKindKeysetsIdenticalMembership {
        data_kind: String,
        keysets: Vec<String>,
        members: Vec<String>,
    },

    KeysetListEmpty,
    KeysetNameBlank,

    SemanticsShadowRootPool {
        carrier: String,
        name: String,
    },

    CyclicSemanticsReference {
        carrier: String,
        cycle: Vec<String>,
    },
}

impl Diagnose for ValidateErrorKind {
    fn message(&self) -> String {
        use ValidateErrorKind::*;
        match self {
            ComplexDataKindInsufficientChildren {
                parent,
                child_count,
            } => format!(
                "complex data kind `{}` has {} child(ren); a Grainset or a Unionset of mode `all` needs at least 1, a Joinset or a Unionset of mode `unique` at least 2",
                parent, child_count
            ),
            CompositionNestingTooDeep { root, depth } => format!(
                "composition `{root}` nests {depth} compositions deep; ADR 0054's \
                 `MAX_COMPOSITION_DEPTH = {MAX_COMPOSITION_DEPTH}` bounds authored nesting",
            ),
            EmptyModel => "model has no data kinds (empty model)".to_string(),
            DuplicateDataKindName { name, occurrences } => format!(
                "duplicate data-kind name `{}` ({} occurrences)",
                name,
                occurrences.len()
            ),
            DuplicateSharedSemanticsName {
                carrier,
                name,
                occurrences,
            } => format!(
                "duplicate `{}` entry `{}` ({} occurrences)",
                carrier,
                name,
                occurrences.len()
            ),
            DuplicateEntityId { id, occurrences } => format!(
                "duplicate entity id `{}` ({} occurrences) (SR-12)",
                id,
                occurrences.len()
            ),
            MissingEntityId { path } => {
                format!(
                    "entity `{}` is missing required `id` (strict profile)",
                    path
                )
            }
            InterfaceMemberPathNotUnique {
                data_kind,
                collection,
                name,
                occurrences,
            } => format!(
                "`{}.{}` exposes `{}` {} times; a scope-qualified name holds one entity",
                data_kind,
                collection,
                name,
                occurrences.len()
            ),
            SemanticsRefMissingExpr { carrier, name } => format!(
                "ref site for {} `{}` and root-pool entry both lack `expr:`",
                carrier, name
            ),
            OrphanSharedSemantics { carrier, name } => format!(
                "root-pool {} `{}` is referenced from no data kind's interface",
                carrier, name
            ),
            RelationshipMissingCardinality { relationship } => format!(
                "relationship `{}` is missing `cardinality:`; omit it only under `on: cross`",
                relationship
            ),
            RelationshipDanglingEndpoint {
                relationship,
                side,
                endpoint,
            } => format!(
                "relationship `{}` references unknown {} endpoint `{}`",
                relationship, side, endpoint
            ),
            TemporalLeafMissingGranularity { data_kind } => format!(
                "leaf `{}` has `temporal:` but is missing `granularity:`",
                data_kind
            ),
            TemporalGranularityOnComplex { data_kind } => format!(
                "root composition `{}` may not author `temporal.granularity:`; only a \
                 composition that is a constituent of another may",
                data_kind
            ),
            GrainsetChildMissingGranularity { grainset, child } => format!(
                "grainset child `{}.{}` must author its own `temporal.granularity:`",
                grainset, child
            ),
            RelationshipSymmetricCardinalityIncomplete {
                relationship,
                missing,
            } => format!(
                "relationship `{}` has symmetric cardinality but is missing `{}:`",
                relationship, missing
            ),
            RelationshipManyToManyCrossFilterDirectional { relationship } => format!(
                "relationship `{}`: many-to-many `cross_filter:` may not be Left/Right",
                relationship
            ),
            RelationshipCrossWithPreservedSide {
                relationship,
                optional,
            } => {
                let side = match optional {
                    Optional::None => "none",
                    Optional::Left => "left",
                    Optional::Right => "right",
                    Optional::Both => "both",
                };
                format!(
                    "relationship `{relationship}` has `on: cross` and `optional: {side}`; \
                     a cross join preserves no side, so `optional:` beside it is `none`"
                )
            }
            DuplicateRelationshipName {
                relationship,
                scope,
                occurrences,
            } => format!(
                "duplicate relationship name `{}` in scope `{}` ({} occurrences); a Joinset-local shadow of a root-level name is a different scope and stays legal",
                relationship, scope, occurrences
            ),
            RelationshipJoinBasisEmpty { relationship } => format!(
                "relationship `{}`: an empty `on:` list is not a join basis; use `on_cross()` to declare there is none",
                relationship
            ),
            RelationshipJoinKeyDuplicated { relationship, key } => format!(
                "relationship `{}`: join key `{}` is repeated",
                relationship, key
            ),
            DataKindKeysetsIdenticalMembership {
                data_kind,
                keysets,
                members,
            } => format!(
                "data kind `{}`: keysets {} have identical members ({}); rename or change \
                 one rather than declaring the same advice twice",
                data_kind,
                keysets
                    .iter()
                    .map(|k| format!("`{k}`"))
                    .collect::<Vec<_>>()
                    .join(" and "),
                members.join(", ")
            ),
            KeysetListEmpty => {
                "a keyset list naming no keyset is not a membership; `Keyness::key()` — \
                 `with_key()` at a site — states `default` membership"
                    .to_string()
            }
            KeysetNameBlank => {
                "a keyset name cannot be blank; nothing could name the keyset it would mint"
                    .to_string()
            }
            SemanticsShadowRootPool { carrier, name } => format!(
                "{} `{}` shadows root-pool entry; use `ref:` + override instead",
                carrier, name
            ),
            CyclicSemanticsReference { carrier, cycle } => {
                format!("cycle in `{}` references: {}", carrier, cycle.join(" → "))
            }
        }
    }

    fn severity_default(&self) -> Severity {
        match self {
            ValidateErrorKind::SemanticsShadowRootPool { .. } => Severity::Warning,
            _ => Severity::Error,
        }
    }
}
