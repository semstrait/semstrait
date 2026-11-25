use std::collections::BTreeMap;

use semstrait_common::str_newtype;
use semstrait_common::EntityId;

use super::semantic_source::SourceRole;
use crate::error::{CompileError, ReferenceSite};

str_newtype!(SourceAlias);

#[derive(Debug)]
pub(crate) struct SourceAliases {
    pub(crate) semantic: BTreeMap<EntityId, SourceAlias>,
    pub(crate) physical: BTreeMap<EntityId, SourceAlias>,
}

impl SourceAliases {
    pub(crate) fn derive<'a>(
        semantic_sources: impl IntoIterator<Item = (&'a EntityId, SourceRole)>,
        physical_sources: impl IntoIterator<Item = &'a EntityId>,
    ) -> Self {
        Self {
            semantic: indexed(
                semantic_sources
                    .into_iter()
                    .map(|(id, source_role)| (id, format!("ss{}", source_role.letter()))),
            ),
            physical: indexed(physical_sources.into_iter().map(|id| (id, "ps".to_owned()))),
        }
    }

    pub(crate) fn semantic_alias(&self, id: &EntityId) -> Result<&SourceAlias, CompileError> {
        self.semantic
            .get(id)
            .ok_or_else(|| CompileError::StructuralViolation {
                site: ReferenceSite::SemanticSource(id.clone()),
                rule: "every planned semantic source has an alias",
            })
    }

    pub(crate) fn physical_alias(&self, id: &EntityId) -> Result<&SourceAlias, CompileError> {
        self.physical
            .get(id)
            .ok_or_else(|| CompileError::StructuralViolation {
                site: ReferenceSite::Source(id.clone()),
                rule: "every physical source has an alias",
            })
    }
}

fn indexed<'a>(
    prefixed: impl Iterator<Item = (&'a EntityId, String)>,
) -> BTreeMap<EntityId, SourceAlias> {
    prefixed
        .collect::<BTreeMap<_, _>>()
        .into_iter()
        .enumerate()
        .map(|(index, (id, prefix))| (id.clone(), SourceAlias(format!("{prefix}{index}"))))
        .collect()
}
