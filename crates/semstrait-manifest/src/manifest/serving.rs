use std::collections::{BTreeMap, BTreeSet};

use semstrait_common::{DataKindName, EntityId};

use super::semantic_signature::SemanticSignature;
use super::semantic_source::{RoleDefinition, SemanticSource};

#[derive(Clone, Copy)]
pub(crate) struct Serving<'a> {
    sources: &'a BTreeMap<EntityId, SemanticSource>,
    signatures: &'a BTreeMap<EntityId, SemanticSignature>,
}

impl<'a> Serving<'a> {
    pub(crate) const fn new(
        sources: &'a BTreeMap<EntityId, SemanticSource>,
        signatures: &'a BTreeMap<EntityId, SemanticSignature>,
    ) -> Self {
        Self {
            sources,
            signatures,
        }
    }

    pub(crate) const fn sources(&self) -> &'a BTreeMap<EntityId, SemanticSource> {
        self.sources
    }

    pub(crate) const fn signatures(&self) -> &'a BTreeMap<EntityId, SemanticSignature> {
        self.signatures
    }

    pub(crate) fn served(&self, id: &EntityId) -> BTreeSet<&'a EntityId> {
        let mut served = BTreeSet::new();
        let mut seen = BTreeSet::new();
        let mut pending = vec![id];
        while let Some(id) = pending.pop() {
            let Some(entry) = self.sources.get(id).filter(|_| seen.insert(id)) else {
                continue;
            };
            served.extend(
                (entry.semantic_signature_ids.iter())
                    .filter_map(|signature| self.signatures.get(signature))
                    .map(|signature| &signature.definition_id),
            );
            if let RoleDefinition::Grainset { grains, .. } = &entry.role_definition {
                pending.extend(grains);
            }
        }
        served
    }

    pub(crate) fn name(&self, id: &EntityId) -> Option<&'a DataKindName> {
        self.sources.get(id).map(|entry| &entry.name)
    }

    pub(crate) fn serves(&self, id: &EntityId, definition: &EntityId) -> bool {
        self.served(id).contains(definition)
    }

    pub(crate) fn maps_whole(&self, id: &EntityId, names: &BTreeSet<&EntityId>) -> bool {
        self.maps_whole_reading(id, names, &|_, _| Vec::new())
    }

    pub(crate) fn maps_whole_reading(
        &self,
        id: &EntityId,
        names: &BTreeSet<&EntityId>,
        computed_reads: ComputedReads<'_>,
    ) -> bool {
        let Some(entry) = self.sources.get(id) else {
            return false;
        };
        match &entry.role_definition {
            RoleDefinition::Grainset { grains, .. } => {
                let own: BTreeSet<&EntityId> = (entry.semantic_signature_ids.iter())
                    .filter_map(|signature| self.signatures.get(signature))
                    .map(|signature| &signature.definition_id)
                    .collect();
                let (computed, others): (BTreeSet<&EntityId>, BTreeSet<&EntityId>) =
                    names.iter().copied().partition(|name| own.contains(name));
                if computed.is_empty() {
                    return self.some_grain_maps_whole(grains, names, computed_reads);
                }
                computed.into_iter().all(|definition| {
                    let reads = computed_reads(id, definition);
                    !reads.is_empty()
                        && reads.iter().all(|read| {
                            let whole = read.iter().chain(others.iter().copied()).collect();
                            self.some_grain_maps_whole(grains, &whole, computed_reads)
                        })
                })
            }
            role => {
                names.is_subset(&self.served(id))
                    && (role.operands().into_iter()).all(|operand| {
                        let served = self.served(operand);
                        let part = names.iter().copied().filter(|name| served.contains(name));
                        self.maps_whole_reading(operand, &part.collect(), computed_reads)
                    })
            }
        }
    }

    fn some_grain_maps_whole(
        &self,
        grains: &[EntityId],
        names: &BTreeSet<&EntityId>,
        computed_reads: ComputedReads<'_>,
    ) -> bool {
        (grains.iter()).any(|grain| self.maps_whole_reading(grain, names, computed_reads))
    }
}

pub(crate) type ComputedReads<'r> = &'r dyn Fn(&EntityId, &EntityId) -> Vec<BTreeSet<EntityId>>;
