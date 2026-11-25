use std::collections::{BTreeMap, BTreeSet};

use semstrait_common::EntityId;
use semstrait_common::{DataKindName, Granularity, SemanticsName};
use semstrait_ir::{
    Cardinality, Integrity, JoinType, JoinsetJoinType, MultisetOp, PhysicalExpr, TemporalType,
};
use semstrait_manifest::{
    Coverage, KeptField, PhysicalExprId, RoleDefinition, SemanticBitmask, SemanticDefinition,
    SemanticManifest, SemanticSignature, SemanticSource, SourceAlias, SourceColumn,
    StorageProperties,
};
use semstrait_sql::expr::{Expr, Literal, NamedReference};

use crate::tree::column_leaves;

pub(crate) mod sealed {
    pub trait Sealed {}
}

pub trait ReadModel: sealed::Sealed {
    fn bit_of(&self, name: &str) -> Option<u32>;
    fn name_of(&self, bit: u32) -> Option<SemanticsName>;
    fn public_sources(&self) -> BTreeMap<DataKindName, (SourceAlias, SemanticBitmask)>;
    fn originated_by(
        &self,
        root: &SourceAlias,
    ) -> BTreeMap<DataKindName, (SourceAlias, SemanticBitmask)>;
    fn operand_closure(&self, source: &SourceAlias) -> BTreeSet<SourceAlias>;
    fn operands(&self, source: &SourceAlias) -> Vec<SourceAlias>;
    fn name(&self, source: &SourceAlias) -> Option<DataKindName>;
    fn originators(&self, source: &SourceAlias) -> Vec<DataKindName>;
    fn signature(&self, source: &SourceAlias, bit: u32) -> Option<SemanticSignature>;
    fn coverage_of(&self, source: &SourceAlias, bit: u32) -> Coverage;
    fn definition(&self, bit: u32) -> Option<SemanticDefinition>;
    fn definition_of(&self, id: &EntityId) -> Option<SemanticDefinition>;
    fn source_kind(&self, source: &SourceAlias) -> Option<RoleDefinition>;
    fn alias(&self, name: &str) -> Option<SourceAlias>;
    fn mask(&self, source: &SourceAlias) -> SemanticBitmask;
    fn granularity(&self, source: &SourceAlias) -> Option<Granularity>;
    fn time_axis(&self, source: &SourceAlias) -> Option<TimeAxis>;
    fn keysets(&self, source: &SourceAlias) -> Vec<SemanticBitmask>;
    fn keyset_named(&self, source: &SourceAlias, named: &EntityId) -> Option<SemanticBitmask>;
    fn expr_tree(
        &self,
        source: &SourceAlias,
        root: &PhysicalExprId,
        kept: &[KeptField],
    ) -> Option<PhysicalExpr>;
    fn linked(
        &self,
        source: &SourceAlias,
        root: &PhysicalExprId,
    ) -> Option<Vec<SemanticDefinition>>;
    fn aggregate_definitions(
        &self,
        source: &SourceAlias,
        root: &PhysicalExprId,
        kept: &[PhysicalExprId],
    ) -> Option<Vec<SemanticDefinition>>;
    fn reached(
        &self,
        source: &SourceAlias,
        roots: &[PhysicalExprId],
    ) -> Option<Vec<SemanticDefinition>>;
    fn physical_source(&self, source: &SourceAlias) -> Option<PhysicalSource>;
    fn relationship(&self, source: &SourceAlias) -> Option<Relationship>;
}

pub(crate) fn is_dataset(read_model: &(impl ReadModel + ?Sized), source: &SourceAlias) -> bool {
    matches!(
        read_model.source_kind(source),
        Some(RoleDefinition::Dataset { .. })
    )
}

pub(crate) fn is_grainset(read_model: &(impl ReadModel + ?Sized), source: &SourceAlias) -> bool {
    matches!(
        read_model.source_kind(source),
        Some(RoleDefinition::Grainset { .. })
    )
}

pub(crate) fn is_unionset(read_model: &(impl ReadModel + ?Sized), source: &SourceAlias) -> bool {
    matches!(
        read_model.source_kind(source),
        Some(RoleDefinition::Unionset { .. })
    )
}

pub(crate) fn is_unique(read_model: &(impl ReadModel + ?Sized), source: &SourceAlias) -> bool {
    matches!(
        read_model.source_kind(source),
        Some(RoleDefinition::Unionset {
            mode: MultisetOp::UnionDistinct,
            ..
        })
    )
}

pub(crate) fn literal_on(
    read_model: &(impl ReadModel + ?Sized),
    source: &SourceAlias,
    bit: u32,
) -> Option<Literal> {
    match read_model.source_kind(source)? {
        RoleDefinition::Dataset { .. } => {
            let signature = read_model.signature(source, bit)?;
            match read_model.expr_tree(source, &signature.expression_id, &[])? {
                Expr::Literal(literal) => Some(literal),
                _ => None,
            }
        }
        RoleDefinition::Joinset { .. } => {
            let relationship = read_model.relationship(source)?;
            agreed(
                (relationship.ends().into_iter())
                    .filter(|(end, _)| read_model.mask(end).contains(bit))
                    .map(|(end, preserved)| preserved.then(|| literal_on(read_model, end, bit))?),
            )
        }
        RoleDefinition::Unionset { .. } | RoleDefinition::Grainset { .. } => agreed(
            (read_model.operands(source).iter())
                .map(|operand| literal_on(read_model, operand, bit)),
        ),
        _ => None,
    }
}

fn agreed(literals: impl IntoIterator<Item = Option<Literal>>) -> Option<Literal> {
    let mut literals = literals.into_iter();
    let first = literals.next()??;
    literals
        .all(|literal| literal.as_ref() == Some(&first))
        .then_some(first)
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeAxis {
    Timeseries(u32),
    Events(u32),
    Snapshot(u32),
    Scd { valid_from: u32, valid_to: u32 },
}

impl TimeAxis {
    pub(crate) fn dimensions(self) -> SemanticBitmask {
        match self {
            Self::Timeseries(bit) | Self::Events(bit) | Self::Snapshot(bit) => {
                SemanticBitmask::from_positions([bit])
            }
            Self::Scd {
                valid_from,
                valid_to,
            } => SemanticBitmask::from_positions([valid_from, valid_to]),
        }
    }

    pub(crate) fn single(self) -> Option<u32> {
        match self {
            Self::Timeseries(bit) | Self::Events(bit) | Self::Snapshot(bit) => Some(bit),
            Self::Scd { .. } => None,
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalSource {
    pub name: String,
    pub storage: StorageProperties,
    pub relation: String,
    pub columns: Vec<SourceColumn>,
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct Relationship {
    pub from: SourceAlias,
    pub to: SourceAlias,
    pub join_type: JoinsetJoinType,
    pub on: Option<PhysicalExpr>,
    pub filter: Option<PhysicalExpr>,
    pub cardinality: Cardinality,
    pub integrity: Integrity,
}

impl Relationship {
    pub(crate) fn ends(&self) -> [(&SourceAlias, bool); 2] {
        let [from, to] = match self.join_type {
            JoinsetJoinType::Join(JoinType::Inner) | JoinsetJoinType::Cross => [true, true],
            JoinsetJoinType::Join(JoinType::Left) => [true, false],
            JoinsetJoinType::Join(JoinType::Right) => [false, true],
            JoinsetJoinType::Join(_) => [false, false],
        };
        [(&self.from, from), (&self.to, to)]
    }

    pub(crate) fn repeats(&self, end: &SourceAlias) -> bool {
        match self.cardinality {
            Cardinality::OneToOne => false,
            Cardinality::OneToMany => end == &self.from,
            Cardinality::ManyToOne => end == &self.to,
            _ => true,
        }
    }

    pub(crate) fn may_read_alone(&self, read: &SourceAlias) -> bool {
        let keeps_every_row = match self.join_type {
            JoinsetJoinType::Join(JoinType::Inner) => {
                matches!(self.integrity, Integrity::Enforced | Integrity::Assumed)
            }
            JoinsetJoinType::Join(JoinType::Left) => read == &self.from,
            JoinsetJoinType::Join(JoinType::Right) => read == &self.to,
            _ => false,
        };
        keeps_every_row && self.filter.is_none()
    }

    pub(crate) fn joins_as(&self, other: &Self) -> bool {
        (self.join_type, &self.on, &self.filter) == (other.join_type, &other.on, &other.filter)
    }

    pub(crate) fn condition_names(&self) -> Vec<&str> {
        (self.on.iter())
            .flat_map(column_leaves)
            .map(NamedReference::name)
            .collect()
    }

    pub(crate) fn trees(&self) -> impl Iterator<Item = &PhysicalExpr> {
        self.on.iter().chain(&self.filter)
    }
}

pub(crate) struct ManifestReadModel<'m> {
    manifest: &'m SemanticManifest,
}

impl<'m> ManifestReadModel<'m> {
    pub(crate) fn new(manifest: &'m SemanticManifest) -> Self {
        Self { manifest }
    }

    fn source(&self, alias: &SourceAlias) -> Option<&'m SemanticSource> {
        (self.manifest.semantic_sources.values()).find(|source| &source.alias == alias)
    }

    fn definition_at(&self, bit: u32) -> Option<&'m SemanticDefinition> {
        (self.manifest.semantic_definitions.entries.values())
            .find(|definition| definition.bit_position == bit)
    }

    fn signatures(&self, source: &SourceAlias) -> Option<Vec<&'m SemanticSignature>> {
        Some(self.manifest.signatures_of(self.source(source)?).collect())
    }

    fn keyset_masks(&self, source: &SourceAlias) -> Vec<(&'m EntityId, SemanticBitmask)> {
        let signatures = self.signatures(source).unwrap_or_default();
        let keysets: BTreeSet<&EntityId> = (signatures.iter().copied())
            .flat_map(|signature| &signature.keysets)
            .collect();
        (keysets.into_iter())
            .map(|keyset| {
                let members = (signatures.iter())
                    .filter(|signature| signature.keysets.contains(keyset))
                    .map(|signature| &signature.definition_id);
                (keyset, self.manifest.semantic_definitions.mask_of(members))
            })
            .collect()
    }

    fn definitions_of(&self, signatures: &[&SemanticSignature]) -> Option<Vec<SemanticDefinition>> {
        let definitions = &self.manifest.semantic_definitions.entries;
        (signatures.iter())
            .map(|signature| definitions.get(&signature.definition_id).cloned())
            .collect()
    }
}

impl sealed::Sealed for ManifestReadModel<'_> {}

impl ReadModel for ManifestReadModel<'_> {
    fn bit_of(&self, name: &str) -> Option<u32> {
        (self.manifest.semantic_definitions.by_name(name)).map(|definition| definition.bit_position)
    }

    fn name_of(&self, bit: u32) -> Option<SemanticsName> {
        self.definition_at(bit)
            .map(|definition| definition.name.clone())
    }

    fn public_sources(&self) -> BTreeMap<DataKindName, (SourceAlias, SemanticBitmask)> {
        (self.manifest.semantic_sources.values())
            .filter(|source| source.public)
            .map(|source| {
                let held = (source.alias.clone(), source.semantic_bitmask.clone());
                (source.name.clone(), held)
            })
            .collect()
    }

    fn originated_by(
        &self,
        root: &SourceAlias,
    ) -> BTreeMap<DataKindName, (SourceAlias, SemanticBitmask)> {
        let mut origins: Vec<&EntityId> =
            self.source(root).map(|root| &root.id).into_iter().collect();
        let mut units = BTreeMap::new();
        while let Some(origin) = origins.pop() {
            let originated = (self.manifest.semantic_sources.values())
                .filter(|unit| unit.originators.contains(origin));
            for unit in originated {
                let held = (unit.alias.clone(), unit.semantic_bitmask.clone());
                if units.insert(unit.name.clone(), held).is_none() {
                    origins.push(&unit.id);
                }
            }
        }
        units
    }

    fn operand_closure(&self, source: &SourceAlias) -> BTreeSet<SourceAlias> {
        let mut operands: Vec<&EntityId> = (self.source(source))
            .map(|source| source.role_definition.operands())
            .unwrap_or_default();
        let mut closure = BTreeSet::new();
        while let Some(id) = operands.pop() {
            if let Some(operand) = self.manifest.semantic_sources.get(id) {
                if closure.insert(operand.alias.clone()) {
                    operands.extend(operand.role_definition.operands());
                }
            }
        }
        closure
    }

    fn operands(&self, source: &SourceAlias) -> Vec<SourceAlias> {
        let mut operands: Vec<&SemanticSource> = (self.source(source).into_iter())
            .flat_map(|source| source.role_definition.operands())
            .filter_map(|id| self.manifest.semantic_sources.get(id))
            .collect();
        operands.sort_by(|left, right| left.name.cmp(&right.name));
        (operands.into_iter())
            .map(|operand| operand.alias.clone())
            .collect()
    }

    fn name(&self, source: &SourceAlias) -> Option<DataKindName> {
        self.source(source).map(|source| source.name.clone())
    }

    fn originators(&self, source: &SourceAlias) -> Vec<DataKindName> {
        let Some(source) = self.source(source) else {
            return Vec::new();
        };
        (source.originators.iter())
            .filter_map(|id| self.manifest.semantic_sources.get(id))
            .map(|originator| originator.name.clone())
            .collect()
    }

    fn signature(&self, source: &SourceAlias, bit: u32) -> Option<SemanticSignature> {
        let definition = self.definition_at(bit)?;
        (self.manifest.signatures_of(self.source(source)?))
            .find(|signature| signature.definition_id == definition.id)
            .cloned()
    }

    fn coverage_of(&self, source: &SourceAlias, bit: u32) -> Coverage {
        match (self.definition_at(bit), self.source(source)) {
            (Some(definition), Some(source)) => {
                self.manifest.coverage_of(&definition.id, &source.id)
            }
            _ => Coverage::NullFill,
        }
    }

    fn definition(&self, bit: u32) -> Option<SemanticDefinition> {
        self.definition_at(bit).cloned()
    }

    fn definition_of(&self, id: &EntityId) -> Option<SemanticDefinition> {
        self.manifest.semantic_definitions.get(id).cloned()
    }

    fn source_kind(&self, source: &SourceAlias) -> Option<RoleDefinition> {
        self.source(source)
            .map(|source| source.role_definition.clone())
    }

    fn alias(&self, name: &str) -> Option<SourceAlias> {
        (self.manifest.semantic_sources.values())
            .find(|source| source.name.as_str() == name)
            .map(|source| source.alias.clone())
    }

    fn mask(&self, source: &SourceAlias) -> SemanticBitmask {
        (self.source(source)).map_or_else(SemanticBitmask::empty, |source| {
            source.semantic_bitmask.clone()
        })
    }

    fn granularity(&self, source: &SourceAlias) -> Option<Granularity> {
        self.source(source)?.temporal.as_ref()?.granularity
    }

    fn time_axis(&self, source: &SourceAlias) -> Option<TimeAxis> {
        let bit = |id: &EntityId| {
            (self.manifest.semantic_definitions.get(id)).map(|definition| definition.bit_position)
        };
        Some(
            match &self.source(source)?.temporal.as_ref()?.temporal_type {
                TemporalType::Timeseries { occurred_at } => TimeAxis::Timeseries(bit(occurred_at)?),
                TemporalType::Events { event_time } => TimeAxis::Events(bit(event_time)?),
                TemporalType::Snapshot { snapshotted_at } => {
                    TimeAxis::Snapshot(bit(snapshotted_at)?)
                }
                TemporalType::Scd {
                    valid_from,
                    valid_to,
                    ..
                } => TimeAxis::Scd {
                    valid_from: bit(valid_from)?,
                    valid_to: bit(valid_to)?,
                },
                _ => return None,
            },
        )
    }

    fn keysets(&self, source: &SourceAlias) -> Vec<SemanticBitmask> {
        (self.keyset_masks(source).into_iter())
            .map(|(_, mask)| mask)
            .collect()
    }

    fn keyset_named(&self, source: &SourceAlias, named: &EntityId) -> Option<SemanticBitmask> {
        let keysets = &self.manifest.keysets;
        let name = &keysets.get(named)?.name;
        (self.keyset_masks(source).into_iter())
            .find(|(keyset, _)| (keysets.get(*keyset)).is_some_and(|keyset| &keyset.name == name))
            .map(|(_, mask)| mask)
    }

    fn expr_tree(
        &self,
        source: &SourceAlias,
        root: &PhysicalExprId,
        kept: &[KeptField],
    ) -> Option<PhysicalExpr> {
        (self.manifest.expressions).reassemble(root, &self.signatures(source)?, kept)
    }

    fn linked(
        &self,
        source: &SourceAlias,
        root: &PhysicalExprId,
    ) -> Option<Vec<SemanticDefinition>> {
        let signatures = self.signatures(source)?;
        self.definitions_of(&(self.manifest.expressions).linked(root, &signatures)?)
    }

    fn aggregate_definitions(
        &self,
        source: &SourceAlias,
        root: &PhysicalExprId,
        kept: &[PhysicalExprId],
    ) -> Option<Vec<SemanticDefinition>> {
        let signatures = self.signatures(source)?;
        let held = (self.manifest.expressions).aggregate_signatures(root, &signatures, kept)?;
        self.definitions_of(&held)
    }

    fn reached(
        &self,
        source: &SourceAlias,
        roots: &[PhysicalExprId],
    ) -> Option<Vec<SemanticDefinition>> {
        let signatures = self.signatures(source)?;
        self.definitions_of(&(self.manifest.expressions).reached(roots, &signatures)?)
    }

    fn physical_source(&self, source: &SourceAlias) -> Option<PhysicalSource> {
        let RoleDefinition::Dataset { source_id, .. } = &self.source(source)?.role_definition
        else {
            return None;
        };
        let stored = self.manifest.sources.get(source_id)?;
        Some(PhysicalSource {
            name: stored.name.clone(),
            storage: stored.storage.clone(),
            relation: stored.locator(),
            columns: stored.schema().columns.clone(),
        })
    }

    fn relationship(&self, source: &SourceAlias) -> Option<Relationship> {
        let RoleDefinition::Joinset {
            from,
            to,
            on_expr_id,
            filter_expr_id,
            join_type,
            cardinality,
            integrity,
            ..
        } = &self.source(source)?.role_definition
        else {
            return None;
        };
        let alias = |id| Some(self.manifest.semantic_sources.get(id)?.alias.clone());
        let tree = |root: &Option<PhysicalExprId>| {
            (root.as_ref()).map_or(Some(None), |root| {
                self.expr_tree(source, root, &[]).map(Some)
            })
        };
        Some(Relationship {
            from: alias(from)?,
            to: alias(to)?,
            join_type: *join_type,
            on: tree(on_expr_id)?,
            filter: tree(filter_expr_id)?,
            cardinality: *cardinality,
            integrity: *integrity,
        })
    }
}
