use crate::model::entity_btree_map::{EntityBTreeMap, NamedEntity};
use crate::model::identity::{
    interface_member_path, resolve_entity_id, IdentityProfile, MintLedger,
};
use crate::model::types::entities::dimension::DimensionSite;
use crate::model::types::entities::filter::DataKindFilter;
use crate::model::types::entities::measure::MeasureSite;
use crate::model::types::entities::metric::MetricSite;
use crate::model::types::entities::semantic_body::{AuthoringBody, SemanticSite};
use bon::Builder;
use semstrait_common::diagnostic::{Location, SourceId, Span};
use semstrait_common::identity::EntityNamespace;
use semstrait_common::EntityId;
use semstrait_common::SemanticsName;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, Builder)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
#[builder(
    start_fn = builder,
    finish_fn = build,
    state_mod(name = semantic_interface_builder, vis = "pub(crate)"),
)]
pub struct SemanticInterface {
    #[cfg_attr(
        feature = "schemars",
        schemars(
            description = "The Dimensions this data kind exposes, each either declared inline or referenced from the pool."
        )
    )]
    #[serde(default, skip_serializing_if = "EntityBTreeMap::is_empty")]
    #[builder(field)]
    pub dimensions: EntityBTreeMap<DimensionSite>,

    #[serde(default, skip_serializing_if = "EntityBTreeMap::is_empty")]
    #[builder(field)]
    pub measures: EntityBTreeMap<MeasureSite>,

    #[serde(default, skip_serializing_if = "EntityBTreeMap::is_empty")]
    #[builder(field)]
    pub metrics: EntityBTreeMap<MetricSite>,

    #[cfg_attr(
        feature = "schemars",
        schemars(
            description = "Named filters this data kind declares; a Measure's or Metric's own `filter:` is separate."
        )
    )]
    #[serde(default, skip_serializing_if = "EntityBTreeMap::is_empty")]
    #[builder(field)]
    pub filters: EntityBTreeMap<DataKindFilter>,
}

impl SemanticInterface {
    pub fn is_empty(&self) -> bool {
        self.dimensions.is_empty()
            && self.measures.is_empty()
            && self.metrics.is_empty()
            && self.filters.is_empty()
    }

    pub fn keysets(&self) -> BTreeMap<String, Vec<SemanticsName>> {
        let mut sites: Vec<&DimensionSite> = self.dimensions.values().collect();
        sites.sort_by(|a, b| a.name().cmp(b.name()));

        let mut out: BTreeMap<String, Vec<SemanticsName>> = BTreeMap::new();
        for site in sites {
            for keyset in site.keyness().keyset_membership() {
                out.entry(keyset).or_default().push(site.name().clone());
            }
        }
        out
    }
}

pub(crate) trait InterfaceMember: NamedEntity {
    fn id_mut(&mut self) -> &mut Option<EntityId>;
}

impl<I: AuthoringBody, R: AuthoringBody> InterfaceMember for SemanticSite<I, R> {
    fn id_mut(&mut self) -> &mut Option<EntityId> {
        &mut self.body_mut().id
    }
}

impl InterfaceMember for DataKindFilter {
    fn id_mut(&mut self) -> &mut Option<EntityId> {
        &mut self.id
    }
}

fn provisional_key<T: InterfaceMember>(member: &T) -> EntityId {
    member
        .entity_id()
        .unwrap_or_else(|| member.entity_name())
        .to_owned()
}

pub(crate) fn resolve_collection<T: InterfaceMember>(
    members: impl IntoIterator<Item = T>,
    data_kind: &str,
    collection: &str,
    profile: IdentityProfile,
    location: &Location,
    ledger: &mut MintLedger,
) -> EntityBTreeMap<T> {
    let mut resolved: BTreeMap<EntityId, T> = BTreeMap::new();
    for mut member in members {
        let path = interface_member_path(data_kind, collection, member.entity_name());
        let id = resolve_entity_id(
            member.id_mut(),
            EntityNamespace::InterfaceMember,
            profile,
            &path,
            location,
            ledger,
        );
        resolved.insert(id, member);
    }
    resolved.into()
}

fn rekey<T: InterfaceMember>(
    collection: &mut EntityBTreeMap<T>,
    data_kind: &str,
    label: &str,
    profile: IdentityProfile,
    location: &Location,
    ledger: &mut MintLedger,
) -> EntityBTreeMap<T> {
    let members = std::mem::take(collection).into_inner().into_values();
    resolve_collection(members, data_kind, label, profile, location, ledger)
}

impl SemanticInterface {
    pub(crate) fn resolve_member_ids(
        &mut self,
        data_kind: &str,
        profile: IdentityProfile,
        location: &Location,
        ledger: &mut MintLedger,
    ) {
        self.dimensions = rekey(
            &mut self.dimensions,
            data_kind,
            "dimensions",
            profile,
            location,
            ledger,
        );
        self.measures = rekey(
            &mut self.measures,
            data_kind,
            "measures",
            profile,
            location,
            ledger,
        );
        self.metrics = rekey(
            &mut self.metrics,
            data_kind,
            "metrics",
            profile,
            location,
            ledger,
        );
        self.filters = rekey(
            &mut self.filters,
            data_kind,
            "filters",
            profile,
            location,
            ledger,
        );
    }

    pub(crate) fn resolve_member_ids_for_code_build(&mut self, data_kind: &str) {
        self.resolve_member_ids(
            data_kind,
            IdentityProfile::ConvenienceGenerateMissing,
            &Location::new(SourceId::unknown(), Span::unknown()),
            &mut MintLedger::default(),
        );
    }
}

impl SemanticInterface {
    pub fn with_dimension(mut self, e: DimensionSite) -> Self {
        self.dimensions.insert(provisional_key(&e), e);
        self
    }
    pub fn with_dimensions(mut self, items: impl IntoIterator<Item = DimensionSite>) -> Self {
        self.dimensions = EntityBTreeMap::default();
        for e in items {
            self = self.with_dimension(e);
        }
        self
    }
    pub fn with_measure(mut self, e: MeasureSite) -> Self {
        self.measures.insert(provisional_key(&e), e);
        self
    }
    pub fn with_measures(mut self, items: impl IntoIterator<Item = MeasureSite>) -> Self {
        self.measures = EntityBTreeMap::default();
        for e in items {
            self = self.with_measure(e);
        }
        self
    }
    pub fn with_metric(mut self, e: MetricSite) -> Self {
        self.metrics.insert(provisional_key(&e), e);
        self
    }
    pub fn with_metrics(mut self, items: impl IntoIterator<Item = MetricSite>) -> Self {
        self.metrics = EntityBTreeMap::default();
        for e in items {
            self = self.with_metric(e);
        }
        self
    }
    pub fn with_filter(mut self, f: DataKindFilter) -> Self {
        self.filters.insert(provisional_key(&f), f);
        self
    }
    pub fn with_filters(mut self, items: impl IntoIterator<Item = DataKindFilter>) -> Self {
        self.filters = EntityBTreeMap::default();
        for f in items {
            self = self.with_filter(f);
        }
        self
    }
}

impl<S: semantic_interface_builder::State> SemanticInterfaceBuilder<S> {
    pub fn dimensions(mut self, items: impl IntoIterator<Item = DimensionSite>) -> Self {
        self.dimensions = replace_sites(items);
        self
    }
    pub fn measures(mut self, items: impl IntoIterator<Item = MeasureSite>) -> Self {
        self.measures = replace_sites(items);
        self
    }
    pub fn metrics(mut self, items: impl IntoIterator<Item = MetricSite>) -> Self {
        self.metrics = replace_sites(items);
        self
    }
    pub fn filters(mut self, items: impl IntoIterator<Item = DataKindFilter>) -> Self {
        self.filters = SemanticInterface::default().with_filters(items).filters;
        self
    }
    pub fn dimension(mut self, e: DimensionSite) -> Self {
        self.dimensions.insert(provisional_key(&e), e);
        self
    }
    pub fn measure(mut self, e: MeasureSite) -> Self {
        self.measures.insert(provisional_key(&e), e);
        self
    }
    pub fn metric(mut self, e: MetricSite) -> Self {
        self.metrics.insert(provisional_key(&e), e);
        self
    }
    pub fn filter(mut self, f: DataKindFilter) -> Self {
        self.filters.insert(provisional_key(&f), f);
        self
    }
}

fn replace_sites<I: AuthoringBody, R: AuthoringBody>(
    items: impl IntoIterator<Item = SemanticSite<I, R>>,
) -> EntityBTreeMap<SemanticSite<I, R>> {
    let mut out = EntityBTreeMap::default();
    for site in items {
        out.insert(provisional_key(&site), site);
    }
    out
}
