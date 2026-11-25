use crate::error::build::ModelBuildErrorKind;
use crate::error::parse::ParseErrorKind;
use crate::error::validate::ValidateErrorKind;
use crate::model::builder::dedup;
use crate::model::identity::{derive_root_id, resolve_entity_id, IdentityProfile, MintLedger};
use crate::model::types::data_kinds::{Dataset, Grainset, Joinset, Unionset};
use crate::model::types::entities::{AiContext, Dimension, Measure, Metric, Relationship};
use crate::model::SemanticModel;
use crate::validation::validate;
use crate::yaml::data_kinds::{YamlDataset, YamlGrainset, YamlJoinset, YamlUnionset};
use semstrait_common::diagnostic::{
    split_by_severity, Diagnostic, Diagnostics, Location, SourceId, Span,
};
use semstrait_common::identity::EntityNamespace;
use semstrait_common::EntityId;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct SemanticModelBuilder {
    profile: IdentityProfile,
    pub(crate) id: Option<EntityId>,
    name: String,
    description: Option<String>,
    ai_context: Option<AiContext>,
    labels: Vec<String>,
    pub(crate) datasets: BTreeMap<EntityId, Dataset>,
    pub(crate) grainsets: BTreeMap<EntityId, Grainset>,
    pub(crate) unionsets: BTreeMap<EntityId, Unionset>,
    pub(crate) joinsets: BTreeMap<EntityId, Joinset>,
    pub(crate) dimensions: BTreeMap<EntityId, Dimension>,
    pub(crate) measures: BTreeMap<EntityId, Measure>,
    pub(crate) metrics: BTreeMap<EntityId, Metric>,
    pub(crate) relationships: BTreeMap<EntityId, Relationship>,
    pub(crate) id_locations: BTreeMap<EntityId, Vec<Location>>,
    pub(crate) ledger: MintLedger,
    parse_warnings: Diagnostics<ParseErrorKind>,
}

impl SemanticModel {
    pub fn builder() -> SemanticModelBuilder {
        SemanticModelBuilder::default()
    }
}

fn synthetic_location() -> Location {
    Location::new(SourceId::unknown(), Span::unknown())
}

impl SemanticModelBuilder {
    pub(crate) fn with_profile(profile: IdentityProfile) -> Self {
        Self {
            profile,
            ..Self::default()
        }
    }

    pub fn id(mut self, id: impl Into<EntityId>) -> Self {
        if self.id.is_none() {
            self.id = Some(id.into());
        }
        self
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        if self.name.is_empty() {
            self.name = name.into();
        }
        self
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        if self.description.is_none() {
            self.description = Some(description.into());
        }
        self
    }

    pub fn ai_context(mut self, ai_context: AiContext) -> Self {
        if self.ai_context.is_none() {
            self.ai_context = Some(ai_context);
        }
        self
    }

    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.labels.push(label.into());
        self
    }

    pub fn labels(mut self, labels: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.labels.extend(labels.into_iter().map(Into::into));
        self
    }

    pub fn dataset(mut self, d: Dataset) -> Self {
        self.insert_dataset(d, synthetic_location());
        self
    }
    pub fn datasets(mut self, items: impl IntoIterator<Item = Dataset>) -> Self {
        for d in items {
            self.insert_dataset(d, synthetic_location());
        }
        self
    }
    pub fn grainset(mut self, g: Grainset) -> Self {
        self.insert_grainset(g, synthetic_location());
        self
    }
    pub fn grainsets(mut self, items: impl IntoIterator<Item = Grainset>) -> Self {
        for g in items {
            self.insert_grainset(g, synthetic_location());
        }
        self
    }
    pub fn unionset(mut self, u: Unionset) -> Self {
        self.insert_unionset(u, synthetic_location());
        self
    }
    pub fn unionsets(mut self, items: impl IntoIterator<Item = Unionset>) -> Self {
        for u in items {
            self.insert_unionset(u, synthetic_location());
        }
        self
    }
    pub fn joinset(mut self, j: Joinset) -> Self {
        self.insert_joinset(j, synthetic_location());
        self
    }
    pub fn joinsets(mut self, items: impl IntoIterator<Item = Joinset>) -> Self {
        for j in items {
            self.insert_joinset(j, synthetic_location());
        }
        self
    }
    pub fn dimension(mut self, d: Dimension) -> Self {
        self.insert_dimension(d, synthetic_location());
        self
    }
    pub fn dimensions(mut self, items: impl IntoIterator<Item = Dimension>) -> Self {
        for d in items {
            self.insert_dimension(d, synthetic_location());
        }
        self
    }
    pub fn measure(mut self, m: Measure) -> Self {
        self.insert_measure(m, synthetic_location());
        self
    }
    pub fn measures(mut self, items: impl IntoIterator<Item = Measure>) -> Self {
        for m in items {
            self.insert_measure(m, synthetic_location());
        }
        self
    }
    pub fn metric(mut self, m: Metric) -> Self {
        self.insert_metric(m, synthetic_location());
        self
    }
    pub fn metrics(mut self, items: impl IntoIterator<Item = Metric>) -> Self {
        for m in items {
            self.insert_metric(m, synthetic_location());
        }
        self
    }
    pub fn relationship(mut self, r: Relationship) -> Self {
        self.insert_relationship(r, synthetic_location());
        self
    }
    pub fn relationships(mut self, items: impl IntoIterator<Item = Relationship>) -> Self {
        for r in items {
            self.insert_relationship(r, synthetic_location());
        }
        self
    }

    pub(crate) fn dataset_at(mut self, d: YamlDataset, loc: Location) -> Self {
        let d = d.resolve(self.profile, &loc, &mut self.ledger);
        self.insert_dataset(d, loc);
        self
    }
    pub(crate) fn grainset_at(mut self, g: YamlGrainset, loc: Location) -> Self {
        let g = g.resolve(self.profile, &loc, &mut self.ledger);
        self.insert_grainset(g, loc);
        self
    }
    pub(crate) fn unionset_at(mut self, u: YamlUnionset, loc: Location) -> Self {
        let u = u.resolve(self.profile, &loc, &mut self.ledger);
        self.insert_unionset(u, loc);
        self
    }
    pub(crate) fn joinset_at(mut self, j: YamlJoinset, loc: Location) -> Self {
        let j = j.resolve(self.profile, &loc, &mut self.ledger);
        self.insert_joinset(j, loc);
        self
    }
    pub(crate) fn dimension_at(mut self, d: Dimension, loc: Location) -> Self {
        self.insert_dimension(d, loc);
        self
    }
    pub(crate) fn measure_at(mut self, m: Measure, loc: Location) -> Self {
        self.insert_measure(m, loc);
        self
    }
    pub(crate) fn metric_at(mut self, m: Metric, loc: Location) -> Self {
        self.insert_metric(m, loc);
        self
    }
    pub(crate) fn relationship_at(mut self, r: Relationship, loc: Location) -> Self {
        self.insert_relationship(r, loc);
        self
    }

    fn insert_dataset(&mut self, mut d: Dataset, loc: Location) {
        let id = resolve_entity_id(
            &mut d.body.base.id,
            EntityNamespace::DataKind,
            self.profile,
            &d.body.base.name,
            &loc,
            &mut self.ledger,
        );
        self.id_locations.entry(id.clone()).or_default().push(loc);
        self.datasets.insert(id, d);
    }
    fn insert_grainset(&mut self, mut g: Grainset, loc: Location) {
        let id = resolve_entity_id(
            &mut g.body.base.id,
            EntityNamespace::DataKind,
            self.profile,
            &g.body.base.name,
            &loc,
            &mut self.ledger,
        );
        self.id_locations.entry(id.clone()).or_default().push(loc);
        self.grainsets.insert(id, g);
    }
    fn insert_unionset(&mut self, mut u: Unionset, loc: Location) {
        let id = resolve_entity_id(
            &mut u.body.base.id,
            EntityNamespace::DataKind,
            self.profile,
            &u.body.base.name,
            &loc,
            &mut self.ledger,
        );
        self.id_locations.entry(id.clone()).or_default().push(loc);
        self.unionsets.insert(id, u);
    }
    fn insert_joinset(&mut self, mut j: Joinset, loc: Location) {
        let id = resolve_entity_id(
            &mut j.body.base.id,
            EntityNamespace::DataKind,
            self.profile,
            &j.body.base.name,
            &loc,
            &mut self.ledger,
        );
        self.id_locations.entry(id.clone()).or_default().push(loc);
        self.joinsets.insert(id, j);
    }
    fn insert_dimension(&mut self, mut d: Dimension, loc: Location) {
        let name = d.name().as_str().to_owned();
        let id = resolve_entity_id(
            d.id_mut(),
            EntityNamespace::Dimension,
            self.profile,
            &name,
            &loc,
            &mut self.ledger,
        );
        self.id_locations.entry(id.clone()).or_default().push(loc);
        self.dimensions.insert(id, d);
    }
    fn insert_measure(&mut self, mut m: Measure, loc: Location) {
        let name = m.name().as_str().to_owned();
        let id = resolve_entity_id(
            m.id_mut(),
            EntityNamespace::Measure,
            self.profile,
            &name,
            &loc,
            &mut self.ledger,
        );
        self.id_locations.entry(id.clone()).or_default().push(loc);
        self.measures.insert(id, m);
    }
    fn insert_metric(&mut self, mut m: Metric, loc: Location) {
        let name = m.name().as_str().to_owned();
        let id = resolve_entity_id(
            m.id_mut(),
            EntityNamespace::Metric,
            self.profile,
            &name,
            &loc,
            &mut self.ledger,
        );
        self.id_locations.entry(id.clone()).or_default().push(loc);
        self.metrics.insert(id, m);
    }
    fn insert_relationship(&mut self, mut r: Relationship, loc: Location) {
        let id = resolve_entity_id(
            &mut r.id,
            EntityNamespace::Relationship,
            self.profile,
            &r.name,
            &loc,
            &mut self.ledger,
        );
        self.id_locations.entry(id.clone()).or_default().push(loc);
        self.relationships.insert(id, r);
    }
}

impl SemanticModelBuilder {
    #[allow(dead_code)]
    pub(crate) fn name_view(&self) -> &str {
        &self.name
    }

    pub(crate) fn push_parse_warning(&mut self, diagnostic: Diagnostic<ParseErrorKind>) {
        self.parse_warnings.push(diagnostic);
    }
}

impl SemanticModelBuilder {
    pub fn build(
        mut self,
    ) -> Result<(SemanticModel, Diagnostics<ModelBuildErrorKind>), Diagnostics<ModelBuildErrorKind>>
    {
        let mut validate_diags: Diagnostics<ValidateErrorKind> = Vec::new();
        dedup::collect_duplicate_data_kinds(&self, &mut validate_diags);
        dedup::collect_duplicate_shared_semantics(&self, &mut validate_diags);
        dedup::collect_duplicate_entity_ids(&self, &mut validate_diags);
        dedup::collect_duplicate_relationship_names(&self, &mut validate_diags);
        dedup::collect_repeated_interface_member_paths(&self, &mut validate_diags);
        for path in self.ledger.missing_ids() {
            validate_diags.push(Diagnostic::new(ValidateErrorKind::MissingEntityId {
                path: path.clone(),
            }));
        }

        if self.id.is_none() {
            if self.profile == IdentityProfile::StrictRequireProvided {
                validate_diags.push(Diagnostic::new(ValidateErrorKind::MissingEntityId {
                    path: "<model root>".to_owned(),
                }));
            } else {
                self.id = Some(derive_root_id(&self.name));
            }
        }

        let mut lifted: Diagnostics<ModelBuildErrorKind> = std::mem::take(&mut self.parse_warnings)
            .into_iter()
            .map(|d| d.map_kind(ModelBuildErrorKind::Parse))
            .collect();

        let model = self.into_model();

        match validate(&model) {
            Ok(extra) => validate_diags.extend(extra),
            Err(extra) => validate_diags.extend(extra),
        }

        lifted.extend(
            validate_diags
                .into_iter()
                .map(|d| d.map_kind(ModelBuildErrorKind::Validate)),
        );

        let (errors, warnings) = split_by_severity(lifted);
        if errors.is_empty() {
            Ok((model, warnings))
        } else {
            let mut combined = errors;
            combined.extend(warnings);
            Err(combined)
        }
    }

    fn into_model(self) -> SemanticModel {
        SemanticModel {
            id: self.id,
            name: self.name,
            description: self.description,
            ai_context: self.ai_context,
            labels: self.labels,
            datasets: self.datasets.into(),
            grainsets: self.grainsets.into(),
            unionsets: self.unionsets.into(),
            joinsets: self.joinsets.into(),
            dimensions: self.dimensions.into(),
            measures: self.measures.into(),
            metrics: self.metrics.into(),
            relationships: self.relationships.into(),
        }
    }
}
