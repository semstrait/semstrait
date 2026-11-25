use crate::model::types::data_kinds::base::{ComplexExtras, DataKindBase, LeafExtras};
use crate::model::types::data_kinds::dataset::{Dataset, DatasetBody, NestedDataset};
use crate::model::types::data_kinds::grainset::{Grainset, GrainsetBody, NestedGrainset};
use crate::model::types::data_kinds::joinset::{Joinset, JoinsetBody, NestedJoinset};
use crate::model::types::data_kinds::storage::{PartitionDef, StorageConfig, StorageFormat};
use crate::model::types::data_kinds::unionset::{
    NestedUnionset, UnionMode, Unionset, UnionsetBody,
};
use crate::model::types::entities::ai::AiContext;
use crate::model::types::entities::dimension::DimensionSite;
use crate::model::types::entities::filter::DataKindFilter;
use crate::model::types::entities::mapping::SemanticMapping;
use crate::model::types::entities::measure::MeasureSite;
use crate::model::types::entities::metric::MetricSite;
use crate::model::types::entities::relationship::Relationship;
use crate::model::types::entities::semantic_interface::SemanticInterface;
use crate::model::types::entities::temporal::TemporalShape;
use bon::bon;
use std::mem;

#[bon]
impl Dataset {
    #[builder(
        builder_type(name = DatasetBuilder, vis = "pub"),
        state_mod(name = dataset_builder, vis = "pub(crate)"),
        finish_fn = build,
    )]
    pub fn new(
        #[builder(start_fn, into)] name: String,
        #[builder(field)] extras: LeafExtras,
        #[builder(field)] mut semantic_interface: SemanticInterface,
        #[builder(into)] description: Option<String>,
        ai_context: Option<AiContext>,
    ) -> Self {
        semantic_interface.resolve_member_ids_for_code_build(&name);
        Dataset {
            body: DatasetBody {
                base: DataKindBase::rooted(name, extras),
            },
            description,
            ai_context,
            semantic_interface,
        }
    }
}

impl<S: dataset_builder::State> DatasetBuilder<S> {
    pub fn extras(mut self, e: LeafExtras) -> Self {
        self.extras = e;
        self
    }
    pub fn semantic_interface(mut self, s: SemanticInterface) -> Self {
        self.semantic_interface = s;
        self
    }

    pub fn catalog(mut self, c: impl Into<String>) -> Self {
        self.extras = mem::take(&mut self.extras).with_catalog(c);
        self
    }
    pub fn storage(mut self, s: StorageConfig) -> Self {
        self.extras = mem::take(&mut self.extras).with_storage(s);
        self
    }
    pub fn format(mut self, f: StorageFormat) -> Self {
        self.extras = mem::take(&mut self.extras).with_format(f);
        self
    }
    pub fn path(mut self, p: impl Into<String>) -> Self {
        self.extras = mem::take(&mut self.extras).with_path(p);
        self
    }
    pub fn paths(mut self, items: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.extras = mem::take(&mut self.extras).with_paths(items);
        self
    }
    pub fn table(mut self, t: impl Into<String>) -> Self {
        self.extras = mem::take(&mut self.extras).with_table(t);
        self
    }
    pub fn tables(mut self, items: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.extras = mem::take(&mut self.extras).with_tables(items);
        self
    }
    pub fn partition_def(mut self, entry: PartitionDef) -> Self {
        self.extras = mem::take(&mut self.extras).with_partition_def(entry);
        self
    }
    pub fn semantic_mapping(mut self, m: SemanticMapping) -> Self {
        self.extras = mem::take(&mut self.extras).with_semantic_mapping(m);
        self
    }
    pub fn temporal(mut self, t: TemporalShape) -> Self {
        self.extras = mem::take(&mut self.extras).with_temporal(t);
        self
    }

    pub fn dimension(mut self, e: DimensionSite) -> Self {
        self.semantic_interface = mem::take(&mut self.semantic_interface).with_dimension(e);
        self
    }
    pub fn dimensions(mut self, items: impl IntoIterator<Item = DimensionSite>) -> Self {
        self.semantic_interface = mem::take(&mut self.semantic_interface).with_dimensions(items);
        self
    }
    pub fn measure(mut self, e: MeasureSite) -> Self {
        self.semantic_interface = mem::take(&mut self.semantic_interface).with_measure(e);
        self
    }
    pub fn measures(mut self, items: impl IntoIterator<Item = MeasureSite>) -> Self {
        self.semantic_interface = mem::take(&mut self.semantic_interface).with_measures(items);
        self
    }
    pub fn metric(mut self, e: MetricSite) -> Self {
        self.semantic_interface = mem::take(&mut self.semantic_interface).with_metric(e);
        self
    }
    pub fn metrics(mut self, items: impl IntoIterator<Item = MetricSite>) -> Self {
        self.semantic_interface = mem::take(&mut self.semantic_interface).with_metrics(items);
        self
    }
    pub fn filter(mut self, f: DataKindFilter) -> Self {
        self.semantic_interface = mem::take(&mut self.semantic_interface).with_filter(f);
        self
    }
    pub fn filters(mut self, items: impl IntoIterator<Item = DataKindFilter>) -> Self {
        self.semantic_interface = mem::take(&mut self.semantic_interface).with_filters(items);
        self
    }
}

#[bon]
impl NestedDataset {
    #[builder(
        builder_type(name = NestedDatasetBuilder, vis = "pub"),
        state_mod(name = nested_dataset_builder, vis = "pub(crate)"),
        finish_fn = build,
    )]
    pub fn new(
        #[builder(start_fn, into)] name: String,
        #[builder(field)] extras: LeafExtras,
    ) -> Self {
        NestedDataset {
            body: DatasetBody {
                base: DataKindBase::nested(name, extras),
            },
        }
    }
}

impl<S: nested_dataset_builder::State> NestedDatasetBuilder<S> {
    pub fn extras(mut self, e: LeafExtras) -> Self {
        self.extras = e;
        self
    }

    pub fn catalog(mut self, c: impl Into<String>) -> Self {
        self.extras = mem::take(&mut self.extras).with_catalog(c);
        self
    }
    pub fn storage(mut self, s: StorageConfig) -> Self {
        self.extras = mem::take(&mut self.extras).with_storage(s);
        self
    }
    pub fn format(mut self, f: StorageFormat) -> Self {
        self.extras = mem::take(&mut self.extras).with_format(f);
        self
    }
    pub fn path(mut self, p: impl Into<String>) -> Self {
        self.extras = mem::take(&mut self.extras).with_path(p);
        self
    }
    pub fn paths(mut self, items: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.extras = mem::take(&mut self.extras).with_paths(items);
        self
    }
    pub fn table(mut self, t: impl Into<String>) -> Self {
        self.extras = mem::take(&mut self.extras).with_table(t);
        self
    }
    pub fn tables(mut self, items: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.extras = mem::take(&mut self.extras).with_tables(items);
        self
    }
    pub fn partition_def(mut self, entry: PartitionDef) -> Self {
        self.extras = mem::take(&mut self.extras).with_partition_def(entry);
        self
    }
    pub fn semantic_mapping(mut self, m: SemanticMapping) -> Self {
        self.extras = mem::take(&mut self.extras).with_semantic_mapping(m);
        self
    }
    pub fn temporal(mut self, t: TemporalShape) -> Self {
        self.extras = mem::take(&mut self.extras).with_temporal(t);
        self
    }
}

#[bon]
impl Grainset {
    #[builder(
        builder_type(name = GrainsetBuilder, vis = "pub"),
        state_mod(name = grainset_builder, vis = "pub(crate)"),
        finish_fn = build,
    )]
    pub fn new(
        #[builder(start_fn, into)] name: String,
        #[builder(field)] datasets: Vec<NestedDataset>,
        #[builder(field)] unionsets: Vec<NestedUnionset>,
        #[builder(field)] joinsets: Vec<NestedJoinset>,
        #[builder(field)] extras: ComplexExtras,
        #[builder(field)] mut semantic_interface: SemanticInterface,
        #[builder(into)] description: Option<String>,
        ai_context: Option<AiContext>,
    ) -> Self {
        semantic_interface.resolve_member_ids_for_code_build(&name);
        Grainset {
            body: GrainsetBody {
                base: DataKindBase::rooted(name, extras),
                datasets,
                unionsets,
                joinsets,
            },
            description,
            ai_context,
            semantic_interface,
        }
    }
}

impl<S: grainset_builder::State> GrainsetBuilder<S> {
    pub fn dataset(mut self, child: NestedDataset) -> Self {
        self.datasets.push(child);
        self
    }
    pub fn datasets(mut self, children: impl IntoIterator<Item = NestedDataset>) -> Self {
        self.datasets.extend(children);
        self
    }
    pub fn unionset(mut self, child: NestedUnionset) -> Self {
        self.unionsets.push(child);
        self
    }
    pub fn unionsets(mut self, children: impl IntoIterator<Item = NestedUnionset>) -> Self {
        self.unionsets.extend(children);
        self
    }
    pub fn joinset(mut self, child: NestedJoinset) -> Self {
        self.joinsets.push(child);
        self
    }
    pub fn joinsets(mut self, children: impl IntoIterator<Item = NestedJoinset>) -> Self {
        self.joinsets.extend(children);
        self
    }

    pub fn extras(mut self, e: ComplexExtras) -> Self {
        self.extras = e;
        self
    }
    pub fn semantic_interface(mut self, s: SemanticInterface) -> Self {
        self.semantic_interface = s;
        self
    }

    pub fn temporal(mut self, t: TemporalShape) -> Self {
        self.extras = mem::take(&mut self.extras).with_temporal(t);
        self
    }

    pub fn dimension(mut self, e: DimensionSite) -> Self {
        self.semantic_interface = mem::take(&mut self.semantic_interface).with_dimension(e);
        self
    }
    pub fn dimensions(mut self, items: impl IntoIterator<Item = DimensionSite>) -> Self {
        self.semantic_interface = mem::take(&mut self.semantic_interface).with_dimensions(items);
        self
    }
    pub fn measure(mut self, e: MeasureSite) -> Self {
        self.semantic_interface = mem::take(&mut self.semantic_interface).with_measure(e);
        self
    }
    pub fn measures(mut self, items: impl IntoIterator<Item = MeasureSite>) -> Self {
        self.semantic_interface = mem::take(&mut self.semantic_interface).with_measures(items);
        self
    }
    pub fn metric(mut self, e: MetricSite) -> Self {
        self.semantic_interface = mem::take(&mut self.semantic_interface).with_metric(e);
        self
    }
    pub fn metrics(mut self, items: impl IntoIterator<Item = MetricSite>) -> Self {
        self.semantic_interface = mem::take(&mut self.semantic_interface).with_metrics(items);
        self
    }
    pub fn filter(mut self, f: DataKindFilter) -> Self {
        self.semantic_interface = mem::take(&mut self.semantic_interface).with_filter(f);
        self
    }
    pub fn filters(mut self, items: impl IntoIterator<Item = DataKindFilter>) -> Self {
        self.semantic_interface = mem::take(&mut self.semantic_interface).with_filters(items);
        self
    }
}

#[bon]
impl NestedGrainset {
    #[builder(
        builder_type(name = NestedGrainsetBuilder, vis = "pub"),
        state_mod(name = nested_grainset_builder, vis = "pub(crate)"),
        finish_fn = build,
    )]
    pub fn new(
        #[builder(start_fn, into)] name: String,
        #[builder(field)] datasets: Vec<NestedDataset>,
        #[builder(field)] unionsets: Vec<NestedUnionset>,
        #[builder(field)] joinsets: Vec<NestedJoinset>,
        #[builder(field)] extras: ComplexExtras,
    ) -> Self {
        NestedGrainset {
            body: GrainsetBody {
                base: DataKindBase::nested(name, extras),
                datasets,
                unionsets,
                joinsets,
            },
        }
    }
}

impl<S: nested_grainset_builder::State> NestedGrainsetBuilder<S> {
    pub fn dataset(mut self, child: NestedDataset) -> Self {
        self.datasets.push(child);
        self
    }
    pub fn datasets(mut self, items: impl IntoIterator<Item = NestedDataset>) -> Self {
        self.datasets.extend(items);
        self
    }
    pub fn unionset(mut self, child: NestedUnionset) -> Self {
        self.unionsets.push(child);
        self
    }
    pub fn unionsets(mut self, items: impl IntoIterator<Item = NestedUnionset>) -> Self {
        self.unionsets.extend(items);
        self
    }
    pub fn joinset(mut self, child: NestedJoinset) -> Self {
        self.joinsets.push(child);
        self
    }
    pub fn joinsets(mut self, items: impl IntoIterator<Item = NestedJoinset>) -> Self {
        self.joinsets.extend(items);
        self
    }

    pub fn extras(mut self, e: ComplexExtras) -> Self {
        self.extras = e;
        self
    }
    pub fn temporal(mut self, t: TemporalShape) -> Self {
        self.extras = mem::take(&mut self.extras).with_temporal(t);
        self
    }
}

#[bon]
impl Unionset {
    #[builder(
        builder_type(name = UnionsetBuilder, vis = "pub"),
        state_mod(name = unionset_builder, vis = "pub"),
        finish_fn = build,
    )]
    pub fn new(
        #[builder(start_fn, into)] name: String,
        #[builder(field)] datasets: Vec<NestedDataset>,
        #[builder(field)] grainsets: Vec<NestedGrainset>,
        #[builder(field)] joinsets: Vec<NestedJoinset>,
        #[builder(field)] extras: ComplexExtras,
        #[builder(field)] mut semantic_interface: SemanticInterface,
        #[builder(default)] mode: UnionMode,
        #[builder(into)] description: Option<String>,
        ai_context: Option<AiContext>,
    ) -> Self {
        semantic_interface.resolve_member_ids_for_code_build(&name);
        Unionset {
            body: UnionsetBody {
                base: DataKindBase::rooted(name, extras),
                mode,
                datasets,
                grainsets,
                joinsets,
            },
            description,
            ai_context,
            semantic_interface,
        }
    }
}

impl<S: unionset_builder::State> UnionsetBuilder<S> {
    pub fn dataset(mut self, child: NestedDataset) -> Self {
        self.datasets.push(child);
        self
    }
    pub fn grainset(mut self, child: NestedGrainset) -> Self {
        self.grainsets.push(child);
        self
    }
    pub fn joinset(mut self, child: NestedJoinset) -> Self {
        self.joinsets.push(child);
        self
    }

    pub fn extras(mut self, e: ComplexExtras) -> Self {
        self.extras = e;
        self
    }
    pub fn semantic_interface(mut self, s: SemanticInterface) -> Self {
        self.semantic_interface = s;
        self
    }

    pub fn temporal(mut self, t: TemporalShape) -> Self {
        self.extras = mem::take(&mut self.extras).with_temporal(t);
        self
    }

    pub fn dimension(mut self, e: DimensionSite) -> Self {
        self.semantic_interface = mem::take(&mut self.semantic_interface).with_dimension(e);
        self
    }
    pub fn dimensions(mut self, items: impl IntoIterator<Item = DimensionSite>) -> Self {
        self.semantic_interface = mem::take(&mut self.semantic_interface).with_dimensions(items);
        self
    }
    pub fn measure(mut self, e: MeasureSite) -> Self {
        self.semantic_interface = mem::take(&mut self.semantic_interface).with_measure(e);
        self
    }
    pub fn measures(mut self, items: impl IntoIterator<Item = MeasureSite>) -> Self {
        self.semantic_interface = mem::take(&mut self.semantic_interface).with_measures(items);
        self
    }
    pub fn metric(mut self, e: MetricSite) -> Self {
        self.semantic_interface = mem::take(&mut self.semantic_interface).with_metric(e);
        self
    }
    pub fn metrics(mut self, items: impl IntoIterator<Item = MetricSite>) -> Self {
        self.semantic_interface = mem::take(&mut self.semantic_interface).with_metrics(items);
        self
    }
    pub fn filter(mut self, f: DataKindFilter) -> Self {
        self.semantic_interface = mem::take(&mut self.semantic_interface).with_filter(f);
        self
    }
    pub fn filters(mut self, items: impl IntoIterator<Item = DataKindFilter>) -> Self {
        self.semantic_interface = mem::take(&mut self.semantic_interface).with_filters(items);
        self
    }
}

impl<S: unionset_builder::State> UnionsetBuilder<S>
where
    S::Mode: unionset_builder::IsUnset,
{
    pub fn union_all(self) -> UnionsetBuilder<unionset_builder::SetMode<S>> {
        self.mode(UnionMode::All)
    }
    pub fn union_unique(self) -> UnionsetBuilder<unionset_builder::SetMode<S>> {
        self.mode(UnionMode::Unique)
    }
}

#[bon]
impl NestedUnionset {
    #[builder(
        builder_type(name = NestedUnionsetBuilder, vis = "pub"),
        state_mod(name = nested_unionset_builder, vis = "pub"),
        finish_fn = build,
    )]
    pub fn new(
        #[builder(start_fn, into)] name: String,
        #[builder(field)] datasets: Vec<NestedDataset>,
        #[builder(field)] grainsets: Vec<NestedGrainset>,
        #[builder(field)] joinsets: Vec<NestedJoinset>,
        #[builder(field)] extras: ComplexExtras,
        #[builder(default)] mode: UnionMode,
    ) -> Self {
        NestedUnionset {
            body: UnionsetBody {
                base: DataKindBase::nested(name, extras),
                mode,
                datasets,
                grainsets,
                joinsets,
            },
        }
    }
}

impl<S: nested_unionset_builder::State> NestedUnionsetBuilder<S> {
    pub fn dataset(mut self, child: NestedDataset) -> Self {
        self.datasets.push(child);
        self
    }
    pub fn datasets(mut self, items: impl IntoIterator<Item = NestedDataset>) -> Self {
        self.datasets.extend(items);
        self
    }
    pub fn grainset(mut self, child: NestedGrainset) -> Self {
        self.grainsets.push(child);
        self
    }
    pub fn grainsets(mut self, items: impl IntoIterator<Item = NestedGrainset>) -> Self {
        self.grainsets.extend(items);
        self
    }
    pub fn joinset(mut self, child: NestedJoinset) -> Self {
        self.joinsets.push(child);
        self
    }
    pub fn joinsets(mut self, items: impl IntoIterator<Item = NestedJoinset>) -> Self {
        self.joinsets.extend(items);
        self
    }

    pub fn extras(mut self, e: ComplexExtras) -> Self {
        self.extras = e;
        self
    }
    pub fn temporal(mut self, t: TemporalShape) -> Self {
        self.extras = mem::take(&mut self.extras).with_temporal(t);
        self
    }
}

impl<S: nested_unionset_builder::State> NestedUnionsetBuilder<S>
where
    S::Mode: nested_unionset_builder::IsUnset,
{
    pub fn union_all(self) -> NestedUnionsetBuilder<nested_unionset_builder::SetMode<S>> {
        self.mode(UnionMode::All)
    }
    pub fn union_unique(self) -> NestedUnionsetBuilder<nested_unionset_builder::SetMode<S>> {
        self.mode(UnionMode::Unique)
    }
}

#[bon]
impl Joinset {
    #[builder(
        builder_type(name = JoinsetBuilder, vis = "pub"),
        state_mod(name = joinset_builder, vis = "pub(crate)"),
        finish_fn = build,
    )]
    pub fn new(
        #[builder(start_fn, into)] name: String,
        #[builder(field)] relationships: Vec<Relationship>,
        #[builder(field)] datasets: Vec<NestedDataset>,
        #[builder(field)] grainsets: Vec<NestedGrainset>,
        #[builder(field)] unionsets: Vec<NestedUnionset>,
        #[builder(field)] extras: ComplexExtras,
        #[builder(field)] mut semantic_interface: SemanticInterface,
        #[builder(into)] description: Option<String>,
        ai_context: Option<AiContext>,
    ) -> Self {
        semantic_interface.resolve_member_ids_for_code_build(&name);
        Joinset {
            body: JoinsetBody {
                base: DataKindBase::rooted(name, extras),
                relationships,
                datasets,
                grainsets,
                unionsets,
            },
            description,
            ai_context,
            semantic_interface,
        }
    }
}

impl<S: joinset_builder::State> JoinsetBuilder<S> {
    pub fn relationship(mut self, r: Relationship) -> Self {
        self.relationships.push(r);
        self
    }
    pub fn relationships(mut self, items: impl IntoIterator<Item = Relationship>) -> Self {
        self.relationships.extend(items);
        self
    }
    pub fn dataset(mut self, child: NestedDataset) -> Self {
        self.datasets.push(child);
        self
    }
    pub fn grainset(mut self, child: NestedGrainset) -> Self {
        self.grainsets.push(child);
        self
    }
    pub fn unionset(mut self, child: NestedUnionset) -> Self {
        self.unionsets.push(child);
        self
    }

    pub fn extras(mut self, e: ComplexExtras) -> Self {
        self.extras = e;
        self
    }
    pub fn semantic_interface(mut self, s: SemanticInterface) -> Self {
        self.semantic_interface = s;
        self
    }

    pub fn temporal(mut self, t: TemporalShape) -> Self {
        self.extras = mem::take(&mut self.extras).with_temporal(t);
        self
    }

    pub fn dimension(mut self, e: DimensionSite) -> Self {
        self.semantic_interface = mem::take(&mut self.semantic_interface).with_dimension(e);
        self
    }
    pub fn dimensions(mut self, items: impl IntoIterator<Item = DimensionSite>) -> Self {
        self.semantic_interface = mem::take(&mut self.semantic_interface).with_dimensions(items);
        self
    }
    pub fn measure(mut self, e: MeasureSite) -> Self {
        self.semantic_interface = mem::take(&mut self.semantic_interface).with_measure(e);
        self
    }
    pub fn measures(mut self, items: impl IntoIterator<Item = MeasureSite>) -> Self {
        self.semantic_interface = mem::take(&mut self.semantic_interface).with_measures(items);
        self
    }
    pub fn metric(mut self, e: MetricSite) -> Self {
        self.semantic_interface = mem::take(&mut self.semantic_interface).with_metric(e);
        self
    }
    pub fn metrics(mut self, items: impl IntoIterator<Item = MetricSite>) -> Self {
        self.semantic_interface = mem::take(&mut self.semantic_interface).with_metrics(items);
        self
    }
    pub fn filter(mut self, f: DataKindFilter) -> Self {
        self.semantic_interface = mem::take(&mut self.semantic_interface).with_filter(f);
        self
    }
    pub fn filters(mut self, items: impl IntoIterator<Item = DataKindFilter>) -> Self {
        self.semantic_interface = mem::take(&mut self.semantic_interface).with_filters(items);
        self
    }
}

#[bon]
impl NestedJoinset {
    #[builder(
        builder_type(name = NestedJoinsetBuilder, vis = "pub"),
        state_mod(name = nested_joinset_builder, vis = "pub(crate)"),
        finish_fn = build,
    )]
    pub fn new(
        #[builder(start_fn, into)] name: String,
        #[builder(field)] relationships: Vec<Relationship>,
        #[builder(field)] datasets: Vec<NestedDataset>,
        #[builder(field)] grainsets: Vec<NestedGrainset>,
        #[builder(field)] unionsets: Vec<NestedUnionset>,
        #[builder(field)] extras: ComplexExtras,
    ) -> Self {
        NestedJoinset {
            body: JoinsetBody {
                base: DataKindBase::nested(name, extras),
                relationships,
                datasets,
                grainsets,
                unionsets,
            },
        }
    }
}

impl<S: nested_joinset_builder::State> NestedJoinsetBuilder<S> {
    pub fn relationship(mut self, r: Relationship) -> Self {
        self.relationships.push(r);
        self
    }
    pub fn relationships(mut self, items: impl IntoIterator<Item = Relationship>) -> Self {
        self.relationships.extend(items);
        self
    }
    pub fn dataset(mut self, child: NestedDataset) -> Self {
        self.datasets.push(child);
        self
    }
    pub fn datasets(mut self, items: impl IntoIterator<Item = NestedDataset>) -> Self {
        self.datasets.extend(items);
        self
    }
    pub fn grainset(mut self, child: NestedGrainset) -> Self {
        self.grainsets.push(child);
        self
    }
    pub fn grainsets(mut self, items: impl IntoIterator<Item = NestedGrainset>) -> Self {
        self.grainsets.extend(items);
        self
    }
    pub fn unionset(mut self, child: NestedUnionset) -> Self {
        self.unionsets.push(child);
        self
    }
    pub fn unionsets(mut self, items: impl IntoIterator<Item = NestedUnionset>) -> Self {
        self.unionsets.extend(items);
        self
    }

    pub fn extras(mut self, e: ComplexExtras) -> Self {
        self.extras = e;
        self
    }
    pub fn temporal(mut self, t: TemporalShape) -> Self {
        self.extras = mem::take(&mut self.extras).with_temporal(t);
        self
    }
}
