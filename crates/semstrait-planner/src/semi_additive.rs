use std::iter;

use semstrait_common::SemanticsName;
use semstrait_common::ValidateError;
use semstrait_ir::{AdditivityType, PhysicalExpr, SemiAdditivityStrategy};
use semstrait_manifest::{PhysicalExprId, SemanticDefinition, SourceAlias};
use semstrait_sql::expr::{NamedReference, NullOrdering, Over, SortDir, SortKey};

use crate::fan_out::datasets;
use crate::frame::Frame;
use crate::grain::SourceGrain;
use crate::read_model::ReadModel;
use crate::steps::window;
use crate::strategy::StrategyInput;
use crate::{BoundName, PlanError, UnbuildableState};

#[derive(Debug, Clone)]
pub(crate) struct RankMarker {
    pub(crate) by: BoundName,
    pub(crate) axes: Vec<BoundName>,
    dir: SortDir,
    read: Vec<(SemanticsName, SourceAlias)>,
}

impl RankMarker {
    pub(crate) fn of(
        read_model: &dyn ReadModel,
        input: &StrategyInput,
        definition: &SemanticDefinition,
        datasets: &[SourceAlias],
    ) -> Result<Option<Self>, PlanError> {
        let Some(AdditivityType::Semi(semi)) = definition.role_definition.additivity() else {
            return Ok(None);
        };
        let single = |source: &SourceAlias| read_model.time_axis(source)?.single();
        let by = match &semi.by {
            Some(by) => input.bound_by_id(read_model, by)?,
            None => (datasets.iter().find_map(single))
                .or_else(|| single(&input.alias))
                .and_then(|bit| {
                    Some(BoundName {
                        name: read_model.name_of(bit)?,
                        bit,
                    })
                })
                .ok_or_else(|| input.broken())?,
        };
        let dataset = |bit: u32| {
            let mapping: Vec<_> = (datasets.iter())
                .filter(|dataset| read_model.mask(dataset).contains(bit))
                .collect();
            match mapping[..] {
                [] if datasets.is_empty() => Ok(None),
                [dataset] => Ok(Some(dataset)),
                _ => (mapping.into_iter())
                    .find(|dataset| single(dataset) == Some(bit))
                    .map(Some)
                    .ok_or_else(|| input.broken()),
            }
        };
        let source = dataset(by.bit)?.unwrap_or(&input.alias);
        let axes: Vec<BoundName> = (semi.axes.iter())
            .map(|axis| input.bound_by_id(read_model, axis))
            .collect::<Result<_, _>>()?;
        let asked = (input.names.iter())
            .find(|name| name.bit == by.bit)
            .map(|name| input.requested_granularity.get(&name.name).copied());
        if SourceGrain::of_source(read_model, source).kept(asked, by.bit) {
            return Ok(None);
        }
        let dir = match semi.strategy {
            SemiAdditivityStrategy::Latest => SortDir::Desc {
                nulls: NullOrdering::Last,
            },
            SemiAdditivityStrategy::Earliest => SortDir::Asc {
                nulls: NullOrdering::Last,
            },
            strategy => {
                return Err(input.unbuildable(UnbuildableState::SemiAdditiveStrategy(strategy)))
            }
        };
        let read = (iter::once(&by).chain(&axes))
            .filter_map(|name| {
                let read = dataset(name.bit).transpose()?;
                Some(read.map(|dataset| (name.name.clone(), dataset.clone())))
            })
            .collect::<Result<_, PlanError>>()?;
        Ok(Some(Self {
            by,
            axes,
            dir,
            read,
        }))
    }

    pub(crate) fn of_binding(
        read_model: &dyn ReadModel,
        input: &StrategyInput,
        operands: &[Frame],
        binding: &BoundName,
        kept: &[PhysicalExprId],
    ) -> Result<Vec<Option<Self>>, PlanError> {
        let definitions = input.aggregate_definitions(read_model, binding, kept)?;
        let mut datasets = datasets(read_model, input, operands, binding)?.into_iter();
        (definitions.iter())
            .map(|definition| {
                let datasets = datasets.next().unwrap_or_default();
                Self::of(read_model, input, definition, &datasets)
            })
            .collect()
    }

    pub(crate) fn reads(&self) -> impl Iterator<Item = &BoundName> {
        iter::once(&self.by).chain(&self.axes)
    }

    pub(crate) fn leaf(&self, name: &SemanticsName) -> Option<NamedReference> {
        let (_, dataset) = self.read.iter().find(|(read, _)| read == name)?;
        NamedReference::qualified(dataset.as_str(), name.as_str())
    }

    pub(crate) fn window(
        &self,
        partition_by: Vec<PhysicalExpr>,
        by: PhysicalExpr,
    ) -> Result<PhysicalExpr, ValidateError> {
        let over = Over {
            partition_by,
            order_by: vec![SortKey {
                expr: by,
                dir: self.dir,
            }],
            window_frame: None,
        };
        window("rank", over)
    }
}
