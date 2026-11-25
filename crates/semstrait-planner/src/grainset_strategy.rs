use semstrait_ir::Delivery;
use semstrait_manifest::{SemanticBitmask, SourceAlias};

use crate::binding::BoundName;
use crate::grain::{RequestGrain, SourceGrain};
use crate::predicate::leaves_beyond;
use crate::read_model::ReadModel;
use crate::resolver::{nearest_children, Structural};
use crate::routing::{not_servable, route, Route, Routed};
use crate::strategy::{GrainsetArm, Strategy, StrategyInput};
use crate::{Combine, Decision, Operand, PlanError};

#[derive(Debug)]
pub(crate) struct GrainsetStrategy;

impl Strategy<GrainsetArm> for GrainsetStrategy {
    fn decide(
        &self,
        read_model: &dyn ReadModel,
        arm: &GrainsetArm,
        input: &StrategyInput,
    ) -> Result<Decision, PlanError> {
        let (grainset, request, routed) = routed(read_model, &arm.children, input)?;
        let (stand_in, operands, combine) = match routed.map(|routed| routed.route) {
            Some(Route::Child(child)) => (
                Some(input.name(read_model, &child)?),
                Vec::new(),
                Combine::PassThrough,
            ),
            Some(Route::DrillAcross { chosen, .. })
                if input.delivery == Delivery::FinalAggregates =>
            {
                let demanded = demanded_names(input);
                let operand = |(source, served): (SourceAlias, SemanticBitmask)| {
                    let names = (demanded.iter())
                        .filter(|name| served.contains(name.bit))
                        .map(|name| name.name.clone())
                        .collect();
                    Operand {
                        source,
                        names,
                        null_filled: Vec::new(),
                    }
                };
                let operands = chosen.into_iter().map(operand).collect();
                (None, operands, Combine::DrillAcross)
            }
            _ => {
                let (path, source) = (input.path.clone(), input.semantic_source.clone());
                let demanded = demanded_names(input);
                let nearest = nearest_children(read_model, &Structural, &demanded, &input.alias);
                return Err(not_servable(&grainset, &request, path, source, nearest));
            }
        };
        Ok(Decision {
            stand_in,
            operands,
            combine,
            physical_read: None,
        })
    }
}

fn demanded_names(input: &StrategyInput) -> Vec<BoundName> {
    (input.names.iter())
        .chain(leaves_beyond(&input.names, &input.predicates))
        .filter(|name| input.demanded.contains(name.bit))
        .cloned()
        .collect()
}

pub(crate) fn routed(
    read_model: &dyn ReadModel,
    children: &[SourceAlias],
    input: &StrategyInput,
) -> Result<(SourceGrain, RequestGrain, Option<Routed>), PlanError> {
    let selected = (input.names.iter()).map(|name| {
        (
            name.bit,
            input.requested_granularity.get(&name.name).copied(),
        )
    });
    let request = RequestGrain::new(read_model, selected);
    let grainset = SourceGrain::of_demand(read_model, input);
    let conditions = input.condition_reads(read_model)?;
    let routed = route(
        read_model,
        &grainset,
        children,
        &input.demanded,
        &conditions,
        &request,
        input.delivery,
    )
    .map_err(|unrouted| unrouted.error(read_model, &input.path, &input.semantic_source))?;
    Ok((grainset, request, routed))
}
