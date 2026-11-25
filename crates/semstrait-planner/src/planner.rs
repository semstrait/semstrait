use semstrait_common::{Diagnostic, Diagnostics};
use semstrait_ir::SemanticPlan;
use semstrait_manifest::SemanticManifest;

use crate::binding::bind;
use crate::build::plan_root;
use crate::fan_out::{grain, placed};
use crate::frame::decide;
use crate::read_model::{ManifestReadModel, Relationship};
use crate::resolver::{resolve, tied_stand_in, Structural};
use crate::strategy::{DatasetArm, GrainsetArm, Strategies, Strategy, StrategyInput, UnionsetArm};
use crate::{Advisory, PlanError, Request, RequestContext};

#[non_exhaustive]
#[derive(Debug, Default)]
pub struct Planner {
    strategies: Strategies,
}

impl Planner {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_dataset_strategy(mut self, strategy: Box<dyn Strategy<DatasetArm>>) -> Self {
        self.strategies.dataset = strategy;
        self
    }

    #[must_use]
    pub fn with_joinset_strategy(mut self, strategy: Box<dyn Strategy<Relationship>>) -> Self {
        self.strategies.joinset = strategy;
        self
    }

    #[must_use]
    pub fn with_unionset_strategy(mut self, strategy: Box<dyn Strategy<UnionsetArm>>) -> Self {
        self.strategies.unionset = strategy;
        self
    }

    #[must_use]
    pub fn with_grainset_strategy(mut self, strategy: Box<dyn Strategy<GrainsetArm>>) -> Self {
        self.strategies.grainset = strategy;
        self
    }

    #[allow(
        clippy::type_complexity,
        reason = "ADR 0064's fail-fast form, spelled out as a caller binds it"
    )]
    pub fn plan(
        &self,
        manifest: &SemanticManifest,
        request: &Request,
        context: &RequestContext,
    ) -> Result<(SemanticPlan, Diagnostics<Advisory>), (Diagnostic<PlanError>, Diagnostics<Advisory>)>
    {
        let refused = |error| (Diagnostic::new(error), Vec::new());
        let read_model = ManifestReadModel::new(manifest);
        let bound = bind(&read_model, request, manifest.dialect).map_err(refused)?;
        let (resolution, advisories) = resolve(
            &read_model,
            &Structural,
            request.from.as_ref(),
            &request.select,
            &bound.demanded(),
        )
        .map_err(refused)?;
        let mut advisories: Diagnostics<Advisory> =
            advisories.into_iter().map(Diagnostic::new).collect();
        let (compositions, fit) = (resolution.compositions, resolution.granularity_fit);
        let alternatives = resolution.alternatives.clone();
        let input = StrategyInput::root(resolution, bound);
        let grain = grain(&read_model, &input.names);
        let planned = decide(&self.strategies, &read_model, &grain, input).and_then(|root| {
            let may_stand_in = |unit: &_| root.input.may_stand_in(&read_model, unit);
            let stand_in = root
                .decided()
                .and_then(|decided| decided.decision.stand_in.as_ref());
            let tie = stand_in.and_then(|stand_in| {
                tied_stand_in(&alternatives, stand_in, &root.input.path, may_stand_in)
            });
            advisories.extend(tie.map(Diagnostic::new));
            let fan_out = placed(&read_model, &grain, &root)?;
            advisories.extend(fan_out.into_iter().map(Diagnostic::new));
            let unique = root.unique_without_keyset(&read_model);
            advisories.extend(unique.into_iter().map(Diagnostic::new));
            let tied = root.tied_children(&read_model)?;
            advisories.extend(tied.into_iter().map(Diagnostic::new));
            let mut advised = Vec::new();
            let plan = plan_root(
                &read_model,
                root,
                compositions,
                fit,
                request,
                context,
                &mut advised,
            );
            advisories.extend(advised.into_iter().map(Diagnostic::new));
            plan
        });
        let validated = planned.map_err(Diagnostic::new).and_then(|plan| {
            (plan.validate().map(|()| plan))
                .map_err(|refusal| refusal.map_kind(|error| PlanError::PlanRefused { error }))
        });
        match validated {
            Ok(plan) => Ok((plan, advisories)),
            Err(error) => Err((error, advisories)),
        }
    }
}
