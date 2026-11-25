use crate::read_model::ReadModel;
use crate::strategy::{Strategy, StrategyInput, UnionsetArm};
use crate::{Combine, Decision, PlanError};

#[derive(Debug)]
pub(crate) struct UnionsetStrategy;

impl Strategy<UnionsetArm> for UnionsetStrategy {
    fn decide(
        &self,
        read_model: &dyn ReadModel,
        arm: &UnionsetArm,
        input: &StrategyInput,
    ) -> Result<Decision, PlanError> {
        let branches = arm.branches(read_model, input)?;
        let operands = (branches.read.into_iter())
            .map(|branch| arm.operand(read_model, branch, &branches.names));
        Ok(Decision {
            stand_in: None,
            operands: operands.collect(),
            combine: Combine::Stack,
            physical_read: None,
        })
    }
}
