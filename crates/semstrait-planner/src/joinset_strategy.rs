use semstrait_common::SemanticsName;

use crate::read_model::{ReadModel, Relationship};
use crate::strategy::{qualifies, Strategy, StrategyInput};
use crate::{Combine, Decision, Operand, PlanError};

#[derive(Debug)]
pub(crate) struct JoinsetStrategy;

impl Strategy<Relationship> for JoinsetStrategy {
    fn decide(
        &self,
        read_model: &dyn ReadModel,
        relationship: &Relationship,
        input: &StrategyInput,
    ) -> Result<Decision, PlanError> {
        let stand_in = (input.stand_ins.iter()).find(|unit| input.may_stand_in(read_model, unit));
        if let Some(stand_in) = stand_in {
            return Ok(Decision {
                stand_in: Some(stand_in.clone()),
                operands: Vec::new(),
                combine: Combine::PassThrough,
                physical_read: None,
            });
        }
        let mut operands = [&relationship.from, &relationship.to].map(|operand| Operand {
            source: operand.clone(),
            names: Vec::new(),
            null_filled: Vec::new(),
        });
        for (binding, columns) in input.columns(read_model)? {
            for column in columns {
                let operand = (operands.iter_mut())
                    .find(|operand| qualifies(&operand.source, &column))
                    .ok_or_else(|| input.broken_tree(&binding))?;
                let served = SemanticsName::new(column.name());
                if !operand.names.contains(&served) {
                    operand.names.push(served);
                }
            }
        }
        let mut kept = None;
        for end in [&relationship.from, &relationship.to] {
            if input.reads_alone(read_model, relationship, end)? {
                kept = Some(end);
                break;
            }
        }
        Ok(Decision {
            stand_in: None,
            operands: (operands.into_iter())
                .filter(|operand| kept.is_none_or(|kept| kept == &operand.source))
                .collect(),
            combine: match kept {
                Some(_) => Combine::PassThrough,
                None => relationship.join_type.into(),
            },
            physical_read: None,
        })
    }
}
