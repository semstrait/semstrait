use std::collections::{BTreeMap, BTreeSet};
use std::iter;

use semstrait_common::SemanticsName;
use semstrait_manifest::{PhysicalExprId, SemanticSignature};

use super::{FrameStages, Stages, Subtree};
use crate::advisory::advise;
use crate::formula::volatile;
use crate::frame::{Decided, Frame, FrameKind};
use crate::read_model::ReadModel;
use crate::strategy::StrategyInput;
use crate::tree::column_leaves;
use crate::{Advisory, BoundName, Combine, PlanError};

pub(super) fn in_two_reads(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    decided: &Decided,
    stages: &FrameStages,
    advised: &mut Vec<Advisory>,
) -> Result<(), PlanError> {
    let own = (!stages.read.aggregates.is_empty()).then(|| (&stages.read, Vec::new()));
    let reads: Vec<_> = (own.into_iter())
        .chain(stages.subtrees.iter().flat_map(reads))
        .collect();
    if reads.len() < 2 {
        return Ok(());
    }
    let before = stages.read.before_aggregation.as_ref();
    let shared = joins_draw(read_model, input, decided)? || before.is_some_and(volatile);
    let drawn = (reads.iter())
        .map(|(read, _)| {
            let mut drawn = BTreeSet::new();
            for row in &read.row_level {
                drawn.extend(reached_volatile(read_model, input, &row.semantics)?);
            }
            Ok(drawn)
        })
        .collect::<Result<Vec<_>, PlanError>>()?;
    for (at, ((_, metrics), here)) in reads.iter().zip(&drawn).enumerate() {
        let elsewhere = |value| {
            (drawn.iter().enumerate()).any(|(other, held)| other != at && held.contains(value))
        };
        if shared || here.iter().any(elsewhere) {
            for metric in metrics {
                let path = input.path.clone();
                let metric = (*metric).clone();
                advise(advised, Advisory::VolatileInTwoReads { path, metric });
            }
        }
    }
    Ok(())
}

pub(super) fn copied(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    stages: &FrameStages,
    advised: &mut Vec<Advisory>,
) -> Result<(), PlanError> {
    let subtrees = stages.subtrees.iter().flat_map(reads);
    for read in iter::once(&stages.read).chain(subtrees.map(|(read, _)| read)) {
        let computing = (read.row_level.iter().chain(&read.derived))
            .filter_map(|field| volatile(&field.tree).then_some(&field.semantics));
        let (mut computed, mut own) = (BTreeMap::<SemanticsName, usize>::new(), BTreeSet::new());
        for semantics in computing {
            for drawn in reached_volatile(read_model, input, semantics)? {
                if &drawn == semantics && !own.insert(drawn.clone()) {
                    continue;
                }
                *computed.entry(drawn).or_default() += 1;
            }
        }
        for (semantics, _) in computed.into_iter().filter(|(_, count)| *count > 1) {
            let path = input.path.clone();
            advise(advised, Advisory::VolatileCopied { path, semantics });
        }
    }
    Ok(())
}

fn reads(subtree: &Subtree) -> Vec<(&Stages, Vec<&SemanticsName>)> {
    let metrics = (subtree.values.iter()).map(|value| &value.metric).collect();
    iter::once((&subtree.step1, metrics))
        .chain(subtree.linked.iter().flat_map(reads))
        .collect()
}

fn joins_draw(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    decided: &Decided,
) -> Result<bool, PlanError> {
    let own = match decided.decision.combine {
        Combine::Join | Combine::Cross => {
            let relationship = input.relationship(read_model)?;
            let trees: Vec<_> = relationship.trees().collect();
            let mut own = (trees.iter().copied())
                .chain(&decided.above_join)
                .any(volatile);
            for leaf in trees.into_iter().flat_map(column_leaves) {
                own = own || draws(read_model, input, &input.bound(read_model, leaf.name())?)?;
            }
            own
        }
        _ => false,
    };
    (decided.operands.iter()).try_fold(own, |held, operand| {
        Ok(held || frame_draws(read_model, operand)?)
    })
}

fn frame_draws(read_model: &dyn ReadModel, frame: &Frame) -> Result<bool, PlanError> {
    match &frame.kind {
        FrameKind::Decided(decided) => joins_draw(read_model, &frame.input, decided),
        FrameKind::Compute(compute) => (compute.groups.iter()).try_fold(false, |held, group| {
            Ok(held || frame_draws(read_model, &group.frame)?)
        }),
    }
}

fn reached_volatile(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    semantics: &SemanticsName,
) -> Result<Vec<SemanticsName>, PlanError> {
    let Some(bit) = read_model.bit_of(semantics.as_str()) else {
        return Ok(Vec::new());
    };
    let Some(signature) = read_model.signature(&input.alias, bit) else {
        return Ok(Vec::new());
    };
    let binding = BoundName {
        name: semantics.clone(),
        bit,
    };
    let reached = (read_model.reached(&input.alias, &entries(&signature)))
        .ok_or_else(|| input.broken_tree(&binding))?;
    let mut drawn = Vec::new();
    for definition in reached {
        let binding = BoundName::of(definition);
        if draws(read_model, input, &binding)? {
            drawn.push(binding.name);
        }
    }
    Ok(drawn)
}

fn draws(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    binding: &BoundName,
) -> Result<bool, PlanError> {
    let beneath = read_model.operand_closure(&input.alias);
    for source in iter::once(&input.alias).chain(&beneath) {
        let Some(signature) = read_model.signature(source, binding.bit) else {
            continue;
        };
        for root in entries(&signature) {
            let tree = (read_model.expr_tree(source, &root, &[]))
                .ok_or_else(|| input.broken_tree(binding))?;
            if volatile(&tree) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn entries(signature: &SemanticSignature) -> Vec<PhysicalExprId> {
    iter::once(signature.expression_id)
        .chain(signature.filter_expr_id)
        .collect()
}
