use crate::plan::common::Anchor;
use crate::plan::node::PlanNode;
use crate::primitives::Name;

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticPlan {
    pub root: PlanNode,
    pub output_names: Vec<Name>,
}

impl SemanticPlan {
    pub fn new(mut root: PlanNode, output_names: Vec<Name>) -> Self {
        number(&mut root, &mut 1);
        Self { root, output_names }
    }
}

fn number(node: &mut PlanNode, next: &mut u128) {
    node.common_mut().anchor = Anchor(*next);
    *next += 1;
    for input in node.children_mut() {
        number(input, next);
    }
}
