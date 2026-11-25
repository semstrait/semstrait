mod rewrite;
mod whole_call;

use std::iter;

use semstrait_adapter::{Extensions, InvalidRewrite, SubstraitRules};
use semstrait_common::Release;
use semstrait_ir::{
    FilterNode, JoinNode, MultisetNode, Parameter, PhysicalExpr, PlanNode, ProjectNode, ScanNode,
    Schema,
};
use semstrait_sql::expr::{AggregateAllowed, ScalarCall};
use substrait::proto::read_rel::{LocalFiles, ReadType};
use substrait::proto::set_rel::SetOp;
use substrait::proto::{rel, Expression, FunctionArgument, ReadRel, Rel, SetRel, Type};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataFusionRules {
    release: Release,
}

impl DataFusionRules {
    pub const fn new(release: Release) -> Self {
        Self { release }
    }

    pub fn release(&self) -> Release {
        self.release
    }
}

impl Default for DataFusionRules {
    fn default() -> Self {
        Self::new(Release::DATAFUSION_REFERENCE)
    }
}

impl SubstraitRules for DataFusionRules {
    fn rewrite_node(&self, node: &PlanNode) -> Option<PlanNode> {
        match node {
            PlanNode::Join(join) => filter_above(join),
            PlanNode::Multiset(union) => projected_branches(union),
            _ => None,
        }
    }

    fn rewrite_expression(&self, expr: &PhysicalExpr, _input: &Schema) -> Option<PhysicalExpr> {
        rewrite::expression(expr)
    }

    fn whole_call(
        &self,
        call: &ScalarCall<Parameter, AggregateAllowed>,
        arguments: &[FunctionArgument],
        output_type: &Type,
        extensions: &mut Extensions,
    ) -> Result<Option<Expression>, InvalidRewrite> {
        whole_call::whole_call(call, arguments, output_type, extensions, self.release)
    }

    fn reader(&self, _scan: &ScanNode, mut read: ReadRel) -> Rel {
        match read.read_type.take() {
            Some(ReadType::NamedTable(mut table))
                if table.names.first().is_some_and(String::is_empty) =>
            {
                table.names.remove(0);
                read_rel(ReadRel {
                    read_type: Some(ReadType::NamedTable(table)),
                    ..read
                })
            }
            Some(ReadType::LocalFiles(LocalFiles {
                items,
                advanced_extension,
            })) if items.len() > 1 => {
                let inputs = (items.into_iter())
                    .map(|item| {
                        let files = LocalFiles {
                            items: vec![item],
                            advanced_extension: advanced_extension.clone(),
                        };
                        read_rel(ReadRel {
                            read_type: Some(ReadType::LocalFiles(files)),
                            ..read.clone()
                        })
                    })
                    .collect();
                Rel {
                    rel_type: Some(rel::RelType::Set(SetRel {
                        inputs,
                        op: SetOp::UnionAll.into(),
                        ..SetRel::default()
                    })),
                }
            }
            read_type => read_rel(ReadRel { read_type, ..read }),
        }
    }
}

fn filter_above(join: &JoinNode) -> Option<PlanNode> {
    let filter = join.post_join_filter.clone()?;
    let (left, right) = (join.left.as_ref().clone(), join.right.as_ref().clone());
    let mut inner = JoinNode::new(left, right, join.join_type, join.on.clone(), None).ok()?;
    inner.common.anchor = join.common.anchor;
    inner.common.hint = join.common.hint.clone();
    inner.common.semantic_meta = join.common.semantic_meta.clone();
    let filtered = PlanNode::from(FilterNode::new(inner, filter).ok()?);
    match &join.common.emit_form {
        Some(emit_form) => filtered.with_emit_form(emit_form.clone()).ok(),
        None => Some(filtered),
    }
}

fn projected_branches(union: &MultisetNode) -> Option<PlanNode> {
    let projected = |branch: &PlanNode| {
        matches!(branch, PlanNode::Project(_)) || branch.common().emit_form.is_some()
    };
    let (first, later) = union.inputs.split_first()?;
    if later.iter().all(projected) {
        return None;
    }
    let later = later.iter().map(|branch| {
        if projected(branch) {
            Ok(branch.clone())
        } else {
            ProjectNode::new(branch.clone(), Vec::new()).map(PlanNode::from)
        }
    });
    let branches = iter::once(Ok(first.clone())).chain(later);
    let mut named = MultisetNode::new(branches.collect::<Result<_, _>>().ok()?, union.op).ok()?;
    named.common = union.common.clone();
    Some(named.into())
}

fn read_rel(read: ReadRel) -> Rel {
    Rel {
        rel_type: Some(rel::RelType::Read(Box::new(read))),
    }
}
