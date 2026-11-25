use semstrait_ir::{Parameter, PhysicalExpr, PlanNode, ScanNode, Schema};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{AggregateAllowed, ScalarCall};
use substrait::proto::{rel, Expression, FunctionArgument, ReadRel, Rel, Type};

use crate::error::InvalidRewrite;
use crate::extensions::Extensions;
use crate::functions::FunctionName;

pub trait SubstraitRules {
    fn rewrite_node(&self, _node: &PlanNode) -> Option<PlanNode> {
        None
    }

    fn rewrite_expression(&self, _expr: &PhysicalExpr, _input: &Schema) -> Option<PhysicalExpr> {
        None
    }

    fn function(&self, _canonical: &str, faithful: FunctionName) -> FunctionName {
        faithful
    }

    fn whole_call(
        &self,
        _call: &ScalarCall<Parameter, AggregateAllowed>,
        _arguments: &[FunctionArgument],
        _output_type: &Type,
        _extensions: &mut Extensions,
    ) -> Result<Option<Expression>, InvalidRewrite> {
        Ok(None)
    }

    fn reader(&self, _scan: &ScanNode, read: ReadRel) -> Rel {
        Rel {
            rel_type: Some(rel::RelType::Read(Box::new(read))),
        }
    }

    fn type_variation(&self, _data_type: &DataType) -> u32 {
        0
    }
}
