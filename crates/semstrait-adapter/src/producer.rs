#![allow(
    clippy::result_large_err,
    reason = "every stage returns its Diagnostic envelope unboxed"
)]

use std::collections::BTreeMap;
use std::iter;

use semstrait_catalog::FileFormat;
use semstrait_common::diagnostic::{Diagnostic, Diagnostics};
use semstrait_common::Tree;
use semstrait_ir::{
    ComputationType, FilesScan, Hint, JoinType, MultisetOp, NodeCommon, Parameter, PhysicalExpr,
    PlanNode, Reader, Schema, SemanticPlan, TableScan, TextReadOptions,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{
    AggregateAllowed, AggregateCall, BinaryOpKind, Call, CastFailure, Expr, FunctionRef, LikeKind,
    Literal, NullOrdering, Operation, OperationKind, OptionSetting, Over, ScalarCall, SortDir,
    SortKey, Typed, UnaryOpKind, WindowBound, WindowCall, WindowFrame, WindowFrameKind,
};
use semstrait_sql::functions::{
    ArgumentForm, ClauseRule, Declaration, FunctionType, OptionDecl, OptionName, ReturnRule,
    Signature,
};
use substrait::proto::aggregate_function::AggregationInvocation;
use substrait::proto::aggregate_rel::{Grouping, Measure};
use substrait::proto::expression::cast::FailureBehavior;
use substrait::proto::expression::field_reference::{ReferenceType, RootReference, RootType};
use substrait::proto::expression::if_then::IfClause;
use substrait::proto::expression::reference_segment::{self, StructField};
use substrait::proto::expression::window_function::bound::{
    CurrentRow, Following, Kind as BoundKind, Preceding, Unbounded,
};
use substrait::proto::expression::window_function::{Bound, BoundsType};
use substrait::proto::expression::{
    self, nested, FieldReference, IfThen, ReferenceSegment, RexType, ScalarFunction,
    SingularOrList, WindowFunction,
};
use substrait::proto::extensions::AdvancedExtension;
use substrait::proto::fetch_rel::{CountMode, OffsetMode};
use substrait::proto::function_argument::ArgType;
use substrait::proto::r#type::Kind;
use substrait::proto::read_rel::local_files::file_or_files::{
    DelimiterSeparatedTextReadOptions, FileFormat as ItemFormat, OrcReadOptions,
    ParquetReadOptions, PathType,
};
use substrait::proto::read_rel::local_files::FileOrFiles;
use substrait::proto::read_rel::{self, LocalFiles, ReadType};
use substrait::proto::rel::RelType;
use substrait::proto::rel_common::{self, hint, EmitKind};
use substrait::proto::sort_field::{SortDirection, SortKind};
use substrait::proto::{
    join_rel, plan_rel, set_rel, AggregateFunction, AggregateRel, AggregationPhase, CrossRel,
    Expression, FetchRel, FilterRel, FunctionArgument, FunctionOption, JoinRel, Plan, PlanRel,
    ProjectRel, ReadRel, Rel, RelCommon, RelRoot, SetRel, SortField, SortRel, Type,
};
use substrait::version::version_with_producer;

use crate::error::{number, InvalidRewrite, VocabularyPart};
use crate::explain::trace;
use crate::extensions::Extensions;
use crate::functions::{self, Argument, Form, Operands};
use crate::literal::literal;
use crate::rules::SubstraitRules;
use crate::types::{named_struct, proto_type, zoned};

pub struct SubstraitProducer<'r> {
    rules: &'r dyn SubstraitRules,
}

impl<'r> SubstraitProducer<'r> {
    pub fn new(rules: &'r dyn SubstraitRules) -> Self {
        Self { rules }
    }

    #[allow(
        clippy::type_complexity,
        reason = "ADR 0005's fail-fast form, spelled out as a caller binds it"
    )]
    pub fn produce(
        &self,
        plan: &SemanticPlan,
    ) -> Result<
        (Plan, Diagnostics<InvalidRewrite>),
        (Diagnostic<InvalidRewrite>, Diagnostics<InvalidRewrite>),
    > {
        (self.production(plan, Production::Executed))
            .map(|plan| (plan, Vec::new()))
            .map_err(|error| (error, Vec::new()))
    }

    pub(crate) fn production(
        &self,
        plan: &SemanticPlan,
        production: Production,
    ) -> Result<Plan, Diagnostic<InvalidRewrite>> {
        let mut walk = Walk {
            rules: self.rules,
            production,
            extensions: Extensions::default(),
        };
        let root = RelRoot {
            input: Some(walk.rel(&plan.root)?),
            names: (plan.output_names.iter())
                .map(|name| name.as_str().to_owned())
                .collect(),
        };
        let (extension_urns, extensions) = walk.extensions.declarations();
        Ok(Plan {
            version: Some(version_with_producer("semstrait")),
            extension_urns,
            extensions,
            relations: vec![PlanRel {
                rel_type: Some(plan_rel::RelType::Root(root)),
            }],
            ..Plan::default()
        })
    }
}

pub(crate) enum Production {
    Executed,
    Explain,
}

struct Walk<'r> {
    rules: &'r dyn SubstraitRules,
    production: Production,
    extensions: Extensions,
}

impl Walk<'_> {
    fn rel(&mut self, offered: &PlanNode) -> Result<Rel, Diagnostic<InvalidRewrite>> {
        let replacement = self.rules.rewrite_node(offered);
        let node = match &replacement {
            Some(node) => {
                let inputs = node.children().into_iter().cloned().collect();
                let rebuilt = node.clone().with_new_children(inputs).ok();
                let derived = rebuilt
                    .as_ref()
                    .map(|rebuilt| &rebuilt.common().output_schema);
                if derived != Some(&offered.common().output_schema) {
                    return Err(offered.diagnostic(InvalidRewrite::SchemaChanged));
                }
                node
            }
            None => offered,
        };
        let inputs = (node.children().into_iter())
            .map(|input| self.rel(input))
            .collect::<Result<_, _>>()?;
        self.relation(node, inputs)
            .map_err(|error| node.diagnostic(error))
    }

    fn relation(&mut self, node: &PlanNode, inputs: Vec<Rel>) -> Result<Rel, InvalidRewrite> {
        fn fields(input: &PlanNode) -> &Schema {
            &input.common().output_schema
        }
        let mut inputs = inputs.into_iter();
        let mut next = || inputs.next().map(Box::new);
        let common = Some(rel_common(node.common())?);
        let advanced_extension = self.advanced_extension(node.common());
        let rel_type = match node {
            PlanNode::Scan(scan) => {
                let read = ReadRel {
                    common,
                    advanced_extension,
                    base_schema: Some(named_struct(&scan.schema, self.rules)?),
                    filter: (scan.filter.as_ref())
                        .map(|filter| self.expression(filter, &scan.schema).map(Box::new))
                        .transpose()?,
                    read_type: Some(self.read_type(&scan.reader, &scan.schema)?),
                    ..ReadRel::default()
                };
                return Ok(self.rules.reader(scan, read));
            }
            PlanNode::Filter(filter) => RelType::Filter(Box::new(FilterRel {
                common,
                input: next(),
                condition: Some(Box::new(
                    self.expression(&filter.filter, fields(&filter.input))?,
                )),
                advanced_extension,
            })),
            PlanNode::Project(project) => RelType::Project(Box::new(ProjectRel {
                common,
                input: next(),
                expressions: self.expressions(&project.expressions, fields(&project.input))?,
                advanced_extension,
            })),
            PlanNode::Aggregate(aggregate) => {
                let fields = fields(&aggregate.input);
                let grouping_expressions = self.expressions(&aggregate.groupings, fields)?;
                let grouping = Grouping {
                    expression_references: (0..grouping_expressions.len())
                        .map(number)
                        .collect::<Result<_, _>>()?,
                    ..Grouping::default()
                };
                RelType::Aggregate(Box::new(AggregateRel {
                    common,
                    input: next(),
                    groupings: vec![grouping],
                    measures: (aggregate.aggregates.iter())
                        .map(|(_, call)| self.measure(call, fields))
                        .collect::<Result<_, _>>()?,
                    grouping_expressions,
                    advanced_extension,
                }))
            }
            PlanNode::Sort(sort) => RelType::Sort(Box::new(SortRel {
                common,
                input: next(),
                sorts: self.sorts(&sort.order_by, fields(&sort.input))?,
                advanced_extension,
            })),
            PlanNode::Fetch(fetch) => RelType::Fetch(Box::new(FetchRel {
                common,
                input: next(),
                offset_mode: Some(OffsetMode::OffsetExpr(Box::new(
                    self.literal(&Literal::BigInt(fetch.offset))?,
                ))),
                count_mode: (fetch
                    .limit
                    .map(|limit| self.literal(&Literal::BigInt(limit))))
                .transpose()?
                .map(|count| CountMode::CountExpr(Box::new(count))),
                advanced_extension,
            })),
            PlanNode::Join(join) => RelType::Join(Box::new(JoinRel {
                common,
                left: next(),
                right: next(),
                expression: Some(Box::new(self.expression(&join.on, &join.concatenated())?)),
                post_join_filter: (join.post_join_filter.as_ref())
                    .map(|filter| {
                        self.expression(filter, &join.natural_output())
                            .map(Box::new)
                    })
                    .transpose()?,
                r#type: match join.join_type {
                    JoinType::Inner => join_rel::JoinType::Inner,
                    JoinType::Left => join_rel::JoinType::Left,
                    JoinType::Right => join_rel::JoinType::Right,
                    JoinType::Full => join_rel::JoinType::Outer,
                    _ => {
                        return Err(InvalidRewrite::UnknownForm {
                            part: VocabularyPart::JoinType,
                        })
                    }
                }
                .into(),
                advanced_extension,
            })),
            PlanNode::Cross(_) => RelType::Cross(Box::new(CrossRel {
                common,
                left: next(),
                right: next(),
                advanced_extension,
            })),
            PlanNode::Multiset(multiset) => RelType::Set(SetRel {
                common,
                inputs: inputs.collect(),
                op: match multiset.op {
                    MultisetOp::UnionAll => set_rel::SetOp::UnionAll,
                    MultisetOp::UnionDistinct => set_rel::SetOp::UnionDistinct,
                    _ => {
                        return Err(InvalidRewrite::UnknownForm {
                            part: VocabularyPart::SetOperation,
                        })
                    }
                }
                .into(),
                advanced_extension,
            }),
            _ => {
                return Err(InvalidRewrite::NoLowering {
                    node: node.type_name(),
                })
            }
        };
        Ok(Rel {
            rel_type: Some(rel_type),
        })
    }

    fn advanced_extension(&self, common: &NodeCommon) -> Option<AdvancedExtension> {
        match self.production {
            Production::Executed => None,
            Production::Explain => trace(&common.semantic_meta),
        }
    }

    fn read_type(&mut self, reader: &Reader, schema: &Schema) -> Result<ReadType, InvalidRewrite> {
        Ok(match reader {
            Reader::VirtualTable(table) => ReadType::VirtualTable(read_rel::VirtualTable {
                expressions: (table.expressions.iter())
                    .map(|row| {
                        let fields = self.expressions(row, schema)?;
                        Ok(nested::Struct { fields })
                    })
                    .collect::<Result<_, _>>()?,
                ..read_rel::VirtualTable::default()
            }),
            Reader::FilesScan(files) => ReadType::LocalFiles(local_files(files)),
            Reader::TableScan(TableScan::NamedTable { names, .. }) => {
                ReadType::NamedTable(read_rel::NamedTable {
                    names: names.to_vec(),
                    advanced_extension: None,
                })
            }
            _ => {
                return Err(InvalidRewrite::UnknownForm {
                    part: VocabularyPart::Reader,
                })
            }
        })
    }

    fn expressions(
        &mut self,
        exprs: &[PhysicalExpr],
        input: &Schema,
    ) -> Result<Vec<Expression>, InvalidRewrite> {
        (exprs.iter())
            .map(|expr| self.expression(expr, input))
            .collect()
    }

    fn expression(
        &mut self,
        offered: &PhysicalExpr,
        input: &Schema,
    ) -> Result<Expression, InvalidRewrite> {
        let replacement = self.rules.rewrite_expression(offered, input);
        let expr = match &replacement {
            Some(expr) => {
                let offered = type_of(input, offered)?.data_type;
                let replacement = type_of(input, expr)?.data_type;
                if offered != replacement {
                    return Err(InvalidRewrite::TypeChanged {
                        offered,
                        replacement,
                    });
                }
                expr
            }
            None => offered,
        };
        let rex_type = match expr {
            Expr::Literal(value) => return self.literal(value),
            Expr::Column(leaf) => {
                let position =
                    (input.position(leaf)).ok_or_else(|| InvalidRewrite::UnboundColumn {
                        column: leaf.clone(),
                    })?;
                field_reference(position)?
            }
            Expr::BinaryOp { op, left, right } => {
                let operands = [self.operand(left, input)?, self.operand(right, input)?];
                return self.binary(*op, operands, &type_of(input, expr)?);
            }
            Expr::UnaryOp { op, operand } => {
                let operands = Operands::of(&[type_of(input, operand)?.data_type]);
                let form = functions::unary(*op, operands);
                let arguments = vec![self.value(operand, input)?];
                return self.operation(op.operation(), form, arguments, &type_of(input, expr)?);
            }
            Expr::Cast {
                input: value,
                target,
                on_failure,
            } => {
                let failure = match on_failure {
                    CastFailure::Error => FailureBehavior::ThrowException,
                    CastFailure::Null => FailureBehavior::ReturnNull,
                    _ => {
                        return Err(InvalidRewrite::UnknownForm {
                            part: VocabularyPart::CastFailure,
                        })
                    }
                };
                let to = proto_type(target, type_of(input, expr)?.nullable, self.rules)?;
                return Ok(cast(self.expression(value, input)?, to, failure));
            }
            Expr::Case { whens, else_ } => RexType::IfThen(Box::new(IfThen {
                ifs: (whens.iter())
                    .map(|(when, then)| {
                        Ok(IfClause {
                            r#if: Some(self.expression(when, input)?),
                            then: Some(self.expression(then, input)?),
                        })
                    })
                    .collect::<Result<_, InvalidRewrite>>()?,
                r#else: (else_.as_deref())
                    .map(|else_| self.expression(else_, input).map(Box::new))
                    .transpose()?,
            })),
            Expr::InList {
                value,
                list,
                negated,
            } => {
                let in_list = Expression {
                    rex_type: Some(RexType::SingularOrList(Box::new(SingularOrList {
                        value: Some(Box::new(self.expression(value, input)?)),
                        options: self.expressions(list, input)?,
                    }))),
                };
                return self.negated(*negated, in_list, &type_of(input, expr)?);
            }
            Expr::Between {
                value,
                low,
                high,
                negated,
            } => {
                let typed = type_of(input, expr)?;
                let value = self.operand(value, input)?;
                let (low, high) = (self.operand(low, input)?, self.operand(high, input)?);
                let compared = |bound: &(Expression, Typed)| {
                    Typed::new(DataType::Boolean, value.1.nullable || bound.1.nullable)
                };
                let (from, to) = (compared(&low), compared(&high));
                let gte = self.binary(BinaryOpKind::GtEq, [value.clone(), low], &from)?;
                let lte = self.binary(BinaryOpKind::LtEq, [value, high], &to)?;
                let both = [(gte, from), (lte, to)];
                let between = self.binary(BinaryOpKind::And, both, &typed)?;
                return self.negated(*negated, between, &typed);
            }
            Expr::Like {
                value,
                pattern,
                kind,
            } => {
                let typed = type_of(input, expr)?;
                let (like, negated) = match kind {
                    LikeKind::NotLike => (LikeKind::Like, true),
                    LikeKind::NotILike => (LikeKind::ILike, true),
                    kind => (*kind, false),
                };
                let arguments = vec![self.value(value, input)?, self.value(pattern, input)?];
                let like =
                    self.operation(like.operation(), functions::like(like), arguments, &typed)?;
                return self.negated(negated, like, &typed);
            }
            Expr::IsNull(operand) => {
                let operation = OperationKind::IsNull.operation();
                let arguments = vec![self.value(operand, input)?];
                let typed = type_of(input, expr)?;
                return self.operation(operation, functions::IS_NULL, arguments, &typed);
            }
            Expr::Function(Call::Scalar(call)) => {
                let typed = type_of(input, expr)?;
                let output_type = self.output(&typed)?;
                let function = FunctionRef::Scalar(call.function());
                let lowered = self.arguments(function, call.args(), input)?;
                let whole =
                    (self.rules).whole_call(call, &lowered, &output_type, &mut self.extensions)?;
                return match whole {
                    Some(whole) => Ok(whole),
                    None => self.faithful(call, lowered, input, &typed, output_type),
                };
            }
            Expr::Function(Call::Window(call)) => {
                RexType::WindowFunction(self.window(call, input, &type_of(input, expr)?)?)
            }
            Expr::Function(Call::Aggregate(..)) => {
                return Err(InvalidRewrite::AggregateOutsideEntry)
            }
            _ => {
                return Err(InvalidRewrite::UnknownForm {
                    part: VocabularyPart::Expression,
                })
            }
        };
        Ok(Expression {
            rex_type: Some(rex_type),
        })
    }

    fn literal(&mut self, value: &Literal) -> Result<Expression, InvalidRewrite> {
        Ok(Expression {
            rex_type: Some(RexType::Literal(literal(value, self.rules)?)),
        })
    }

    fn value(
        &mut self,
        expr: &PhysicalExpr,
        input: &Schema,
    ) -> Result<FunctionArgument, InvalidRewrite> {
        self.expression(expr, input).map(argument)
    }

    fn operation(
        &mut self,
        operation: &Operation,
        form: Form,
        arguments: Vec<FunctionArgument>,
        typed: &Typed,
    ) -> Result<Expression, InvalidRewrite> {
        let declaration = &operation.canonical;
        let options = options(&form, declaration.options, &[]);
        let output_type = self.output(typed)?;
        self.scalar(declaration.name, &form, options, arguments, output_type)
    }

    fn operand(
        &mut self,
        expr: &PhysicalExpr,
        input: &Schema,
    ) -> Result<(Expression, Typed), InvalidRewrite> {
        Ok((self.expression(expr, input)?, type_of(input, expr)?))
    }

    fn binary(
        &mut self,
        op: BinaryOpKind,
        operands: [(Expression, Typed); 2],
        typed: &Typed,
    ) -> Result<Expression, InvalidRewrite> {
        let operation = op.operation();
        let declaration = &operation.canonical;
        let (lowered, types): (Vec<_>, Vec<_>) = operands.into_iter().unzip();
        let (form, arguments) = self.overloaded(
            |operands| functions::binary(op, operands),
            (&declaration.signature, &declaration.returns),
            &typed.data_type,
            &types,
            lowered.into_iter().map(argument).collect(),
        )?;
        self.operation(operation, form, arguments, typed)
    }

    fn overloaded(
        &self,
        form: impl FnOnce(Operands) -> Form,
        (signature, returns): (&Signature, &ReturnRule),
        result: &DataType,
        types: &[Typed],
        lowered: Vec<FunctionArgument>,
    ) -> Result<(Form, Vec<FunctionArgument>), InvalidRewrite> {
        let data_types: Vec<_> = (types.iter())
            .map(|typed| typed.data_type.clone())
            .collect();
        let form = form(Operands::of(&data_types));
        let overloads = form.overload_types(signature, returns, result, &data_types);
        let arguments = (lowered.into_iter().zip(types).zip(overloads))
            .map(|((lowered, typed), to)| match (lowered.arg_type, to) {
                (Some(ArgType::Value(value)), Some(to)) => {
                    let to = proto_type(&to, typed.nullable, self.rules)?;
                    Ok(argument(cast(value, to, FailureBehavior::ThrowException)))
                }
                (arg_type, _) => Ok(FunctionArgument { arg_type }),
            })
            .collect::<Result<_, InvalidRewrite>>()?;
        Ok((form, arguments))
    }

    fn negated(
        &mut self,
        negated: bool,
        expr: Expression,
        typed: &Typed,
    ) -> Result<Expression, InvalidRewrite> {
        if !negated {
            return Ok(expr);
        }
        let not = UnaryOpKind::Not.operation();
        self.operation(not, functions::NOT, vec![argument(expr)], typed)
    }

    fn anchor(&mut self, canonical: &str, form: &Form) -> Result<u32, InvalidRewrite> {
        let name = self.rules.function(canonical, form.name);
        self.extensions.declare(name)
    }

    fn scalar(
        &mut self,
        canonical: &str,
        form: &Form,
        options: Vec<FunctionOption>,
        arguments: Vec<FunctionArgument>,
        output_type: Type,
    ) -> Result<Expression, InvalidRewrite> {
        let function_reference = self.anchor(canonical, form)?;
        Ok(scalar_function(
            function_reference,
            arguments,
            options,
            output_type,
        ))
    }

    fn output(&self, typed: &Typed) -> Result<Type, InvalidRewrite> {
        proto_type(&typed.data_type, typed.nullable, self.rules)
    }

    fn faithful(
        &mut self,
        call: &ScalarCall<Parameter, AggregateAllowed>,
        lowered: Vec<FunctionArgument>,
        input: &Schema,
        typed: &Typed,
        output_type: Type,
    ) -> Result<Expression, InvalidRewrite> {
        let declaration = &call.function().canonical;
        match declaration.name {
            "date_trunc" => return self.date_trunc(call, lowered, output_type),
            "extract" => return self.extract(lowered, output_type),
            _ => {}
        }
        let types = types(call.args(), input)?;
        let (function_reference, arguments, options) = self.call(
            declaration,
            declaration.specific.options,
            call.options(),
            &types,
            lowered,
            &typed.data_type,
        )?;
        Ok(scalar_function(
            function_reference,
            arguments,
            options,
            output_type,
        ))
    }

    fn date_trunc(
        &mut self,
        call: &ScalarCall<Parameter, AggregateAllowed>,
        lowered: Vec<FunctionArgument>,
        output_type: Type,
    ) -> Result<Expression, InvalidRewrite> {
        let canonical = "date_trunc";
        let shape = || InvalidRewrite::CallShape {
            function: canonical,
        };
        let unbound = |error| InvalidRewrite::UnboundCall { error };
        let function = FunctionRef::Scalar(call.function());
        let mut bound: BTreeMap<_, _> = (function.parameters(call.args()).map_err(unbound)?)
            .into_iter()
            .map(|parameter| parameter.name)
            .zip(lowered)
            .collect();
        let value = bound.remove("value").ok_or_else(shape)?;
        let calendar = functions::calendar(
            enum_value(bound.get("granularity")).ok_or_else(shape)?,
            enum_value(bound.get("week_start")),
        )
        .ok_or_else(shape)?;
        let rounding = [
            enumerated("FLOOR"),
            enumerated(calendar.unit),
            enumerated(calendar.origin),
            argument(self.literal(&Literal::BigInt(calendar.multiple))?),
        ];
        if matches!(output_type.kind, Some(Kind::Date(_))) {
            let anchor = argument(self.literal(&Literal::Date(calendar.anchor.to_owned()))?);
            let arguments = iter::once(value).chain(rounding).chain([anchor]).collect();
            let form = &functions::ROUND_CALENDAR;
            return self.scalar(canonical, form, Vec::new(), arguments, output_type);
        }
        let timezone = match bound.remove("timezone") {
            Some(timezone) => timezone,
            None => {
                let default = (function.parameter("timezone").map_err(unbound)?).default();
                argument(self.literal(&default.ok_or_else(shape)?)?)
            }
        };
        let instant = zoned(output_type.clone());
        let utc = argument(self.literal(&Literal::String("UTC".to_owned()))?);
        let form = &functions::ASSUME_TIMEZONE;
        let utc = self.scalar(
            canonical,
            form,
            Vec::new(),
            vec![value, utc],
            instant.clone(),
        )?;
        let arguments = (iter::once(argument(utc)).chain(rounding))
            .chain([timezone.clone()])
            .collect();
        let form = &functions::ROUND_CALENDAR;
        let bucket = self.scalar(canonical, form, Vec::new(), arguments, instant)?;
        let arguments = vec![argument(bucket), timezone];
        let form = &functions::LOCAL_TIMESTAMP;
        self.scalar(canonical, form, Vec::new(), arguments, output_type)
    }

    fn extract(
        &mut self,
        lowered: Vec<FunctionArgument>,
        output_type: Type,
    ) -> Result<Expression, InvalidRewrite> {
        let mut lowered = lowered.into_iter();
        let (component, indexed) = enum_value(lowered.next().as_ref())
            .and_then(functions::component)
            .ok_or(InvalidRewrite::CallShape {
                function: "extract",
            })?;
        let arguments = (iter::once(enumerated(component)))
            .chain(indexed.then(|| enumerated("ONE")))
            .chain(lowered)
            .collect();
        self.scalar(
            "extract",
            &functions::EXTRACT,
            Vec::new(),
            arguments,
            output_type,
        )
    }

    fn arguments(
        &mut self,
        function: FunctionRef<AggregateAllowed>,
        args: &[PhysicalExpr],
        input: &Schema,
    ) -> Result<Vec<FunctionArgument>, InvalidRewrite> {
        let parameters =
            (function.parameters(args)).map_err(|error| InvalidRewrite::UnboundCall { error })?;
        (args.iter().zip(parameters))
            .map(|(arg, parameter)| match (&parameter.form, arg) {
                (ArgumentForm::Enum(_), Expr::Literal(Literal::String(value))) => {
                    Ok(enumerated(value))
                }
                _ => self.value(arg, input),
            })
            .collect()
    }

    fn call<F: FunctionType, N: OptionName>(
        &mut self,
        declaration: &Declaration<F>,
        canonical: &[OptionDecl<N>],
        settings: &[OptionSetting],
        types: &[Typed],
        lowered: Vec<FunctionArgument>,
        result: &DataType,
    ) -> Result<(u32, Vec<FunctionArgument>, Vec<FunctionOption>), InvalidRewrite> {
        let name = declaration.name;
        let (form, lowered) = self.overloaded(
            |operands| functions::function(name, operands, types.len()),
            (&declaration.signature, &declaration.returns),
            result,
            types,
            lowered,
        )?;
        let arguments = if form.arguments.is_empty() {
            lowered
        } else {
            let mut lowered: Vec<_> = lowered.into_iter().map(Some).collect();
            let mut take = |position: usize| lowered.get_mut(position).and_then(Option::take);
            (form.arguments.iter())
                .map(|slot| {
                    Ok(match *slot {
                        Argument::Call(position) => {
                            take(position).ok_or(InvalidRewrite::CallShape { function: name })?
                        }
                        Argument::CallOr(position, constant) => match take(position) {
                            Some(lowered) => lowered,
                            None => argument(self.literal(&constant.literal())?),
                        },
                        Argument::Literal(constant) => argument(self.literal(&constant.literal())?),
                        Argument::Enum(value) => enumerated(value),
                    })
                })
                .collect::<Result<_, InvalidRewrite>>()?
        };
        let options = options(&form, canonical, settings);
        Ok((self.anchor(name, &form)?, arguments, options))
    }

    fn measure(
        &mut self,
        call: &AggregateCall<Parameter, AggregateAllowed>,
        input: &Schema,
    ) -> Result<Measure, InvalidRewrite> {
        let declaration = &call.function().declarations.canonical;
        let function = FunctionRef::Aggregate(call.function(), AggregateAllowed);
        let lowered = self.arguments(function, call.args(), input)?;
        let entry = Expr::Function(Call::Aggregate(call.clone(), AggregateAllowed));
        let typed = type_of(input, &entry)?;
        let types = types(call.args(), input)?;
        let (function_reference, arguments, options) = self.call(
            declaration,
            declaration.specific.options,
            call.options(),
            &types,
            lowered,
            &typed.data_type,
        )?;
        let invocation = if call.distinct() {
            AggregationInvocation::Distinct
        } else {
            AggregationInvocation::All
        };
        Ok(Measure {
            measure: Some(AggregateFunction {
                function_reference,
                arguments,
                options,
                output_type: Some(self.output(&typed)?),
                phase: AggregationPhase::InitialToResult.into(),
                sorts: self.sorts(call.order_by(), input)?,
                invocation: invocation.into(),
                ..AggregateFunction::default()
            }),
            filter: (call.filter())
                .map(|filter| self.expression(filter, input))
                .transpose()?,
        })
    }

    fn window(
        &mut self,
        call: &WindowCall<Parameter, AggregateAllowed>,
        input: &Schema,
        typed: &Typed,
    ) -> Result<WindowFunction, InvalidRewrite> {
        let declaration = &call.function().canonical;
        let lowered = self.arguments(FunctionRef::Window(call.function()), call.args(), input)?;
        let types = types(call.args(), input)?;
        let (function_reference, arguments, options) = self.call(
            declaration,
            declaration.specific.options,
            call.options(),
            &types,
            lowered,
            &typed.data_type,
        )?;
        let over = call.over();
        let (bounds_type, lower_bound, upper_bound) =
            bounds(over, declaration.specific.clauses.window_frame)?;
        Ok(WindowFunction {
            function_reference,
            arguments,
            options,
            output_type: Some(self.output(typed)?),
            phase: AggregationPhase::InitialToResult.into(),
            sorts: self.sorts(&over.order_by, input)?,
            invocation: AggregationInvocation::All.into(),
            partitions: self.expressions(&over.partition_by, input)?,
            bounds_type: bounds_type.into(),
            lower_bound,
            upper_bound,
            ..WindowFunction::default()
        })
    }

    fn sorts(
        &mut self,
        keys: &[SortKey<Parameter, AggregateAllowed>],
        input: &Schema,
    ) -> Result<Vec<SortField>, InvalidRewrite> {
        (keys.iter())
            .map(|key| {
                Ok(SortField {
                    expr: Some(self.expression(&key.expr, input)?),
                    sort_kind: Some(SortKind::Direction(direction(key.dir)?.into())),
                })
            })
            .collect()
    }
}

fn argument(expr: Expression) -> FunctionArgument {
    arg(ArgType::Value(expr))
}

fn enumerated(value: &str) -> FunctionArgument {
    arg(ArgType::Enum(value.to_owned()))
}

fn enum_value(argument: Option<&FunctionArgument>) -> Option<&str> {
    match argument?.arg_type.as_ref()? {
        ArgType::Enum(value) => Some(value),
        _ => None,
    }
}

fn arg(arg_type: ArgType) -> FunctionArgument {
    FunctionArgument {
        arg_type: Some(arg_type),
    }
}

fn options<N: OptionName>(
    form: &Form,
    canonical: &[OptionDecl<N>],
    settings: &[OptionSetting],
) -> Vec<FunctionOption> {
    let find = |options: &[(&'static str, &'static str)], name: &str| {
        (options.iter()).find_map(|&(option, value)| (option == name).then_some(value))
    };
    let row = (form.options.iter()).map(|&(name, _)| name);
    (form.declared.iter().copied())
        .chain(row.filter(|name| !form.declared.contains(name)))
        .filter_map(|name| {
            let setting = (settings.iter()).find(|setting| setting.name == name);
            let declaration = (canonical.iter()).find(|option| option.name.name() == name);
            let value = (setting.map(|setting| setting.value.as_str()))
                .or_else(|| find(form.options, name))
                .or_else(|| declaration.map(|option| option.canonical))
                .or_else(|| find(functions::CANONICAL_OPTIONS, name))?;
            Some(FunctionOption {
                name: name.to_owned(),
                preference: vec![value.to_owned()],
            })
        })
        .collect()
}

fn scalar_function(
    function_reference: u32,
    arguments: Vec<FunctionArgument>,
    options: Vec<FunctionOption>,
    output_type: Type,
) -> Expression {
    Expression {
        rex_type: Some(RexType::ScalarFunction(ScalarFunction {
            function_reference,
            arguments,
            options,
            output_type: Some(output_type),
            ..ScalarFunction::default()
        })),
    }
}

fn cast(input: Expression, to: Type, failure: FailureBehavior) -> Expression {
    Expression {
        rex_type: Some(RexType::Cast(Box::new(expression::Cast {
            r#type: Some(to),
            input: Some(Box::new(input)),
            failure_behavior: failure.into(),
        }))),
    }
}

fn types(args: &[PhysicalExpr], input: &Schema) -> Result<Vec<Typed>, InvalidRewrite> {
    (args.iter()).map(|arg| type_of(input, arg)).collect()
}

fn type_of(input: &Schema, expr: &PhysicalExpr) -> Result<Typed, InvalidRewrite> {
    input
        .type_of(expr)
        .map_err(|error| InvalidRewrite::Untyped { error })
}

fn direction(dir: SortDir) -> Result<SortDirection, InvalidRewrite> {
    Ok(match dir {
        SortDir::Asc {
            nulls: NullOrdering::First,
        } => SortDirection::AscNullsFirst,
        SortDir::Asc {
            nulls: NullOrdering::Last,
        } => SortDirection::AscNullsLast,
        SortDir::Desc {
            nulls: NullOrdering::First,
        } => SortDirection::DescNullsFirst,
        SortDir::Desc {
            nulls: NullOrdering::Last,
        } => SortDirection::DescNullsLast,
        _ => {
            return Err(InvalidRewrite::UnknownForm {
                part: VocabularyPart::SortDirection,
            })
        }
    })
}

fn bounds(
    over: &Over<Parameter, AggregateAllowed>,
    rule: ClauseRule,
) -> Result<(BoundsType, Option<Bound>, Option<Bound>), InvalidRewrite> {
    let ends_at_current_row = rule == ClauseRule::Allowed && !over.order_by.is_empty();
    let sql_default = ends_at_current_row.then_some(WindowFrame::new(
        WindowFrameKind::Range,
        WindowBound::UnboundedPreceding,
        WindowBound::CurrentRow,
    ));
    let Some(window_frame) = over.window_frame.or(sql_default) else {
        return Ok((BoundsType::Unspecified, None, None));
    };
    let unfit = || InvalidRewrite::UnknownForm {
        part: VocabularyPart::WindowFrame,
    };
    let bounds_type = match window_frame.kind {
        WindowFrameKind::Rows => BoundsType::Rows,
        WindowFrameKind::Range => BoundsType::Range,
        _ => return Err(unfit()),
    };
    let offset = |offset| i64::try_from(offset).map_err(|_| unfit());
    let bound = |bound| {
        let kind = match bound {
            WindowBound::UnboundedPreceding | WindowBound::UnboundedFollowing => {
                BoundKind::Unbounded(Unbounded {})
            }
            WindowBound::CurrentRow | WindowBound::Preceding(0) | WindowBound::Following(0) => {
                BoundKind::CurrentRow(CurrentRow {})
            }
            WindowBound::Preceding(n) => BoundKind::Preceding(Preceding { offset: offset(n)? }),
            WindowBound::Following(n) => BoundKind::Following(Following { offset: offset(n)? }),
            _ => return Err(unfit()),
        };
        Ok(Some(Bound { kind: Some(kind) }))
    };
    Ok((
        bounds_type,
        bound(window_frame.start)?,
        bound(window_frame.end)?,
    ))
}

fn rel_common(common: &NodeCommon) -> Result<RelCommon, InvalidRewrite> {
    let emit_kind = match &common.emit_form {
        None => EmitKind::Direct(rel_common::Direct {}),
        Some(emit_form) => EmitKind::Emit(rel_common::Emit {
            output_mapping: (emit_form.output_mapping.iter().copied())
                .map(number)
                .collect::<Result<_, _>>()?,
        }),
    };
    Ok(RelCommon {
        emit_kind: Some(emit_kind),
        hint: common.hint.as_ref().map(hint),
        advanced_extension: None,
    })
}

fn field_reference(position: usize) -> Result<RexType, InvalidRewrite> {
    let segment = reference_segment::ReferenceType::StructField(Box::new(StructField {
        field: number(position)?,
        child: None,
    }));
    Ok(RexType::Selection(Box::new(FieldReference {
        reference_type: Some(ReferenceType::DirectReference(ReferenceSegment {
            reference_type: Some(segment),
        })),
        root_type: Some(RootType::RootReference(RootReference {})),
    })))
}

fn local_files(files: &FilesScan) -> LocalFiles {
    let folder_read = files.paths.is_empty() && files.pattern.is_none();
    let paths = if folder_read {
        vec![PathType::UriFolder(files.base_path.clone())]
    } else {
        (files.paths.iter().cloned())
            .map(PathType::UriFile)
            .collect()
    };
    let file_format = item_format(files.format, files.format_options.as_deref());
    LocalFiles {
        items: (paths.into_iter())
            .map(|path_type| FileOrFiles {
                path_type: Some(path_type),
                file_format: file_format.clone(),
                ..FileOrFiles::default()
            })
            .collect(),
        advanced_extension: None,
    }
}

fn item_format(
    format: Option<FileFormat>,
    options: Option<&TextReadOptions>,
) -> Option<ItemFormat> {
    match (format?, options) {
        (FileFormat::Parquet, _) => Some(ItemFormat::Parquet(ParquetReadOptions {})),
        (FileFormat::Orc, _) => Some(ItemFormat::Orc(OrcReadOptions {})),
        (FileFormat::Csv, Some(options)) => {
            Some(ItemFormat::Text(DelimiterSeparatedTextReadOptions {
                field_delimiter: options.field_delimiter.clone(),
                max_line_size: options.max_line_size,
                quote: options.quote.clone(),
                header_lines_to_skip: options.header_lines_to_skip,
                escape: options.escape.clone(),
                value_treated_as_null: options.value_treated_as_null.clone(),
            }))
        }
        _ => None,
    }
}

fn hint(hint: &Hint) -> rel_common::Hint {
    let computation_type = |computation_type: ComputationType| {
        i32::from(match computation_type {
            ComputationType::Hashtable => hint::ComputationType::Hashtable,
            ComputationType::BloomFilter => hint::ComputationType::BloomFilter,
            _ => hint::ComputationType::Unknown,
        })
    };
    rel_common::Hint {
        stats: hint.stats.map(|stats| hint::Stats {
            row_count: stats.row_count,
            record_size: stats.record_size,
            advanced_extension: None,
        }),
        constraint: hint.constraint.map(|_| hint::RuntimeConstraint {
            advanced_extension: None,
        }),
        alias: hint.alias.clone().unwrap_or_default(),
        output_names: hint.output_names.clone(),
        advanced_extension: None,
        saved_computations: (hint.saved_computations.iter())
            .map(|saved| hint::SavedComputation {
                computation_id: saved.id,
                r#type: computation_type(saved.computation_type),
                advanced_extension: None,
            })
            .collect(),
        loaded_computations: (hint.loaded_computations.iter())
            .map(|loaded| hint::LoadedComputation {
                computation_id_reference: loaded.id,
                r#type: computation_type(loaded.computation_type),
                advanced_extension: None,
            })
            .collect(),
    }
}
