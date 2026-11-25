use std::iter;

use semstrait_common::Dialect;

use super::call::{bind, CallOperands};
use super::literal::{literal_type, read_untyped};
use super::{
    Aggregation, Call, CallClause, CastFailure, Expr, Leaf, Literal, NamedReference, Operation,
    OperationDeclaration, OperationKind, OptionSetting, TypeError, UntypedLiteral,
};
use crate::datatype::family::{one_family, value_casts_to};
use crate::datatype::{DataType, TypeFamily, TypeOrFamily, TypeSystem};
use crate::functions::signature::{
    at_t, check_typed_arguments, derives_from_operands, given, nullable, one_exact_type,
    option_values, result_type,
};
use crate::functions::{
    ArgumentRefusal, ClauseRule, Constraint, Declaration, Function, FunctionType, Nullability,
    OptionDecl, OptionName, ParameterDecl, ParameterType, ReturnRule, Signature, BOOLEAN,
};
use crate::DocumentDialect;

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Typed {
    pub data_type: DataType,
    pub nullable: bool,
    pub family: Option<TypeFamily>,
}

impl Typed {
    pub fn new(data_type: DataType, nullable: bool) -> Self {
        Self {
            family: data_type.family(),
            data_type,
            nullable,
        }
    }

    pub(crate) fn fits(&self, parameter: &ParameterType) -> bool {
        parameter.takes_value(&self.data_type, self.family.as_ref())
    }

    pub(crate) fn named(&self) -> TypeOrFamily {
        TypeOrFamily::of(&self.data_type, self.family.as_ref())
    }
}

pub fn check_cast(
    from: &DataType,
    family: Option<&TypeFamily>,
    to: &DataType,
) -> Result<(), TypeError> {
    if value_casts_to(from, family, to) {
        Ok(())
    } else {
        Err(TypeError::Cast {
            from: TypeOrFamily::of(from, family),
            to: to.clone(),
        })
    }
}

pub fn is_condition(data_type: &DataType, family: Option<&TypeFamily>) -> bool {
    BOOLEAN.takes_value(data_type, family)
}

fn check_clause(
    name: &'static str,
    clause: CallClause,
    takes: &'static ParameterType,
    got: Typed,
) -> Result<(), TypeError> {
    if got.fits(takes) {
        Ok(())
    } else {
        Err(TypeError::Clause {
            name,
            clause,
            takes,
            got: got.named(),
        })
    }
}

fn refused(name: &'static str, refusal: ArgumentRefusal) -> TypeError {
    match refusal {
        ArgumentRefusal::Argument {
            parameter,
            takes,
            got,
        } => TypeError::Argument {
            name,
            parameter,
            takes,
            got,
        },
        ArgumentRefusal::Arguments { types } => TypeError::Arguments { name, types },
        ArgumentRefusal::TypeVariable { bound, got } => {
            TypeError::TypeVariable { name, bound, got }
        }
        ArgumentRefusal::EnumRefused {
            parameter,
            value,
            got,
        } => TypeError::EnumRefused {
            name,
            parameter,
            value,
            got,
        },
        ArgumentRefusal::ParameterRefused { parameter } => {
            TypeError::ParameterRefused { name, parameter }
        }
        ArgumentRefusal::ParameterRequired { parameter } => {
            TypeError::ParameterRequired { name, parameter }
        }
    }
}

pub trait InputTypes<L: Leaf> {
    fn column(&self, column: &NamedReference) -> Typed;

    fn leaf(&self, leaf: &L) -> Typed;
}

pub fn type_tree<L: Leaf, A: Aggregation>(
    mut tree: Expr<L, A>,
    types: &dyn InputTypes<L>,
    dialect: DocumentDialect,
    type_system: &dyn TypeSystem,
    context: Option<&DataType>,
) -> Result<(Expr<L, A>, Typed), TypeError> {
    let typing = Typing {
        types,
        dialect,
        type_system,
    };
    let typed = typing.node(&mut tree, context)?;
    Ok((tree, typed))
}

struct Typing<'a, L: Leaf> {
    types: &'a dyn InputTypes<L>,
    dialect: DocumentDialect,
    type_system: &'a dyn TypeSystem,
}

type Arguments = (Vec<(DataType, Option<TypeFamily>)>, Vec<bool>);

impl<L: Leaf> Typing<'_, L> {
    fn node<A: Aggregation>(
        &self,
        expr: &mut Expr<L, A>,
        context: Option<&DataType>,
    ) -> Result<Typed, TypeError> {
        match expr {
            Expr::Literal(literal) => Ok(literal_type(literal)),
            Expr::Column(column) => Ok(self.types.column(column)),
            Expr::Leaf(leaf) => match leaf.untyped() {
                None => Ok(self.types.leaf(leaf)),
                Some(UntypedLiteral(literal)) => {
                    let literal = read_untyped(literal, context)?;
                    let typed = literal_type(&literal);
                    *expr = Expr::Literal(literal);
                    Ok(typed)
                }
            },
            Expr::BinaryOp { op, left, right } => {
                self.operation(op.operation(), &mut [left, right], context)
            }
            Expr::UnaryOp { op, operand } => {
                self.operation(op.operation(), &mut [operand], context)
            }
            Expr::Like {
                value,
                pattern,
                kind,
            } => self.operation(kind.operation(), &mut [value, pattern], context),
            Expr::Between {
                value, low, high, ..
            } => self.operation(
                OperationKind::Between.operation(),
                &mut [value, low, high],
                context,
            ),
            Expr::InList { value, list, .. } => {
                let mut operands: Vec<&mut Expr<L, A>> =
                    iter::once(&mut **value).chain(list).collect();
                self.operation(OperationKind::InList.operation(), &mut operands, context)
            }
            Expr::IsNull(operand) => {
                self.operation(OperationKind::IsNull.operation(), &mut [operand], context)
            }
            Expr::Case { whens, else_ } => {
                let (whens, thens) = whens.iter_mut().map(|(when, then)| (when, then)).unzip();
                self.case(whens, thens, else_.as_deref_mut(), context)
            }
            Expr::Cast {
                input,
                target,
                on_failure,
            } => {
                let input = self.node(input, None)?;
                check_cast(&input.data_type, input.family.as_ref(), target)?;
                let fails_to_null = match on_failure {
                    CastFailure::Error => false,
                    CastFailure::Null => true,
                };
                Ok(Typed::new(target.clone(), input.nullable || fails_to_null))
            }
            Expr::Function(call) => self.call(call, context),
        }
    }

    fn operation<A: Aggregation>(
        &self,
        operation: &'static Operation,
        operands: &mut [&mut Expr<L, A>],
        context: Option<&DataType>,
    ) -> Result<Typed, TypeError> {
        let declaration = self.operation_declaration(operation)?;
        let signature = &declaration.signature;
        let binding: Vec<_> = (signature.parameters.iter().enumerate())
            .chain(signature.tail())
            .take(operands.len())
            .collect();
        let parts = Parts {
            name: operation.canonical.name,
            signature,
            returns: &declaration.returns,
            nullability: &declaration.nullability,
            options: option_values(declaration.options, |_| None),
            within_group: None,
        };
        (self.apply(&parts, &binding, operands, context)).map(|(typed, _)| typed)
    }

    fn call<A: Aggregation>(
        &self,
        call: &mut Call<L, A>,
        context: Option<&DataType>,
    ) -> Result<Typed, TypeError> {
        let parts = self.function(call)?;
        let binding = bind(parts.name, parts.signature, call.args()).map_err(TypeError::Call)?;
        let window_frame = match call {
            Call::Window(call) => call.over().window_frame,
            Call::Scalar(_) | Call::Aggregate(..) => None,
        };
        let CallOperands {
            mut args,
            filter,
            partition_by,
            order_by,
        } = call.operands_mut();
        let (typed, defaults) = self.apply(&parts, &binding, &mut args, context)?;
        if let Some(filter) = filter {
            let got = self.node(filter, Some(&DataType::Boolean))?;
            check_clause(parts.name, CallClause::Filter, &BOOLEAN, got)?;
        }
        for key in partition_by {
            self.node(key, None)?;
        }
        for key in order_by {
            let got = self.node(key, None)?;
            if let Some(window_frame) = window_frame {
                window_frame.check_order_key(parts.name, got.family.as_ref())?;
            }
            if let Some(takes) = parts.within_group {
                check_clause(parts.name, CallClause::WithinGroup, takes, got)?;
            }
        }
        for (written, (index, default)) in defaults.into_iter().enumerate() {
            let before = binding.iter().filter(|(at, _)| *at < index).count();
            (call.insert_argument(before + written, Expr::Literal(default)))
                .map_err(TypeError::Call)?;
        }
        Ok(typed)
    }

    fn function<A: Aggregation>(&self, call: &Call<L, A>) -> Result<Parts, TypeError> {
        let (name, settings) = (call.name(), call.options());
        Ok(match call {
            Call::Scalar(call) => {
                let declaration = self.function_declaration(call.function())?;
                Parts::of(name, declaration, declaration.specific.options, settings)
            }
            Call::Aggregate(call, _) => {
                let declaration = self.function_declaration(&call.function().declarations)?;
                let clauses = &declaration.specific.clauses;
                Parts {
                    within_group: (clauses.within_group != ClauseRule::Refused)
                        .then_some(&clauses.within_group_key),
                    ..Parts::of(name, declaration, declaration.specific.options, settings)
                }
            }
            Call::Window(call) => {
                let declaration = self.function_declaration(call.function())?;
                Parts::of(name, declaration, declaration.specific.options, settings)
            }
        })
    }

    fn function_declaration<F: FunctionType>(
        &self,
        function: &'static Function<F>,
    ) -> Result<&'static Declaration<F>, TypeError> {
        let canonical = &function.canonical;
        self.in_dialect(canonical, |on| function.on(on), canonical.name)
    }

    fn operation_declaration(
        &self,
        operation: &'static Operation,
    ) -> Result<&'static OperationDeclaration, TypeError> {
        let canonical = &operation.canonical;
        self.in_dialect(canonical, |on| operation.on(on), canonical.name)
    }

    fn apply<A: Aggregation>(
        &self,
        parts: &Parts,
        binding: &[(usize, &'static ParameterDecl)],
        args: &mut [&mut Expr<L, A>],
        context: Option<&DataType>,
    ) -> Result<(Typed, Vec<(usize, Literal)>), TypeError> {
        let constraints = parts.signature.constraints;
        let derives = derives_from_operands(parts.returns);
        let binds = |known: &DataType| {
            let family = known.family();
            (derives || constraints.contains(&Constraint::OneFamily))
                && constraints.iter().all(|constraint| match constraint {
                    Constraint::Families(rows) => rows
                        .iter()
                        .any(|row| row.iter().all(|want| family.as_ref() == Some(want))),
                    Constraint::OneFamily | Constraint::EnumRefused { .. } => true,
                })
        };
        let (types, nullables) = self.arguments(args, |known| {
            let t_args = (binding.iter().zip(known)).filter(|((_, parameter), _)| at_t(parameter));
            let returns_t = *parts.returns == ReturnRule::Var;
            let t = given(
                t_args.map(|(_, known)| *known),
                |_| true,
                context.filter(|_| returns_t),
            );
            let others = given(known.iter().copied(), binds, context.filter(|_| derives));
            (binding.iter())
                .map(|(_, parameter)| match &parameter.ty {
                    ParameterType::Exact(exact) => Some(exact.data_type()),
                    ParameterType::Var(_) => t.clone(),
                    _ => others.clone(),
                })
                .collect()
        })?;
        let strings: Vec<_> = (args.iter())
            .map(|arg| match &**arg {
                Expr::Literal(Literal::String(value)) => Some(value.as_str()),
                _ => None,
            })
            .collect();
        let (t, defaults) = check_typed_arguments(parts.signature, binding, &types, &strings)
            .map_err(|refusal| refused(parts.name, refusal))?;
        let nullable = nullable(parts.nullability, &parts.options, binding, &nullables);
        let (data_type, kept) = result_type(self.type_system, parts.returns, binding, types, t);
        let typed = Typed {
            family: data_type.family().or(kept),
            data_type,
            nullable,
        };
        Ok((typed, defaults))
    }

    fn case<'tree, A: Aggregation>(
        &self,
        whens: Vec<&'tree mut Expr<L, A>>,
        mut branches: Vec<&'tree mut Expr<L, A>>,
        else_: Option<&'tree mut Expr<L, A>>,
        context: Option<&DataType>,
    ) -> Result<Typed, TypeError> {
        for when in whens {
            let when = self.node(when, Some(&DataType::Boolean))?;
            if !is_condition(&when.data_type, when.family.as_ref()) {
                return Err(TypeError::CaseWhen { got: when.named() });
            }
        }
        let no_else = else_.is_none();
        branches.extend(else_);
        let (types, nullables) = self.arguments(&mut branches, |known| {
            vec![given(known.iter().copied(), |_| true, context); known.len()]
        })?;
        let data_type = one_exact_type(&types)
            .map_err(|(first, other)| TypeError::CaseBranches { first, other })?;
        Ok(Typed {
            nullable: no_else || nullables.contains(&true),
            family: (data_type.family())
                .or_else(|| one_family(types.iter().map(|(_, family)| family.as_ref()))),
            data_type,
        })
    }

    fn arguments<A: Aggregation>(
        &self,
        args: &mut [&mut Expr<L, A>],
        contexts: impl FnOnce(&[Option<&DataType>]) -> Vec<Option<DataType>>,
    ) -> Result<Arguments, TypeError> {
        let known = args
            .iter_mut()
            .map(|arg| {
                (!self.needs_context(arg))
                    .then(|| self.node(arg, None))
                    .transpose()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let types: Vec<_> = (known.iter())
            .map(|known| known.as_ref().map(|known| &known.data_type))
            .collect();
        let contexts = contexts(&types);
        (args.iter_mut().zip(known).zip(contexts))
            .map(|((arg, known), context)| {
                let typed = known.map_or_else(|| self.node(arg, context.as_ref()), Ok)?;
                Ok(((typed.data_type, typed.family), typed.nullable))
            })
            .collect()
    }

    fn needs_context<A: Aggregation>(&self, expr: &Expr<L, A>) -> bool {
        let theirs =
            |returns: &ReturnRule| derives_from_operands(returns) || *returns == ReturnRule::Var;
        let derives = |operation| {
            self.operation_declaration(operation)
                .is_ok_and(|declaration| theirs(&declaration.returns))
        };
        match expr {
            Expr::Leaf(leaf) => leaf.untyped().is_some(),
            Expr::BinaryOp { op, left, right } => {
                derives(op.operation()) && self.needs_context(left) && self.needs_context(right)
            }
            Expr::UnaryOp { op, operand } => derives(op.operation()) && self.needs_context(operand),
            Expr::Case { whens, else_ } => (whens.iter().map(|(_, then)| then))
                .chain(else_.as_deref())
                .all(|branch| self.needs_context(branch)),
            Expr::Function(call) => {
                self.function(call).is_ok_and(|parts| theirs(parts.returns))
                    && call.args().iter().all(|arg| self.needs_context(arg))
            }
            _ => false,
        }
    }

    fn in_dialect<D>(
        &self,
        canonical: &'static D,
        on: impl FnOnce(Dialect) -> Option<&'static D>,
        name: &'static str,
    ) -> Result<&'static D, TypeError> {
        match self.dialect {
            DocumentDialect::Canonical => Ok(canonical),
            DocumentDialect::Dialect(dialect) => {
                on(dialect).ok_or(TypeError::NotInDialect { name, dialect })
            }
        }
    }
}

struct Parts {
    name: &'static str,
    signature: &'static Signature,
    returns: &'static ReturnRule,
    nullability: &'static Nullability,
    options: Vec<&'static str>,
    within_group: Option<&'static ParameterType>,
}

impl Parts {
    fn of<F: FunctionType, N: OptionName>(
        name: &'static str,
        declaration: &'static Declaration<F>,
        options: &'static [OptionDecl<N>],
        settings: &[OptionSetting],
    ) -> Self {
        Self {
            name,
            signature: &declaration.signature,
            returns: &declaration.returns,
            nullability: &declaration.nullability,
            options: option_values(options, |name| {
                (settings.iter())
                    .find(|setting| setting.name == name)
                    .map(|setting| setting.value.as_str())
            }),
            within_group: None,
        }
    }
}
