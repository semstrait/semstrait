use std::collections::BTreeSet;

use semstrait_common::SemanticsName;
use semstrait_common::Tree;
use semstrait_ir::SemanticRole;
use semstrait_model::UnresolvedSemanticExpr as SemanticExpr;
use semstrait_sql::datatype::{CanonicalTypeSystem, DataType, TypeOrFamily};
use semstrait_sql::expr::{
    check_cast, is_condition, map_nodes, type_tree, AggregateAllowed, AggregateCall, Call,
    CallError, CastFailure, Expr, FunctionRef, InputTypes, Leaf, Literal, NamedReference,
    RebuildError, SemanticLeaf, Typed, UntypedLiteral,
};
use semstrait_sql::functions::{registry, AggregateFunction};
use semstrait_sql::DocumentDialect;

use super::columns::ColumnTypes;
use crate::compile::types::{LeafPlace, LoweringCtx, Provenance, Reference, ResolvedExprEntry};
use crate::error::CompileError;
use crate::identity::{mint_expr_id, mint_signature_id};
use crate::manifest::expression::{ExprRoot, Link};

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum LoweredLeaf {
    Untyped(UntypedLiteral),
    Splice {
        tree: Expr<Link>,
        data_type: DataType,
    },
}

impl Leaf for LoweredLeaf {
    fn untyped(&self) -> Option<&UntypedLiteral> {
        match self {
            Self::Untyped(literal) => Some(literal),
            Self::Splice { .. } => None,
        }
    }
}

pub(crate) type LoweredExpr = Expr<LoweredLeaf>;

struct LoweredTypes<'a>(&'a ColumnTypes);

impl InputTypes<LoweredLeaf> for LoweredTypes<'_> {
    fn column(&self, column: &NamedReference) -> Typed {
        Typed::new(self.0.of(column.name()), true)
    }

    fn leaf(&self, leaf: &LoweredLeaf) -> Typed {
        let data_type = match leaf {
            LoweredLeaf::Splice { data_type, .. } => data_type.clone(),
            LoweredLeaf::Untyped(UntypedLiteral(literal)) => literal.data_type(),
        };
        Typed::new(data_type, true)
    }
}

#[derive(Debug)]
pub(crate) struct Lowered {
    tree: LoweredExpr,
    referenced_columns: BTreeSet<String>,
}

impl Lowered {
    pub(crate) fn new(
        tree: LoweredExpr,
        referenced_columns: impl IntoIterator<Item = String>,
    ) -> Self {
        Self {
            tree,
            referenced_columns: referenced_columns.into_iter().collect(),
        }
    }

    pub(crate) fn aggregated(
        self,
        aggregation: (&'static AggregateFunction, bool),
        semantic: &str,
    ) -> Result<Self, CompileError> {
        Ok(Self {
            tree: wrap_aggregate(self.tree, aggregation, semantic)?,
            ..self
        })
    }

    pub(crate) fn second_step(
        self,
        aggregation: (&'static AggregateFunction, bool),
        columns: &ColumnTypes,
        declared: Option<&DataType>,
        node: &str,
    ) -> Result<ResolvedExprEntry, CompileError> {
        let call = (self.aggregated(aggregation, node)?).typed(columns, declared, node)?;
        let Expr::Function(Call::Aggregate(aggregate, _)) = &call.expr else {
            return Err(unsupported(
                node,
                "a typed call of an `agg:` is an aggregate call",
            ));
        };
        let formula = (aggregate.args().first().cloned())
            .ok_or_else(|| unsupported(node, "the call of an `agg:` holds its formula"))?;
        Ok(ResolvedExprEntry {
            expr: formula,
            argument_type: None,
            ..reconcile_boundary(call, declared, node)?
        })
    }

    pub(crate) fn typed(
        self,
        columns: &ColumnTypes,
        context: Option<&DataType>,
        node: &str,
    ) -> Result<ResolvedExprEntry, CompileError> {
        if let (Expr::Literal(literal), Some(declared)) = (&self.tree, context) {
            let from = literal.data_type();
            if &from != declared && from.family() == declared.family() && !from.widens_to(declared)
            {
                return Err(CompileError::NarrowingLiteral {
                    node: node.to_owned(),
                    literal: literal.clone(),
                    declared: declared.clone(),
                });
            }
        }
        let type_of = |tree, context| {
            type_tree(
                tree,
                &LoweredTypes(columns),
                DocumentDialect::Canonical,
                &CanonicalTypeSystem,
                context,
            )
            .map_err(|error| CompileError::TypeError {
                node: node.to_owned(),
                error,
            })
        };
        let (tree, typed) = type_of(self.tree, context)?;
        utc_only(&tree, node)?;
        let argument_type = match &tree {
            Expr::Function(call @ Call::Aggregate(..)) => (call.args().first())
                .map(|argument| type_of(argument.clone(), None).map(|(_, typed)| typed.data_type))
                .transpose()?,
            _ => None,
        };
        let expr = map_nodes(
            &tree,
            &mut |expr| match expr {
                Expr::Leaf(LoweredLeaf::Splice { tree, .. }) => Some(Ok(tree.clone())),
                Expr::Leaf(LoweredLeaf::Untyped(_)) => Some(Err(unsupported(
                    node,
                    "a typed tree holds no untyped literal",
                ))),
                _ => None,
            },
            &|error| refused_rebuild(node, error, |error| unsupported(node, &error.to_string())),
        )?;
        Ok(ResolvedExprEntry {
            expr,
            inferred_type: typed.data_type,
            family: typed.family,
            argument_type,
            referenced_columns: self.referenced_columns.into_iter().collect(),
            provenance: Provenance::new(),
        })
    }

    pub(crate) fn typed_condition(
        self,
        columns: &ColumnTypes,
        node: &str,
    ) -> Result<ResolvedExprEntry, CompileError> {
        let entry = self.typed(columns, Some(&DataType::Boolean), node)?;
        let (data_type, family) = (&entry.inferred_type, entry.family.as_ref());
        if is_condition(data_type, family) {
            return Ok(entry);
        }
        Err(CompileError::TypeInferenceFailure {
            node: node.to_owned(),
            reason: format!(
                "expected Boolean, found {}",
                TypeOrFamily::of(data_type, family)
            ),
        })
    }
}

fn utc_only(tree: &LoweredExpr, node: &str) -> Result<(), CompileError> {
    let timezone = match tree {
        Expr::Function(call) if call.name() == "date_trunc" => {
            call.argument("timezone").and_then(Expr::literal)
        }
        _ => None,
    };
    match timezone {
        Some(timezone) if !matches!(timezone, Literal::String(zone) if zone == "UTC") => {
            Err(CompileError::DateTruncTimezoneNotUtc {
                node: node.to_owned(),
                timezone: timezone.clone(),
            })
        }
        _ => (tree.children().into_iter()).try_for_each(|operand| utc_only(operand, node)),
    }
}

pub(crate) fn resolve(
    expr: &SemanticExpr,
    ctx: &mut LoweringCtx<'_>,
) -> Result<Lowered, CompileError> {
    let owner = ctx.owner;
    let mut columns = BTreeSet::new();
    let tree = map_nodes(
        expr,
        &mut |node| lower_node(node, ctx, &mut columns),
        &|error| {
            refused_rebuild(owner.as_str(), error, |error| {
                unsupported(
                    owner.as_str(),
                    &format!("the body holds a form this build cannot lower: {error}"),
                )
            })
        },
    )?;
    Ok(Lowered::new(tree, columns))
}

pub(crate) fn reconcile_boundary(
    entry: ResolvedExprEntry,
    declared: Option<&DataType>,
    node: &str,
) -> Result<ResolvedExprEntry, CompileError> {
    let Some(declared) = declared else {
        if entry.inferred_type == DataType::Unknown {
            return Err(CompileError::TypeInferenceFailure {
                node: node.to_string(),
                reason: "the tree has no type, and its semantics declares no data_type to \
                         cast it to"
                    .to_string(),
            });
        }
        return Ok(entry);
    };
    if &entry.inferred_type == declared {
        return Ok(entry);
    }
    check_cast(&entry.inferred_type, entry.family.as_ref(), declared).map_err(|error| {
        CompileError::TypeError {
            node: node.to_owned(),
            error,
        }
    })?;
    Ok(ResolvedExprEntry {
        expr: cast_to_declared(entry.expr, declared),
        inferred_type: declared.clone(),
        family: declared.family(),
        ..entry
    })
}

pub(super) fn cast_to_declared<L: Leaf>(tree: Expr<L>, declared: &DataType) -> Expr<L> {
    Expr::Cast {
        input: Box::new(tree),
        target: declared.clone(),
        on_failure: CastFailure::Error,
    }
}

pub(crate) fn wrap_aggregate<L: Leaf>(
    input: Expr<L>,
    aggregation: (&'static AggregateFunction, bool),
    semantic: &str,
) -> Result<Expr<L>, CompileError> {
    let (function, distinct) = aggregation;
    let call = AggregateCall::builder(function, vec![input])
        .distinct(distinct)
        .build()
        .map_err(|error| {
            let suffix = if distinct { "_distinct" } else { "" };
            refused_call(
                &format!(
                    "measure {semantic} agg {}{suffix}",
                    function.declarations.canonical.name
                ),
                error,
            )
        })?;
    Ok(Expr::Function(Call::Aggregate(call, AggregateAllowed)))
}

pub(crate) fn catalog_function(
    name: &'static str,
    node: &str,
) -> Result<FunctionRef<AggregateAllowed>, CompileError> {
    registry()
        .get(name)
        .ok_or_else(|| unsupported(node, "the function catalog declares no such function"))
}

pub(crate) fn refused_call(node: &str, error: CallError) -> CompileError {
    CompileError::CallRefused {
        node: node.to_owned(),
        error,
    }
}

pub(crate) fn refused_rebuild(
    owner: &str,
    error: RebuildError,
    other: impl FnOnce(RebuildError) -> CompileError,
) -> CompileError {
    match error {
        RebuildError::Call(error) => refused_call(owner, error),
        error => other(error),
    }
}

fn unsupported(node: &str, reason: &str) -> CompileError {
    CompileError::TypeInferenceFailure {
        node: node.to_owned(),
        reason: reason.to_owned(),
    }
}

pub(crate) fn refused_window(owner: &str) -> CompileError {
    CompileError::TypeInferenceFailure {
        node: format!("{owner} expr window"),
        reason: "a window call's partition and order name a late-bound request axis, and \
                 what binds that axis is undetermined; an authored window call is refused \
                 rather than lowered onto a placeholder"
            .to_owned(),
    }
}

fn lower_node(
    expr: &SemanticExpr,
    ctx: &mut LoweringCtx<'_>,
    columns: &mut BTreeSet<String>,
) -> Option<Result<LoweredExpr, CompileError>> {
    Some(match expr {
        Expr::Leaf(leaf) => lower_leaf(leaf, ctx, columns),
        Expr::Column(column) => lower_column(column, ctx, columns),
        Expr::Function(Call::Window(_)) => Err(refused_window(ctx.owner.as_str())),
        _ => return None,
    })
}

fn lower_column(
    column: &NamedReference,
    ctx: &LoweringCtx<'_>,
    columns: &mut BTreeSet<String>,
) -> Result<LoweredExpr, CompileError> {
    if ctx.manual_mapping {
        return Err(CompileError::ColumnInSemanticExprUnderManualMapping {
            binding: ctx.binding.clone(),
        });
    }
    if let Some(unit) = ctx.composed_filter_unit {
        return Err(CompileError::ColumnInComposedFilter {
            unit: unit.to_string(),
            semantic: ctx.owner.as_str().to_owned(),
            column: column.name().to_owned(),
        });
    }
    columns.insert(column.name().to_owned());
    Ok(Expr::Column(column.clone()))
}

fn lower_leaf(
    leaf: &SemanticLeaf,
    ctx: &mut LoweringCtx<'_>,
    columns: &mut BTreeSet<String>,
) -> Result<LoweredExpr, CompileError> {
    let (name, role, accessor) = match leaf {
        SemanticLeaf::Untyped(literal) => {
            return Ok(Expr::Leaf(LoweredLeaf::Untyped(literal.clone())))
        }
        SemanticLeaf::Field(name) => (name, None, false),
        SemanticLeaf::Dimension(name, accessor) => {
            (name, Some(SemanticRole::Dimension), accessor.is_some())
        }
        SemanticLeaf::Measure(name, accessor) => {
            (name, Some(SemanticRole::Measure), accessor.is_some())
        }
        SemanticLeaf::Metric(name, accessor) => {
            (name, Some(SemanticRole::Metric), accessor.is_some())
        }
        _ => {
            return Err(unsupported(
                "<unknown semantic leaf>",
                "unsupported SemanticLeaf variant in body lowering",
            ))
        }
    };
    if accessor {
        return Err(refused_accessor(name, ctx));
    }
    substitute_semantic_leaf(name, role, ctx, columns)
}

fn refused_accessor(name: &NamedReference, ctx: &LoweringCtx<'_>) -> CompileError {
    CompileError::TypeInferenceFailure {
        node: format!("{} expr leaf {}", ctx.owner.as_str(), name.name()),
        reason: "a semantic leaf accessor (Previous / Next / Lag / Lead / Delta / \
                 PercentChange) lowers to a window over a late-bound request axis, and \
                 what binds that axis is undetermined; the accessor is refused rather \
                 than lowered onto a placeholder"
            .to_owned(),
    }
}

fn substitute_semantic_leaf(
    name: &NamedReference,
    authored_role: Option<SemanticRole>,
    ctx: &mut LoweringCtx<'_>,
    columns: &mut BTreeSet<String>,
) -> Result<LoweredExpr, CompileError> {
    let model_name = SemanticsName::from(name.name());

    let Some(definition) = ctx.all_semantics.get(&model_name) else {
        return Err(CompileError::UnknownReference {
            name: name.name().to_owned(),
            scope: crate::error::ReferenceScope::DataKind(ctx.binding.clone()),
        });
    };
    if let Some(authored_role) = authored_role {
        if definition.role() != authored_role {
            return Err(CompileError::SemanticKindMismatch {
                authored_kind: authored_role,
                registered_kind: definition.role(),
                name: name.name().to_owned(),
            });
        }
    }

    if definition.role() == SemanticRole::Filter
        && ctx.all_semantics.get(ctx.owner).map(|owner| owner.role()) != Some(SemanticRole::Filter)
    {
        return Err(CompileError::NamedFilterReference {
            semantics: ctx.owner.as_str().to_owned(),
            filter: name.name().to_owned(),
        });
    }

    if ctx.row_level
        && matches!(
            definition.role(),
            SemanticRole::Measure | SemanticRole::Metric
        )
    {
        return Err(CompileError::AggregateReferenceInRowLevelBody {
            semantics: ctx.owner.as_str().to_owned(),
            referenced: name.name().to_owned(),
        });
    }

    if !ctx.recursion.enter(model_name.clone()) {
        let mut cycle: Vec<String> = ctx
            .recursion
            .stack()
            .iter()
            .map(|n| n.as_str().to_string())
            .collect();
        cycle.push(name.name().to_owned());
        ctx.recursion.exit(&model_name);
        return Err(CompileError::CyclicReference { cycle });
    }

    let result = match (ctx.resolved.lookup(&model_name, ctx.binding), ctx.place) {
        (Some(entry), _) => {
            columns.extend(entry.referenced_columns.iter().cloned());
            let tree = match ctx.reference {
                Reference::Link => Expr::Leaf(Link(mint_expr_id(
                    &mint_signature_id(ctx.binding, &definition.id),
                    ExprRoot::SignatureExpression,
                ))),
                Reference::Inline => entry.expr.clone(),
            };
            Ok(Expr::Leaf(LoweredLeaf::Splice {
                tree,
                data_type: entry.inferred_type.clone(),
            }))
        }
        (None, Some(LeafPlace::UnionsetBranch)) => {
            let data_type = definition.data_type.clone().unwrap_or(DataType::Boolean);
            Ok(Expr::Leaf(LoweredLeaf::Splice {
                tree: Expr::Literal(Literal::Null(data_type.clone())),
                data_type,
            }))
        }
        (None, _) => Err(CompileError::UnknownReference {
            name: name.name().to_owned(),
            scope: crate::error::ReferenceScope::Appearance(ctx.binding.clone()),
        }),
    };

    ctx.recursion.exit(&model_name);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use semstrait_common::EntityId;
    use semstrait_model::Agg;
    use semstrait_sql::expr::{BinaryOpKind, Literal, MeasureAccessor, ScalarCall};

    use crate::compile::types::{
        RecursionState, ResolvedExprKey, ResolvedExprTable, SemanticsIndex,
    };
    use crate::manifest::semantic_definition::{SemanticDefinition, SemanticRoleDefinition};

    const BINDING: &str = "bind-a";

    struct Fixture {
        all_semantics: SemanticsIndex<'static>,
        resolved: ResolvedExprTable,
        binding: EntityId,
        owner: SemanticsName,
        row_level: bool,
        manual_mapping: bool,
        composed_filter_unit: Option<&'static str>,
        place: Option<LeafPlace>,
        reference: Reference,
        recursion: RecursionState,
    }

    impl Fixture {
        fn new() -> Self {
            Self {
                all_semantics: SemanticsIndex::new(),
                resolved: ResolvedExprTable::new(),
                binding: EntityId::from(BINDING),
                owner: SemanticsName::from("owner"),
                row_level: false,
                manual_mapping: false,
                composed_filter_unit: None,
                place: None,
                reference: Reference::Link,
                recursion: RecursionState::new(),
            }
        }

        fn ctx(&mut self) -> LoweringCtx<'_> {
            LoweringCtx {
                all_semantics: &self.all_semantics,
                resolved: &self.resolved,
                reference: self.reference,
                binding: &self.binding,
                owner: &self.owner,
                row_level: self.row_level,
                manual_mapping: self.manual_mapping,
                composed_filter_unit: self.composed_filter_unit,
                place: self.place,
                recursion: &mut self.recursion,
            }
        }

        fn entry(&mut self, expr: &SemanticExpr) -> Result<ResolvedExprEntry, CompileError> {
            resolve(expr, &mut self.ctx())?.typed(&ColumnTypes::default(), None, "owner")
        }
    }

    fn reference(name: &str) -> NamedReference {
        NamedReference::unqualified(name).expect("a name")
    }

    fn col_leaf(name: &str) -> SemanticExpr {
        Expr::Column(reference(name))
    }

    fn lit(literal: Literal) -> SemanticExpr {
        Expr::Literal(literal)
    }

    fn int_lit(v: i32) -> SemanticExpr {
        lit(Literal::Integer(v))
    }

    fn measure_leaf(name: &str) -> SemanticExpr {
        Expr::Leaf(SemanticLeaf::Measure(reference(name), None))
    }

    fn column(name: &str) -> Expr<Link> {
        Expr::Column(reference(name))
    }

    fn link_to(name: &str) -> Expr<Link> {
        Expr::Leaf(Link(mint_expr_id(
            &mint_signature_id(BINDING, name),
            ExprRoot::SignatureExpression,
        )))
    }

    fn semantics_index(defs: Vec<SemanticDefinition>) -> SemanticsIndex<'static> {
        let mut map: BTreeMap<SemanticsName, &'static SemanticDefinition> = BTreeMap::new();
        for def in defs {
            let leaked: &'static SemanticDefinition = Box::leak(Box::new(def));
            map.insert(leaked.name.clone(), leaked);
        }
        SemanticsIndex::from_map(map)
    }

    fn aggregation_of(name: &str) -> (&'static AggregateFunction, bool) {
        let agg = Agg::of_name(name).expect("a catalog aggregate");
        (agg.function(), agg.distinct())
    }

    fn sum_over(name: &str) -> Expr<Link> {
        wrap_aggregate(column(name), aggregation_of("sum"), "test").expect("sum wraps")
    }

    fn resolved_entry(
        expr: Expr<Link>,
        inferred_type: DataType,
        column: &str,
    ) -> ResolvedExprEntry {
        ResolvedExprEntry {
            expr,
            family: inferred_type.family(),
            inferred_type,
            argument_type: None,
            referenced_columns: vec![column.to_string()],
            provenance: Provenance::new(),
        }
    }

    fn def(name: &str, role: SemanticRole, ty: Option<DataType>) -> SemanticDefinition {
        SemanticDefinition::new(
            name,
            SemanticsName::from(name),
            ty,
            SemanticRoleDefinition::plain(role),
        )
    }

    #[test]
    fn literal_lowers_and_types() {
        let entry = Fixture::new()
            .entry(&int_lit(42))
            .expect("literal resolves");
        assert_eq!(entry.expr, Expr::Literal(Literal::Integer(42)));
        assert_eq!(entry.inferred_type, DataType::Integer);
        assert!(entry.referenced_columns.is_empty());
    }

    #[test]
    fn column_auto_lowers_and_harvests() {
        let entry = Fixture::new()
            .entry(&col_leaf("amount"))
            .expect("column resolves");
        assert_eq!(entry.expr, column("amount"));
        assert_eq!(entry.referenced_columns, vec!["amount".to_string()]);
        assert_eq!(entry.inferred_type, DataType::Unknown);
    }

    #[test]
    fn field_resolving_to_dimension_redispatches() {
        let mut fx = Fixture::new();
        fx.all_semantics = semantics_index(vec![def(
            "region",
            SemanticRole::Dimension,
            Some(DataType::String),
        )]);
        fx.resolved.insert(
            ResolvedExprKey::new(SemanticsName::from("region"), "bind-a"),
            resolved_entry(column("region_col"), DataType::String, "region_col"),
        );
        let field = Expr::Leaf(SemanticLeaf::Field(reference("region")));
        let entry = fx.entry(&field).expect("field resolves");
        assert_eq!(entry.expr, link_to("region"));
        assert_eq!(entry.inferred_type, DataType::String);
        assert_eq!(entry.referenced_columns, vec!["region_col".to_string()]);
    }

    #[test]
    fn field_unknown_is_unknown_reference() {
        let field = Expr::Leaf(SemanticLeaf::Field(reference("nope")));
        let err = Fixture::new()
            .entry(&field)
            .expect_err("unknown field errors");
        assert!(matches!(err, CompileError::UnknownReference { .. }));
    }

    #[test]
    fn typed_leaf_links_resolved_target() {
        let mut fx = Fixture::new();
        fx.all_semantics = semantics_index(vec![def(
            "revenue",
            SemanticRole::Measure,
            Some(DataType::BigInt),
        )]);
        fx.resolved.insert(
            ResolvedExprKey::new(SemanticsName::from("revenue"), "bind-a"),
            resolved_entry(sum_over("amount"), DataType::BigInt, "amount"),
        );
        let entry = fx
            .entry(&measure_leaf("revenue"))
            .expect("measure resolves");
        assert_eq!(entry.expr, link_to("revenue"));
        assert_eq!(entry.inferred_type, DataType::BigInt);
        assert_eq!(entry.referenced_columns, vec!["amount".to_string()]);
    }

    fn two_resolved_measures() -> Fixture {
        let mut fx = Fixture::new();
        fx.all_semantics = semantics_index(vec![
            def("revenue", SemanticRole::Measure, Some(DataType::BigInt)),
            def("cost", SemanticRole::Measure, Some(DataType::BigInt)),
        ]);
        for (name, column) in [("revenue", "amount"), ("cost", "spend")] {
            fx.resolved.insert(
                ResolvedExprKey::new(SemanticsName::from(name), "bind-a"),
                resolved_entry(sum_over(column), DataType::BigInt, column),
            );
        }
        fx
    }

    fn revenue_minus_cost() -> SemanticExpr {
        measure_leaf("revenue") - measure_leaf("cost")
    }

    #[test]
    fn a_metric_over_measures_links_each_measures_entry() {
        let entry = two_resolved_measures()
            .entry(&revenue_minus_cost())
            .expect("the metric body resolves");
        assert_eq!(entry.expr, link_to("revenue") - link_to("cost"));
        assert_eq!(entry.inferred_type, DataType::BigInt);
    }

    #[test]
    fn an_inlined_reference_holds_the_referenced_tree() {
        let mut fx = two_resolved_measures();
        fx.reference = Reference::Inline;
        let entry = fx
            .entry(&revenue_minus_cost())
            .expect("the metric body resolves");
        assert_eq!(entry.expr, sum_over("amount") - sum_over("spend"));
    }

    #[test]
    fn typed_leaf_kind_mismatch_errors() {
        let mut fx = Fixture::new();
        fx.all_semantics = semantics_index(vec![def(
            "revenue",
            SemanticRole::Measure,
            Some(DataType::BigInt),
        )]);
        let leaf = Expr::Leaf(SemanticLeaf::Dimension(reference("revenue"), None));
        match fx.entry(&leaf).expect_err("kind mismatch") {
            CompileError::SemanticKindMismatch {
                authored_kind,
                registered_kind,
                name,
            } => {
                assert_eq!(authored_kind, SemanticRole::Dimension);
                assert_eq!(registered_kind, SemanticRole::Measure);
                assert_eq!(name, "revenue");
            }
            other => panic!("expected SemanticKindMismatch, got {other:?}"),
        }
    }

    #[test]
    fn an_accessor_bearing_leaf_is_refused() {
        let mut fx = Fixture::new();
        fx.all_semantics = semantics_index(vec![def(
            "revenue",
            SemanticRole::Measure,
            Some(DataType::BigInt),
        )]);
        let leaf = Expr::Leaf(SemanticLeaf::Measure(
            reference("revenue"),
            Some(MeasureAccessor::Previous),
        ));
        match fx
            .entry(&leaf)
            .expect_err("an accessor has no lowering here")
        {
            CompileError::TypeInferenceFailure { node, reason } => {
                assert!(node.contains("revenue"), "the leaf is named: {node}");
                assert!(
                    reason.contains("accessor"),
                    "the reason is precise: {reason}"
                );
            }
            other => panic!("expected TypeInferenceFailure, got {other:?}"),
        }
    }

    #[test]
    fn a_row_level_body_refuses_an_aggregated_reference() {
        let mut fx = two_resolved_measures();
        fx.row_level = true;
        fx.owner = SemanticsName::from("full_name");
        match fx
            .entry(&revenue_minus_cost())
            .expect_err("row-level refuses")
        {
            CompileError::AggregateReferenceInRowLevelBody {
                semantics,
                referenced,
            } => {
                assert_eq!(semantics, "full_name");
                assert_eq!(referenced, "revenue");
            }
            other => panic!("expected AggregateReferenceInRowLevelBody, got {other:?}"),
        }
    }

    #[test]
    fn a_row_level_body_refuses_a_scalar_bodied_metric_reference() {
        let mut fx = Fixture::new();
        fx.all_semantics = semantics_index(vec![def(
            "is_eu",
            SemanticRole::Metric,
            Some(DataType::Boolean),
        )]);
        fx.resolved.insert(
            ResolvedExprKey::new(SemanticsName::from("is_eu"), "bind-a"),
            resolved_entry(column("c"), DataType::Boolean, "c"),
        );
        fx.row_level = true;
        fx.owner = SemanticsName::from("net");
        let leaf = Expr::Leaf(SemanticLeaf::Metric(reference("is_eu"), None));
        match fx
            .entry(&leaf)
            .expect_err("a row-level ctx refuses a Metric reference")
        {
            CompileError::AggregateReferenceInRowLevelBody {
                semantics,
                referenced,
            } => {
                assert_eq!(semantics, "net");
                assert_eq!(referenced, "is_eu");
            }
            other => panic!("expected AggregateReferenceInRowLevelBody, got {other:?}"),
        }
    }

    #[test]
    fn a_column_under_manual_mapping_is_refused() {
        let mut fx = Fixture::new();
        fx.manual_mapping = true;
        match fx
            .entry(&col_leaf("amount"))
            .expect_err("manual mapping refuses")
        {
            CompileError::ColumnInSemanticExprUnderManualMapping { binding } => {
                assert_eq!(binding, EntityId::from("bind-a"));
            }
            other => panic!("expected ColumnInSemanticExprUnderManualMapping, got {other:?}"),
        }
    }

    #[test]
    fn a_column_under_a_composed_filter_unit_is_refused() {
        let mut fx = Fixture::new();
        fx.composed_filter_unit = Some("sales");
        match fx
            .entry(&col_leaf("is_active"))
            .expect_err("composed filter refuses")
        {
            CompileError::ColumnInComposedFilter {
                unit,
                semantic,
                column,
            } => {
                assert_eq!(unit, "sales");
                assert_eq!(semantic, "owner");
                assert_eq!(column, "is_active");
            }
            other => panic!("expected ColumnInComposedFilter, got {other:?}"),
        }
    }

    #[test]
    fn the_measure_wrap_carries_the_family_and_its_type() {
        let mut fx = Fixture::new();
        let summed = resolve(&int_lit(7), &mut fx.ctx())
            .and_then(|body| body.aggregated(aggregation_of("sum"), "units"))
            .and_then(|lowered| lowered.typed(&ColumnTypes::default(), None, "units"))
            .expect("sum wraps");
        assert_eq!(summed.inferred_type, DataType::BigInt);
        assert!(matches!(
            &summed.expr,
            Expr::Function(Call::Aggregate(call, _)) if call.function().declarations.canonical.name == "sum"
        ));

        let counted = resolve(&int_lit(7), &mut fx.ctx())
            .and_then(|body| body.aggregated(aggregation_of("count_distinct"), "units"))
            .and_then(|lowered| lowered.typed(&ColumnTypes::default(), None, "units"))
            .expect("count wraps");
        assert_eq!(counted.inferred_type, DataType::BigInt);
        assert!(
            matches!(&counted.expr, Expr::Function(Call::Aggregate(call, _))
                if call.function().declarations.canonical.name == "count" && call.distinct()),
            "`count_distinct` is `count` with DISTINCT"
        );
    }

    #[test]
    fn a_refused_call_keeps_its_call_error() {
        let corr = aggregation_of("corr");
        match wrap_aggregate(column("x"), corr, "total") {
            Err(CompileError::CallRefused { node, error }) => {
                assert_eq!(node, "measure total agg corr");
                assert_eq!(
                    error,
                    CallError::TooFewArguments {
                        function: "corr",
                        min: 2,
                        max: Some(2),
                    }
                );
            }
            other => panic!("expected CallRefused, got {other:?}"),
        }
    }

    #[test]
    fn a_rebuilt_call_that_its_constructor_refuses_keeps_its_call_error() {
        let Ok(FunctionRef::Scalar(round)) = catalog_function("round", "owner") else {
            panic!("`round` is a scalar function of the catalog");
        };
        let args = vec![column("x"), Expr::Literal(Literal::Integer(2))];
        let call = ScalarCall::builder(round, args)
            .build()
            .expect("`round(x, 2)` fits its signature");
        let tree: Expr<Link> = Expr::Function(Call::Scalar(call));
        let rebuilt = map_nodes(
            &tree,
            &mut |node| match node {
                Expr::Literal(_) => Some(Ok(column("s"))),
                _ => None,
            },
            &|error| {
                refused_rebuild("owner", error, |error| {
                    unsupported("owner", &error.to_string())
                })
            },
        );
        match rebuilt {
            Err(CompileError::CallRefused { node, error }) => {
                assert_eq!(node, "owner");
                assert_eq!(
                    error,
                    CallError::NotALiteral {
                        function: "round",
                        parameter: "scale",
                    }
                );
            }
            other => panic!("expected CallRefused, got {other:?}"),
        }
    }

    #[test]
    fn referenced_columns_dedup() {
        let expr = Expr::BinaryOp {
            op: BinaryOpKind::Add,
            left: Box::new(col_leaf("x")),
            right: Box::new(col_leaf("x")),
        };
        let entry = Fixture::new().entry(&expr).expect("dedup");
        assert_eq!(entry.referenced_columns, vec!["x".to_string()]);
    }

    #[test]
    fn self_referential_measure_is_cyclic() {
        let mut fx = Fixture::new();
        fx.all_semantics = semantics_index(vec![def(
            "revenue",
            SemanticRole::Measure,
            Some(DataType::BigInt),
        )]);
        fx.recursion.enter(SemanticsName::from("revenue"));
        let err = fx.entry(&measure_leaf("revenue")).expect_err("cycle");
        assert!(matches!(err, CompileError::CyclicReference { .. }));
    }
}
