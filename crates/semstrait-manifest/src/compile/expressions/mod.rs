use std::borrow::Cow;

use semstrait_model::SemanticMappingValue;
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{map_nodes, Expr, Literal, NamedReference, UntypedLiteral};
use semstrait_sql::functions::AggregateFunction;

use super::qualifiers::qualified_column;
use super::sources::ResolvedSource;
use super::types::ResolvedExprEntry;
use columns::ColumnTypes;
use lower::{cast_to_declared, refused_rebuild, Lowered, LoweredExpr, LoweredLeaf};
use semstrait_common::EntityId;

use crate::manifest::semantic_definition::{DimensionType, MetadataSource};

pub mod bodies;
pub(super) mod columns;
pub(super) mod grainset_metrics;
pub mod lower;
pub(super) mod projection;
use crate::error::{CompileError, ReferenceSite};
use crate::manifest::expression::{ExprRoot, Link, ManifestExpressions, PhysicalExprId};
use crate::manifest::expression_layer::ExprLayer;
use crate::manifest::physical_source::StorageTarget;
use crate::manifest::semantic_definition::{
    casts_an_observed_column, is_count, SemanticDefinition, SemanticRoleDefinition,
};
use crate::manifest::source_alias::SourceAlias;

#[derive(Debug, Clone)]
pub(super) struct MappedLeaf {
    tree: LoweredExpr,
    referenced_columns: Vec<String>,
}

impl MappedLeaf {
    fn column(semantic: &str, column: &str) -> Result<Self, CompileError> {
        let reference =
            NamedReference::unqualified(column).ok_or_else(|| CompileError::EmptyColumnPart {
                node: format!("semantic_mapping {semantic}"),
                parts: vec![column.to_owned()],
            })?;
        Ok(Self {
            tree: Expr::Column(reference),
            referenced_columns: vec![column.to_owned()],
        })
    }

    pub(super) fn mapped_column(&self) -> Option<&str> {
        match &self.tree {
            Expr::Column(column) => Some(column.name()),
            _ => None,
        }
    }

    fn realised(
        self,
        aggregation: Option<(&'static AggregateFunction, bool)>,
        declared: Option<&DataType>,
        columns: &ColumnTypes,
        name: &str,
    ) -> Result<ResolvedExprEntry, CompileError> {
        let casts = |column: &str, declared: &DataType| {
            if columns.is_observed(column) {
                aggregation.is_some_and(|(function, _)| casts_an_observed_column(function))
                    && columns.of(column) != *declared
            } else {
                !aggregation.is_some_and(|(function, _)| is_count(function))
            }
        };
        let tree = match (declared, self.mapped_column()) {
            (Some(declared), Some(column)) if casts(column, declared) => {
                cast_to_declared(self.tree, declared)
            }
            _ => self.tree,
        };
        let lowered = Lowered::new(tree, self.referenced_columns);
        let lowered = match aggregation {
            Some(aggregation) => lowered.aggregated(aggregation, name)?,
            None => lowered,
        };
        lowered.typed(columns, declared, name)
    }
}

fn lower_mapping_value(
    semantic: &str,
    value: &SemanticMappingValue,
) -> Result<MappedLeaf, CompileError> {
    let literal = |tree| MappedLeaf {
        tree,
        referenced_columns: Vec::new(),
    };
    match value {
        SemanticMappingValue::Column(column) => MappedLeaf::column(semantic, column),
        SemanticMappingValue::Literal(tagged) => Ok(literal(Expr::Literal(tagged.clone()))),
        SemanticMappingValue::Untyped(untyped) => {
            Ok(literal(Expr::Leaf(LoweredLeaf::Untyped(untyped.clone()))))
        }
        SemanticMappingValue::Expr(_) => Err(unlowerable_mapping(
            semantic,
            "expr",
            "an authored physical body reaches the pool only through the authored-body \
             lowering control flow, which no ADR wires",
        )),
        SemanticMappingValue::Metadata(_) => Err(unlowerable_mapping(
            semantic,
            "metadata",
            "a metadata recipe is compile-synthesised and is not an authorable \
             `semantic_mapping` form; a metadata Dimension realises from its own category",
        )),
        _ => Err(unlowerable_mapping(
            semantic,
            "unknown",
            "the form has no ratified lowering in this crate",
        )),
    }
}

pub(super) fn unlowerable_mapping(semantic: &str, form: &str, reason: &str) -> CompileError {
    CompileError::TypeInferenceFailure {
        node: format!("semantic_mapping {semantic} ({form})"),
        reason: reason.to_owned(),
    }
}

pub(super) fn realise(
    definition: &SemanticDefinition,
    source: &ResolvedSource,
    leaf: Option<MappedLeaf>,
    columns: &ColumnTypes,
) -> Result<Option<ResolvedExprEntry>, CompileError> {
    let name = definition.name.as_str();
    let declared = definition.data_type.as_ref();
    if let SemanticRoleDefinition::Dimension {
        category: DimensionType::Metadata(body),
    } = &definition.role_definition
    {
        return metadata_entry(name, declared, &body.source, source).map(Some);
    }
    let aggregation = definition.role_definition.aggregation();
    leaf.map(|leaf| leaf.realised(aggregation, declared, columns, name))
        .transpose()
}

pub(super) fn needs_mapping_entry(definition: &SemanticDefinition) -> bool {
    match &definition.role_definition {
        SemanticRoleDefinition::Dimension {
            category: DimensionType::Metadata(_),
        } => false,
        SemanticRoleDefinition::Dimension { .. } | SemanticRoleDefinition::Measure { .. } => true,
        _ => false,
    }
}

pub(super) fn mapped_leaf(
    definition: &SemanticDefinition,
    mapped: Option<&SemanticMappingValue>,
) -> Result<Option<MappedLeaf>, CompileError> {
    if !needs_mapping_entry(definition) {
        return Ok(None);
    }
    let name = definition.name.as_str();
    match mapped {
        Some(value) => lower_mapping_value(name, value).map(Some),
        None => MappedLeaf::column(name, name).map(Some),
    }
}

pub(super) fn emit_source_attached(
    expr: &Expr<Link>,
    owner: &str,
    root: ExprRoot,
    consumed_by: &EntityId,
    alias: &SourceAlias,
    pool: &mut ManifestExpressions,
) -> Result<PhysicalExprId, CompileError> {
    let qualified = map_nodes(
        expr,
        &mut |node| match node {
            Expr::Column(column) => {
                Some(qualified_column(alias.as_str(), column.name(), owner).map(Expr::Column))
            }
            Expr::Leaf(link) => Some(Ok(Expr::Leaf(*link))),
            _ => None,
        },
        &|error| {
            refused_rebuild(owner, error, |_| CompileError::StructuralViolation {
                site: ReferenceSite::Source(consumed_by.clone()),
                rule: "every form of a Dataset's tree can be qualified by its physical source's \
                       alias",
            })
        },
    )?;
    store(qualified, owner, root, [consumed_by.clone()], pool)
}

pub(super) fn emit_composition_attached(
    expr: Expr<Link>,
    owner: &str,
    root: ExprRoot,
    pool: &mut ManifestExpressions,
) -> Result<PhysicalExprId, CompileError> {
    store(expr, owner, root, [], pool)
}

pub(super) fn emit_reading_final_values(
    expr: Expr<Link>,
    owner: &str,
    root: ExprRoot,
    pool: &mut ManifestExpressions,
    column: &dyn Fn(&NamedReference) -> ExprLayer,
) -> Result<PhysicalExprId, CompileError> {
    store_reading(expr, owner, root, [], pool, column)
}

fn store(
    expr: Expr<Link>,
    owner: &str,
    root: ExprRoot,
    consumers: impl IntoIterator<Item = EntityId>,
    pool: &mut ManifestExpressions,
) -> Result<PhysicalExprId, CompileError> {
    store_reading(expr, owner, root, consumers, pool, &|_| ExprLayer::Scalar)
}

fn store_reading(
    expr: Expr<Link>,
    owner: &str,
    root: ExprRoot,
    consumers: impl IntoIterator<Item = EntityId>,
    pool: &mut ManifestExpressions,
    column: &dyn Fn(&NamedReference) -> ExprLayer,
) -> Result<PhysicalExprId, CompileError> {
    let row_level =
        |tree: &Expr<Link>| pool.layer_of_reading(tree, column) == Some(ExprLayer::Scalar);
    let written = (!row_level(&expr)).then(|| expr.clone());
    let folded = expr.fold_with(&|Link(id)| match pool.linked_root(id)?.expr() {
        Expr::Literal(literal) => Some(Cow::Borrowed(literal)),
        _ => None,
    });
    let stored = match written {
        Some(written) if row_level(&folded) => written,
        _ => folded,
    };
    pool.emit_linked_reading(&stored, owner, root, consumers, column)
}

fn metadata_entry(
    name: &str,
    declared: Option<&DataType>,
    extraction: &MetadataSource,
    source: &ResolvedSource,
) -> Result<ResolvedExprEntry, CompileError> {
    let MetadataSource::Path(reference) = extraction;
    let token = reference.token;
    let segment = location_segment(source, token).ok_or_else(|| {
        CompileError::MetadataSegmentUnavailable {
            semantic: name.to_owned(),
            physical_source: source.id.clone(),
            locator: source.locator(),
            token,
        }
    })?;
    Lowered::new(
        Expr::Leaf(LoweredLeaf::Untyped(UntypedLiteral(Literal::String(
            segment.to_owned(),
        )))),
        [],
    )
    .typed(&ColumnTypes::default(), declared, name)
}

fn location_segment(source: &ResolvedSource, token: u32) -> Option<&str> {
    let StorageTarget::Path { base_path, .. } = &source.storage.target else {
        return None;
    };
    base_path.segments().nth(usize::try_from(token).ok()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::catalog::table_ref_from_fqn;
    use crate::manifest::physical_source::{StorageProperties, Uri};

    fn source_over(target: StorageTarget) -> ResolvedSource {
        ResolvedSource::model_fed_for_test(StorageProperties::bare(target))
    }

    fn path_source(base_path: &str) -> ResolvedSource {
        source_over(StorageTarget::path(Uri::from(base_path), Vec::new()))
    }

    fn table_source(locator: &str) -> ResolvedSource {
        source_over(StorageTarget::table(table_ref_from_fqn(locator)))
    }

    #[test]
    fn a_location_segment_is_the_whole_raw_segment() {
        let uri = path_source("s3://bucket/orders/year=2024/");
        assert_eq!(location_segment(&uri, 0), Some("bucket"));
        assert_eq!(location_segment(&uri, 2), Some("year=2024"));
        assert_eq!(location_segment(&uri, 3), None);

        let local = path_source("file:///data/orders/year=2024/");
        assert_eq!(location_segment(&local, 0), Some("data"));
        assert_eq!(location_segment(&local, 2), Some("year=2024"));
    }

    #[test]
    fn a_table_source_has_no_location_segment() {
        assert_eq!(location_segment(&table_source("warehouse.orders"), 0), None);
    }

    #[test]
    fn the_retired_partition_form_does_not_decode() {
        let error = serde_json::from_str::<MetadataSource>(r#"{"partition":{"column":"dt"}}"#)
            .expect_err("the retired form does not decode");
        assert!(
            error.to_string().contains("unknown variant `partition`"),
            "{error}"
        );
    }
}
