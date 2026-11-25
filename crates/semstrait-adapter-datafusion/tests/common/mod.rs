#![allow(dead_code)]
#![allow(
    clippy::result_large_err,
    reason = "a function in `plan`'s place returns what `plan` returns, its PlanError unboxed"
)]

pub mod case;
mod consumer;
mod data;
pub mod declaration;
pub mod golden;
pub mod refusal;
pub mod runner;

use std::cmp::Ordering;
use std::error::Error;
use std::fmt::Debug;
use std::path::{Path, PathBuf};

use datafusion::arrow::datatypes::Fields;
use datafusion::prelude::SessionContext;
use datafusion::scalar::ScalarValue;
use datafusion_substrait::extensions::Extensions as DataFusionExtensions;
use datafusion_substrait::logical_plan::consumer::from_substrait_plan_with_consumer;
use prost::Message;
use semstrait_adapter::{SubstraitProducer, SubstraitRules};
use semstrait_adapter_datafusion::DataFusionRules;
use semstrait_common::Release;
use semstrait_ir::{
    EmitForm, IrErrorKind, Name, Parameter, PhysicalExpr, PlanNode, ProjectNode, Reader, ScanNode,
    Schema, SchemaColumn, SemanticPlan,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{
    AggregateAllowed, AggregateCall, Call, Expr, FunctionRef, IntoLiteral, NamedReference,
};
use semstrait_sql::functions::{registry, AggregateFunction};
use substrait::proto::extensions::AdvancedExtension;
use substrait::proto::read_rel::local_files::file_or_files::PathType;
use substrait::proto::read_rel::ReadType;
use substrait::proto::{plan_rel, rel, Plan, ReadRel, Rel, RelCommon};
use substrait::version::version_with_producer;

use consumer::FilesConsumer;

pub struct Faithful;

impl SubstraitRules for Faithful {}

pub fn rules() -> DataFusionRules {
    match std::env::var("SEMSTRAIT_DATAFUSION_RELEASE") {
        Ok(release) => DataFusionRules::new(Release(release.parse().expect("a major release"))),
        Err(_) => DataFusionRules::default(),
    }
}

pub fn fixture(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(env!("CARGO_CRATE_NAME"))
        .join(relative)
}

pub fn name(name: &str) -> Name {
    Name::new(name).expect("a name")
}

pub fn schema(columns: &[(&str, DataType)]) -> Schema {
    let columns = (columns.iter()).map(|(name, data_type)| SchemaColumn {
        name: (*name).to_owned(),
        data_type: data_type.clone(),
        nullable: true,
    });
    Schema {
        columns: columns.collect(),
    }
}

pub fn scanned(
    reader: impl Into<Reader>,
    columns: &[(&str, DataType)],
    alias: Option<&str>,
) -> Result<PlanNode, IrErrorKind> {
    let scan = ScanNode::new(reader, schema(columns))?;
    let leaves = (columns.iter())
        .map(|(column, _)| {
            let field =
                alias.map_or_else(|| (*column).to_owned(), |alias| format!("{alias}.{column}"));
            (name(&field), leaf(&["ps0", column]))
        })
        .collect();
    project(scan, leaves)
}

pub fn leaf(parts: &[&str]) -> PhysicalExpr {
    Expr::Column(NamedReference::new(parts.iter().copied()).expect("a name path"))
}

pub fn literal(value: impl IntoLiteral) -> PhysicalExpr {
    Expr::Literal(value.into_literal())
}

pub fn aggregate_function(function: &str) -> &'static AggregateFunction {
    let Some(FunctionRef::Aggregate(function, AggregateAllowed)) = registry().get(function) else {
        panic!("the catalog declares the aggregate `{function}`");
    };
    function
}

pub fn aggregate(
    function: &str,
    args: Vec<PhysicalExpr>,
    filter: Option<PhysicalExpr>,
) -> AggregateCall<Parameter, AggregateAllowed> {
    AggregateCall::builder(aggregate_function(function), args)
        .maybe_filter(filter)
        .build()
        .expect("an aggregate call")
}

pub fn scalar(function: &str, args: Vec<PhysicalExpr>) -> PhysicalExpr {
    let function = registry().get(function).expect("a catalog function");
    Expr::Function(Call::new(function, args).expect("a scalar call"))
}

pub fn project(
    input: impl Into<PlanNode>,
    fields: Vec<(Name, PhysicalExpr)>,
) -> Result<PlanNode, IrErrorKind> {
    let input = input.into();
    let passed = input.common().output_schema.columns.len();
    let emit = EmitForm::new(passed..passed + fields.len());
    PlanNode::from(ProjectNode::new(input, fields)?).with_emit_form(emit)
}

pub fn plan(root: impl Into<PlanNode>) -> SemanticPlan {
    let root = root.into();
    let names = (root.common().output_schema.columns.iter())
        .map(|field| name(&field.name))
        .collect();
    SemanticPlan::new(root, names)
}

pub type Row = Vec<ScalarValue>;

pub enum Rows {
    InOrder(Vec<Row>),
    Multiset(Vec<Row>),
    OrderedBy(Vec<usize>, Vec<Row>),
}

impl Rows {
    pub fn assert(self, case: &str, rows: Vec<Row>) {
        match self {
            Self::InOrder(expected) => assert_eq!(rows, expected, "{case}: the rows, in order"),
            Self::Multiset(expected) => Self::OrderedBy(Vec::new(), expected).assert(case, rows),
            Self::OrderedBy(keys, expected) => {
                let runs = |rows: &[Row]| -> Vec<Vec<Row>> {
                    let equal = |left: &Row, right: &Row| keys.iter().all(|&k| left[k] == right[k]);
                    let order = |left: &Row, right: &Row| {
                        left.partial_cmp(right).unwrap_or(Ordering::Equal)
                    };
                    (rows.chunk_by(equal))
                        .map(|run| {
                            let mut run = run.to_vec();
                            run.sort_by(order);
                            run
                        })
                        .collect()
                };
                assert_eq!(
                    runs(&rows),
                    runs(&expected),
                    "{case}: the rows, in the order of the fields {keys:?}, rows of equal keys \
                     as a multiset"
                );
            }
        }
    }
}

fn parts(relation: &Rel) -> (Vec<&Rel>, Option<&RelCommon>, Option<&AdvancedExtension>) {
    use rel::RelType;
    macro_rules! parts {
        ($rel:ident) => {
            parts!($rel, $rel.input.as_deref().into_iter().collect())
        };
        (binary $rel:ident) => {
            parts!(
                $rel,
                [$rel.left.as_deref(), $rel.right.as_deref()]
                    .into_iter()
                    .flatten()
                    .collect()
            )
        };
        ($rel:ident, $inputs:expr) => {
            (
                $inputs,
                $rel.common.as_ref(),
                $rel.advanced_extension.as_ref(),
            )
        };
    }
    match &relation.rel_type {
        Some(RelType::Read(rel)) => parts!(rel, Vec::new()),
        Some(RelType::Filter(rel)) => parts!(rel),
        Some(RelType::Project(rel)) => parts!(rel),
        Some(RelType::Aggregate(rel)) => parts!(rel),
        Some(RelType::Sort(rel)) => parts!(rel),
        Some(RelType::Fetch(rel)) => parts!(rel),
        Some(RelType::Set(rel)) => parts!(rel, rel.inputs.iter().collect()),
        Some(RelType::Join(rel)) => parts!(binary rel),
        Some(RelType::Cross(rel)) => parts!(binary rel),
        other => panic!("a relation that the wire check reads, not {other:?}"),
    }
}

pub fn relations(plan: &Plan) -> Vec<&Rel> {
    fn tree(relation: &Rel) -> Vec<&Rel> {
        let (inputs, ..) = parts(relation);
        std::iter::once(relation)
            .chain(inputs.into_iter().flat_map(tree))
            .collect()
    }
    (plan.relations.iter())
        .filter_map(|relation| match &relation.rel_type {
            Some(plan_rel::RelType::Root(root)) => root.input.as_ref(),
            _ => None,
        })
        .flat_map(tree)
        .collect()
}

pub fn files_read(read: &ReadRel) -> Vec<PathType> {
    let Some(ReadType::LocalFiles(files)) = &read.read_type else {
        panic!("a LocalFiles read");
    };
    (files.items.iter())
        .map(|item| item.path_type.clone().expect("a path"))
        .collect()
}

pub fn wire(case: &str, plan: &Plan) {
    let version = plan.version.as_ref();
    assert_eq!(
        version.map(|version| (
            version.major_number,
            version.minor_number,
            version.patch_number
        )),
        Some((0, 85, 0)),
        "{case}: the spec release"
    );
    assert_eq!(
        plan.version,
        Some(version_with_producer("semstrait")),
        "{case}: the version"
    );
    for relation in relations(plan) {
        let (_, common, own) = parts(relation);
        let common = common.and_then(|common| common.advanced_extension.as_ref());
        assert!(
            own.is_none() && common.is_none(),
            "{case}: no relation carries trace: {relation:?}"
        );
    }
}

pub fn produced(
    semantic: &SemanticPlan,
    rules: &dyn SubstraitRules,
) -> Result<Plan, Box<dyn Error>> {
    let produce = || adapted(SubstraitProducer::new(rules).produce(semantic));
    let plan = produce()?;
    assert_eq!(
        plan.encode_to_vec(),
        produce()?.encode_to_vec(),
        "two productions are byte-equal"
    );
    Ok(plan)
}

pub async fn executed(
    session: &SessionContext,
    plan: &Plan,
) -> Result<(Fields, Vec<Row>), Box<dyn Error>> {
    let state = session.state();
    let plan =
        datafusion_substrait::substrait::proto::Plan::decode(plan.encode_to_vec().as_slice())?;
    let plan = &plan;
    let extensions = DataFusionExtensions::try_from(&plan.extensions)?;
    let consumer = FilesConsumer::new(&extensions, &state, session);
    let logical = from_substrait_plan_with_consumer(&consumer, plan).await?;
    let frame = session.execute_logical_plan(logical).await?;
    let fields = frame.schema().fields().clone();
    let batches = frame.collect().await?;
    let rows = (batches.iter())
        .flat_map(|batch| {
            (0..batch.num_rows()).map(move |row| {
                (batch.columns().iter())
                    .map(|column| ScalarValue::try_from_array(column, row))
                    .collect::<Result<Row, _>>()
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((fields, rows))
}

pub fn adapted<Output, Refusal: Into<Box<dyn Error>>, Warning: Debug>(
    output: Result<(Output, Vec<Warning>), (Refusal, Vec<Warning>)>,
) -> Result<Output, Box<dyn Error>> {
    let (output, warnings) = match output {
        Ok((output, warnings)) => (Ok(output), warnings),
        Err((refusal, warnings)) => (Err(refusal.into()), warnings),
    };
    assert!(warnings.is_empty(), "adapt raises no warning: {warnings:?}");
    output
}
