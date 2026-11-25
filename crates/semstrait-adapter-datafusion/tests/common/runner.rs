use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::{Duration, UNIX_EPOCH};

use datafusion::arrow::datatypes::Fields;
use datafusion::prelude::SessionContext;
use semstrait_adapter::explain;
use semstrait_common::{Diagnostic, Diagnostics};
use semstrait_ir::SemanticPlan;
use semstrait_manifest::{compile, CompileError, ManifestProvenance, SemanticManifest};
use semstrait_model::parse;
use semstrait_planner::{Advisory, PlanError, Planner, Request, RequestContext};
use serde_yaml::Value;
use substrait::proto::Plan;

use super::case::{Case, Expect};
use super::declaration::{Declaration, Declared};
use super::golden::Golden;
use super::refusal::Stage;
use super::{adapted, executed, fixture, produced, rules, wire, Faithful, Rows};

pub type Planned =
    Result<(SemanticPlan, Diagnostics<Advisory>), (Diagnostic<PlanError>, Diagnostics<Advisory>)>;

pub type PlanFn = fn(&SemanticManifest, &Request, &RequestContext) -> Planned;

pub fn planned(
    manifest: &SemanticManifest,
    request: &Request,
    context: &RequestContext,
) -> Planned {
    Planner::new().plan(manifest, request, context)
}

type Compiled = Result<
    (SemanticManifest, Diagnostics<CompileError>),
    (Diagnostic<CompileError>, Diagnostics<CompileError>),
>;

const MODEL: &str = "model.yaml";
const DECLARATION: &str = "data.yaml";

fn files(model: &str) -> [PathBuf; 2] {
    [MODEL, DECLARATION].map(|file| fixture(&format!("{model}/{file}")))
}

fn declaration_of(model: Option<&str>) -> PathBuf {
    match model {
        Some(model) => {
            let [_, declaration] = files(model);
            declaration
        }
        None => fixture(DECLARATION),
    }
}

async fn compilation(
    model: &Path,
    declaration: &Path,
    input_data: &BTreeMap<String, Vec<Vec<Value>>>,
) -> Result<(Compiled, Declaration, Declared), Box<dyn Error>> {
    let named = model.display();
    let (built, warnings) = (parse(&std::fs::read_to_string(model)?))
        .map_err(|diagnostics| format!("{named}: parse refuses: {diagnostics:?}"))?
        .build()
        .map_err(|diagnostics| format!("{named}: build refuses: {diagnostics:?}"))?;
    assert!(warnings.is_empty(), "{named}: build warns: {warnings:?}");
    let mut declaration = Declaration::read(declaration)?;
    declaration.apply_input_data(input_data)?;
    let declared = declaration.build().await?;
    let provenance = ManifestProvenance::new(
        UNIX_EPOCH + Duration::from_secs(1_700_000_000),
        UNIX_EPOCH + Duration::from_secs(1_800_000_000),
    );
    let compiled = compile(built, Some(&declared.catalog), &declared.bucket, provenance).await;
    Ok((compiled, declaration, declared))
}

async fn clean(
    model: &Path,
    declaration: &Path,
    input_data: &BTreeMap<String, Vec<Vec<Value>>>,
) -> Result<(SemanticManifest, Declared), Box<dyn Error>> {
    let (compiled, declaration, declared) = compilation(model, declaration, input_data).await?;
    let (manifest, warnings) = compiled.map_err(|(error, _)| error)?;
    let named = model.display();
    assert!(warnings.is_empty(), "{named}: compile warns: {warnings:?}");
    declaration.check(&manifest)?;
    Ok((manifest, declared))
}

pub async fn compiled(model: &str) -> Result<(SemanticManifest, Declared), Box<dyn Error>> {
    let [model, declaration] = files(model);
    clean(&model, &declaration, &BTreeMap::new()).await
}

pub async fn corpus(model: &str) -> Result<(SemanticManifest, Declared), Box<dyn Error>> {
    let corpus = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data");
    let [model, declaration] =
        ["yaml", "data.yaml"].map(|ext| corpus.join(format!("{model}.{ext}")));
    clean(&model, &declaration, &BTreeMap::new()).await
}

pub fn listing(listed: &[&str]) -> Result<(), Box<dyn Error>> {
    fn files_under(folder: &Path) -> std::io::Result<Vec<PathBuf>> {
        let entries = std::fs::read_dir(folder)?.map(|entry| {
            let path = entry?.path();
            if path.is_dir() {
                files_under(&path)
            } else {
                Ok(vec![path])
            }
        });
        Ok(entries.collect::<std::io::Result<Vec<_>>>()?.concat())
    }
    let root = fixture("");
    let problems = (files_under(&root)?.into_iter())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "yaml")
        })
        .filter(|path| !path.ends_with(MODEL) && !path.ends_with(DECLARATION));
    let mut cases = BTreeSet::new();
    for path in problems {
        let file = path.strip_prefix(&root)?.with_extension("");
        let file = file.to_string_lossy().into_owned();
        for key in Case::problem(&file)?.keys() {
            cases.insert(format!("{file}/{key}"));
        }
    }
    let listed: BTreeSet<_> = listed.iter().map(|&case| case.to_owned()).collect();
    let unlisted: Vec<_> = cases.difference(&listed).collect();
    let missing: Vec<_> = listed.difference(&cases).collect();
    if unlisted.is_empty() && missing.is_empty() {
        return Ok(());
    }
    let failure = format!(
        "case files that no runner lists: {unlisted:?}; listed cases with no file: {missing:?}"
    );
    Err(failure.into())
}

#[allow(clippy::type_complexity, reason = "the two arms of `Planned`, by kind")]
fn kinds(
    planned: &Planned,
) -> Result<(&SemanticPlan, Vec<&Advisory>), (&PlanError, Vec<&Advisory>)> {
    fn advisories(advisories: &[Diagnostic<Advisory>]) -> Vec<&Advisory> {
        advisories.iter().map(|advisory| &advisory.kind).collect()
    }
    match planned {
        Ok((plan, advised)) => Ok((plan, advisories(advised))),
        Err((error, advised)) => Err((&error.kind, advisories(advised))),
    }
}

pub async fn run(case: &str, plan: PlanFn) -> Result<Golden, Box<dyn Error>> {
    let (_, golden) = ran(case, plan).await?;
    Ok(golden)
}

pub async fn hand_built(case: &str, plan: PlanFn) -> Result<(Plan, Golden), Box<dyn Error>> {
    let (ran, golden) = ran(case, plan).await?;
    let (semantic, production) = ran.ok_or_else(|| format!("{case}: a rows case"))?;
    produced(&semantic, &Faithful)?;
    Ok((production, golden))
}

type Ran = Option<(SemanticPlan, Plan)>;

async fn ran(case: &str, plan: PlanFn) -> Result<(Ran, Golden), Box<dyn Error>> {
    let read = Case::read(case)?;
    let model = (read.model()).ok_or_else(|| format!("{case}: the case names no model"))?;
    let [model, declaration] = files(model);
    if let Expect::Error(refusal) = &read.expect {
        match refusal.stage {
            Stage::Parse => {
                let refused = parse(&std::fs::read_to_string(&model)?).err();
                return Ok((None, refusal.compared(case, &refused.unwrap_or_default())?));
            }
            Stage::Compile => {
                let (compiled, ..) = compilation(&model, &declaration, read.input_data()).await?;
                let refused = compiled.err().map(|(error, _)| error);
                return Ok((None, refusal.compared(case, refused.as_slice())?));
            }
            Stage::Plan => {}
        }
    }
    let (manifest, declared) = clean(&model, &declaration, read.input_data()).await?;
    let (request, context) = (read.request()?, read.context()?);
    let planned = plan(&manifest, &request, &context);
    assert_eq!(
        kinds(&planned),
        kinds(&plan(&manifest, &request, &context)),
        "{case}: two plannings are equal"
    );
    if let Expect::Error(refusal) = &read.expect {
        let refused = planned.err().map(|(error, _)| error);
        return Ok((None, refusal.compared(case, refused.as_slice())?));
    }
    let (semantic, production) = validated(case, planned)?;
    let text = rows(
        case,
        &semantic,
        &production,
        &declared.session()?,
        |fields| {
            let output: Vec<_> = fields.iter().map(|field| field.name().as_str()).collect();
            assert_eq!(
                output,
                read.names()?,
                "{case}: the output names are the select names"
            );
            (read.rows(fields)).map_err(|error| format!("{case}: {error}").into())
        },
    )
    .await?;
    let golden = Golden::case(case, "plan", text)?;
    Ok((Some((semantic, production)), golden))
}

pub async fn gate(case: &str, semantic: &SemanticPlan) -> Result<(Plan, Golden), Box<dyn Error>> {
    let read = Case::read(case)?;
    let mut declaration = Declaration::read(&declaration_of(read.model()))?;
    declaration.apply_input_data(read.input_data())?;
    let declared = declaration.build().await?;
    let production = production_of(semantic)?;
    produced(semantic, &Faithful)?;
    let text = rows(
        case,
        semantic,
        &production,
        &declared.session()?,
        |fields| {
            let output: Vec<_> = fields.iter().map(|field| field.name().as_str()).collect();
            assert_eq!(
                output,
                read.names()?,
                "{case}: the output names are the select names"
            );
            (read.rows(fields)).map_err(|error| format!("{case}: {error}").into())
        },
    )
    .await?;
    Ok((production, Golden::case(case, "plan", text)?))
}

async fn rows(
    case: &str,
    semantic: &SemanticPlan,
    production: &Plan,
    session: &SessionContext,
    expected: impl FnOnce(&Fields) -> Result<Rows, Box<dyn Error>>,
) -> Result<String, Box<dyn Error>> {
    wire(case, production);
    let (fields, rows) = executed(session, production).await?;
    expected(&fields)?.assert(case, rows);
    adapted(explain(semantic, &rules()))
}

pub fn validated(case: &str, planned: Planned) -> Result<(SemanticPlan, Plan), Box<dyn Error>> {
    let (semantic, _) = planned.map_err(|(error, _)| format!("{case}: plan refuses: {error}"))?;
    let production = production_of(&semantic)?;
    Ok((semantic, production))
}

fn production_of(semantic: &SemanticPlan) -> Result<Plan, Box<dyn Error>> {
    semantic.validate()?;
    produced(semantic, &rules())
}

pub async fn production(case: &str) -> Result<Plan, Box<dyn Error>> {
    let read = Case::read(case)?;
    let model = (read.model()).ok_or_else(|| format!("{case}: the case names no model"))?;
    let (manifest, _) = compiled(model).await?;
    let (_, production) = validated(case, planned(&manifest, &read.request()?, &read.context()?))?;
    Ok(production)
}
