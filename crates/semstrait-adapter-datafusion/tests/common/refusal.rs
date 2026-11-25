use std::collections::BTreeMap;
use std::error::Error;

use semstrait_common::{Diagnose, Diagnostic, Granularity};
use semstrait_manifest::CompileError;
use semstrait_model::ParseErrorKind;
use semstrait_planner::PlanError;
use serde::Deserialize;
use serde_yaml::Value;

use super::golden::Golden;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Stage {
    Parse,
    Compile,
    Plan,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Refusal {
    pub stage: Stage,
    variant: String,
    #[serde(default)]
    fields: BTreeMap<String, Value>,
}

pub trait Render: Diagnose {
    fn render(&self) -> Option<(&'static str, BTreeMap<&'static str, Value>)>;
}

impl Render for ParseErrorKind {
    fn render(&self) -> Option<(&'static str, BTreeMap<&'static str, Value>)> {
        match self {
            Self::MeasureMissingAgg { carrier, name } => Some((
                "MeasureMissingAgg",
                BTreeMap::from([
                    ("carrier", carrier.as_str().into()),
                    ("name", name.as_str().into()),
                ]),
            )),
            _ => None,
        }
    }
}

impl Render for CompileError {
    fn render(&self) -> Option<(&'static str, BTreeMap<&'static str, Value>)> {
        match self {
            Self::ColumnNotInSourceSchema {
                column,
                physical_source_id,
            } => Some((
                "ColumnNotInSourceSchema",
                BTreeMap::from([
                    ("column", column.as_str().into()),
                    ("physical_source_id", physical_source_id.as_str().into()),
                ]),
            )),
            _ => None,
        }
    }
}

impl Render for PlanError {
    fn render(&self) -> Option<(&'static str, BTreeMap<&'static str, Value>)> {
        match self {
            Self::NoSignature {
                path,
                semantic_source,
                semantics,
            } => Some((
                "NoSignature",
                BTreeMap::from([
                    ("path", path.to_string().split(" / ").collect()),
                    ("semantic_source", semantic_source.as_str().into()),
                    ("semantics", semantics.as_str().into()),
                ]),
            )),
            Self::GrainNotServable {
                path,
                source,
                requested,
                source_granularity,
                ..
            } => Some((
                "GrainNotServable",
                BTreeMap::from([
                    ("path", path.to_string().split(" / ").collect()),
                    ("source", source.as_str().into()),
                    ("requested", granularity(*requested)),
                    ("source_granularity", granularity(*source_granularity)),
                ]),
            )),
            Self::UngroupedMetricFormula { path, metric, name } => Some((
                "UngroupedMetricFormula",
                BTreeMap::from([
                    ("path", path.to_string().split(" / ").collect()),
                    ("metric", metric.as_str().into()),
                    ("name", name.as_str().into()),
                ]),
            )),
            _ => None,
        }
    }
}

fn granularity(granularity: Option<Granularity>) -> Value {
    granularity.map_or(Value::Null, |granularity| granularity.to_string().into())
}

impl Refusal {
    pub fn compared<E: Render>(
        &self,
        case: &str,
        refused: &[Diagnostic<E>],
    ) -> Result<Golden, Box<dyn Error>> {
        let [error] = refused else {
            let refused: Vec<_> = refused.iter().map(ToString::to_string).collect();
            return Err(format!("{case}: one {:?} refusal, not {refused:?}", self.stage).into());
        };
        let (variant, fields) = (error.kind.render())
            .ok_or_else(|| format!("{case}: no rendering of `{}`", error.kind.message()))?;
        assert_eq!(variant, self.variant, "{case}: the variant");
        for (name, value) in &self.fields {
            assert_eq!(
                fields.get(name.as_str()),
                Some(value),
                "{case}: the field `{name}`"
            );
        }
        Golden::case(case, "message", error.to_string())
    }
}
