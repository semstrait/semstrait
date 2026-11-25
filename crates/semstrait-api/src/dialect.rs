use semstrait_adapter::SubstraitRules;
#[cfg(feature = "datafusion")]
use semstrait_common::Release;
use semstrait_common::{Diagnose, Dialect, DialectRelease, Severity};
use thiserror::Error;

use crate::error::{refusal, Refusal, SemstraitError};

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum NoEngineAdapter {
    #[error("the dialect {0:?} has no engine adapter")]
    NoRules(Dialect),
    #[error("the dialect DataFusion needs the `datafusion` feature")]
    DataFusionFeatureOff,
}

impl Diagnose for NoEngineAdapter {
    fn message(&self) -> String {
        self.to_string()
    }

    fn severity_default(&self) -> Severity {
        Severity::Error
    }
}

struct Faithful;

impl SubstraitRules for Faithful {}

pub(crate) fn rules(dialect: DialectRelease) -> Result<Box<dyn SubstraitRules>, Refusal> {
    let refused = |kind| Err(refusal(SemstraitError::NoEngineAdapter(kind)));
    match dialect.dialect {
        Dialect::Substrait => Ok(Box::new(Faithful)),
        #[cfg(feature = "datafusion")]
        Dialect::DataFusion => {
            let release = dialect.release.unwrap_or(Release::DATAFUSION_REFERENCE);
            Ok(Box::new(
                semstrait_adapter_datafusion::DataFusionRules::new(release),
            ))
        }
        #[cfg(not(feature = "datafusion"))]
        Dialect::DataFusion => refused(NoEngineAdapter::DataFusionFeatureOff),
        _ => refused(NoEngineAdapter::NoRules(dialect.dialect)),
    }
}
