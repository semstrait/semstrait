pub use semstrait_api::load;
pub use semstrait_api::{
    ContextPart, Input, Key, Order, RequestPart, Selected, Semstrait, SemstraitBuilder,
    SemstraitError, WeekStart, WireError,
};
pub use semstrait_catalog::CatalogProvider;
pub use semstrait_common::io::backends::local_fs::LocalFileSystem;
pub use semstrait_common::{
    DataKindName, Diagnostic, Diagnostics, Dialect, DialectRelease, FileSystem, Granularity,
    Release, SemanticsName,
};
pub use semstrait_manifest::SemanticManifest;
pub use semstrait_model::UnresolvedSemanticExpr;
pub use semstrait_planner::{
    DimensionModifier, Direction, OrderKey, Request, RequestContext, Selection, Timezone, Weekday,
};

use substrait::proto::Plan;

#[allow(
    clippy::result_large_err,
    reason = "run returns ADR 0005's fail-fast form unboxed, as the verbs it calls do"
)]
pub async fn run(
    input: Input,
    request: &Request,
    context: &RequestContext,
    dialect: impl Into<DialectRelease>,
) -> Result<
    (Plan, Diagnostics<SemstraitError>),
    (Diagnostic<SemstraitError>, Diagnostics<SemstraitError>),
> {
    let semstrait = SemstraitBuilder::default().build();
    let (manifest, warnings) = load(&semstrait, input).await?;
    let joined = |later: Diagnostics<SemstraitError>| warnings.into_iter().chain(later).collect();
    match semstrait.plan(&manifest, request, context, dialect) {
        Ok((plan, later)) => Ok((plan, joined(later))),
        Err((refusal, later)) => Err((refusal, joined(later))),
    }
}
