use std::sync::Arc;
use std::time::SystemTime;

use semstrait_adapter::{explain, InvalidRewrite, SubstraitProducer};
use semstrait_catalog::CatalogProvider;
use semstrait_common::io::backends::local_fs::LocalFileSystem;
use semstrait_common::io::{
    FileSystemRepository, Fingerprintable, InMemoryFileSystem, Path, Repository,
    RepositoryErrorKind,
};
use semstrait_common::{Diagnostics, DialectRelease, EntityId, FileSystem};
use semstrait_manifest::{compile, mint_model_id, ManifestProvenance, SemanticManifest};
use semstrait_model::{parse, validate, CatalogsConfig, SemanticModel};
use semstrait_planner::{Advisory, PlanError, Planner, Request, RequestContext};
use substrait::proto::Plan;

use crate::catalogs::connected;
use crate::dialect::rules;
use crate::error::{joined, lifted, refusal, refused, Outcome, Refusal, SemstraitError, Staged};

pub struct Semstrait {
    catalog: Option<Arc<dyn CatalogProvider>>,
    filesystem: Arc<dyn FileSystem>,
    manifests: FileSystemRepository<SemanticManifest>,
}

#[derive(Default)]
pub struct SemstraitBuilder {
    catalog: Option<Arc<dyn CatalogProvider>>,
    filesystem: Option<Arc<dyn FileSystem>>,
}

impl SemstraitBuilder {
    pub fn with_catalog(mut self, catalog: Arc<dyn CatalogProvider>) -> Self {
        self.catalog = Some(catalog);
        self
    }

    pub async fn with_catalogs(
        self,
        catalogs: &CatalogsConfig,
        alias: Option<&str>,
    ) -> Result<Self, Refusal> {
        let catalog = connected(catalogs, alias)
            .await
            .map_err(|kind| refusal(SemstraitError::Catalog(kind)))?;
        Ok(self.with_catalog(catalog))
    }

    pub fn with_filesystem(mut self, filesystem: Arc<dyn FileSystem>) -> Self {
        self.filesystem = Some(filesystem);
        self
    }

    pub fn build(self) -> Semstrait {
        Semstrait {
            catalog: self.catalog,
            filesystem: self
                .filesystem
                .unwrap_or_else(|| Arc::new(LocalFileSystem::new("/"))),
            manifests: FileSystemRepository::new(
                Arc::new(InMemoryFileSystem::new()),
                Path::new("manifests"),
            ),
        }
    }
}

impl Semstrait {
    pub fn catalog(&self) -> Option<&dyn CatalogProvider> {
        self.catalog.as_deref()
    }

    pub async fn compile_from_yaml(&self, yaml: &str) -> Outcome<SemanticManifest> {
        let (model, warnings) = built(yaml)?;
        self.compiled(model, warnings).await
    }

    pub async fn compile_from_model(&self, model: SemanticModel) -> Outcome<SemanticManifest> {
        let warnings =
            validate(&model).map_err(|errors| refused(errors, SemstraitError::Validate))?;
        self.compiled(model, lifted(warnings, SemstraitError::Validate).collect())
            .await
    }

    pub async fn exists(&self, model: &SemanticModel) -> Result<bool, Refusal> {
        self.manifests.exists(&key(model)).await.map_err(stored)
    }

    pub async fn save_manifest(&self, manifest: &SemanticManifest) -> Result<(), Refusal> {
        self.manifests.save(manifest).await.map_err(stored)
    }

    pub async fn load_manifest(&self, model: &SemanticModel) -> Outcome<SemanticManifest> {
        validated(self.manifests.load(&key(model)).await.map_err(stored)?)
    }

    pub fn validate(
        &self,
        manifest: &SemanticManifest,
        request: &Request,
        context: &RequestContext,
    ) -> Result<Diagnostics<SemstraitError>, Refusal> {
        planned(Planner::new().plan(manifest, request, context)).map(|(_, advisories)| advisories)
    }

    pub fn explain(
        &self,
        manifest: &SemanticManifest,
        request: &Request,
        context: &RequestContext,
        dialect: impl Into<DialectRelease>,
    ) -> Outcome<String> {
        let rules = rules(dialect.into())?;
        adapted(Planner::new().plan(manifest, request, context), |plan| {
            explain(plan, rules.as_ref())
        })
    }

    pub fn plan(
        &self,
        manifest: &SemanticManifest,
        request: &Request,
        context: &RequestContext,
        dialect: impl Into<DialectRelease>,
    ) -> Outcome<Plan> {
        let rules = rules(dialect.into())?;
        adapted(Planner::new().plan(manifest, request, context), |plan| {
            SubstraitProducer::new(rules.as_ref()).produce(plan)
        })
    }

    pub(crate) async fn compiled(
        &self,
        model: SemanticModel,
        warnings: Diagnostics<SemstraitError>,
    ) -> Outcome<SemanticManifest> {
        let now = SystemTime::now();
        let provenance = ManifestProvenance::new(now, now);
        let compiled = compile(model, self.catalog(), self.filesystem.as_ref(), provenance).await;
        joined(
            warnings,
            compiled,
            SemstraitError::Compile,
            SemstraitError::Compile,
        )
    }
}

fn key(model: &SemanticModel) -> EntityId {
    mint_model_id(&model.fingerprint())
}

pub(crate) fn built(yaml: &str) -> Outcome<SemanticModel> {
    let builder = parse(yaml).map_err(|errors| refused(errors, SemstraitError::Parse))?;
    let (model, warnings) = builder
        .build()
        .map_err(|errors| refused(errors, SemstraitError::Build))?;
    Ok((model, lifted(warnings, SemstraitError::Build).collect()))
}

pub(crate) fn validated(manifest: SemanticManifest) -> Outcome<SemanticManifest> {
    let warnings =
        (manifest.validate()).map_err(|errors| refused(errors, SemstraitError::Compile))?;
    Ok((
        manifest,
        lifted(warnings, SemstraitError::Compile).collect(),
    ))
}

fn stored(kind: RepositoryErrorKind) -> Refusal {
    refusal(SemstraitError::Repository(kind))
}

fn planned<P>(staged: Staged<P, PlanError, Advisory>) -> Outcome<P> {
    joined(
        Vec::new(),
        staged,
        SemstraitError::Plan,
        SemstraitError::Advisory,
    )
}

fn adapted<P, T>(
    staged: Staged<P, PlanError, Advisory>,
    adapt: impl FnOnce(&P) -> Staged<T, InvalidRewrite, InvalidRewrite>,
) -> Outcome<T> {
    let (plan, advisories) = planned(staged)?;
    joined(
        advisories,
        adapt(&plan),
        SemstraitError::Adapt,
        SemstraitError::Adapt,
    )
}
