use std::io::{self, Read, Write};
use std::iter::once;
use std::process::ExitCode;
use std::sync::Arc;

use axum::Router;
use clap::{Args, Parser, Subcommand};
use prost::Message;
use semstrait_common::io::{Persistable, RepositoryErrorKind};
use semstrait_common::DialectRelease;
use semstrait_common::{DataKindName, SemanticsName};
use semstrait_manifest::SemanticManifest;
use semstrait_model::{parse_catalogs, CatalogsParseErrorKind};
use semstrait_planner::{Request, RequestContext, Selection};
use tokio::net::TcpListener;

use crate::dialect::rules;
use crate::error::{beside, lifted, refusal, refused, wired, Outcome, SemstraitError};
use crate::input::Input;
use crate::load::load;
use crate::orchestrator::{Semstrait, SemstraitBuilder};
use crate::reader::{read, read_stdin, write};
use crate::rest::{explanation, router};
use crate::wire::RequestPart;

#[derive(Debug, Parser)]
#[command(name = "semstrait", version, about)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Compile(Compile),
    Validate(Requested),
    Explain {
        #[command(flatten)]
        requested: Requested,
        #[arg(long)]
        json: bool,
    },
    Plan {
        #[command(flatten)]
        requested: Requested,
        #[arg(long)]
        output: String,
    },
    Serve(Serve),
}

#[derive(Debug, Args)]
pub struct Compile {
    #[arg(long)]
    pub input: String,
    #[command(flatten)]
    pub catalogs: Catalogs,
    #[arg(long)]
    pub output: Option<String>,
}

#[derive(Debug, Args)]
pub struct Catalogs {
    #[arg(long = "catalogs")]
    pub path: Option<String>,
    #[arg(long = "catalog", requires = "path")]
    pub alias: Option<String>,
}

#[derive(Debug, Args)]
pub struct Model {
    #[arg(long)]
    pub model: String,
    #[command(flatten)]
    pub catalogs: Catalogs,
    #[arg(long, default_value = "substrait")]
    pub dialect: DialectRelease,
}

#[derive(Debug, Args)]
pub struct Serve {
    #[command(flatten)]
    pub model: Model,
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,
    #[arg(long, default_value_t = 8080)]
    pub port: u16,
}

#[derive(Debug, Args)]
pub struct Requested {
    #[command(flatten)]
    pub model: Model,
    #[arg(
        long,
        required_unless_present = "select",
        conflicts_with_all = ["from", "select", "filters"]
    )]
    pub request: Option<String>,
    #[arg(long)]
    pub from: Option<DataKindName>,
    #[arg(long, num_args = 1..)]
    pub select: Vec<SemanticsName>,
    #[arg(long, num_args = 1..)]
    pub filters: Vec<SemanticsName>,
}

impl Cli {
    pub async fn run(
        self,
        stdin: impl Read,
        out: &mut impl Write,
        err: &mut impl Write,
    ) -> io::Result<ExitCode> {
        let printed = match self.command {
            Command::Compile(compile) => compile.compiled().await,
            Command::Validate(requested) => requested.validated(stdin).await,
            Command::Explain { requested, json } => requested.explained(stdin, json).await,
            Command::Plan { requested, output } => requested.planned(stdin, &output).await,
            Command::Serve(serve) => return serve.served(err).await,
        };
        let Some(printed) = reported(err, printed)? else {
            return Ok(ExitCode::FAILURE);
        };
        out.write_all(&printed)?;
        Ok(ExitCode::SUCCESS)
    }
}

impl Compile {
    async fn compiled(self) -> Outcome<Vec<u8>> {
        let yaml = read(&self.input).await?;
        let (semstrait, warnings) = self.catalogs.connected().await?;
        let (manifest, warnings) = beside(warnings, semstrait.compile_from_yaml(&yaml).await)?;
        let printed = match manifest.to_bytes() {
            Ok(json) => written(self.output.as_deref(), json.into()).await,
            Err(actual) => Err(refusal(SemstraitError::Repository(
                RepositoryErrorKind::EncodingMismatch { actual },
            ))),
        };
        beside(warnings, printed)
    }
}

impl Catalogs {
    async fn connected(self) -> Outcome<Semstrait> {
        let builder = SemstraitBuilder::default();
        let Some(path) = self.path else {
            return Ok((builder.build(), Vec::new()));
        };
        let lift = |kind: CatalogsParseErrorKind| SemstraitError::Build(kind.into());
        let parsed = parse_catalogs(&read(&path).await?);
        let (catalogs, warnings) = parsed.map_err(|errors| refused(errors, lift))?;
        let connected = builder
            .with_catalogs(&catalogs, self.alias.as_deref())
            .await;
        let built = connected.map(|builder| (builder.build(), Vec::new()));
        beside(lifted(warnings, lift).collect(), built)
    }
}

impl Model {
    async fn loaded(self) -> Outcome<(Semstrait, SemanticManifest, DialectRelease)> {
        let (semstrait, warnings) = self.catalogs.connected().await?;
        let loaded = load(&semstrait, Input::ModelPath(self.model)).await;
        let (manifest, warnings) = beside(warnings, loaded)?;
        Ok(((semstrait, manifest, self.dialect), warnings))
    }
}

impl Requested {
    async fn validated(self, stdin: impl Read) -> Outcome<Vec<u8>> {
        (self.asked(stdin, |semstrait, manifest, request, context, _| {
            let advisories = semstrait.validate(manifest, request, context)?;
            Ok((b"valid\n".to_vec(), advisories))
        }))
        .await
    }

    async fn explained(self, stdin: impl Read, json: bool) -> Outcome<Vec<u8>> {
        (self.asked(stdin, |semstrait, manifest, request, context, dialect| {
            let (text, advisories) = semstrait.explain(manifest, request, context, dialect)?;
            let printed = match json {
                true => format!("{:#}\n", explanation(&text, &advisories)),
                false => text,
            };
            Ok((printed.into_bytes(), advisories))
        }))
        .await
    }

    async fn planned(self, stdin: impl Read, output: &str) -> Outcome<Vec<u8>> {
        let (plan, advisories) =
            (self.asked(stdin, |semstrait, manifest, request, context, dialect| {
                semstrait.plan(manifest, request, context, dialect)
            }))
            .await?;
        let written = written(Some(output), plan.encode_to_vec()).await;
        beside(advisories, written)
    }

    async fn asked<T>(
        self,
        stdin: impl Read,
        verb: impl FnOnce(
            &Semstrait,
            &SemanticManifest,
            &Request,
            &RequestContext,
            DialectRelease,
        ) -> Outcome<T>,
    ) -> Outcome<T> {
        let (request, context) = match self.request.as_deref() {
            Some(path) => {
                let document = match path {
                    "-" => read_stdin(stdin)?,
                    path => read(path).await?,
                };
                RequestPart::read_yaml(&document).map_err(wired)?
            }
            None => {
                let select = self.select.into_iter().map(Selection::Field).collect();
                let request = Request::new(select).with_filters(self.filters);
                let request = self.from.into_iter().fold(request, Request::with_from);
                (request, RequestContext::default())
            }
        };
        let ((semstrait, manifest, dialect), warnings) = self.model.loaded().await?;
        beside(
            warnings,
            verb(&semstrait, &manifest, &request, &context, dialect),
        )
    }
}

impl Serve {
    pub async fn router(self) -> Outcome<Router> {
        let ((semstrait, manifest, dialect), warnings) = self.model.loaded().await?;
        let served = rules(dialect).map(|_| {
            (
                router(Arc::new(semstrait), Arc::new(manifest), dialect),
                Vec::new(),
            )
        });
        beside(warnings, served)
    }

    async fn served(self, err: &mut impl Write) -> io::Result<ExitCode> {
        let listener = TcpListener::bind((self.host.as_str(), self.port)).await?;
        let Some(router) = reported(err, self.router().await)? else {
            return Ok(ExitCode::FAILURE);
        };
        writeln!(err, "listening on http://{}", listener.local_addr()?)?;
        axum::serve(listener, router).await?;
        Ok(ExitCode::SUCCESS)
    }
}

async fn written(output: Option<&str>, bytes: Vec<u8>) -> Outcome<Vec<u8>> {
    let printed = match output {
        Some(path) => write(path, bytes).await.map(|()| Vec::new())?,
        None => [bytes, b"\n".to_vec()].concat(),
    };
    Ok((printed, Vec::new()))
}

fn reported<T>(err: &mut impl Write, outcome: Outcome<T>) -> io::Result<Option<T>> {
    let (output, diagnostics) = match outcome {
        Ok((output, warnings)) => (Some(output), warnings),
        Err((head, rest)) => (None, once(head).chain(rest).collect()),
    };
    for diagnostic in &diagnostics {
        writeln!(err, "{diagnostic}")?;
    }
    Ok(output)
}
