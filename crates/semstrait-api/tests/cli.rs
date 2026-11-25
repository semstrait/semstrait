#![cfg(feature = "cli")]

mod common;

use std::iter::once;
use std::process::ExitCode;

use axum::body::Body;
use axum::http::Request;
use clap::error::ErrorKind::{
    ArgumentConflict, MissingRequiredArgument, UnknownArgument, ValueValidation,
};
use clap::Parser;
use common::{only_retired_keys, written, SALES};
use http_body_util::BodyExt;
use prost::Message;
use semstrait_api::cli::{Cli, Command};
use semstrait_api::{NoEngineAdapter, SemstraitError};
use semstrait_common::{Dialect, DialectRelease, Release};
use semstrait_manifest::SemanticManifest;
use serde_json::{json, Value};
use substrait::proto::Plan;
use tempfile::{tempdir, TempDir};
use tower::ServiceExt;

const RETIRED_KEYS: &str = "warning: data kind `orders`: the `keys:` block is retired and \
                            reaches nothing; mark each identifying dimension site `is_key: true` \
                            instead (at <inline> [0..0])\n";

const AMBIGUOUS_CANDIDATES: &str = "warning: at `orders`: `orders` is chosen by name order over \
                                    `returns`: name one in `from`\n";

const UNKNOWN_REVENUE: &str = "error: `revenue` is not in the semantic definitions\n";

fn parsed(args: &[&str]) -> Result<Cli, clap::Error> {
    Cli::try_parse_from(once("semstrait").chain(args.iter().copied()))
}

async fn ran(args: &[&str], stdin: &str) -> (ExitCode, String, String) {
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let code = (parsed(args).expect("the arguments parse"))
        .run(stdin.as_bytes(), &mut out, &mut err)
        .await
        .expect("stdout and stderr take every write");
    let text = |bytes| String::from_utf8(bytes).expect("the output is text");
    (code, text(out), text(err))
}

fn sales_file() -> (TempDir, String) {
    let dir = tempdir().expect("a temporary directory");
    let model = written(&dir, "sales.yaml", SALES);
    (dir, model)
}

fn output_in(dir: &TempDir, name: &str) -> String {
    dir.path().join(name).display().to_string()
}

fn manifest(json: &[u8]) -> SemanticManifest {
    serde_json::from_slice(json).expect("the output is manifest JSON")
}

#[test]
fn the_parser_refuses_a_missing_required_flag_a_refused_combination_and_a_removed_flag() {
    let missing = [
        "compile",
        "validate --select region",
        "validate --model m.yaml",
        "plan --model m.yaml --select region",
        "serve",
        "compile --input m.yaml --catalog polaris",
        "explain --model m.yaml --select region --catalog polaris",
    ];
    let conflicting = [
        "validate --model m.yaml --request r.yaml --select region",
        "validate --model m.yaml --request r.yaml --from orders",
        "validate --model m.yaml --request r.yaml --filters paid_only",
    ];
    let invalid = ["explain --model m.yaml --select region --dialect ansi"];
    let removed = [
        "explain --model m.yaml --select region --output plan",
        "explain --model m.yaml --select region --output sql",
        "explain --model m.yaml --select region --engine ansi",
        "serve --model m.yaml --engine datafusion",
        "serve --model m.yaml --grpc-port 50051",
    ];
    let cases = [
        (MissingRequiredArgument, &missing[..]),
        (ArgumentConflict, &conflicting),
        (ValueValidation, &invalid),
        (UnknownArgument, &removed),
    ];

    for (kind, lines) in cases {
        for line in lines {
            let args: Vec<_> = line.split_whitespace().collect();

            let refused = parsed(&args).expect_err("the parser refuses the arguments");

            assert_eq!(refused.kind(), kind, "{line}");
        }
    }
}

#[test]
fn the_dialect_defaults_to_substrait_and_serve_to_port_8080_on_every_address() {
    let explain = parsed(&["explain", "--model", "m.yaml", "--select", "region"]);
    let serve = parsed(&["serve", "--model", "m.yaml"]);

    let (Command::Explain { requested, json }, Command::Serve(serve)) = (
        explain.expect("explain parses").command,
        serve.expect("serve parses").command,
    ) else {
        panic!("the commands are explain and serve");
    };
    assert_eq!(
        (json, requested.model.dialect, requested.model.catalogs.path),
        (false, DialectRelease::from(Dialect::Substrait), None)
    );
    assert_eq!(
        (serve.host.as_str(), serve.port, serve.model.dialect),
        ("0.0.0.0", 8080, DialectRelease::from(Dialect::Substrait))
    );
}

#[test]
fn the_parser_reads_several_names_each_dialect_and_the_address_serve_is_given() {
    let explain = parsed(&[
        "explain",
        "--model",
        "m.yaml",
        "--select",
        "region",
        "revenue",
        "--filters",
        "paid_only",
        "web_only",
        "--dialect",
        "iceberg",
    ]);
    let serve = parsed(&[
        "serve",
        "--model",
        "m.yaml",
        "--host",
        "127.0.0.1",
        "--port",
        "9090",
        "--dialect",
        "datafusion",
    ]);

    let (Command::Explain { requested, .. }, Command::Serve(serve)) = (
        explain.expect("explain parses").command,
        serve.expect("serve parses").command,
    ) else {
        panic!("the commands are explain and serve");
    };
    assert_eq!(
        (requested.select, requested.filters, requested.model.dialect),
        (
            vec!["region".into(), "revenue".into()],
            vec!["paid_only".into(), "web_only".into()],
            DialectRelease::from(Dialect::Iceberg)
        )
    );
    assert_eq!(
        (serve.host.as_str(), serve.port, serve.model.dialect),
        ("127.0.0.1", 9090, DialectRelease::from(Dialect::DataFusion))
    );
}

#[test]
fn the_parser_accepts_an_explicit_dialect_release_and_refuses_one_below_the_oldest() {
    let explain = parsed(&[
        "explain",
        "--model",
        "m.yaml",
        "--select",
        "region",
        "--dialect",
        "datafusion=53",
    ]);
    let Command::Explain { requested, .. } = explain.expect("explain parses").command else {
        panic!("the command is explain");
    };
    assert_eq!(
        requested.model.dialect,
        DialectRelease {
            dialect: Dialect::DataFusion,
            release: Some(Release(53)),
        }
    );

    let refused = parsed(&[
        "explain",
        "--model",
        "m.yaml",
        "--select",
        "region",
        "--dialect",
        "datafusion=52",
    ])
    .expect_err("a release below the oldest is refused");

    assert_eq!(refused.kind(), ValueValidation);
    assert!(refused.to_string().contains("datafusion=52"), "{refused}");
}

#[test]
fn the_binary_exits_with_the_parser_s_code_on_an_argument_error_and_1_on_a_refusal() {
    let exited = |args: &[&str]| {
        let mut binary = std::process::Command::new(env!("CARGO_BIN_EXE_semstrait"));
        let output = binary.args(args).output().expect("the binary runs");
        let stderr = String::from_utf8(output.stderr).expect("stderr is text");
        (output.status.code(), stderr)
    };

    let (code, _) = exited(&["compile"]);
    let refused = exited(&["compile", "--input", "absent.yaml"]);

    assert_eq!(code, Some(2));
    let not_found = "error: not found: absent.yaml (at absent.yaml [0..0])\n";
    assert_eq!(refused, (Some(1), not_found.into()));
}

#[tokio::test]
async fn compile_writes_the_manifest_json_of_the_semantic_model_to_stdout() {
    let (_dir, model) = sales_file();

    let (code, out, err) = ran(&["compile", "--input", &model], "").await;

    assert_eq!(code, ExitCode::SUCCESS);
    let manifest = manifest(out.as_bytes());
    assert_eq!(manifest.metadata.source_model_uri.as_deref(), Some("sales"));
    assert_eq!(err, RETIRED_KEYS);
}

#[tokio::test]
async fn compile_with_output_writes_the_manifest_json_to_the_file_and_nothing_to_stdout() {
    let (dir, model) = sales_file();
    let output = output_in(&dir, "sales.json");

    let (code, out, _) = ran(&["compile", "--input", &model, "--output", &output], "").await;

    assert_eq!((code, out.as_str()), (ExitCode::SUCCESS, ""));
    let manifest = manifest(&std::fs::read(output).expect("the output file is written"));
    assert_eq!(manifest.metadata.source_model_uri.as_deref(), Some("sales"));
}

#[tokio::test]
async fn compile_of_a_model_that_parse_refuses_exits_1_with_its_errors_on_stderr() {
    let dir = tempdir().expect("a temporary directory");
    let model = written(&dir, "broken.yaml", "semantic_model: [");

    let compiled = ran(&["compile", "--input", &model], "").await;

    let syntax = "error: YAML syntax error: semantic_model: invalid type: sequence, expected \
                  struct YamlSemanticModel at line 1 column 17\n";
    assert_eq!(compiled, (ExitCode::FAILURE, "".into(), syntax.into()));
}

#[tokio::test]
async fn the_only_entry_or_the_named_entry_of_catalogs_yaml_that_is_refused_exits_1() {
    let (dir, model) = sales_file();
    let unity = "catalogs: {unity: {type: unity, name: main, url: https://unity.example, \
                 auth: {method: bearer, token: token}}}";
    let catalogs = written(&dir, "catalogs.yaml", unity);
    let broken = written(&dir, "broken.yaml", "catalogs: [");
    let validate = ["validate", "--model", &model, "--select", "region"];
    let cases = [
        (
            vec!["compile", "--input", &model, "--catalogs", &catalogs],
            "error: catalog `unity`: no provider has the type `unity`; the types are polaris \
             and iceberg_rest\n",
        ),
        (
            [
                &validate[..],
                &["--catalogs", &catalogs, "--catalog", "lake"],
            ]
            .concat(),
            "error: no entry of catalogs.yaml has the catalog alias `lake`; its entries are \
             [\"unity\"]\n",
        ),
        (
            [&validate[..], &["--catalogs", &broken]].concat(),
            "error: catalogs YAML syntax error: catalogs: invalid type: sequence, expected a map \
             at line 1 column 11\n",
        ),
    ];

    for (args, refused) in cases {
        let ran = ran(&args, "").await;

        assert_eq!(
            ran,
            (ExitCode::FAILURE, "".into(), refused.into()),
            "{args:?}"
        );
    }
}

#[tokio::test]
async fn validate_prints_valid_and_the_advisories_of_an_accepted_request() {
    let (_dir, model) = sales_file();

    let validated = ran(&["validate", "--model", &model, "--select", "region"], "").await;

    let advisories = [RETIRED_KEYS, AMBIGUOUS_CANDIDATES].concat();
    assert_eq!(validated, (ExitCode::SUCCESS, "valid\n".into(), advisories));
}

#[tokio::test]
async fn the_quick_flags_give_the_request_its_from_and_its_filters() {
    let (_dir, model) = sales_file();
    let validate = ["validate", "--model", &model, "--select", "region"];

    let from = ran(&[&validate[..], &["--from", "orders"]].concat(), "").await;
    let filters = ran(&[&validate[..], &["--filters", "paid_only"]].concat(), "").await;

    assert_eq!(
        from,
        (ExitCode::SUCCESS, "valid\n".into(), RETIRED_KEYS.into())
    );
    let unknown_filter = "error: `paid_only` is not in the semantic definitions\n";
    let refused = [unknown_filter, RETIRED_KEYS].concat();
    assert_eq!(filters, (ExitCode::FAILURE, "".into(), refused));
}

#[tokio::test]
async fn a_request_document_is_read_from_its_file_or_from_stdin() {
    let (dir, model) = sales_file();
    let request = written(
        &dir,
        "request.json",
        r#"{"from": "orders", "select": ["region"]}"#,
    );

    for (path, stdin) in [
        (request.as_str(), ""),
        ("-", "from: orders\nselect: [region]\n"),
    ] {
        let validated = ran(&["validate", "--model", &model, "--request", path], stdin).await;

        let accepted = (ExitCode::SUCCESS, "valid\n".into(), RETIRED_KEYS.into());
        assert_eq!(validated, accepted, "{path}");
    }
}

#[tokio::test]
async fn a_request_document_with_an_unknown_key_or_a_refused_where_exits_1_with_the_wire_error() {
    let (_dir, model) = sales_file();
    let args = ["validate", "--model", &model, "--request", "-"];
    for (document, refused) in [
        (
            "select: [region]\ncolour: 1\n",
            "error: the Request document is refused: unknown field `colour`, expected one of \
             `from`, `select`, `where`, `filters`, `order_by`, `limit`, `offset`, `context` at \
             line 2 column 1\n",
        ),
        (
            "select: [region]\nwhere: {nope: [channel]}\n",
            "error: the Request document is refused: where: unknown reserved tag: `nope` at \
             line 2 column 8\n",
        ),
    ] {
        let validated = ran(&args, document).await;

        assert_eq!(validated, (ExitCode::FAILURE, "".into(), refused.into()));
    }
}

#[tokio::test]
async fn explain_prints_the_explain_text_of_an_accepted_request() {
    let (_dir, model) = sales_file();

    let (code, out, err) = ran(&["explain", "--model", &model, "--select", "region"], "").await;

    assert_eq!(code, ExitCode::SUCCESS);
    assert!(out.contains("Read[\"\".warehouse.fact_rows => region:string?]"));
    assert_eq!(err, [RETIRED_KEYS, AMBIGUOUS_CANDIDATES].concat());
}

#[tokio::test]
async fn explain_with_json_prints_the_rest_body_of_the_text_and_the_advisories() {
    let (_dir, model) = sales_file();
    let args = ["explain", "--model", &model, "--select", "region", "--json"];

    let (code, out, _) = ran(&args, "").await;
    let (_, text, _) = ran(&args[..5], "").await;

    assert_eq!(code, ExitCode::SUCCESS);
    let body: Value = serde_json::from_str(&out).expect("the output is JSON");
    let ambiguous = json!({
        "message": "at `orders`: `orders` is chosen by name order over `returns`: \
                    name one in `from`",
        "severity": "warning",
    });
    assert_eq!(body, json!({ "text": text, "advisories": [ambiguous] }));
}

#[tokio::test]
async fn plan_writes_the_protobuf_bytes_of_the_substrait_plan_of_an_accepted_request() {
    let (dir, model) = sales_file();
    let output = output_in(&dir, "plan.pb");
    let args = [
        "plan", "--model", &model, "--select", "region", "--output", &output,
    ];

    let (code, out, err) = ran(&args, "").await;

    assert_eq!((code, out.as_str()), (ExitCode::SUCCESS, ""));
    assert_eq!(err, [RETIRED_KEYS, AMBIGUOUS_CANDIDATES].concat());
    let bytes = std::fs::read(output).expect("the plan file is written");
    let plan = Plan::decode(&bytes[..]).expect("the file decodes into a Substrait plan");
    assert_eq!(plan.relations.len(), 1);
}

#[tokio::test]
async fn validate_explain_and_plan_of_a_refused_request_exit_1_with_its_errors_on_stderr() {
    let (dir, model) = sales_file();
    let output = output_in(&dir, "plan.pb");
    let commands = [
        &["validate"][..],
        &["explain"],
        &["explain", "--json"],
        &["plan", "--output", &output],
    ];

    for command in commands {
        let args = [command, &["--model", &model, "--select", "revenue"]].concat();

        let ran = ran(&args, "").await;

        let refused = [UNKNOWN_REVENUE, RETIRED_KEYS].concat();
        assert_eq!(ran, (ExitCode::FAILURE, "".into(), refused), "{command:?}");
    }
    assert!(!dir.path().join("plan.pb").exists());
}

async fn explained_in(dialect: &str) -> (ExitCode, String, String) {
    let (_dir, model) = sales_file();
    let args = ["explain", "--model", &model, "--select", "region"];
    ran(&[&args[..], &["--dialect", dialect]].concat(), "").await
}

#[tokio::test]
async fn the_iceberg_dialect_exits_1_with_the_typed_error() {
    let explained = explained_in("iceberg").await;

    let no_adapter = "error: the dialect Iceberg has no engine adapter\n";
    let refused = [no_adapter, RETIRED_KEYS].concat();
    assert_eq!(explained, (ExitCode::FAILURE, "".into(), refused));
}

#[cfg(not(feature = "datafusion"))]
#[tokio::test]
async fn the_datafusion_dialect_exits_1_with_the_typed_error_without_the_datafusion_feature() {
    let explained = explained_in("datafusion").await;

    let no_adapter = "error: the dialect DataFusion needs the `datafusion` feature\n";
    let refused = [no_adapter, RETIRED_KEYS].concat();
    assert_eq!(explained, (ExitCode::FAILURE, "".into(), refused));
}

#[cfg(feature = "datafusion")]
#[tokio::test]
async fn the_datafusion_dialect_explains_with_the_datafusion_rules_with_the_datafusion_feature() {
    let (code, out, _) = explained_in("datafusion").await;

    assert_eq!(code, ExitCode::SUCCESS);
    assert!(out.contains("Read[warehouse.fact_rows => region:string?]"));
}

#[tokio::test]
async fn serve_builds_its_router_from_the_loaded_manifest_without_a_socket() {
    let (_dir, model) = sales_file();
    let parsed = parsed(&["serve", "--model", &model]).expect("the arguments parse");
    let Command::Serve(serve) = parsed.command else {
        panic!("the command is serve, not {:?}", parsed.command);
    };

    let (router, warnings) = serve.router().await.expect("the model loads");
    let schema = Request::get("/schema")
        .body(Body::empty())
        .expect("the request builds");
    let response = router.oneshot(schema).await.expect("the router answers");

    assert!(only_retired_keys(warnings));
    let body = response
        .into_body()
        .collect()
        .await
        .expect("the body reads");
    let schema: Value = serde_json::from_slice(&body.to_bytes()).expect("the body is JSON");
    let source =
        |name| json!({"name": name, "dimensions": ["region"], "measures": [], "metrics": []});
    assert_eq!(schema, json!([source("orders"), source("returns")]));
}

#[tokio::test]
async fn serve_refuses_at_start_a_dialect_with_no_engine_adapter() {
    let (_dir, model) = sales_file();
    let parsed = parsed(&["serve", "--model", &model, "--dialect", "iceberg"]);
    let Command::Serve(serve) = parsed.expect("the arguments parse").command else {
        panic!("the command is serve");
    };

    let (refusal, warnings) = serve.router().await.expect_err("the dialect is refused");

    let no_rules = NoEngineAdapter::NoRules(Dialect::Iceberg);
    assert!(matches!(refusal.kind, SemstraitError::NoEngineAdapter(kind) if kind == no_rules));
    assert!(only_retired_keys(warnings));
}
