use semstrait_common::diagnostic::{Diagnostics, Location, Severity, SourceId};
use semstrait_model::{ModelBuildErrorKind, ParseErrorKind, SemanticModel};
use tempfile::TempDir;

const PARITY_YAML: &str = r#"
semantic_model:
  name: parity-model
  dimensions:
    - name: country
      data_type: string
      type: categorical
  datasets:
    - name: orders
      extras:
        semantic_mapping: auto
      dimensions:
        - ref: country
    - name: shopify
      extras:
        semantic_mapping: auto
      dimensions:
        - name: country
          data_type: string
          type: categorical
"#;

const BAD_IDENT_YAML: &str = r#"
semantic_model:
  name: bad-ident-test
  dimensions:
    - name: bad-name
      data_type: string
      type: categorical
"#;

fn kind_list(diags: &Diagnostics<ModelBuildErrorKind>) -> Vec<ModelBuildErrorKind> {
    diags.iter().map(|d| d.kind.clone()).collect()
}

fn triples(
    diags: &Diagnostics<ModelBuildErrorKind>,
) -> Vec<(ModelBuildErrorKind, Severity, Option<Location>)> {
    diags
        .iter()
        .map(|d| (d.kind.clone(), d.severity, d.location.clone()))
        .collect()
}

#[test]
fn str_bytes_file_parity_same_model_and_diagnostics() {
    let (model_str, diags_str) = SemanticModel::loader()
        .from_yaml_str(PARITY_YAML, SourceId::named("model.yaml"))
        .build()
        .expect("str build");

    let (model_bytes, diags_bytes) = SemanticModel::loader()
        .from_yaml_bytes(PARITY_YAML.as_bytes(), SourceId::named("model.yaml"))
        .build()
        .expect("bytes build");

    let tmp = TempDir::new().expect("create temp dir");
    let file_path = tmp.path().join("model.yaml");
    std::fs::write(&file_path, PARITY_YAML).expect("write temp fixture");
    let (model_file, diags_file) = SemanticModel::loader()
        .from_yaml_file(file_path.to_str().expect("utf8 temp path"))
        .build()
        .expect("file build");

    assert_eq!(
        model_str, model_bytes,
        "bytes model diverges from str model"
    );
    assert_eq!(model_str, model_file, "file model diverges from str model");

    assert_eq!(kind_list(&diags_str), kind_list(&diags_bytes));
    assert_eq!(kind_list(&diags_str), kind_list(&diags_file));
    assert_eq!(diags_str.len(), 1);
}

#[test]
fn bytes_source_id_propagates_to_location_same_as_str() {
    let source = SourceId::named("s3://bucket/model.yaml");

    let err_str = SemanticModel::loader()
        .from_yaml_str(BAD_IDENT_YAML, source.clone())
        .build()
        .expect_err("SR-8 violation must fail build");
    let err_bytes = SemanticModel::loader()
        .from_yaml_bytes(BAD_IDENT_YAML.as_bytes(), source.clone())
        .build()
        .expect_err("SR-8 violation must fail build");

    assert_eq!(triples(&err_str), triples(&err_bytes));

    let located = err_bytes
        .iter()
        .find(|d| {
            matches!(
                &d.kind,
                ModelBuildErrorKind::Parse(ParseErrorKind::InvalidIdentifier { .. })
            )
        })
        .expect("expected an InvalidIdentifier diagnostic")
        .location
        .as_ref()
        .expect("InvalidIdentifier always carries a Location");
    assert_eq!(
        located.source, source,
        "bytes payload's SourceId must reach Location verbatim"
    );

    let tmp = TempDir::new().expect("create temp dir");
    let file_path = tmp.path().join("model.yaml");
    std::fs::write(&file_path, BAD_IDENT_YAML).expect("write temp fixture");
    let err_file = SemanticModel::loader()
        .from_yaml_file(file_path.to_str().expect("utf8 temp path"))
        .build()
        .expect_err("SR-8 violation must fail build");
    assert_eq!(kind_list(&err_str), kind_list(&err_file));
}

#[test]
fn malformed_utf8_stages_as_source_io_invalid_data() {
    let bad_bytes: &[u8] = &[b's', b'e', b'm', 0xFF, b'x'];
    let source = SourceId::named("s3://bucket/model.yaml");

    let err = SemanticModel::loader()
        .from_yaml_bytes(bad_bytes, source.clone())
        .build()
        .expect_err("non-UTF-8 payload must fail build, not panic");

    assert_eq!(err.len(), 1);
    match &err[0].kind {
        ModelBuildErrorKind::SourceIo { path, error } => {
            assert_eq!(path, &source.to_string());
            assert_eq!(*error, std::io::ErrorKind::InvalidData);
        }
        other => panic!("expected SourceIo, got {other:?}"),
    }
}

#[test]
fn valid_utf8_invalid_yaml_still_surfaces_parse_diagnostic() {
    let bad_yaml = b"semantic_model: [unterminated, flow";

    let err = SemanticModel::loader()
        .from_yaml_bytes(bad_yaml, SourceId::named("model.yaml"))
        .build()
        .expect_err("invalid YAML must fail build");

    assert!(
        err.iter().any(|d| matches!(
            &d.kind,
            ModelBuildErrorKind::Parse(ParseErrorKind::YamlSyntax { .. })
        )),
        "expected a Parse(YamlSyntax) diagnostic, got {err:?}"
    );
    assert!(
        !err.iter()
            .any(|d| matches!(&d.kind, ModelBuildErrorKind::SourceIo { .. })),
        "byte-validation must not misclassify a YAML-syntax error as SourceIo"
    );
}
