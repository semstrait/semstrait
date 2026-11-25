use std::fmt::Debug;
use std::marker::PhantomData;

use semstrait_api::{CatalogConnectionError, Semstrait, SemstraitBuilder, SemstraitError};
use semstrait_catalog::CatalogId;
use semstrait_common::Diagnose;
use semstrait_model::parse_catalogs;

const POLARIS: &str = r#"
  polaris:
    type: polaris
    name: warehouse
    url: https://polaris.example/api/catalog
    realm: POLARIS
    auth:
      method: oauth2
      client_id: id
      client_secret: secret
"#;

const LAKE: &str = r#"
  lake:
    type: iceberg_rest
    name: lake
    url: https://lake.example/api/catalog
    auth:
      method: bearer
      token: token
"#;

#[cfg(not(feature = "aws"))]
const SECRETS: &str = r#"
  polaris:
    type: polaris
    name: warehouse
    url: https://polaris.example/api/catalog
    auth:
      method: aws_secrets
      secret_arn: arn:aws:secretsmanager:eu-west-1:000000000000:secret:polaris
"#;

const UNITY: &str = r#"
  unity:
    type: unity
    name: main
    url: https://unity.example
    auth:
      method: bearer
      token: token
"#;

async fn connected(
    entries: &[&str],
    alias: Option<&str>,
) -> Result<Option<CatalogId>, CatalogConnectionError> {
    let yaml = format!("catalogs:{}", entries.concat());
    let (catalogs, _) = parse_catalogs(&yaml).expect("catalogs.yaml parses");
    match SemstraitBuilder::default()
        .with_catalogs(&catalogs, alias)
        .await
    {
        Ok(builder) => Ok(builder.build().catalog().map(|catalog| catalog.id())),
        Err((refusal, _)) => match refusal.kind {
            SemstraitError::Catalog(kind) => Err(kind),
            other => panic!("the refusal is the catalog connection's, not {other:?}"),
        },
    }
}

#[cfg(feature = "iceberg")]
#[tokio::test]
async fn the_only_entry_of_catalogs_yaml_is_the_catalog_the_semstrait_holds() {
    assert_eq!(
        connected(&[POLARIS], None).await,
        Ok(Some(CatalogId::new("polaris")))
    );
}

#[cfg(feature = "iceberg")]
#[tokio::test]
async fn the_catalog_alias_picks_one_of_several_entries() {
    for alias in ["lake", "polaris"] {
        assert_eq!(
            connected(&[LAKE, POLARIS], Some(alias)).await,
            Ok(Some(CatalogId::new(alias)))
        );
    }
}

#[tokio::test]
async fn several_entries_and_no_catalog_alias_are_refused() {
    assert_eq!(
        connected(&[LAKE, POLARIS], None).await,
        Err(CatalogConnectionError::AliasNotGiven {
            aliases: vec!["lake".into(), "polaris".into()],
        })
    );
}

#[tokio::test]
async fn a_catalogs_yaml_with_no_entry_and_no_catalog_alias_is_refused_for_want_of_one_entry() {
    let refused = connected(&[" {}"], None)
        .await
        .expect_err("no entry connects");

    assert_eq!(
        refused.message(),
        "with no catalog alias, catalogs.yaml must have exactly one entry; its entries are []"
    );
}

#[tokio::test]
async fn a_catalog_alias_that_no_entry_has_is_refused() {
    assert_eq!(
        connected(&[LAKE, POLARIS], Some("unity")).await,
        Err(CatalogConnectionError::AliasNotFound {
            alias: "unity".into(),
            aliases: vec!["lake".into(), "polaris".into()],
        })
    );
}

#[tokio::test]
async fn an_entry_of_an_unknown_type_is_refused() {
    assert_eq!(
        connected(&[UNITY], None).await,
        Err(CatalogConnectionError::UnknownType {
            alias: "unity".into(),
            provider_type: "unity".into(),
        })
    );
}

#[cfg(not(feature = "aws"))]
#[tokio::test]
async fn aws_secrets_without_the_aws_feature_is_refused() {
    assert_eq!(
        connected(&[SECRETS], None).await,
        Err(CatalogConnectionError::AwsFeatureOff {
            alias: "polaris".into(),
        })
    );
}

#[cfg(not(feature = "iceberg"))]
#[tokio::test]
async fn a_polaris_entry_without_the_iceberg_feature_is_refused() {
    assert_eq!(
        connected(&[POLARIS], None).await,
        Err(CatalogConnectionError::IcebergFeatureOff {
            alias: "polaris".into(),
        })
    );
}

struct Probe<T>(PhantomData<T>);

trait NotDebug {
    const DEBUG: bool = false;
}

impl<T> NotDebug for Probe<T> {}

impl<T: Debug> Probe<T> {
    const DEBUG: bool = true;
}

#[test]
fn only_the_types_that_hold_no_credential_are_debug() {
    let debug: Vec<&str> = [
        (
            "CatalogConnectionError",
            Probe::<CatalogConnectionError>::DEBUG,
        ),
        ("Semstrait", Probe::<Semstrait>::DEBUG),
        ("SemstraitBuilder", Probe::<SemstraitBuilder>::DEBUG),
    ]
    .into_iter()
    .filter_map(|(name, debug)| debug.then_some(name))
    .collect();

    assert_eq!(debug, ["CatalogConnectionError"]);
}
