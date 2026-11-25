#[cfg(feature = "aws")]
use semstrait_catalog::secrets::{build_polaris_catalog, PolarisCatalogConfig};
use semstrait_catalog::IcebergRestCatalogProvider;
#[cfg(feature = "aws")]
use semstrait_common::Diagnose;
use semstrait_model::{CatalogAuthMethod, CatalogEntry};

use super::CatalogConnectionError;

pub(super) async fn provider(
    alias: &str,
    entry: &CatalogEntry,
) -> Result<IcebergRestCatalogProvider, CatalogConnectionError> {
    #[cfg(feature = "aws")]
    if let Some(config) = polaris_config(alias, entry) {
        return build_polaris_catalog(&config).await.map_err(|kind| {
            CatalogConnectionError::SecretFetch {
                alias: alias.to_owned(),
                reason: kind.message(),
            }
        });
    }
    let direct =
        IcebergRestCatalogProvider::new(alias, entry.url.as_str()).with_prefix(entry.name.as_str());
    let direct = match &entry.realm {
        Some(realm) => direct.with_custom_header("Polaris-Realm", realm),
        None => direct,
    };
    match &entry.auth {
        CatalogAuthMethod::Oauth2 {
            token_url,
            client_id,
            client_secret,
            scope,
        } => {
            let token_url = token_url
                .clone()
                .unwrap_or_else(|| format!("{}/v1/oauth/tokens", entry.url));
            Ok(direct.with_oauth2(token_url, client_id, client_secret, scope.clone()))
        }
        CatalogAuthMethod::Bearer { token } => Ok(direct.with_bearer_token(token)),
        _ => Err(CatalogConnectionError::UnknownAuth {
            alias: alias.to_owned(),
        }),
    }
}

#[cfg(feature = "aws")]
fn polaris_config<'a>(alias: &'a str, entry: &'a CatalogEntry) -> Option<PolarisCatalogConfig<'a>> {
    let CatalogAuthMethod::AwsSecrets {
        secret_arn,
        region,
        scope,
        aws_access_key_id,
        aws_secret_access_key,
        aws_session_token,
        aws_profile,
        ..
    } = &entry.auth
    else {
        return None;
    };
    let given = |credential: &'a Option<String>| credential.as_deref().filter(|it| !it.is_empty());
    Some(PolarisCatalogConfig {
        id: alias,
        catalog_url: &entry.url,
        secret_arn,
        aws_region: region.as_deref(),
        warehouse: Some(&entry.name),
        realm: entry.realm.as_deref(),
        scope: scope.as_deref(),
        aws_profile: given(aws_profile),
        aws_access_key_id: given(aws_access_key_id),
        aws_secret_access_key: given(aws_secret_access_key),
        aws_session_token: given(aws_session_token),
    })
}

#[cfg(all(test, feature = "aws"))]
mod tests {
    use semstrait_catalog::secrets::PolarisCatalogConfig;
    use semstrait_model::{parse_catalogs, CatalogEntry};

    use super::polaris_config;

    fn polaris(yaml: &str) -> CatalogEntry {
        let (mut catalogs, _) = parse_catalogs(yaml).expect("catalogs.yaml parses");
        catalogs
            .catalogs
            .remove("polaris")
            .expect("the entry is polaris")
    }

    fn credentials<'a>(config: &PolarisCatalogConfig<'a>) -> [Option<&'a str>; 4] {
        [
            config.aws_profile,
            config.aws_access_key_id,
            config.aws_secret_access_key,
            config.aws_session_token,
        ]
    }

    #[test]
    fn the_polaris_config_of_test_data_is_named_by_its_catalog_alias() {
        let entry = polaris(include_str!("../../tests/fixtures/catalogs.yaml"));
        let config = polaris_config("polaris", &entry).expect("the entry reads Secrets Manager");

        assert_eq!(
            (config.id, config.warehouse, config.realm),
            ("polaris", Some("example_warehouse"), Some("POLARIS"))
        );
        assert_eq!(credentials(&config), [None; 4]);
    }

    #[test]
    fn an_empty_aws_string_reads_as_absent_and_a_given_one_is_kept() {
        for (value, read) in [("", None), ("given", Some("given"))] {
            let entry = polaris(&format!(
                r#"
catalogs:
  polaris:
    type: polaris
    name: warehouse
    url: https://polaris.example
    auth:
      method: aws_secrets
      secret_arn: arn
      aws_profile: "{value}"
      aws_access_key_id: "{value}"
      aws_secret_access_key: "{value}"
      aws_session_token: "{value}"
"#
            ));
            let config =
                polaris_config("polaris", &entry).expect("the entry reads Secrets Manager");

            assert_eq!(credentials(&config), [read; 4]);
        }
    }
}
