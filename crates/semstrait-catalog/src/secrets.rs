use crate::CatalogProviderErrorKind;
use aws_config::BehaviorVersion;
use aws_config::Region;

#[derive(Clone)]
pub struct PolarisSecret {
    pub client_id: String,
    pub client_secret: String,
}

pub async fn resolve_polaris_secret(
    secret_arn: &str,
    region: Option<&str>,
) -> Result<PolarisSecret, CatalogProviderErrorKind> {
    resolve_polaris_secret_with_credentials(secret_arn, region, None, None, None, None).await
}

pub async fn resolve_polaris_secret_with_credentials(
    secret_arn: &str,
    region: Option<&str>,
    profile: Option<&str>,
    access_key_id: Option<&str>,
    secret_access_key: Option<&str>,
    session_token: Option<&str>,
) -> Result<PolarisSecret, CatalogProviderErrorKind> {
    let mut config_loader = aws_config::defaults(BehaviorVersion::v2026_01_12());
    if let Some(r) = region {
        config_loader = config_loader.region(Region::new(r.to_owned()));
    }
    if let (Some(key_id), Some(secret_key)) = (access_key_id, secret_access_key) {
        let creds = aws_credential_types::Credentials::new(
            key_id,
            secret_key,
            session_token.map(|s| s.to_owned()),
            None,
            "semstrait",
        );
        config_loader = config_loader.credentials_provider(creds);
    } else if let Some(p) = profile {
        config_loader = config_loader.profile_name(p);
    }
    let config = config_loader.load().await;

    let client = aws_sdk_secretsmanager::Client::new(&config);
    let output = client
        .get_secret_value()
        .secret_id(secret_arn)
        .send()
        .await
        .map_err(|e| CatalogProviderErrorKind::ConnectionFailed {
            msg: format!("AWS Secrets Manager error: {e:?}"),
        })?;

    let secret_string =
        output
            .secret_string()
            .ok_or_else(|| CatalogProviderErrorKind::Internal {
                msg: "secret is binary, expected JSON string".to_owned(),
            })?;

    parse_polaris_secret(secret_string)
}

pub(crate) fn parse_polaris_secret(json: &str) -> Result<PolarisSecret, CatalogProviderErrorKind> {
    let parsed: serde_json::Value =
        serde_json::from_str(json).map_err(|e| CatalogProviderErrorKind::Internal {
            msg: format!("failed to parse secret JSON: {e}"),
        })?;

    let client_id = parsed["client_id"]
        .as_str()
        .ok_or_else(|| CatalogProviderErrorKind::Internal {
            msg: "secret JSON missing 'client_id' field".to_owned(),
        })?
        .to_owned();

    let client_secret = parsed["client_secret"]
        .as_str()
        .ok_or_else(|| CatalogProviderErrorKind::Internal {
            msg: "secret JSON missing 'client_secret' field".to_owned(),
        })?
        .to_owned();

    Ok(PolarisSecret {
        client_id,
        client_secret,
    })
}

pub struct PolarisCatalogConfig<'a> {
    pub id: &'a str,
    pub catalog_url: &'a str,
    pub secret_arn: &'a str,
    pub aws_region: Option<&'a str>,
    pub warehouse: Option<&'a str>,
    pub realm: Option<&'a str>,
    pub scope: Option<&'a str>,
    pub aws_profile: Option<&'a str>,
    pub aws_access_key_id: Option<&'a str>,
    pub aws_secret_access_key: Option<&'a str>,
    pub aws_session_token: Option<&'a str>,
}

pub async fn build_polaris_catalog(
    config: &PolarisCatalogConfig<'_>,
) -> Result<crate::IcebergRestCatalogProvider, CatalogProviderErrorKind> {
    let secret = resolve_polaris_secret_with_credentials(
        config.secret_arn,
        config.aws_region,
        config.aws_profile,
        config.aws_access_key_id,
        config.aws_secret_access_key,
        config.aws_session_token,
    )
    .await?;

    let token_url = format!("{}/v1/oauth/tokens", config.catalog_url);

    let oauth_scope = config.scope.unwrap_or("PRINCIPAL_ROLE:ALL");
    let mut catalog = crate::IcebergRestCatalogProvider::new(config.id, config.catalog_url)
        .with_oauth2(
            token_url,
            secret.client_id,
            secret.client_secret,
            Some(oauth_scope.to_owned()),
        );

    if let Some(wh) = config.warehouse {
        catalog = catalog.with_prefix(wh);
    }

    if let Some(r) = config.realm {
        catalog = catalog.with_custom_header("Polaris-Realm", r);
    }

    Ok(catalog)
}

#[cfg(test)]
mod tests {
    use super::*;
    use semstrait_common::diagnostic::Diagnose;

    #[test]
    fn test_parse_valid_secret() {
        let json = r#"{"client_id": "my-id", "client_secret": "my-secret"}"#;
        let secret = parse_polaris_secret(json).unwrap();
        assert_eq!(secret.client_id, "my-id");
        assert_eq!(secret.client_secret, "my-secret");
    }

    #[test]
    fn test_parse_extra_fields_ignored() {
        let json = r#"{"client_id": "id", "client_secret": "secret", "extra": "field"}"#;
        let secret = parse_polaris_secret(json).unwrap();
        assert_eq!(secret.client_id, "id");
        assert_eq!(secret.client_secret, "secret");
    }

    #[test]
    fn test_parse_missing_client_id() {
        let json = r#"{"client_secret": "secret"}"#;
        let err = parse_polaris_secret(json).err().unwrap();
        assert!(
            err.message().contains("client_id"),
            "error should mention client_id: {err:?}"
        );
    }

    #[test]
    fn test_parse_missing_client_secret() {
        let json = r#"{"client_id": "id"}"#;
        let err = parse_polaris_secret(json).err().unwrap();
        assert!(
            err.message().contains("client_secret"),
            "error should mention client_secret: {err:?}"
        );
    }

    #[test]
    fn test_parse_invalid_json() {
        let err = parse_polaris_secret("not json").err().unwrap();
        assert!(
            err.message().contains("parse"),
            "error should mention parsing: {err:?}"
        );
    }

    #[test]
    fn test_parse_null_client_id() {
        let json = r#"{"client_id": null, "client_secret": "secret"}"#;
        let err = parse_polaris_secret(json).err().unwrap();
        assert!(err.message().contains("client_id"));
    }
}
