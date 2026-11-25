use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct CatalogsConfig {
    #[cfg_attr(
        feature = "schemars",
        schemars(
            description = "The catalog entries, keyed by an alias the author chooses, e.g. `polaris_prod`."
        )
    )]
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub catalogs: BTreeMap<String, CatalogEntry>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct CatalogEntry {
    #[cfg_attr(
        feature = "schemars",
        schemars(description = "The catalog provider, e.g. `polaris`, `iceberg_rest` or `unity`.")
    )]
    #[serde(rename = "type")]
    pub provider_type: String,

    #[cfg_attr(
        feature = "schemars",
        schemars(
            description = "The catalog or warehouse name, spelled as the provider expects it."
        )
    )]
    pub name: String,

    #[cfg_attr(
        feature = "schemars",
        schemars(description = "The catalog's base URL.")
    )]
    pub url: String,

    #[cfg_attr(
        feature = "schemars",
        schemars(
            description = "A provider-specific realm; Polaris uses it and most providers ignore it."
        )
    )]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub realm: Option<String>,

    #[cfg_attr(
        feature = "schemars",
        schemars(
            description = "The default namespace, used unless a model's catalog reference overrides it."
        )
    )]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_namespace: Option<String>,

    #[cfg_attr(
        feature = "schemars",
        schemars(description = "How to authenticate against the catalog.")
    )]
    pub auth: CatalogAuthMethod,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(tag = "method", rename_all = "snake_case", deny_unknown_fields)]
#[non_exhaustive]
pub enum CatalogAuthMethod {
    #[cfg_attr(
        feature = "schemars",
        schemars(description = "OAuth2 with an explicit client id and secret.")
    )]
    Oauth2 {
        #[cfg_attr(
            feature = "schemars",
            schemars(
                description = "The token endpoint URL; derived from the catalog URL when omitted."
            )
        )]
        #[serde(default, skip_serializing_if = "Option::is_none")]
        token_url: Option<String>,
        client_id: String,
        client_secret: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        scope: Option<String>,
    },
    #[cfg_attr(feature = "schemars", schemars(description = "A fixed bearer token."))]
    Bearer { token: String },
    #[cfg_attr(
        feature = "schemars",
        schemars(
            description = "Fetches the client id and secret from AWS Secrets Manager, then authenticates with OAuth2."
        )
    )]
    AwsSecrets {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        token_url: Option<String>,
        secret_arn: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        region: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        scope: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        aws_access_key_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        aws_secret_access_key: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        aws_session_token: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        aws_profile: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        secret_keys: Option<SecretKeyMapping>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "schemars",
    schemars(
        description = "Custom key names for the Secrets Manager payload; defaults to `client_id`/`client_secret`."
    )
)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct SecretKeyMapping {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_id_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_secret_key: Option<String>,
}
