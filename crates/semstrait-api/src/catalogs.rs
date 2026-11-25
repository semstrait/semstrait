#[cfg(feature = "iceberg")]
mod iceberg;

use std::collections::BTreeMap;
use std::sync::Arc;

use semstrait_catalog::CatalogProvider;
use semstrait_common::{Diagnose, Severity};
#[cfg(not(feature = "aws"))]
use semstrait_model::CatalogAuthMethod;
use semstrait_model::{CatalogEntry, CatalogsConfig};
use thiserror::Error;

#[non_exhaustive]
#[derive(Debug, PartialEq, Error)]
pub enum CatalogConnectionError {
    #[error("with no catalog alias, catalogs.yaml must have exactly one entry; its entries are {aliases:?}")]
    AliasNotGiven { aliases: Vec<String> },
    #[error(
        "no entry of catalogs.yaml has the catalog alias `{alias}`; its entries are {aliases:?}"
    )]
    AliasNotFound { alias: String, aliases: Vec<String> },
    #[error("catalog `{alias}`: no provider has the type `{provider_type}`; the types are polaris and iceberg_rest")]
    UnknownType {
        alias: String,
        provider_type: String,
    },
    #[error("catalog `{alias}`: no provider connects with its auth method")]
    UnknownAuth { alias: String },
    #[error("catalog `{alias}`: an Iceberg REST catalog needs the `iceberg` feature")]
    IcebergFeatureOff { alias: String },
    #[error("catalog `{alias}`: `aws_secrets` auth needs the `aws` feature")]
    AwsFeatureOff { alias: String },
    #[error("catalog `{alias}`: Secrets Manager gave no Polaris credentials: {reason}")]
    SecretFetch { alias: String, reason: String },
}

impl Diagnose for CatalogConnectionError {
    fn message(&self) -> String {
        self.to_string()
    }

    fn severity_default(&self) -> Severity {
        Severity::Error
    }
}

pub(crate) async fn connected(
    catalogs: &CatalogsConfig,
    alias: Option<&str>,
) -> Result<Arc<dyn CatalogProvider>, CatalogConnectionError> {
    let (alias, entry) = picked(&catalogs.catalogs, alias)?;
    match entry.provider_type.as_str() {
        #[cfg(not(feature = "aws"))]
        "polaris" | "iceberg_rest"
            if matches!(entry.auth, CatalogAuthMethod::AwsSecrets { .. }) =>
        {
            Err(CatalogConnectionError::AwsFeatureOff {
                alias: alias.clone(),
            })
        }
        #[cfg(feature = "iceberg")]
        "polaris" | "iceberg_rest" => Ok(Arc::new(iceberg::provider(alias, entry).await?)),
        #[cfg(not(feature = "iceberg"))]
        "polaris" | "iceberg_rest" => Err(CatalogConnectionError::IcebergFeatureOff {
            alias: alias.clone(),
        }),
        other => Err(CatalogConnectionError::UnknownType {
            alias: alias.clone(),
            provider_type: other.to_owned(),
        }),
    }
}

fn picked<'a>(
    entries: &'a BTreeMap<String, CatalogEntry>,
    alias: Option<&str>,
) -> Result<(&'a String, &'a CatalogEntry), CatalogConnectionError> {
    let aliases = || entries.keys().cloned().collect();
    match alias {
        Some(alias) => {
            entries
                .get_key_value(alias)
                .ok_or_else(|| CatalogConnectionError::AliasNotFound {
                    alias: alias.to_owned(),
                    aliases: aliases(),
                })
        }
        None => entries
            .first_key_value()
            .filter(|_| entries.len() == 1)
            .ok_or_else(|| CatalogConnectionError::AliasNotGiven { aliases: aliases() }),
    }
}
