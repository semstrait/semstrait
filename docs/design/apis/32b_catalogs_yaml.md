---
id: 32b-catalogs-yaml
status: ratified
version: 2.2
impl: drift
---


# 32b. `catalogs.yaml` — Catalog Grammar

`32b` fixes the grammar of the sibling `catalogs.yaml` file. A model references a catalog by alias via `extras.catalog:` (authored per `32 §1.3`); the catalog entry itself lives here.

## 1. Root YAML Shape

```yaml
# catalogs.yaml
catalogs:
  polaris_prod:
    type: polaris
    name: fs1henp3k8p1hqo
    url: https://polaris.example.com/api/catalog
    realm: POLARIS
    default_namespace: finance
    auth:
      method: aws_secrets
      secret_arn: arn:aws:secretsmanager:eu-west-1:123456:secret:polaris
      region: eu-west-1
      scope: "PRINCIPAL_ROLE:ALL"

  polaris_dev:
    type: polaris
    name: dev_catalog
    url: https://polaris-dev.example.com/api/catalog
    auth:
      method: oauth2
      client_id: ${POLARIS_CLIENT_ID}
      client_secret: ${POLARIS_CLIENT_SECRET}
      scope: "PRINCIPAL_ROLE:ALL"

  iceberg_local:
    type: iceberg_rest
    name: local_catalog
    url: http://localhost:8181
    auth:
      method: bearer
      token: ${LOCAL_ICEBERG_TOKEN}
```

Root shape: exactly one top-level `catalogs:` key. No other top-level keys are recognized.

Rust root:

```rust
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogsConfig {
    pub catalogs: BTreeMap<String, CatalogEntry>,
}
```

`catalogs` is a `BTreeMap` for I4 (deterministic iteration). `deny_unknown_fields` at the root rejects typos in the top-level key.

---

## 2. `CatalogEntry` Fields

```rust
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogEntry {
    /// Provider type discriminator (e.g. "polaris", "iceberg_rest", "unity").
    /// Free-form string; provider implementations live under feature-gated
    /// modules in `semstrait-catalog` (per `37`).
    #[serde(rename = "type")]
    pub provider_type: String,

    /// Catalog / warehouse name as the provider expects it.
    pub name: String,

    /// Catalog base URL.
    pub url: String,

    /// Provider-specific realm (Polaris uses this; most others ignore).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub realm: Option<String>,

    /// Default namespace used when the model's `CatalogRef` does not
    /// override it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_namespace: Option<String>,

    /// Authentication configuration — see §3.
    pub auth: CatalogAuthMethod,
}
```

### 2.1 Optional vs required

- Required: `type`, `name`, `url`, `auth`.
- Optional: `realm`, `default_namespace`.

Missing required fields at parse → `CatalogsParseErrorKind::CatalogMissingField { alias, field }` (§5.2).

### 2.2 `type:` discriminator

A free-form string rather than a closed enum. Adding a provider is a new feature-gated module in `semstrait-catalog` (per `37`); the model-layer grammar does not need to change when a new provider lands.

Unknown `type:` values pass the model-layer parser; the error surfaces at catalog-resolution time (`33` / `37`) as `CatalogProviderErrorKind::UnknownProviderType` (`37`) when the caller tries to open a provider the runtime hasn't linked in.

---

## 3. Authentication Methods

```rust
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(tag = "method", rename_all = "snake_case", deny_unknown_fields)]
pub enum CatalogAuthMethod {
    Oauth2 {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        token_url: Option<String>,
        client_id: String,
        client_secret: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        scope: Option<String>,
    },
    Bearer {
        token: String,
    },
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
```

### 3.1 `oauth2`

Direct OAuth2 with explicit credentials (or via `${VAR}`).

```yaml
auth:
  method: oauth2
  token_url: https://auth.example.com/oauth/token   # optional; derived from url if absent
  client_id: my_client
  client_secret: ${MY_SECRET}
  scope: "PRINCIPAL_ROLE:ALL"
```

Required: `client_id`, `client_secret`. Optional: `token_url`, `scope`.

### 3.2 `bearer`

Static token.

```yaml
auth:
  method: bearer
  token: ${ACCESS_TOKEN}
```

Required: `token`.

### 3.3 `aws_secrets`

AWS Secrets Manager retrieves `client_id` / `client_secret`, then falls back to the OAuth2 flow.

```yaml
auth:
  method: aws_secrets
  secret_arn: arn:aws:secretsmanager:eu-west-1:123456:secret:polaris
  region: eu-west-1
  scope: "PRINCIPAL_ROLE:ALL"
  secret_keys:
    client_id_key: polaris_client_id
    client_secret_key: polaris_client_secret
```

Required: `secret_arn`. Everything else optional.

`secret_keys` maps custom key names inside a Secrets Manager JSON payload:

```rust
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SecretKeyMapping {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_id_key: Option<String>,      // default: "client_id"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_secret_key: Option<String>,  // default: "client_secret"
}
```

---

## 4. `CatalogRef` — Reference Site Inside a Model

Inside any `extras.catalog:` block in a model (authored per `32 §1.3`), the value is a `CatalogRef` — a **bare alias string** wrapped in a transparent newtype. There is no namespace override at the reference site, no map form, no additional fields:

```rust
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(transparent)]
pub struct CatalogRef {
    /// Alias — keys a `CatalogEntry` in `catalogs.yaml`.
    pub alias: String,
}
```

Only one authoring form exists:

```yaml
extras:
  catalog: polaris_prod
# → CatalogRef { alias: "polaris_prod" }
```

Map-form authoring (`{ alias: ..., namespace: ... }`) is **not recognized** — models author at most one catalog per data kind, and each catalog already declares its effective `default_namespace` at its own entry in `catalogs.yaml`. A model that needs a different namespace authors a separate catalog entry (typically a lightweight dev / prod variant pair).

### 4.1 Resolution precedence

At `compile` time, every `CatalogRef` is matched against `CatalogsConfig.catalogs[alias]`. Unmatched aliases raise a compile `COMP_E_CATALOG_ALIAS_NOT_FOUND` diagnostic (placeholder code, finalised in spec 30 / spec 34) at catalog resolution (per `33`). Namespace precedence collapses to two tiers:

1. `CatalogEntry.default_namespace` — authored at the catalog entry in `catalogs.yaml` (§2).
2. Provider default — whatever the provider implementation returns when both of the above are absent.

## 5. Loader Contract

### 5.1 Signature

```rust
use semstrait_common::diagnostic::Diagnostics;

/// Parse a `catalogs.yaml` document. Pure and synchronous. Accumulating
/// stage per `30 §7.1` (sibling to `32 §9.1`'s `parse`). `${VAR}`
/// substitutions expanded before YAML decoding (§6).
pub fn parse_catalogs(
    input: &str,
) -> Result<
    (CatalogsConfig, Diagnostics<CatalogsParseErrorKind>),
    Diagnostics<CatalogsParseErrorKind>,
>;
```

Pure: same input + same environment → byte-identical output under a canonical serializer.

### 5.2 `CatalogsParseErrorKind` roster

```rust
use semstrait_common::diagnostic::{Diagnose, Severity};
use semstrait_common::Location;

#[non_exhaustive]
pub enum CatalogsParseErrorKind {
    YamlSyntax             { message: String },
    UnsetEnvVar            { var: String },
    UnknownTopLevelKey     { key: String },
    CatalogMissingField    { alias: String, field: String },
    MalformedAuthMethod    { alias: String, reason: String },
    UnknownField           { field: String, parent: String },
}

impl Diagnose for CatalogsParseErrorKind {
    fn message(&self) -> String { /* per-variant human text */ }
    fn severity_default(&self) -> Severity { Severity::Error }
    fn cause(&self) -> Option<&(dyn std::error::Error + 'static)> { None }
}
```

Per `30 §5.1`, the primary source span lives on the wrapping `Diagnostic<CatalogsParseErrorKind>` (the `location: Option<Location>` field on the envelope), not on the variant. Identification is by variant identity per `30 §5`; there is no string-code accessor. Naming and stability rules mirror `32 §9.2`'s `ParseErrorKind`.

### 5.3 Composition with `parse`

`parse_catalogs` is distinct from `parse` (the `semantic_model:` loader, `32 §9`). A caller typically invokes both, then hands the two products to `compile`:

```rust
let (model, parse_warnings)        = semstrait_model::parse(&model_yaml)?;
let (catalogs, catalog_warnings)   = semstrait_model::parse_catalogs(&catalogs_yaml)?;
let (manifest, compile_warnings)   = semstrait_manifest::compile(&model, &catalogs)?;
```

The exact shape of `compile` is `33`'s to pin down.

### 5.4 Reading catalog bytes

`parse_catalogs` takes text; fetching it is the caller's, on the same seam the model uses. `catalogs.yaml` rides the loader's own catalog setters (`32-model-loader §1`) — `from_catalogs_yaml_file` for a local read, `with_catalogs` for a config the caller already parsed:

```rust
let bytes = fs.read(&Path::new("s3://bucket/catalogs.yaml")).await?;   // caller's transport (37)
let (catalogs, warnings) = parse_catalogs(std::str::from_utf8(&bytes)?, source)?;

let (model, diags) = SemanticModel::loader()
    .from_yaml_file("model.yaml")
    .with_catalogs(catalogs)
    .build()?;
```

There is no `load_catalogs` / `dump_catalogs` async wrapper and no `DumpMode`: a catalogs file is authored input, not a persisted artifact, so nothing writes it back. Read failures surface as `ModelBuildErrorKind::SourceIo` and parse failures as `CatalogsParse(_)` (`32-model-loader §1`).

---

## 6. Environment-Variable Substitution

`${IDENT}` tokens in any string field are rewritten to the value of `std::env::var("IDENT")` before YAML decoding. Identical mechanism to `32 §8`; unset variables raise `CatalogsParseErrorKind::UnsetEnvVar`.

```yaml
catalogs:
  polaris_prod:
    type: polaris
    name: ${POLARIS_CATALOG_NAME}
    url: ${POLARIS_URL}
    auth:
      method: bearer
      token: ${POLARIS_TOKEN}
```

Bare `$VAR` is treated as literal text.

---

## 7. Structural Rules

Each rule maps to a typed-kind variant in `CatalogsParseErrorKind` (§5.2) per `30 §5`.


| ID        | Rule                                                                                                                                                    | Kind                                          |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| **SR-C1** | Exactly one `catalogs:` root key; no other top-level keys.                                                                                              | `CatalogsParseErrorKind::UnknownTopLevelKey`  |
| **SR-C2** | Every `CatalogEntry` carries required fields: `type`, `name`, `url`, `auth`.                                                                            | `CatalogsParseErrorKind::CatalogMissingField` |
| **SR-C3** | Every `CatalogAuthMethod` variant carries its variant-specific required fields (e.g. `bearer` must have `token`; `aws_secrets` must have `secret_arn`). | `CatalogsParseErrorKind::MalformedAuthMethod` |
| **SR-C4** | Aliases (keys in `catalogs:`) are unique — `BTreeMap` enforces.                                                                                         | (duplicate-key raised by YAML decoder)        |
| **SR-C5** | `deny_unknown_fields` applied to `CatalogsConfig`, `CatalogEntry`, every `CatalogAuthMethod` variant, and `SecretKeyMapping`.                           | `CatalogsParseErrorKind::UnknownField`        |
| **SR-C6** | `${VAR}` substitution applied before decoding; unset vars fatal.                                                                                        | `CatalogsParseErrorKind::UnsetEnvVar`         |


---

## Testing Strategy

- Root YAML shape correctness (§1 — exactly one `catalogs:` top-level key, BTreeMap deterministic ordering)
- CatalogEntry required fields enforcement (§2.1 — `type`, `name`, `url`, `auth` presence)
- CatalogAuthMethod per-variant required fields (§3 — `oauth2` needs `client_id`/`client_secret`, `bearer` needs `token`, `aws_secrets` needs `secret_arn`)
- CatalogRef transparent newtype shape (§4 — bare alias string, no map-form authoring)
- parse_catalogs loader contract (§5.1 — pure, synchronous, accumulating diagnostics)
- Environment-variable substitution (§6 — `${VAR}` expansion via `std::env::var`, unset var error)
- Structural rules SR-C1 through SR-C6 (§7 — all error kinds covered)

See the shared behavioral harness: [`../testing/50_behavioral_contract_gates.md`](../testing/50_behavioral_contract_gates.md)

*Cross-references use `NN §M.K` for internal sections and full relative paths for other docs.*
