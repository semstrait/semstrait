# `test_data/` — three corpus models, their declarations, five backups

This directory holds the three corpus models, each with its data declaration
beside it, and five backup examples. This file records which is which, how the
gates run and what the corpus migration dropped, so that nobody has to
rediscover it by grepping.

## Live: the corpus models

Each corpus model has a gate in
`crates/semstrait-adapter-datafusion/tests/oracle.rs`, which calls
`runner::corpus` (`tests/common/runner.rs`): zero errors and zero warnings from
parse, validate and compile against the declaration's catalog and store, and the
declaration's type check holds. A corpus gate has no golden.

| Model | Declaration | Gate |
| --- | --- | --- |
| `acme_eu_ad_platform_v3.yaml` | `acme_eu_ad_platform_v3.data.yaml` | `acme_v3_compiles_clean_against_its_declaration` |
| `comprehensive_ecommerce.yaml` | `comprehensive_ecommerce.data.yaml` | `corpus_c3_comprehensive_ecommerce_compiles_clean_against_its_declaration` |
| `e2e_full_coverage.yaml` | `e2e_full_coverage.data.yaml` | `corpus_c2_e2e_full_coverage_compiles_clean_against_its_declaration` |

`semstrait-model`'s parsing tests also read
`acme_eu_ad_platform_v3.yaml`, by `include_str!` in the const `FIXTURE`
of `crates/semstrait-model/tests/model_parsing.rs` and of
`crates/semstrait-model/tests/model_parsing_shapes.rs`.

Every path in a corpus model is a Parquet folder under `s3://semstrait-e2e/`.
A declaration (`crates/semstrait-adapter-datafusion/tests/common/declaration.rs`)
lists each catalog table under `tables:` and one object per folder under
`objects:`, `<base path>part-0.parquet`, inside its `key=value/` folder where
the folder is partitioned. Each lists its typed columns and no rows: the gates
read no row.

The other live models sit in
`crates/semstrait-adapter-datafusion/tests/fixtures/oracle/`, each in its own
folder: its `model.yaml` and, where it has them, its declaration `data.yaml`
and its cases `<case>.yaml`, each with its golden. A folder with no case is
read by a hand-written test.

- `harness/`: the harness model.
- `missing_column/` and `measure_without_agg/`: the failure models. The case of
  `missing_column/` fails at compile; the case of `measure_without_agg/` fails
  at parse, so that model has no declaration.
- Every other folder: a small model of the planned gates, built around the form
  that they test. No planned gate reads a corpus model.

The compile and planner refusals are tests of their own crates, in
`tests/<target>.rs`, whose models are built in the test.

## How the gates run

`semstrait-adapter-datafusion` runs every case; its test target `oracle` runs
the oracle's case files and the three corpus gates:

```sh
cargo test -p semstrait-adapter-datafusion --test oracle corpus_   # the three corpus gates
cargo test -p semstrait-adapter-datafusion --test oracle           # every case and gate of the oracle
```

`make test` (`cargo test --workspace`) and CI (`cargo test --all-features`) run
them with every other test.

## The golden generators

Each test target that holds goldens has one generator, its test
`generate_goldens`; a crate's run reaches all of them. It writes only where a
run asks for it by that name with the switch `SEMSTRAIT_GOLDENS` set; every
other run compares. `--test <target>` limits the run to one target:

```sh
SEMSTRAIT_GOLDENS=1 cargo test -p <crate> [--test <target>] generate_goldens
```

Each path below is under the crate's folder.

| Crate | Writes |
| --- | --- |
| `semstrait-model` | the parse and model-validate message goldens, `tests/fixtures/<target>/<case>.error.txt` |
| `semstrait-manifest` | the compile message goldens, `tests/fixtures/<target>/<case>.error.txt`, and the manifest snapshot, `tests/fixtures/happy_manifest.snapshot.json` (`--test manifest_snapshot`) |
| `semstrait-planner` | the plan-stage message goldens, `tests/fixtures/<target>/<case>.error.txt` |
| `semstrait-adapter-datafusion` | the case goldens, `tests/fixtures/<target>/<case>.explain.txt` (the plan's explain text) or `<case>.error.txt` (the refusal); the oracle's sit beside their cases in `tests/fixtures/oracle/<model>/` |

## Backups: read by nothing

Kept as authoring examples and as provenance for the live models. They are
**not** maintained against the current model surface, so treat each as a
snapshot of what was authorable when it was written, not as a statement of what
validates today.

| File | Shape | Why it is kept |
| --- | --- | --- |
| `acme_eu_ad_platform_v2.yaml` | `semantic_model:` | The predecessor of v3: `acme_eu_ad_platform_v3.yaml:3` says it is "Adapted from `acme_eu_ad_platform_v2.yaml` to the new model surface" |
| `acme_eu_ad_demo.yaml` | `semantic_model:` | A trimmed demo cut of the same platform model |
| `paid_media_kind.yaml` | `semantic_model:` | A paid-media data kind example |
| `crates/semstrait-api/tests/fixtures/catalogs.yaml` | `catalogs:` | A catalog document, not a semantic model: an example of the catalog side |
| `catalogs_demo.yaml` | `catalogs:` | A smaller catalog document |

## Citations of the corpus

A citation records what its author read, so an ADR can cite a corpus line
that has since moved. Each citation below corresponds to the current files as
follows:

- `docs/adr/0057-a-join-conditions-column-leaf-carries-its-constituents-alias.md:51`
  cites `comprehensive_ecommerce.yaml` by line: `:830-838` for the
  `orders_to_customers` relationship, now at `:829-835`, and `:709-724` for the
  `column_mapping` entries, now the `semantic_mapping:` entries of
  `orders_daily` at `:710-717`.
- `docs/adr/0087-the-function-catalog-declares-the-common-sql-functions-that-datafusion-can-run.md:61`
  cites `acme_eu_ad_platform_v3.yaml:261`, a `regexp_match`. In the current
  file the Boolean match is `regexp_like`, first at `:256`.
- `docs/adr/0082-a-physical-read-types-an-unobserved-column-by-the-semantics-mapped-to-it-and-compile-writes-that-type-into-the-schema.md:54`
  names `orders_daily`'s path in `comprehensive_ecommerce.yaml` as
  `s3://warehouse/orders/daily/`. The current path is
  `s3://semstrait-e2e/warehouse/orders/daily/`.

`docs/adr/0053-authoring-direction-is-exclusive-per-root-data-kind.md:57` cites
`acme_eu_ad_platform_v3.yaml`'s `market` by content, a `case` over
`dataset_name` at the root, which still holds.

## `comprehensive_ecommerce.yaml` carries a `domain:` field that no model type has

Eleven Datasets and Grainsets in this file — lines 95, 173, 194, 242, 267, 316,
357, 386, 406, 435 and 506 — carry a `domain:` field, with values `commerce`
(×3), `crm`, `catalog`, `operations` (×2), `traffic` (×3) and `marketing`.

`domain` is **not a field of any model type**: `rg 'domain' crates/semstrait-model/src/`
finds nothing.

**It is not rejected, though — it is silently dropped.** `domain:` sits at
DataKind level, a sibling of `name:` and `description:`, and the Public DataKind
wire mirrors flatten their body (`Dataset` flattens `DatasetBody`, which flattens
`DataKindBase`), so serde cannot carry `deny_unknown_fields` there. This is
stated in `crates/semstrait-model/src/yaml/retired.rs:4-6`: "an unknown
DataKind-level key is dropped." Nor does anything warn: `retired.rs` warns only
for the named retired spellings `keys`, `is_key`, `keyset` and `keysets`, and
`domain` is not one of them. So this document parses and compiles with no
warning, and all eleven values vanish without a diagnostic.

That is why it is recorded here: it may be a **lost requirement** rather than a
typo, and the failure mode is silent. A duplicate of this file in a
second fixture directory had no `domain:` fields at all; ignoring one added blank
line at `:20`, those eleven fields were the only difference between the two
copies before that duplicate was deleted (`diff -B` reduces to exactly the eleven
replacements).

## What the migration dropped

The migration moved each corpus model to today's surface. It dropped each
construct below because parse accepts no form of it today, and kept what the
construct meant where today's surface can say it. `acme_eu_ad_platform_v3.yaml`
dropped nothing.

| Construct | Dropped from | What the model keeps |
| --- | --- | --- |
| `ai_context.entity` | `e2e_full_coverage.yaml`: the model and `transactions` | Each `ai_context`'s `description`, and the model's synonyms |
| A categorical `enum:` | `e2e_full_coverage.yaml`: `country`, `txn_type`, `account_status`, `sensor_type`; `comprehensive_ecommerce.yaml`: `region`, `category`, `customer_tier`, `product_category`, `page_type`, `campaign_channel`, `campaign_status`, `platform` | Each a categorical Dimension; `platform`'s values `web`, `mobile` and `email` stay as the literals its branches map |
| `geo`'s `lat:` and `lon:` | `e2e_full_coverage.yaml`: `location` (`latitude`, `longitude`) | `location`, a geo Dimension |
| A bucket's `column:` | `e2e_full_coverage.yaml`: `amount_tier`; `comprehensive_ecommerce.yaml`: `price_bucket` (each `column: amount`) | The buckets and their bounds; the Dimension reads its own column |
| `associativity:` | `e2e_full_coverage.yaml`: `txn_details` (left), `product_inventory` (right), `account_sensor_full` (left); `comprehensive_ecommerce.yaml`: `order_details`, `traffic_campaigns` (left) | Each relationship's join, as `optional:` |
| A Measure's or a Metric's `constraints:` | `e2e_full_coverage.yaml`: the Measure `constrained_revenue`; `comprehensive_ecommerce.yaml`: the Measures `revenue` and `unique_customers`, the Metric `margin` | Each Measure and Metric, with no constraint: `constraints:` is ADR 0027's surface, whose planner side is not yet specified |
| The `{column, grain}` mapping | `e2e_full_coverage.yaml`: `txn_by_grain`'s `txn_monthly_agg`; `comprehensive_ecommerce.yaml`: `sales`' `orders_monthly` | The temporal Dimension maps `month_start`, and the child declares `granularity: month` |
