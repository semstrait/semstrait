# Semstrait semantic model schema

This directory contains a [JSON Schema](https://json-schema.org/) that describes the structure of semstrait semantic model YAML files.

## Schema file

- **`semantic-model.json`** – Draft 2020-12 schema for the root document. The document must have a top-level `semantic_models` array; each item is a semantic model with `name`, `dataset_groups`, and optional `dimensions`, `metrics`, `dataFilter`.

## Using the schema

### Editor support (VS Code / Cursor)

Associate the schema with semantic model YAML files so the editor can validate and provide completion:

**Workspace** (`.vscode/settings.json` in the repo root):

```json
{
  "yaml.schemas": {
    "schema/semantic-model.json": ["test_data/*.yaml", "**/semantic*.yaml"]
  }
}
```

**User settings:** add the same `yaml.schemas` entry and use an absolute path to `semantic-model.json` if needed.

### Validating a file

1. **As JSON:** If your validator expects JSON, convert YAML to JSON first (e.g. `yq eval -o=j file.yaml` or load in a script and re-serialize).
2. **As YAML:** Many tools (e.g. [yamllint](https://yamllint.readthedocs.io/) with a schema plugin, or editor extensions) can validate YAML against a JSON Schema.

Example with [ajv](https://github.com/ajv-validator/ajv) (Node) after converting YAML to JSON:

```bash
npx ajv validate -s schema/semantic-model.json -d "<(yq eval -o=j test_data/steelwheels.yaml)"
```

### Keeping the schema in sync

The schema mirrors the Rust types in `src/semantic_model/`. When you change those types (e.g. new fields, new source types, or measure/metric expression forms), update `semantic-model.json` so validation and editor support stay accurate.
