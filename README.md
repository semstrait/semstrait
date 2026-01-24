# semstrait

> Compile semantic models to Substrait compute plans

**⚠️ This project is under active development and not yet ready for use.**

## What is semstrait?

semstrait is a Rust library that transforms YAML-based semantic model definitions into [Substrait](https://substrait.io/) compute plans, enabling engine-agnostic analytics.

```
YAML Schema → semstrait → Substrait Plan → Any Engine
                                              ├── DataFusion
                                              ├── DuckDB
                                              ├── Velox
                                              └── ...
```

## Planned Features

- **Semantic Modeling** — Define dimensions, measures, metrics, and joins in YAML
- **Query Resolution** — Resolve business queries against the semantic model
- **Substrait Output** — Generate portable compute plans for any Substrait-compatible engine
- **Lightweight** — Pure Rust library, no runtime server required

## Example (Planned API)

```rust
use semstrait::{Schema, Query, emit_plan};

let schema = Schema::from_file("model.yaml")?;
let query = Query::new("sales")
    .rows(["dates.year", "markets.country"])
    .metrics(["revenue", "quantity"]);

let plan = emit_plan(&schema, &query)?;
// Execute on DataFusion, DuckDB, or any Substrait consumer
```

## Status

🚧 **Pre-release** — API is unstable and documentation is incomplete.

Follow the repo for updates or star it to show interest.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
