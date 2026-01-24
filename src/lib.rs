//! # semstrait
//!
//! Compile semantic models to Substrait compute plans.
//!
//! semstrait transforms YAML-based semantic model definitions into
//! [Substrait](https://substrait.io/) compute plans, enabling engine-agnostic analytics.
//!
//! ## Status
//!
//! **⚠️ This crate is under active development and not yet ready for use.**
//!
//! ## Planned Usage
//!
//! ```ignore
//! use semstrait::{Schema, Query, emit_plan};
//!
//! let schema = Schema::from_file("model.yaml")?;
//! let query = Query::new("sales")
//!     .rows(["dates.year", "markets.country"])
//!     .metrics(["revenue", "quantity"]);
//!
//! let plan = emit_plan(&schema, &query)?;
//! // Execute on DataFusion, DuckDB, or any Substrait consumer
//! ```

// Placeholder - real implementation coming soon
