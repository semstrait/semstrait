//! Table selector module
//!
//! Selects optimal table(s) from a model to serve a query.

mod error;
mod select;

pub use error::SelectError;
pub use select::{select_tables, SelectedTable};
