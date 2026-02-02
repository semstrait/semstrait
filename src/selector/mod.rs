//! Table selector module
//!
//! Selects optimal table(s) from a model to serve a query.
//!
//! Supports two selection modes:
//! - Single table: When one table can satisfy all query requirements
//! - Multi-table JOIN: When measures span multiple tables in the same tableGroup

mod error;
mod select;

pub use error::SelectError;
pub use select::{
    select_tables, 
    select_tables_for_join, 
    SelectedTable, 
    MultiTableSelection,
    TableWithMeasures,
};
