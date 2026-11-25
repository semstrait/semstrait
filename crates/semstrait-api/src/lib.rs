#![allow(
    clippy::result_large_err,
    reason = "every verb returns ADR 0005's fail-fast form unboxed, as the stages it calls do"
)]

mod catalogs;
#[cfg(feature = "cli")]
pub mod cli;
mod dialect;
mod error;
mod input;
mod load;
mod orchestrator;
mod reader;
#[cfg(feature = "rest")]
mod rest;
mod wire;

pub use catalogs::CatalogConnectionError;
pub use dialect::NoEngineAdapter;
pub use error::{InputErrorKind, SemstraitError};
pub use input::Input;
pub use load::load;
pub use orchestrator::{Semstrait, SemstraitBuilder};
#[cfg(feature = "rest")]
pub use rest::router;
pub use wire::{ContextPart, Key, Order, RequestPart, Selected, WeekStart, WireError};
