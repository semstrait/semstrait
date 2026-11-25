pub mod compile;
pub mod error;
pub mod identity;
pub mod manifest;
pub mod repository;
pub mod validate;

pub use error::{
    CompileError, ReferenceScope, ReferenceSite, ReferenceTargetKind, RelationshipSide, ShapeAxis,
};
pub use identity::{
    mint_expr_id, mint_keyset_id, mint_model_id, mint_path_source_id, mint_reduction_leaf_id,
    mint_semantic_id, mint_semantic_source_id, mint_signature_id, mint_table_source_id,
    EXPR_NAMESPACE,
};
pub use manifest::*;
pub use validate::FaultReporting;

pub use compile::{
    Provenance, RecursionState, ResolvedExprEntry, ResolvedExprKey, ResolvedExprTable, SchemaIndex,
    ScopeChain, ScopeFrame, ScopeKind, SemanticsIndex,
};

pub use compile::compile;
