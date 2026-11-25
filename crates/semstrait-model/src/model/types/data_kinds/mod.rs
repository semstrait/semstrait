pub mod base;
pub mod dataset;
pub mod grainset;
pub mod joinset;
pub mod storage;
pub mod unionset;

mod impls;
mod impls_complex;
mod refs;
mod refs_impls;
mod traits;
mod variants;

pub(crate) mod sealed {
    pub trait Sealed {}
}

pub use base::{ComplexExtras, DataKindBase, ExtrasFlavor, LeafExtras};
pub use dataset::{Dataset, DatasetBody, NestedDataset};
pub use grainset::{Grainset, GrainsetBody, NestedGrainset};
pub use joinset::{Joinset, JoinsetBody, NestedJoinset};
pub use refs::{
    AnyDataKindRef, ComplexDataKindRef, NestedDataKindRef, PublicDataKindRef, SimpleDataKindRef,
};
pub use storage::{CatalogRef, PartitionDef, StorageConfig, StorageFormat};
pub use traits::{ComplexDataKind, DataKind, NestedDataKind, PublicDataKind, SimpleDataKind};
pub use unionset::{NestedUnionset, UnionMode, Unionset, UnionsetBody};
pub use variants::{DataKindForm, DataKindVariant};
