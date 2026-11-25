use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum DataKindVariant {
    Dataset,
    Grainset,
    Unionset,
    Joinset,
}

impl DataKindVariant {
    pub fn ordering_key(self) -> u8 {
        match self {
            Self::Dataset => 0,
            Self::Grainset => 1,
            Self::Unionset => 2,
            Self::Joinset => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum DataKindForm {
    Public,
    Nested,
}
