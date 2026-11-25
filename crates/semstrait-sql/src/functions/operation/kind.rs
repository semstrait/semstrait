use serde::{Deserialize, Serialize};

macro_rules! kinds {
    ($($(#[$doc:meta])* $kind:ident { $($(#[$variant_doc:meta])* $variant:ident,)* })*) => {$(
        $(#[$doc])*
        #[non_exhaustive]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(rename_all = "snake_case")]
        pub enum $kind {
            $($(#[$variant_doc])* $variant,)*
        }

        impl $kind {
            pub(crate) const ALL: [Self; [$(stringify!($variant)),*].len()] =
                [$(Self::$variant),*];
        }
    )*};
}

kinds! {
    BinaryOpKind {
        Add,
        Subtract,
        Multiply,
        Divide,
        SafeDivide,
        Mod,
        Eq,
        NotEq,
        Lt,
        LtEq,
        Gt,
        GtEq,
        IsNotDistinctFrom,
        And,
        Or,
    }

    UnaryOpKind {
        Negate,
        Not,
    }

    LikeKind {
        Like,
        NotLike,
        ILike,
        NotILike,
    }
}
