mod declaration;
mod kind;

use semstrait_common::Dialect;

use super::{options, Nullability, OptionDecl, ReturnRule, Signature};

pub use kind::{BinaryOpKind, LikeKind, UnaryOpKind};

#[derive(Debug)]
pub struct Operation {
    pub canonical: OperationDeclaration,
    pub dialects: &'static [(Dialect, Option<OperationDeclaration>)],
}

impl Operation {
    pub fn on(&'static self, dialect: Dialect) -> Option<&'static OperationDeclaration> {
        crate::on_dialect(&self.canonical, self.dialects, dialect)
    }
}

#[non_exhaustive]
#[derive(Debug, PartialEq)]
pub struct OperationDeclaration {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub signature: Signature,
    pub returns: ReturnRule,
    pub nullability: Nullability,
    pub options: &'static [OptionDecl<OperationOption>],
}

options!(
    OperationOption {
        Overflow: "overflow" ["SILENT", "SATURATE", "ERROR"],
        OnDivisionByZero: "on_division_by_zero" ["NULL", "ERROR"],
    }
);

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationKind {
    Binary(BinaryOpKind),
    Unary(UnaryOpKind),
    Like(LikeKind),
    Between,
    InList,
    IsNull,
}

impl OperationKind {
    pub fn all() -> impl Iterator<Item = Self> {
        let binary = BinaryOpKind::ALL.into_iter().map(Self::Binary);
        let unary = UnaryOpKind::ALL.into_iter().map(Self::Unary);
        let like = LikeKind::ALL.into_iter().map(Self::Like);
        binary
            .chain(unary)
            .chain(like)
            .chain([Self::Between, Self::InList, Self::IsNull])
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::all().find(|kind| {
            let canonical = &kind.operation().canonical;
            canonical.name == name || canonical.aliases.contains(&name)
        })
    }
}
