use std::{fmt, iter};

use super::name::{Spelling, MAX_ARRAY_LAYERS};
use super::DataType;
use crate::DocumentDialect;

pub(super) const ELEMENT: &str = "element";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WrittenDataType {
    Spelling(String),
    Number(u64),
    Object(Vec<(String, WrittenDataType)>),
}

impl DataType {
    pub fn resolve_written(written: &WrittenDataType, dialect: DocumentDialect) -> Option<Self> {
        match written {
            WrittenDataType::Spelling(spelling) => Self::resolve(spelling, dialect),
            WrittenDataType::Object(entries) => match &entries[..] {
                [(type_key, WrittenDataType::Object(values))] => {
                    Self::keyed(&type_key.to_ascii_lowercase(), values, dialect).filter(
                        |resolved| {
                            resolved.depth() <= MAX_ARRAY_LAYERS && resolved.is_valid_in(dialect)
                        },
                    )
                }
                _ => None,
            },
            WrittenDataType::Number(_) => None,
        }
    }

    fn depth(&self) -> usize {
        iter::successors(Some(self), |data_type| match data_type {
            Self::Array(element) => Some(&**element),
            _ => None,
        })
        .skip(1)
        .count()
    }

    fn keyed(
        type_key: &str,
        values: &[(String, WrittenDataType)],
        dialect: DocumentDialect,
    ) -> Option<Self> {
        if Self::array_declaration(dialect).is_some_and(|array| array.name == type_key) {
            return match values {
                [(value_key, element)] if value_key == ELEMENT => Some(Self::Array(Box::new(
                    Self::resolve_written(element, dialect)?,
                ))),
                _ => None,
            };
        }
        Self::with_name(type_key, dialect, Spelling::NameOrAlias, |declaration| {
            if declaration.limits().count() != values.len() || values.is_empty() {
                return None;
            }
            declaration
                .limits()
                .map(
                    |(_, key, _)| match values.iter().find(|(value_key, _)| value_key == key) {
                        Some((_, WrittenDataType::Number(number))) => u32::try_from(*number).ok(),
                        _ => None,
                    },
                )
                .collect()
        })
    }
}

impl fmt::Display for WrittenDataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spelling(spelling) => f.write_str(spelling),
            Self::Number(number) => write!(f, "{number}"),
            Self::Object(entries) => {
                let entries: Vec<_> = entries
                    .iter()
                    .map(|(key, value)| format!("{key}: {value}"))
                    .collect();
                write!(f, "{{{}}}", entries.join(", "))
            }
        }
    }
}
