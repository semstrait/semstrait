use super::{Literal, TypeError, Typed};
use crate::datatype::DataType;

pub(crate) fn literal_type(literal: &Literal) -> Typed {
    Typed::new(literal.data_type(), matches!(literal, Literal::Null(_)))
}

pub(crate) fn read_untyped(
    literal: &Literal,
    context: Option<&DataType>,
) -> Result<Literal, TypeError> {
    match context.filter(|context| **context != DataType::Unknown) {
        None => Ok(literal.clone()),
        Some(context) => literal
            .read_as(context)
            .map_err(|error| TypeError::UntypedLiteral {
                literal: literal.clone(),
                context: context.clone(),
                error,
            }),
    }
}
