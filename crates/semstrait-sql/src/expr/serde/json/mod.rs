mod buffer;
mod de;
mod ser;

pub use de::JsonReader;
pub use ser::JsonForm;

macro_rules! keys {
    ($($keyword:ident = $spelling:literal),+; $($key:ident),+) => {
        #[allow(non_upper_case_globals)]
        mod keys {
            $(pub(super) const $keyword: &str = $spelling;)+
            $(pub(super) const $key: &str = stringify!($key);)+
        }
    };
}

keys!(
    r#type = "type", r#else = "else";
    content, op, left, right, operand, input, target, on_failure, whens, value, list, negated,
    low, high, pattern, kind, name, args, options, distinct, filter, order_by, nulls, over, when,
    then, expr, dir, partition_by, window_frame
);

macro_rules! tags {
    ($($tag:ident: $name:literal),*; $($fields:ident: $fields_name:literal),*) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum Tag {
            $($tag,)*
            Fields(Fields),
        }

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum Fields {
            $($fields),*
        }

        impl Tag {
            fn name(self) -> &'static str {
                match self {
                    $(Self::$tag => $name,)*
                    $(Self::Fields(Fields::$fields) => $fields_name,)*
                }
            }

            fn from_name(name: &str) -> Option<Self> {
                match name {
                    $($name => Some(Self::$tag),)*
                    $($fields_name => Some(Self::Fields(Fields::$fields)),)*
                    _ => None,
                }
            }
        }
    };
}

tags!(
    Literal: "literal", Column: "column", IsNull: "is_null";
    BinaryOp: "binary_op", UnaryOp: "unary_op", Cast: "cast", Case: "case", InList: "in_list",
    Between: "between", Like: "like", Scalar: "scalar", Aggregate: "aggregate", Window: "window"
);
