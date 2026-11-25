use semstrait_common::SemanticsName;
use semstrait_sql::datatype::DataType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MappedSemantics<'a> {
    pub name: &'a SemanticsName,
    pub data_type: Option<&'a DataType>,
    pub is_count: bool,
}

pub fn read_type<'a>(
    mapped: impl IntoIterator<Item = MappedSemantics<'a>>,
) -> Option<&'a DataType> {
    let mut typing: Vec<_> = mapped
        .into_iter()
        .filter_map(|semantics| Some((semantics.name, semantics.data_type?, semantics.is_count)))
        .collect();
    let only_counts = typing.iter().all(|&(.., is_count)| is_count);
    typing.retain(|&(.., is_count)| is_count == only_counts);
    typing
        .iter()
        .find(|(_, widest, _)| {
            typing
                .iter()
                .all(|(_, other, _)| other == widest || widens(other, widest))
        })
        .or_else(|| typing.iter().min_by_key(|&&(name, ..)| name))
        .map(|&(_, data_type, _)| data_type)
}

fn widens(from: &DataType, to: &DataType) -> bool {
    use DataType::{BigInt, Date, Decimal, Double, Integer, Real, SmallInt, Time, Timestamp};
    match (from, to) {
        (SmallInt, Integer)
        | (SmallInt | Integer, BigInt)
        | (SmallInt | Integer | BigInt | Real, Double)
        | (SmallInt | Integer | BigInt, Decimal { .. })
        | (Date, Timestamp { .. }) => true,
        (
            Decimal { precision, scale },
            Decimal {
                precision: to_precision,
                scale: to_scale,
            },
        ) => to_precision > precision && to_scale == scale,
        (Time { precision }, Time { precision: higher })
        | (Timestamp { precision }, Timestamp { precision: higher }) => higher > precision,
        _ => false,
    }
}
