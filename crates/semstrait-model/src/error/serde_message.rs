use crate::error::parse::ParseErrorKind;

const UNKNOWN_FIELD: &str = "unknown field `";
const IN_CARRIER: &str = " in `";
const UNKNOWN_CARRIER: &str = "<unknown>";
const NAME_WITH_REF: &str = "both `name:` and `ref:` in `";
const ON_EMPTY: &str = "an empty join basis";
const ON_SPELLING: &str = "unknown relationship `on:` spelling `";
const ON_KEY_DUPLICATED: &str = "relationship `on:` repeats the name `";
const KEYSET_BOTH: &str = "both `keyset:` and `keysets:` in `";
const KEYSET_UNACTIVATED: &str = "names a keyset with `";
const KEYSET_LIST_EMPTY: &str = "declares an empty `keysets:` list in `";
const KEYSET_NAME_BLANK: &str = "names a blank keyset under `";
const DATA_TYPE_REFUSED: &str = "no valid data type is named `";
const LITERAL_REFUSED: &str = "a literal of type `";
const LITERAL_REASON: &str = " is refused: `";
const UNKNOWN_TAG: &str = "unknown reserved tag: `";
const CALL_REFUSED: &str = "the call of `";
const CALL_REASON: &str = " is refused: ";
const AT_LINE: &str = " at line ";
const AGGREGATE_IN_ROW_EXPR: &str = "aggregate tag `";
const UNKNOWN_AGGREGATE: &str = "no catalog aggregate is named `";

pub(crate) fn unknown_field(field: &str, carrier: &str, expected: &[&str]) -> String {
    let expected = expected
        .iter()
        .map(|f| format!("`{f}`"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{UNKNOWN_FIELD}{field}`{IN_CARRIER}{carrier}`, expected one of {expected}")
}

pub(crate) fn site_name_with_ref(carrier: &str) -> String {
    format!(
        "a semantics site declares {NAME_WITH_REF}{carrier}`, but it is either an \
         inline declaration (`name:`) or a reference (`ref:`), never both"
    )
}

pub(crate) fn join_basis_empty() -> String {
    format!(
        "relationship `on: []` declares {ON_EMPTY}; write `on: cross` for no \
         dimension-name basis"
    )
}

pub(crate) fn join_basis_unknown_spelling(spelling: &str) -> String {
    format!(
        "{ON_SPELLING}{spelling}`; expected a list of dimension names, `auto`, or \
         `cross`"
    )
}

pub(crate) fn join_key_duplicated(key: &str) -> String {
    format!("{ON_KEY_DUPLICATED}{key}`")
}

pub(crate) fn keyset_both_spellings(carrier: &str) -> String {
    format!(
        "a dimension site declares {KEYSET_BOTH}{carrier}`, but two spellings of one \
         membership cannot disagree; keep either the singular or the plural"
    )
}

pub(crate) fn keyset_without_is_key(spelling: &str, carrier: &str) -> String {
    format!(
        "a dimension site {KEYSET_UNACTIVATED}{spelling}:`{IN_CARRIER}{carrier}` but no \
         `is_key: true`, so the membership would silently do nothing"
    )
}

pub(crate) fn keyset_list_empty(carrier: &str) -> String {
    format!(
        "a dimension site {KEYSET_LIST_EMPTY}{carrier}`, naming no keyset; write bare \
         `is_key: true` for `default` membership"
    )
}

pub(crate) fn keyset_name_blank(spelling: &str, carrier: &str) -> String {
    format!(
        "a dimension site {KEYSET_NAME_BLANK}{spelling}:`{IN_CARRIER}{carrier}`; a keyset \
         name cannot be empty"
    )
}

pub(crate) fn data_type_refused(spelling: &str) -> String {
    format!(
        "{DATA_TYPE_REFUSED}{spelling}`; write a canonical data type's name or an alias, \
         with the values its type takes"
    )
}

pub(crate) fn literal_refused(data_type: &str, reason: &str) -> String {
    format!("{LITERAL_REFUSED}{data_type}`{LITERAL_REASON}{reason}`")
}

pub(crate) fn unknown_tag(tag: &str) -> String {
    format!("{UNKNOWN_TAG}{tag}`")
}

pub(crate) fn call_refused(function: &str, reason: &str) -> String {
    format!("{CALL_REFUSED}{function}`{CALL_REASON}{reason}")
}

pub(crate) fn aggregate_in_row_expr(function: &str) -> String {
    format!(
        "{AGGREGATE_IN_ROW_EXPR}{function}` is not allowed in an expression body; aggregation \
         is stated through `agg:`"
    )
}

pub(crate) fn unknown_aggregate(name: &str) -> String {
    format!(
        "{UNKNOWN_AGGREGATE}{name}`; write an aggregate function's name or an alias, or \
         `count_distinct`"
    )
}

pub(crate) fn classify(msg: &str) -> Option<ParseErrorKind> {
    if let Some(name) = backticked_after(msg, UNKNOWN_AGGREGATE) {
        return Some(ParseErrorKind::UnknownAggregate { name });
    }
    if let Some(tag) = backticked_after(msg, UNKNOWN_TAG) {
        return Some(ParseErrorKind::UnknownTag { tag });
    }
    if let Some((function, rest)) = split_backticked(msg, CALL_REFUSED) {
        if let Some(reason) = rest.strip_prefix(CALL_REASON) {
            let reason = reason.rfind(AT_LINE).map_or(reason, |end| &reason[..end]);
            return Some(ParseErrorKind::CallRefused {
                function,
                reason: reason.to_owned(),
            });
        }
    }
    if let Some(function) = backticked_after(msg, AGGREGATE_IN_ROW_EXPR) {
        return Some(ParseErrorKind::AggregateInRowExpr { function });
    }
    if let Some(carrier) = backticked_after(msg, NAME_WITH_REF) {
        return Some(ParseErrorKind::SemanticsSiteNameWithRef { carrier });
    }
    if msg.contains(ON_EMPTY) {
        return Some(ParseErrorKind::RelationshipJoinKeysEmpty);
    }
    if let Some(spelling) = backticked_after(msg, ON_SPELLING) {
        return Some(ParseErrorKind::RelationshipJoinKeysUnknownSpelling { spelling });
    }
    if let Some(key) = backticked_after(msg, ON_KEY_DUPLICATED) {
        return Some(ParseErrorKind::RelationshipJoinKeyDuplicated { key });
    }
    if let Some(carrier) = backticked_after(msg, KEYSET_BOTH) {
        return Some(ParseErrorKind::SemanticsSiteKeysetBothSpellings { carrier });
    }
    if let Some((spelling, rest)) = split_backticked(msg, KEYSET_UNACTIVATED) {
        if let Some(carrier) = backticked_after(rest, IN_CARRIER) {
            return Some(ParseErrorKind::SemanticsSiteKeysetWithoutIsKey {
                spelling: spelling.trim_end_matches(':').to_owned(),
                carrier,
            });
        }
    }
    if let Some(spelling) = backticked_after(msg, DATA_TYPE_REFUSED) {
        return Some(ParseErrorKind::InvalidDataType { spelling });
    }
    if let Some((data_type, rest)) = split_backticked(msg, LITERAL_REFUSED) {
        if let Some(reason) = backticked_after(rest, LITERAL_REASON) {
            return Some(ParseErrorKind::InvalidLiteral { data_type, reason });
        }
    }
    if let Some(carrier) = backticked_after(msg, KEYSET_LIST_EMPTY) {
        return Some(ParseErrorKind::SemanticsSiteKeysetListEmpty { carrier });
    }
    if let Some((spelling, rest)) = split_backticked(msg, KEYSET_NAME_BLANK) {
        if let Some(carrier) = backticked_after(rest, IN_CARRIER) {
            return Some(ParseErrorKind::SemanticsSiteKeysetNameBlank {
                spelling: spelling.trim_end_matches(':').to_owned(),
                carrier,
            });
        }
    }
    let (field, rest) = split_backticked(msg, UNKNOWN_FIELD)?;
    let parent = rest
        .strip_prefix(IN_CARRIER)
        .and_then(|r| r.split_once('`'))
        .map_or_else(|| UNKNOWN_CARRIER.to_owned(), |(p, _)| p.to_owned());
    Some(ParseErrorKind::UnknownField { field, parent })
}

pub(crate) fn backticked_after(msg: &str, marker: &str) -> Option<String> {
    split_backticked(msg, marker).map(|(token, _)| token)
}

fn split_backticked<'a>(msg: &'a str, marker: &str) -> Option<(String, &'a str)> {
    let idx = msg.find(marker)?;
    let rest = &msg[idx + marker.len()..];
    let end = rest.find('`')?;
    Some((rest[..end].to_string(), &rest[end + 1..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unknown_field_message_round_trips_producer_to_reader() {
        let msg = unknown_field("kyset", "Dimension", &["id", "name"]);
        assert_eq!(
            classify(&msg),
            Some(ParseErrorKind::UnknownField {
                field: "kyset".to_owned(),
                parent: "Dimension".to_owned(),
            }),
            "produced `{msg}` did not read back"
        );
    }

    #[test]
    fn a_derived_path_message_classifies_with_no_carrier() {
        assert_eq!(
            classify("unknown field `bogus_block`, expected one of `id`, `name`"),
            Some(ParseErrorKind::UnknownField {
                field: "bogus_block".to_owned(),
                parent: UNKNOWN_CARRIER.to_owned(),
            })
        );
    }

    #[test]
    fn a_serde_path_prefix_does_not_hide_the_marker() {
        let msg = format!(
            "semantic_model.datasets[0]: {}",
            unknown_field("typ", "Dimension", &["type"])
        );
        assert_eq!(
            classify(&msg),
            Some(ParseErrorKind::UnknownField {
                field: "typ".to_owned(),
                parent: "Dimension".to_owned(),
            })
        );
    }

    #[test]
    fn a_name_with_ref_message_round_trips_producer_to_reader() {
        let msg = site_name_with_ref("Measure");
        assert_eq!(
            classify(&msg),
            Some(ParseErrorKind::SemanticsSiteNameWithRef {
                carrier: "Measure".to_owned(),
            }),
            "produced `{msg}` did not read back"
        );
    }

    #[test]
    fn a_join_basis_message_round_trips_producer_to_reader() {
        let cases: [(String, ParseErrorKind); 3] = [
            (
                join_basis_empty(),
                ParseErrorKind::RelationshipJoinKeysEmpty,
            ),
            (
                join_basis_unknown_spelling("crossed"),
                ParseErrorKind::RelationshipJoinKeysUnknownSpelling {
                    spelling: "crossed".to_owned(),
                },
            ),
            (
                join_key_duplicated("region"),
                ParseErrorKind::RelationshipJoinKeyDuplicated {
                    key: "region".to_owned(),
                },
            ),
        ];
        for (msg, expected) in cases {
            assert_eq!(classify(&msg), Some(expected), "`{msg}` did not read back");
        }
    }

    #[test]
    fn the_empty_basis_message_names_cross_and_the_spelling_message_names_both() {
        assert!(join_basis_empty().contains("cross"));
        let unknown = join_basis_unknown_spelling("cartesian");
        assert!(unknown.contains("auto") && unknown.contains("cross"));
    }

    #[test]
    fn a_keyness_message_round_trips_producer_to_reader() {
        let cases: [(String, ParseErrorKind); 5] = [
            (
                keyset_list_empty("Dimension"),
                ParseErrorKind::SemanticsSiteKeysetListEmpty {
                    carrier: "Dimension".to_owned(),
                },
            ),
            (
                keyset_name_blank("keysets", "Dimension reference site"),
                ParseErrorKind::SemanticsSiteKeysetNameBlank {
                    spelling: "keysets".to_owned(),
                    carrier: "Dimension reference site".to_owned(),
                },
            ),
            (
                keyset_both_spellings("Dimension"),
                ParseErrorKind::SemanticsSiteKeysetBothSpellings {
                    carrier: "Dimension".to_owned(),
                },
            ),
            (
                keyset_without_is_key("keyset", "Dimension"),
                ParseErrorKind::SemanticsSiteKeysetWithoutIsKey {
                    spelling: "keyset".to_owned(),
                    carrier: "Dimension".to_owned(),
                },
            ),
            (
                keyset_without_is_key("keysets", "Dimension reference site"),
                ParseErrorKind::SemanticsSiteKeysetWithoutIsKey {
                    spelling: "keysets".to_owned(),
                    carrier: "Dimension reference site".to_owned(),
                },
            ),
        ];
        for (msg, expected) in cases {
            assert_eq!(classify(&msg), Some(expected), "`{msg}` did not read back");
        }
    }

    #[test]
    fn the_empty_keysets_message_names_the_bare_activation_spelling() {
        assert!(keyset_list_empty("Dimension").contains("`is_key: true`"));
    }

    #[test]
    fn an_unrelated_message_classifies_as_none() {
        assert_eq!(
            classify("missing field `data_type` at line 3 column 5"),
            None
        );
    }
}
