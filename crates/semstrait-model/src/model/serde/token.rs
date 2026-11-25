use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::Literal;

use super::error::ParseError;
use super::literal::number;

#[derive(Debug, Clone, PartialEq)]
pub enum LeafToken {
    Literal(Literal),
    Name(String),
    NameWithAccessor { name: String, accessor: String },
}

pub fn tokenize_leaf(raw: &str) -> Result<LeafToken, ParseError> {
    let s = raw;

    let literal = match s {
        "null" => Some(Literal::Null(DataType::String)),
        "true" => Some(Literal::Boolean(true)),
        "false" => Some(Literal::Boolean(false)),
        _ => strip_quoted(s, '\'').map(|body| Literal::String(unescape_minimal(body))),
    };
    if let Some(literal) = literal {
        return Ok(LeafToken::Literal(literal));
    }

    if let Some(body) = strip_quoted(s, '"') {
        if body.is_empty() {
            return Err(ParseError::InvalidToken {
                raw: raw.to_owned(),
                reason: "empty quoted identifier",
            });
        }
        return Ok(LeafToken::Name(unescape_minimal(body)));
    }

    if is_number(s) {
        return number(s).map(LeafToken::Literal);
    }

    if s.is_empty() {
        return Err(ParseError::InvalidToken {
            raw: raw.to_owned(),
            reason: "empty identifier",
        });
    }
    if let Some(idx) = s.find('.') {
        let (name, rest) = s.split_at(idx);
        let accessor = &rest[1..];
        if name.is_empty() {
            return Err(ParseError::InvalidToken {
                raw: raw.to_owned(),
                reason: "accessor without name",
            });
        }
        if accessor.is_empty() {
            return Err(ParseError::InvalidToken {
                raw: raw.to_owned(),
                reason: "trailing dot",
            });
        }
        if accessor.contains('.') {
            return Err(ParseError::InvalidToken {
                raw: raw.to_owned(),
                reason: "accessor must not contain a dot; double-quote the name to embed dots",
            });
        }
        return Ok(LeafToken::NameWithAccessor {
            name: name.to_owned(),
            accessor: accessor.to_owned(),
        });
    }

    Ok(LeafToken::Name(s.to_owned()))
}

fn is_number(s: &str) -> bool {
    s.contains(|c: char| c.is_ascii_digit()) && s.parse::<f64>().is_ok()
}

pub(crate) const COMPOUND_CHARS: &str = " \t\n\x0C\r+*/%=<>(),&|!~^;:@#?";

pub(crate) fn is_compound_expr_string(s: &str) -> bool {
    !is_number(s) && s.contains(|c| COMPOUND_CHARS.contains(c))
}

fn strip_quoted(s: &str, q: char) -> Option<&str> {
    let bytes = s.as_bytes();
    if bytes.len() >= 2 && bytes[0] as char == q && bytes[bytes.len() - 1] as char == q {
        Some(&s[1..s.len() - 1])
    } else {
        None
    }
}

fn unescape_minimal(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some(next @ ('\\' | '\'' | '"')) => out.push(next),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn literal(raw: &str) -> Literal {
        match tokenize_leaf(raw).unwrap() {
            LeafToken::Literal(literal) => literal,
            other => panic!("expected a literal for {raw:?}, got {other:?}"),
        }
    }

    #[test]
    fn a_literal_token_is_its_parse_default() {
        for (raw, expected) in [
            ("null", Literal::Null(DataType::String)),
            ("true", Literal::Boolean(true)),
            ("false", Literal::Boolean(false)),
            ("42", Literal::BigInt(42)),
            ("-7", Literal::BigInt(-7)),
            ("1.5", Literal::Double(1.5)),
            ("'Germany'", Literal::String("Germany".into())),
            ("''", Literal::String("".into())),
        ] {
            assert_eq!(literal(raw), expected, "{raw:?}");
        }
    }

    #[test]
    fn double_quoted_dotted_identifier() {
        assert_eq!(
            tokenize_leaf(r#""my.dotted.name""#).unwrap(),
            LeafToken::Name("my.dotted.name".into())
        );
    }

    #[test]
    fn double_quoted_empty_rejects() {
        let err = tokenize_leaf(r#""""#).unwrap_err();
        assert!(matches!(
            err,
            ParseError::InvalidToken {
                reason: "empty quoted identifier",
                ..
            }
        ));
    }

    #[test]
    fn bare_identifier_with_dash_is_one_atom() {
        assert_eq!(
            tokenize_leaf("shopify-shipping_country").unwrap(),
            LeafToken::Name("shopify-shipping_country".into())
        );
    }

    #[test]
    fn bare_identifier_with_dot_splits_accessor() {
        assert_eq!(
            tokenize_leaf("revenue.previous").unwrap(),
            LeafToken::NameWithAccessor {
                name: "revenue".into(),
                accessor: "previous".into(),
            }
        );
    }

    #[test]
    fn multi_dot_in_bare_form_rejects() {
        let err = tokenize_leaf("a.b.c").unwrap_err();
        assert!(matches!(err, ParseError::InvalidToken { .. }));
    }

    #[test]
    fn trailing_dot_rejects() {
        let err = tokenize_leaf("a.").unwrap_err();
        assert!(matches!(
            err,
            ParseError::InvalidToken {
                reason: "trailing dot",
                ..
            }
        ));
    }

    #[test]
    fn empty_input_rejects() {
        let err = tokenize_leaf("").unwrap_err();
        assert!(matches!(
            err,
            ParseError::InvalidToken {
                reason: "empty identifier",
                ..
            }
        ));
    }

    #[test]
    fn escape_sequences_in_string_literal() {
        assert_eq!(literal(r#"'it\'s'"#), Literal::String("it's".into()));
        assert_eq!(literal(r"'one\\two'"), Literal::String("one\\two".into()));
    }

    #[test]
    fn compound_guard_rejects_whitespace() {
        for s in [" ", "a b", "a\tb", "a\nb", "a\rb", "revenue / cost"] {
            assert!(is_compound_expr_string(s), "{s:?} should be compound");
        }
    }

    #[test]
    fn compound_guard_rejects_operator_chars() {
        for s in [
            "a+b", "a*b", "a/b", "a%b", "a=b", "a<b", "a>b", "(a)", "a,b", "a&b", "a|b", "a!b",
            "a~b", "a^b", "a;b", "a:b", "a@b", "a#b", "a?b",
        ] {
            assert!(is_compound_expr_string(s), "{s:?} should be compound");
        }
    }

    #[test]
    fn compound_guard_passes_bare_leaf_forms() {
        for s in [
            "revenue",
            "shopify-shipping_country",
            "country.previous",
            "'Germany'",
            "\"a.b\"",
            "42",
            "1.5",
            "_x",
            "null",
        ] {
            assert!(!is_compound_expr_string(s), "{s:?} should NOT be compound");
        }
    }
}
