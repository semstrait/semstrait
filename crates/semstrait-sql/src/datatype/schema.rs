use std::borrow::Cow;
use std::collections::BTreeMap;

use schemars::{json_schema, JsonSchema, Schema, SchemaGenerator};

use super::declaration::{TypeDeclaration, Values};
use super::name::Spelling;
use super::written::ELEMENT;
use super::DataType;
use crate::DocumentDialect;

impl DataType {
    pub fn resolution_schema(dialect: DocumentDialect, element: Schema) -> Schema {
        Self::spelling_schema(dialect, Spelling::NameOrAlias, Some(element))
    }

    fn spelling_schema(
        dialect: DocumentDialect,
        rule: Spelling,
        element: Option<Schema>,
    ) -> Schema {
        let declarations: Vec<_> = Self::roster()
            .iter()
            .filter_map(|template| template.declarations()?.in_document(dialect))
            .collect();
        let scalars: Vec<_> = declarations
            .iter()
            .map(|declaration| branch(declaration, rule))
            .collect();
        let elements = scalars
            .iter()
            .map(|(_, pattern)| pattern.as_str())
            .collect::<Vec<_>>()
            .join("|");
        let mut branches = Vec::new();
        for (declaration, (title, pattern)) in declarations.iter().zip(&scalars) {
            branches.push(string(title, pattern));
            let values: Vec<_> = declaration
                .limits()
                .map(|(letter, key, limit)| (letter, key, limit.values.schema()))
                .collect();
            if element.is_some() && !values.is_empty() {
                branches.push(object(declaration, rule, values));
            }
        }
        if let Some(array) = Self::array_declaration(dialect) {
            let array_name = spelled(array.name, rule);
            branches.push(string(
                &format!("{}<T>", array.name),
                &format!(r"(?:{array_name}<\s*)+(?:{elements})(?:\s*>)+"),
            ));
            if let Some(element) = element {
                branches.push(object(array, rule, vec![("T", ELEMENT, element)]));
            }
        }
        let description = match rule {
            Spelling::Name => format!(
                "A data type's canonical name, as compile writes it, such as `{}` or `{}`.",
                Self::BigInt,
                Self::Decimal { precision: 18, scale: 2 }
            ),
            Spelling::NameOrAlias => format!(
                "A data type's name or an alias in any case: a string such as `{}`, or an object that names each value.",
                Self::Decimal { precision: 18, scale: 4 }.to_string().to_ascii_uppercase()
            ),
        };
        json_schema!({ "description": description, "oneOf": branches })
    }
}

impl JsonSchema for DataType {
    fn schema_name() -> Cow<'static, str> {
        "DataType".into()
    }

    fn json_schema(_: &mut SchemaGenerator) -> Schema {
        Self::spelling_schema(DocumentDialect::Canonical, Spelling::Name, None)
    }
}

pub fn any_case(name: &str) -> String {
    name.bytes()
        .map(|b| match b {
            b'a'..=b'z' => format!("[{}{}]", char::from(b), char::from(b).to_ascii_uppercase()),
            b'0'..=b'9' | b'_' => char::from(b).to_string(),
            _ => format!("\\x{b:02x}"),
        })
        .collect()
}

fn string(title: &str, pattern: &str) -> Schema {
    json_schema!({ "title": title, "type": "string", "pattern": format!("^{pattern}$") })
}

fn object(
    declaration: &TypeDeclaration,
    rule: Spelling,
    values: Vec<(&str, &str, Schema)>,
) -> Schema {
    let title: Vec<_> = values
        .iter()
        .map(|(letter, key, _)| format!("{key}: {letter}"))
        .collect();
    let required: Vec<_> = values.iter().map(|(_, key, _)| *key).collect();
    let properties: BTreeMap<_, _> = values
        .into_iter()
        .map(|(_, key, schema)| (key, schema))
        .collect();
    json_schema!({
        "title": format!("{{{}: {{{}}}}}", declaration.name, title.join(", ")),
        "type": "object",
        "minProperties": 1,
        "maxProperties": 1,
        "patternProperties": {
            format!("^{}$", names(declaration, rule)): {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": false
            }
        },
        "additionalProperties": false
    })
}

impl Values {
    pub fn schema(self) -> Schema {
        match self {
            Self::Range(low, high) => {
                json_schema!({ "type": "integer", "minimum": low, "maximum": high })
            }
            Self::OneOf(values) => json_schema!({ "enum": values }),
        }
    }
}

fn names(declaration: &TypeDeclaration, rule: Spelling) -> String {
    let names: Vec<_> = rule
        .names(declaration)
        .map(|name| spelled(name, rule))
        .collect();
    format!("(?:{})", names.join("|"))
}

fn branch(declaration: &TypeDeclaration, rule: Spelling) -> (String, String) {
    let limits: Vec<_> = declaration.limits().collect();
    let names = names(declaration, rule);
    if limits.is_empty() {
        return (declaration.name.to_owned(), names);
    }
    let letters: Vec<_> = limits.iter().map(|(letter, ..)| *letter).collect();
    let values: Vec<_> = limits
        .iter()
        .map(|(.., limit)| format!(r"\+?0*(?:{})", digits(limit.values)))
        .collect();
    let bare = limits.iter().all(|(.., limit)| limit.default.is_some());
    (
        format!("{}({})", declaration.name, letters.join(",")),
        format!(
            r"{names}(?:\(\s*{}\s*\)){}",
            values.join(r"\s*,\s*"),
            if bare { "?" } else { "" }
        ),
    )
}

fn spelled(name: &str, rule: Spelling) -> String {
    match rule {
        Spelling::NameOrAlias => any_case(name),
        Spelling::Name => name.to_owned(),
    }
}

fn digits(values: Values) -> String {
    let (mut next, high) = match values {
        Values::Range(low, high) => (u64::from(low), u64::from(high)),
        Values::OneOf(values) => {
            return values
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join("|")
        }
    };
    let mut runs = Vec::new();
    while next <= high {
        let size = (1..)
            .take_while(|&k| {
                next > 0 && next % 10u64.pow(k) == 0 && next + 10u64.pow(k) <= high + 1
            })
            .last()
            .unwrap_or(0);
        let unit = 10u64.pow(size);
        let (first, last) = (
            next / unit,
            (next / unit / 10 * 10 + 9).min((high + 1) / unit - 1),
        );
        let written = first.to_string();
        let prefix = &written[..written.len() - 1];
        let rest = "[0-9]".repeat(size as usize);
        runs.push(format!("{prefix}[{}-{}]{rest}", first % 10, last % 10));
        next = (last + 1) * unit;
    }
    runs.join("|")
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::datatype::WrittenDataType;

    #[test]
    fn each_name_and_alias_of_each_roster_type_passes_the_schema_and_resolves_as_an_object() {
        let schema = serde_json::to_value(DataType::resolution_schema(
            DocumentDialect::Canonical,
            Schema::new_ref("#".to_owned()),
        ))
        .expect("a JSON schema");
        let validator = jsonschema::validator_for(&schema).expect("a valid JSON schema");
        for template in DataType::roster() {
            let declaration = &template.declarations().expect("a declared type").canonical;
            let values: Vec<_> = declaration
                .limits()
                .zip(template.values())
                .map(|((_, key, _), value)| {
                    (
                        key.to_owned(),
                        u64::try_from(value).expect("a natural value"),
                    )
                })
                .collect();
            if values.is_empty() {
                continue;
            }
            let object: serde_json::Map<_, _> = values
                .iter()
                .map(|(key, value)| (key.clone(), (*value).into()))
                .collect();
            let numbers = values
                .iter()
                .map(|(key, value)| (key.clone(), WrittenDataType::Number(*value)));
            for name in iter::once(&declaration.name).chain(declaration.aliases) {
                for key in [name.to_string(), name.to_ascii_uppercase()] {
                    let written = WrittenDataType::Object(vec![(
                        key.clone(),
                        WrittenDataType::Object(numbers.clone().collect()),
                    )]);
                    assert!(
                        validator.is_valid(&serde_json::json!({ key: object })),
                        "`{written}`"
                    );
                    assert_eq!(
                        DataType::resolve_written(&written, DocumentDialect::Canonical).as_ref(),
                        Some(template),
                        "`{written}`"
                    );
                }
            }
        }
    }

    #[test]
    fn each_name_and_alias_of_each_roster_type_passes_the_schema_and_resolves_in_any_case() {
        let schema = serde_json::to_value(DataType::resolution_schema(
            DocumentDialect::Canonical,
            Schema::new_ref("#".to_owned()),
        ))
        .expect("a JSON schema");
        let validator = jsonschema::validator_for(&schema).expect("a valid JSON schema");
        let nested = DataType::Array(Box::new(DataType::Array(Box::new(DataType::decimal()))));
        for template in DataType::roster().iter().chain([&nested]) {
            let declaration = &template.declarations().expect("a declared type").canonical;
            for name in iter::once(&declaration.name).chain(declaration.aliases) {
                let spelling = template.to_string().replacen(declaration.name, name, 1);
                let mixed = spelling
                    .chars()
                    .enumerate()
                    .map(|(at, c)| {
                        if at % 2 == 0 {
                            c.to_ascii_uppercase()
                        } else {
                            c
                        }
                    })
                    .collect();
                for written in [spelling.to_ascii_uppercase(), mixed, spelling] {
                    assert!(
                        validator.is_valid(&serde_json::json!(written)),
                        "`{written}`"
                    );
                    assert_eq!(
                        DataType::resolve(&written, DocumentDialect::Canonical).as_ref(),
                        Some(template),
                        "`{written}`"
                    );
                }
            }
        }
    }
}
