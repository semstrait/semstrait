use super::error::ParseError;
use serde_yaml::{Mapping, Value};

pub(crate) fn lookup<'m>(map: &'m Mapping, key: &str) -> Option<&'m Value> {
    map.get(Value::String(key.into()))
}

pub(crate) fn get_required_str<'m>(
    map: &'m Mapping,
    key: &str,
    field: &'static str,
) -> Result<&'m str, ParseError> {
    let v = lookup(map, key).ok_or(ParseError::MissingField(field))?;
    v.as_str().ok_or(ParseError::InvalidValue {
        field,
        reason: "expected string".into(),
    })
}

pub(crate) fn get_optional_value<'m>(map: &'m Mapping, key: &str) -> Option<&'m Value> {
    lookup(map, key)
}

pub(crate) fn deny_unknown_fields(
    map: &Mapping,
    parent: &'static str,
    allowed: &[&str],
) -> Result<(), ParseError> {
    for (k, _) in map.iter() {
        let key = k.as_str().ok_or_else(|| ParseError::InvalidValue {
            field: parent,
            reason: "non-string field key".into(),
        })?;
        if !allowed.contains(&key) {
            return Err(ParseError::InvalidValue {
                field: parent,
                reason: format!("unknown field `{key}`"),
            });
        }
    }
    Ok(())
}

pub(crate) fn tag(key: &str, body: Value) -> Value {
    let mut m = Mapping::new();
    m.insert(Value::String(key.to_owned()), body);
    Value::Mapping(m)
}

pub(crate) fn map(pairs: Vec<(&str, Value)>) -> Value {
    let mut m = Mapping::new();
    for (k, v) in pairs {
        m.insert(Value::String(k.to_owned()), v);
    }
    Value::Mapping(m)
}

pub(crate) fn seq(items: Vec<Value>) -> Value {
    Value::Sequence(items)
}
