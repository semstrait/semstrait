use serde::de::Error as DeError;
use serde::ser::SerializeMap;
use serde::{Serialize, Serializer};
use serde_yaml::Value;

pub(crate) fn serialize_single_key<S, T>(
    serializer: S,
    tag: &str,
    body: &T,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
    T: Serialize + ?Sized,
{
    let mut map = serializer.serialize_map(Some(1))?;
    map.serialize_entry(tag, body)?;
    map.end()
}

pub(crate) fn single_key_map<E: DeError>(
    value: Value,
    type_name: &str,
) -> Result<(String, Value), E> {
    let map = match value {
        Value::Mapping(m) => m,
        other => {
            return Err(E::custom(format!(
                "{type_name}: expected single-key tagged map, got {other:?}"
            )));
        }
    };
    if map.len() != 1 {
        return Err(E::custom(format!(
            "{type_name}: expected single-key tagged map (e.g. `{{ tag: body }}`), got {} keys",
            map.len()
        )));
    }
    let (k, v) = map.into_iter().next().unwrap();
    let tag = k
        .as_str()
        .ok_or_else(|| E::custom(format!("{type_name}: variant tag must be a string")))?;
    Ok((tag.to_string(), v))
}
