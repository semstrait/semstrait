use serde::{de, Deserialize, Deserializer, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct NamedReference(Vec<String>);

impl NamedReference {
    pub fn new(parts: impl IntoIterator<Item = impl Into<String>>) -> Option<Self> {
        let parts: Vec<String> = parts.into_iter().map(Into::into).collect();
        (!parts.is_empty() && !parts.iter().any(String::is_empty)).then_some(Self(parts))
    }

    pub fn unqualified(name: impl Into<String>) -> Option<Self> {
        Self::new([name.into()])
    }

    pub fn qualified(qualifier: impl Into<String>, name: impl Into<String>) -> Option<Self> {
        Self::new([qualifier.into(), name.into()])
    }

    pub fn parts(&self) -> &[String] {
        &self.0
    }

    pub fn name(&self) -> &str {
        self.0.last().map_or("", String::as_str)
    }
}

impl<'de> Deserialize<'de> for NamedReference {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        const EXPECTED: &str = "a name path of one part or more, none of them empty";
        let parts = Vec::<String>::deserialize(deserializer)?;
        let length = parts.len();
        Self::new(parts).ok_or_else(|| match length {
            0 => de::Error::invalid_length(0, &EXPECTED),
            _ => de::Error::invalid_value(de::Unexpected::Str(""), &EXPECTED),
        })
    }
}
