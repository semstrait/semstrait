use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GlobPattern(String);

impl GlobPattern {
    pub fn new(pattern: impl Into<String>) -> Self {
        Self(pattern.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for GlobPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for GlobPattern {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for GlobPattern {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_and_as_str_round_trip() {
        let p = GlobPattern::new("sales_*");
        assert_eq!(p.as_str(), "sales_*");
    }

    #[test]
    fn display_renders_pattern() {
        assert_eq!(GlobPattern::new("data/**").to_string(), "data/**");
    }

    #[test]
    fn from_str_and_string_agree() {
        assert_eq!(
            GlobPattern::from("a*"),
            GlobPattern::from(String::from("a*"))
        );
    }

    #[test]
    fn eq_and_hash_by_inner_string() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(GlobPattern::new("x?"));
        assert!(set.contains(&GlobPattern::new("x?")));
        assert!(!set.contains(&GlobPattern::new("y?")));
    }
}
