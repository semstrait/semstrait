#[macro_export]
macro_rules! str_newtype {
    ($(#[$extra:meta])* $name:ident) => {
        $(#[$extra])*
        #[derive(
            Debug,
            Clone,
            PartialEq,
            Eq,
            Hash,
            PartialOrd,
            Ord,
            ::serde::Serialize,
            ::serde::Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(pub ::std::string::String);

        impl $name {
            #[must_use]
            pub fn new(s: impl ::core::convert::Into<::std::string::String>) -> Self {
                Self(s.into())
            }

            #[must_use]
            pub fn as_str(&self) -> &str {
                &self.0
            }

            #[must_use]
            pub fn into_string(self) -> ::std::string::String {
                self.0
            }
        }

        impl ::core::convert::From<&str> for $name {
            fn from(s: &str) -> Self {
                Self(::std::borrow::ToOwned::to_owned(s))
            }
        }

        impl ::core::convert::From<::std::string::String> for $name {
            fn from(s: ::std::string::String) -> Self {
                Self(s)
            }
        }

        impl ::core::convert::AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }

        impl ::core::fmt::Display for $name {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                f.write_str(&self.0)
            }
        }
    };
}

str_newtype!(
    #[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
    #[cfg_attr(
        feature = "schemars",
        schemars(description = "The name of a Dimension, Measure, Metric or Filter.")
    )]
    SemanticsName
);
str_newtype!(
    #[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
    #[cfg_attr(
        feature = "schemars",
        schemars(description = "The name of a Dataset, Grainset, Unionset or Joinset.")
    )]
    DataKindName
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newtype_roundtrip_serde() {
        let n: SemanticsName = "revenue".into();
        let json = serde_json::to_string(&n).unwrap();
        assert_eq!(json, "\"revenue\"");
        let back: SemanticsName = serde_json::from_str(&json).unwrap();
        assert_eq!(back, n);
    }

    #[test]
    fn ordering_alphabetical() {
        let mut v = [
            DataKindName::from("zeta"),
            DataKindName::from("alpha"),
            DataKindName::from("mu"),
        ];
        v.sort();
        let names: Vec<&str> = v.iter().map(|n| n.as_str()).collect();
        assert_eq!(names, vec!["alpha", "mu", "zeta"]);
    }
}
