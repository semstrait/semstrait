use std::fmt;

use chrono_tz::Tz;
use thiserror::Error;

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct RequestContext {
    pub timezone: Timezone,
    pub week_start: Weekday,
}

impl RequestContext {
    #[must_use]
    pub fn with_timezone(self, timezone: Timezone) -> Self {
        Self { timezone, ..self }
    }

    #[must_use]
    pub fn with_week_start(self, week_start: Weekday) -> Self {
        Self { week_start, ..self }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Weekday {
    #[default]
    Monday,
    Sunday,
}

impl Weekday {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Monday => "monday",
            Self::Sunday => "sunday",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Timezone(Tz);

impl Timezone {
    pub fn new(name: &str) -> Result<Self, UnknownTimezone> {
        name.parse().map(Self).map_err(|_| UnknownTimezone {
            name: name.to_owned(),
        })
    }

    #[must_use]
    pub fn name(self) -> &'static str {
        self.0.name()
    }
}

impl Default for Timezone {
    fn default() -> Self {
        Self(Tz::UTC)
    }
}

impl fmt::Display for Timezone {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[non_exhaustive]
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[error("`{name}` is not an IANA timezone")]
pub struct UnknownTimezone {
    pub name: String,
}
