use std::fmt;

use serde::{Deserialize, Serialize};

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "schemars",
    schemars(
        description = "A time granularity such as `minute`, `day` or `month`, ordered from finest to coarsest."
    )
)]
#[serde(rename_all = "lowercase")]
pub enum Granularity {
    Minute,
    Hour,
    Day,
    Week,
    Month,
    Quarter,
    Year,
}

impl Granularity {
    pub fn coarseness(self) -> u8 {
        match self {
            Self::Minute => 0,
            Self::Hour => 1,
            Self::Day => 2,
            Self::Week => 3,
            Self::Month => 4,
            Self::Quarter => 5,
            Self::Year => 6,
        }
    }

    pub fn serves(self, requested: Granularity) -> bool {
        if self == Self::Week {
            requested == Self::Week
        } else {
            requested.coarseness() >= self.coarseness()
        }
    }
}

impl fmt::Display for Granularity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Minute => f.write_str("minute"),
            Self::Hour => f.write_str("hour"),
            Self::Day => f.write_str("day"),
            Self::Week => f.write_str("week"),
            Self::Month => f.write_str("month"),
            Self::Quarter => f.write_str("quarter"),
            Self::Year => f.write_str("year"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn granularity_coarseness_is_total_ordered() {
        assert!(Granularity::Minute.coarseness() < Granularity::Hour.coarseness());
        assert!(Granularity::Hour.coarseness() < Granularity::Day.coarseness());
        assert!(Granularity::Day.coarseness() < Granularity::Week.coarseness());
        assert!(Granularity::Week.coarseness() < Granularity::Month.coarseness());
        assert!(Granularity::Month.coarseness() < Granularity::Quarter.coarseness());
        assert!(Granularity::Quarter.coarseness() < Granularity::Year.coarseness());
    }

    #[test]
    fn granularity_coarseness_min_is_minute_max_is_year() {
        assert_eq!(Granularity::Minute.coarseness(), 0);
        assert_eq!(Granularity::Year.coarseness(), 6);
    }

    #[test]
    fn granularity_display_lowercase() {
        assert_eq!(Granularity::Day.to_string(), "day");
        assert_eq!(Granularity::Quarter.to_string(), "quarter");
    }

    const ROSTER: [Granularity; 7] = [
        Granularity::Minute,
        Granularity::Hour,
        Granularity::Day,
        Granularity::Week,
        Granularity::Month,
        Granularity::Quarter,
        Granularity::Year,
    ];

    #[test]
    fn every_granularity_serves_itself_and_every_coarser_one_except_week() {
        let served: [(Granularity, &[Granularity]); 7] = [
            (
                Granularity::Minute,
                &[
                    Granularity::Minute,
                    Granularity::Hour,
                    Granularity::Day,
                    Granularity::Week,
                    Granularity::Month,
                    Granularity::Quarter,
                    Granularity::Year,
                ],
            ),
            (
                Granularity::Hour,
                &[
                    Granularity::Hour,
                    Granularity::Day,
                    Granularity::Week,
                    Granularity::Month,
                    Granularity::Quarter,
                    Granularity::Year,
                ],
            ),
            (
                Granularity::Day,
                &[
                    Granularity::Day,
                    Granularity::Week,
                    Granularity::Month,
                    Granularity::Quarter,
                    Granularity::Year,
                ],
            ),
            (Granularity::Week, &[Granularity::Week]),
            (
                Granularity::Month,
                &[Granularity::Month, Granularity::Quarter, Granularity::Year],
            ),
            (
                Granularity::Quarter,
                &[Granularity::Quarter, Granularity::Year],
            ),
            (Granularity::Year, &[Granularity::Year]),
        ];
        for (source, roster) in served {
            for requested in ROSTER {
                assert_eq!(
                    source.serves(requested),
                    roster.contains(&requested),
                    "{source:?} serving {requested:?}"
                );
            }
        }
    }

    #[test]
    fn granularity_serde_roundtrip() {
        for g in &ROSTER {
            let json = serde_json::to_string(g).unwrap();
            let back: Granularity = serde_json::from_str(&json).unwrap();
            assert_eq!(g, &back);
        }
    }
}
