use semstrait_common::{DataKindName, SemanticsName};
use semstrait_common::{Diagnose, Severity};

use crate::error::quoted;
use crate::frame_path::{at, FramePath};

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum Advisory {
    FanOut {
        path: FramePath,
        operand: DataKindName,
    },
    UniqueWithoutKeyset {
        path: FramePath,
        semantic_source: DataKindName,
    },
    LossyReaggregation {
        path: FramePath,
        semantics: SemanticsName,
    },
    VolatileInTwoReads {
        path: FramePath,
        metric: SemanticsName,
    },
    VolatileCopied {
        path: FramePath,
        semantics: SemanticsName,
    },
    AmbiguousCandidates {
        path: FramePath,
        chosen: DataKindName,
        tied: Vec<DataKindName>,
    },
    BranchNotServable {
        path: FramePath,
        branch: DataKindName,
        lacks: Vec<SemanticsName>,
    },
}

impl Advisory {
    #[must_use]
    pub fn path(&self) -> &FramePath {
        match self {
            Self::FanOut { path, .. }
            | Self::UniqueWithoutKeyset { path, .. }
            | Self::LossyReaggregation { path, .. }
            | Self::VolatileInTwoReads { path, .. }
            | Self::VolatileCopied { path, .. }
            | Self::AmbiguousCandidates { path, .. }
            | Self::BranchNotServable { path, .. } => path,
        }
    }

    fn notice(&self) -> String {
        match self {
            Self::FanOut { operand, .. } => format!(
                "the row identity of `{operand}` is inferred from every Dimension it exposes, \
                 so its rows that agree on each merge into one"
            ),
            Self::UniqueWithoutKeyset {
                semantic_source, ..
            } => format!(
                "the unique Unionset `{semantic_source}` has no guaranteed keyset, so its rows \
                 that agree on every field it compares merge into one"
            ),
            Self::LossyReaggregation { semantics, .. } => format!(
                "`{semantics}` is not additive, and is aggregated again over rows that are \
                 already aggregates"
            ),
            Self::VolatileInTwoReads { metric, .. } => {
                format!("the two reads of the Metric `{metric}` draw a volatile call apart")
            }
            Self::VolatileCopied { semantics, .. } => format!(
                "`{semantics}` calls a volatile function in two expressions, so its readers \
                 read two draws"
            ),
            Self::AmbiguousCandidates { path, chosen, tied } => {
                let tied = quoted(tied);
                if path.0.last() == Some(chosen) {
                    format!("`{chosen}` is chosen by name order over {tied}: name one in `from`")
                } else {
                    format!(
                        "`{chosen}` is chosen by name order over {tied}: make them differ in the model"
                    )
                }
            }
            Self::BranchNotServable { branch, lacks, .. } => format!(
                "`{branch}` is pruned: its children cannot serve the demand, and lack {}",
                quoted(lacks)
            ),
        }
    }
}

pub(crate) fn advise(advised: &mut Vec<Advisory>, advisory: Advisory) {
    if !advised.contains(&advisory) {
        advised.push(advisory);
    }
}

impl Diagnose for Advisory {
    fn message(&self) -> String {
        at(self.path(), self.notice())
    }

    fn severity_default(&self) -> Severity {
        Severity::Warning
    }
}

#[cfg(test)]
mod tests {
    use semstrait_common::Diagnostic;

    use super::*;

    #[test]
    fn every_advisory_names_its_path_and_what_it_advises() {
        let path = FramePath(vec!["orders_customers".into(), "orders".into()]);
        let advisories = [
            (
                Advisory::FanOut {
                    path: path.clone(),
                    operand: "orders".into(),
                },
                vec!["orders"],
            ),
            (
                Advisory::UniqueWithoutKeyset {
                    path: path.clone(),
                    semantic_source: "all_orders".into(),
                },
                vec!["all_orders"],
            ),
            (
                Advisory::LossyReaggregation {
                    path: path.clone(),
                    semantics: "avg_cpc".into(),
                },
                vec!["avg_cpc"],
            ),
            (
                Advisory::VolatileInTwoReads {
                    path: path.clone(),
                    metric: "avg_customer_revenue".into(),
                },
                vec!["avg_customer_revenue"],
            ),
            (
                Advisory::VolatileCopied {
                    path: path.clone(),
                    semantics: "draw".into(),
                },
                vec!["draw"],
            ),
            (
                Advisory::AmbiguousCandidates {
                    path: path.clone(),
                    chosen: "spend_a".into(),
                    tied: vec!["spend_b".into()],
                },
                vec!["spend_a", "spend_b"],
            ),
            (
                Advisory::BranchNotServable {
                    path: path.clone(),
                    branch: "facebook".into(),
                    lacks: vec!["gender".into(), "age".into()],
                },
                vec!["facebook", "gender", "age"],
            ),
        ];
        for (advisory, named) in advisories {
            assert_eq!(advisory.path(), &path);
            let diagnostic = Diagnostic::new(advisory);
            assert_eq!(diagnostic.severity, Severity::Warning);
            let message = diagnostic.kind.message();
            assert!(message.contains(&path.to_string()), "{message}");
            for name in named {
                assert!(message.contains(name), "`{message}` names `{name}`");
            }
        }
    }

    #[test]
    fn an_in_scope_tie_does_not_advise_from() {
        let advisory = Advisory::AmbiguousCandidates {
            path: FramePath(vec!["reports".into()]),
            chosen: "reports.a_child".into(),
            tied: vec!["reports.b_child".into()],
        };
        let message = Diagnostic::new(advisory).kind.message();
        assert!(!message.contains("`from`"), "{message}");
    }

    #[test]
    fn a_branch_not_servable_notice_says_pruned() {
        let advisory = Advisory::BranchNotServable {
            path: FramePath(vec!["reports".into()]),
            branch: "facebook".into(),
            lacks: vec!["gender".into()],
        };
        let message = Diagnostic::new(advisory).kind.message();
        assert!(message.contains("is pruned"), "{message}");
    }
}
