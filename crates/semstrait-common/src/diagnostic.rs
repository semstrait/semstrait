use std::error::Error;
use std::fmt;

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Severity {
    Error,
    Warning,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Error => f.write_str("error"),
            Severity::Warning => f.write_str("warning"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub const fn unknown() -> Self {
        Self { start: 0, end: 0 }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SourceId(SourceIdInner);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum SourceIdInner {
    Unknown,
    Named(Box<str>),
}

impl SourceId {
    pub const fn unknown() -> Self {
        Self(SourceIdInner::Unknown)
    }

    pub fn named(label: impl Into<Box<str>>) -> Self {
        Self(SourceIdInner::Named(label.into()))
    }

    pub fn as_str(&self) -> &str {
        match &self.0 {
            SourceIdInner::Unknown => "<unknown>",
            SourceIdInner::Named(label) => label,
        }
    }
}

impl fmt::Display for SourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Location {
    pub source: SourceId,
    pub span: Span,
}

impl Location {
    pub const fn new(source: SourceId, span: Span) -> Self {
        Self { source, span }
    }
}

pub trait Diagnose {
    fn message(&self) -> String;

    fn severity_default(&self) -> Severity;

    fn cause(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct Diagnostic<K: Diagnose> {
    pub kind: K,
    pub severity: Severity,
    pub location: Option<Location>,
    pub notes: Vec<String>,
}

impl<K: Diagnose> Diagnostic<K> {
    pub fn new(kind: K) -> Self {
        let severity = kind.severity_default();
        Self {
            kind,
            severity,
            location: None,
            notes: Vec::new(),
        }
    }

    pub fn with_severity(mut self, severity: Severity) -> Self {
        self.severity = severity;
        self
    }

    pub fn with_location(mut self, location: Location) -> Self {
        self.location = Some(location);
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    pub fn map_kind<K2, F>(self, f: F) -> Diagnostic<K2>
    where
        K2: Diagnose,
        F: FnOnce(K) -> K2,
    {
        Diagnostic {
            kind: f(self.kind),
            severity: self.severity,
            location: self.location,
            notes: self.notes,
        }
    }
}

pub type Diagnostics<K> = Vec<Diagnostic<K>>;

pub fn split_by_severity<K: Diagnose>(diags: Diagnostics<K>) -> (Diagnostics<K>, Diagnostics<K>) {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    for d in diags {
        if d.severity == Severity::Error {
            errors.push(d);
        } else {
            warnings.push(d);
        }
    }
    (errors, warnings)
}

impl<K: Diagnose> fmt::Display for Diagnostic<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.severity, self.kind.message())?;
        if let Some(loc) = &self.location {
            write!(
                f,
                " (at {} [{}..{}])",
                loc.source, loc.span.start, loc.span.end
            )?;
        }
        for note in &self.notes {
            write!(f, "\n  note: {}", note)?;
        }
        Ok(())
    }
}

impl<K: Diagnose + fmt::Debug> Error for Diagnostic<K> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.kind.cause()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    enum TestKind {
        Bad,
        Warn,
    }

    impl Diagnose for TestKind {
        fn message(&self) -> String {
            match self {
                Self::Bad => "bad".into(),
                Self::Warn => "warn".into(),
            }
        }

        fn severity_default(&self) -> Severity {
            match self {
                Self::Bad => Severity::Error,
                Self::Warn => Severity::Warning,
            }
        }
    }

    fn envelope<K: Diagnose>(kind: K) -> Diagnostic<K> {
        let severity = kind.severity_default();
        Diagnostic {
            kind,
            severity,
            location: None,
            notes: Vec::new(),
        }
    }

    #[test]
    fn severity_is_non_exhaustive_match_compiles() {
        #[allow(unreachable_patterns)]
        fn label(s: Severity) -> &'static str {
            match s {
                Severity::Error => "e",
                Severity::Warning => "w",
                _ => "?",
            }
        }
        assert_eq!(label(Severity::Warning), "w");
        assert_eq!(label(Severity::Error), "e");
    }

    #[test]
    fn severity_default_resolves_via_diagnose_trait() {
        assert_eq!(envelope(TestKind::Bad).severity, Severity::Error);
        assert_eq!(envelope(TestKind::Warn).severity, Severity::Warning);
    }

    #[test]
    fn source_id_unknown_is_const_constructible() {
        const ID: SourceId = SourceId::unknown();
        assert_eq!(ID.as_str(), "<unknown>");
    }

    #[test]
    fn source_id_unknown_eq_unknown() {
        assert_eq!(SourceId::unknown(), SourceId::unknown());
    }

    #[test]
    fn source_id_hashes_consistently_with_eq() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let a = SourceId::unknown();
        let b = SourceId::unknown();
        let mut ha = DefaultHasher::new();
        let mut hb = DefaultHasher::new();
        a.hash(&mut ha);
        b.hash(&mut hb);
        assert_eq!(ha.finish(), hb.finish());
    }

    #[test]
    fn source_id_named_round_trips_label() {
        let id = SourceId::named("model.yaml");
        assert_eq!(id.as_str(), "model.yaml");
        assert_eq!(id.to_string(), "model.yaml");
    }

    #[test]
    fn source_id_named_distinguishes_by_label_and_from_unknown() {
        assert_eq!(SourceId::named("a"), SourceId::named("a"));
        assert_ne!(SourceId::named("a"), SourceId::named("b"));
        assert_ne!(SourceId::named("a"), SourceId::unknown());
    }

    #[test]
    fn span_is_half_open_byte_range() {
        let s = Span { start: 5, end: 10 };
        assert_eq!(s.start, 5);
        assert_eq!(s.end, 10);
    }

    #[test]
    fn span_unknown_is_zero_width() {
        const S: Span = Span::unknown();
        assert_eq!(S, Span { start: 0, end: 0 });
    }

    #[test]
    fn location_carries_source_and_span() {
        let loc = Location {
            source: SourceId::unknown(),
            span: Span { start: 0, end: 0 },
        };
        assert_eq!(loc.source, SourceId::unknown());
        assert_eq!(loc.span.start, 0);
    }

    #[test]
    fn diagnostic_envelope_pub_fields() {
        let d = Diagnostic::<TestKind> {
            kind: TestKind::Bad,
            severity: Severity::Error,
            location: None,
            notes: vec!["hint".into()],
        };
        assert_eq!(d.severity, Severity::Error);
        assert_eq!(d.notes.len(), 1);
    }

    #[test]
    fn diagnostic_new_seeds_severity_from_kind_default() {
        let d = Diagnostic::new(TestKind::Warn);
        assert_eq!(d.severity, Severity::Warning);
        assert!(d.location.is_none());
        assert!(d.notes.is_empty());
    }

    #[test]
    fn diagnostic_with_severity_overrides_default() {
        let d = Diagnostic::new(TestKind::Bad).with_severity(Severity::Warning);
        assert_eq!(d.severity, Severity::Warning);
    }

    #[test]
    fn diagnostic_with_location_attaches_span() {
        let loc = Location {
            source: SourceId::unknown(),
            span: Span { start: 7, end: 11 },
        };
        let d = Diagnostic::new(TestKind::Bad).with_location(loc);
        let span = d.location.expect("location attached").span;
        assert_eq!(span.start, 7);
        assert_eq!(span.end, 11);
    }

    #[test]
    fn diagnostic_with_note_appends_in_order() {
        let d = Diagnostic::new(TestKind::Bad)
            .with_note("first")
            .with_note(String::from("second"));
        assert_eq!(d.notes, vec!["first".to_string(), "second".to_string()]);
    }

    #[test]
    fn display_delegates_to_diagnose_message() {
        let d = envelope(TestKind::Bad);
        assert_eq!(format!("{}", d), "error: bad");
    }

    #[test]
    fn display_does_not_require_kind_debug() {
        let d = envelope(TestKind::Warn);
        let _ = format!("{}", d);
    }

    #[derive(Debug)]
    struct InnerCause;
    impl fmt::Display for InnerCause {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("inner")
        }
    }
    impl Error for InnerCause {}

    #[derive(Debug)]
    struct WrapKind {
        cause: InnerCause,
    }
    impl Diagnose for WrapKind {
        fn message(&self) -> String {
            "wrap".into()
        }
        fn severity_default(&self) -> Severity {
            Severity::Error
        }
        fn cause(&self) -> Option<&(dyn Error + 'static)> {
            Some(&self.cause)
        }
    }

    #[test]
    fn error_impl_chains_to_diagnose_cause() {
        let d = envelope(WrapKind { cause: InnerCause });
        let src = (&d as &dyn Error).source();
        assert!(src.is_some());
    }

    #[derive(Debug)]
    struct BareKind;
    impl Diagnose for BareKind {
        fn message(&self) -> String {
            "bare".into()
        }
        fn severity_default(&self) -> Severity {
            Severity::Error
        }
    }

    #[test]
    fn error_impl_returns_none_when_no_cause() {
        let d = envelope(BareKind);
        let src = (&d as &dyn Error).source();
        assert!(src.is_none());
    }

    #[test]
    fn diagnostics_alias_accepts_vec_methods() {
        let mut v: Diagnostics<TestKind> = vec![envelope(TestKind::Warn)];
        v.push(envelope(TestKind::Bad));
        assert!(!v.is_empty());
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn split_by_severity_partitions_and_preserves_order() {
        let diags: Diagnostics<TestKind> = vec![
            Diagnostic::new(TestKind::Bad),
            Diagnostic::new(TestKind::Warn),
            Diagnostic::new(TestKind::Bad),
        ];
        let (errors, warnings) = split_by_severity(diags);
        assert_eq!(errors.len(), 2);
        assert_eq!(warnings.len(), 1);
        assert!(errors.iter().all(|d| d.severity == Severity::Error));
        assert!(warnings.iter().all(|d| d.severity == Severity::Warning));
    }

    enum FusedKind {
        Wrapped(TestKind),
    }

    impl Diagnose for FusedKind {
        fn message(&self) -> String {
            match self {
                Self::Wrapped(k) => k.message(),
            }
        }
        fn severity_default(&self) -> Severity {
            match self {
                Self::Wrapped(k) => k.severity_default(),
            }
        }
    }

    #[test]
    fn map_kind_rewraps_kind_and_preserves_envelope_metadata() {
        let loc = Location {
            source: SourceId::unknown(),
            span: Span { start: 1, end: 4 },
        };
        let d = Diagnostic::new(TestKind::Warn)
            .with_location(loc)
            .with_note("hint");

        let fused = d.map_kind(FusedKind::Wrapped);

        assert!(matches!(fused.kind, FusedKind::Wrapped(TestKind::Warn)));
        assert_eq!(fused.severity, Severity::Warning);
        assert_eq!(fused.location.as_ref().map(|l| l.span.start), Some(1),);
        assert_eq!(fused.notes, vec!["hint".to_string()]);
    }

    #[test]
    fn diagnose_severity_default_is_overridable_by_caller() {
        let d = Diagnostic::<TestKind> {
            kind: TestKind::Bad,
            severity: Severity::Warning,
            location: None,
            notes: Vec::new(),
        };
        assert_eq!(d.severity, Severity::Warning);
    }
}
