use std::ops::ControlFlow;

use thiserror::Error;

use crate::diagnostic::{Diagnose, Severity};

pub trait Tree: Sized {
    fn children(&self) -> Vec<&Self>;

    fn with_new_children(self, new_children: Vec<Self>) -> Result<Self, ValidateError>;

    fn apply<V: Visitor<Self>>(&self, v: &mut V) -> ControlFlow<V::Output> {
        v.f_down(self)?;
        for child in self.children() {
            child.apply(v)?;
        }
        v.f_up(self)
    }

    fn pre_order(&self) -> impl Iterator<Item = &Self> {
        let mut stack = vec![self];
        std::iter::from_fn(move || {
            let node = stack.pop()?;
            stack.extend(node.children().into_iter().rev());
            Some(node)
        })
    }

    fn transform<F>(self, f: &mut F) -> Result<Self, ValidateError>
    where
        F: FnMut(Self) -> Result<Self, ValidateError>,
        Self: Clone,
    {
        let new_children: Vec<Self> = self
            .children()
            .into_iter()
            .cloned()
            .map(|child| child.transform(f))
            .collect::<Result<Vec<_>, _>>()?;
        let rebuilt = self.with_new_children(new_children)?;
        f(rebuilt)
    }
}

pub trait Visitor<N> {
    type Output;

    fn f_down(&mut self, node: &N) -> ControlFlow<Self::Output>;

    fn f_up(&mut self, node: &N) -> ControlFlow<Self::Output>;
}

pub trait Rewriter<N> {
    fn f_down(&mut self, node: N) -> Result<N, ValidateError>;

    fn f_up(&mut self, node: N) -> Result<N, ValidateError>;
}

#[non_exhaustive]
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ValidateError {
    #[error("with_new_children: child count mismatch — expected {expected}, got {got}")]
    ChildCountMismatch { expected: usize, got: usize },

    #[error("InList requires at least one list element")]
    EmptyInList,

    #[error("Case requires at least one when-branch")]
    EmptyCase,

    #[error("Name cannot be empty")]
    EmptyName,

    #[error("{reason}")]
    CallRefused { reason: String },

    #[error("{reason}")]
    NodeRefused { reason: String },
}

impl Diagnose for ValidateError {
    fn message(&self) -> String {
        format!("{}", self)
    }

    fn severity_default(&self) -> Severity {
        Severity::Error
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct MockTree {
        tag: i32,
        children: Vec<MockTree>,
    }

    impl MockTree {
        fn leaf(tag: i32) -> Self {
            Self {
                tag,
                children: Vec::new(),
            }
        }

        fn parent(tag: i32, children: Vec<Self>) -> Self {
            Self { tag, children }
        }
    }

    impl Tree for MockTree {
        fn children(&self) -> Vec<&Self> {
            self.children.iter().collect()
        }

        fn with_new_children(self, new_children: Vec<Self>) -> Result<Self, ValidateError> {
            if new_children.len() != self.children.len() {
                return Err(ValidateError::ChildCountMismatch {
                    expected: self.children.len(),
                    got: new_children.len(),
                });
            }
            Ok(Self {
                tag: self.tag,
                children: new_children,
            })
        }
    }

    struct TagCollector {
        tags: Vec<i32>,
    }

    impl Visitor<MockTree> for TagCollector {
        type Output = ();

        fn f_down(&mut self, node: &MockTree) -> ControlFlow<Self::Output> {
            self.tags.push(node.tag);
            ControlFlow::Continue(())
        }

        fn f_up(&mut self, _node: &MockTree) -> ControlFlow<Self::Output> {
            ControlFlow::Continue(())
        }
    }

    struct NodeCounter {
        seen: usize,
    }

    impl Visitor<MockTree> for NodeCounter {
        type Output = ();

        fn f_down(&mut self, _node: &MockTree) -> ControlFlow<Self::Output> {
            self.seen += 1;
            ControlFlow::Continue(())
        }

        fn f_up(&mut self, _node: &MockTree) -> ControlFlow<Self::Output> {
            ControlFlow::Continue(())
        }
    }

    fn sample_tree() -> MockTree {
        MockTree::parent(
            1,
            vec![
                MockTree::parent(2, vec![MockTree::leaf(4)]),
                MockTree::leaf(3),
            ],
        )
    }

    #[test]
    fn apply_visits_pre_order_parent_before_children() {
        let tree = sample_tree();
        let mut collector = TagCollector { tags: Vec::new() };
        let _ = tree.apply(&mut collector);
        assert_eq!(collector.tags, vec![1, 2, 4, 3]);
    }

    #[test]
    fn pre_order_gives_each_node_parent_first_in_children_order() {
        let tags: Vec<i32> = sample_tree().pre_order().map(|node| node.tag).collect();
        assert_eq!(tags, vec![1, 2, 4, 3]);
    }

    #[test]
    fn apply_counts_every_node_via_visitor() {
        let tree = sample_tree();
        let mut counter = NodeCounter { seen: 0 };
        let _ = tree.apply(&mut counter);
        assert_eq!(counter.seen, 4);
    }

    #[test]
    fn transform_identity_preserves_tree() {
        let tree = sample_tree();
        let original = tree.clone();
        let mut identity = |t: MockTree| -> Result<MockTree, ValidateError> { Ok(t) };
        let rebuilt = tree.transform(&mut identity).expect("identity rewrite");
        assert_eq!(rebuilt, original);
    }

    #[test]
    fn transform_increments_each_tag() {
        let tree = sample_tree();
        let mut bump = |t: MockTree| -> Result<MockTree, ValidateError> {
            Ok(MockTree {
                tag: t.tag + 10,
                children: t.children,
            })
        };
        let rebuilt = tree.transform(&mut bump).expect("bump rewrite");
        let expected = MockTree::parent(
            11,
            vec![
                MockTree::parent(12, vec![MockTree::leaf(14)]),
                MockTree::leaf(13),
            ],
        );
        assert_eq!(rebuilt, expected);
    }

    #[test]
    fn transform_propagates_callback_error() {
        let tree = sample_tree();
        let mut explode = |t: MockTree| -> Result<MockTree, ValidateError> {
            if t.tag == 4 {
                Err(ValidateError::ChildCountMismatch {
                    expected: 0,
                    got: 99,
                })
            } else {
                Ok(t)
            }
        };
        let result = tree.transform(&mut explode);
        match result {
            Err(ValidateError::ChildCountMismatch { expected, got }) => {
                assert_eq!(expected, 0);
                assert_eq!(got, 99);
            }
            other => panic!("expected ChildCountMismatch, got {:?}", other),
        }
    }

    #[test]
    fn with_new_children_rejects_arity_mismatch() {
        let parent = MockTree::parent(1, vec![MockTree::leaf(2), MockTree::leaf(3)]);
        let result = parent.with_new_children(vec![MockTree::leaf(99)]);
        match result {
            Err(ValidateError::ChildCountMismatch { expected, got }) => {
                assert_eq!(expected, 2);
                assert_eq!(got, 1);
            }
            other => panic!("expected ChildCountMismatch, got {:?}", other),
        }
    }

    #[test]
    fn validate_error_displays_child_count_mismatch() {
        let err = ValidateError::ChildCountMismatch {
            expected: 2,
            got: 3,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("child count mismatch"));
        assert!(msg.contains("expected 2"));
        assert!(msg.contains("got 3"));
    }

    #[test]
    fn validate_error_displays_empty_in_list() {
        let err = ValidateError::EmptyInList;
        let msg = format!("{}", err);
        assert!(msg.contains("InList"));
        assert!(msg.contains("at least one"));
    }

    #[test]
    fn validate_error_displays_empty_case() {
        let err = ValidateError::EmptyCase;
        let msg = format!("{}", err);
        assert!(msg.contains("Case"));
        assert!(msg.contains("at least one"));
    }

    #[test]
    fn validate_error_implements_diagnose() {
        let err = ValidateError::EmptyCase;
        assert!(!err.message().is_empty());
        assert_eq!(err.severity_default(), Severity::Error);
        assert!(err.cause().is_none());
    }

    #[test]
    fn validate_error_displays_empty_name() {
        let err = ValidateError::EmptyName;
        let msg = format!("{}", err);
        assert!(msg.contains("Name"));
        assert!(msg.contains("empty"));
    }
}
