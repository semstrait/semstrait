use std::mem;

use semstrait_common::EntityId;
use semstrait_model::{
    AiContext, DataKindVariant, Dataset, Grainset, GrainsetBody, Joinset, JoinsetBody,
    NestedDataset, NestedGrainset, NestedJoinset, NestedUnionset, SemanticInterface, SemanticModel,
    TemporalShape, Unionset, UnionsetBody,
};

enum Child {
    Dataset(NestedDataset),
    Unionset(NestedUnionset),
    Grainset(NestedGrainset),
    Joinset(NestedJoinset),
}

trait EndingDataKind {
    fn ending_data_kind(&self) -> DataKindVariant;
}

impl EndingDataKind for NestedDataset {
    fn ending_data_kind(&self) -> DataKindVariant {
        DataKindVariant::Dataset
    }
}

macro_rules! ending_data_kind {
    ($ty:ty, $variant:ident) => {
        impl EndingDataKind for $ty {
            fn ending_data_kind(&self) -> DataKindVariant {
                self.body
                    .sole_child_ending_data_kind()
                    .unwrap_or(DataKindVariant::$variant)
            }
        }
    };
}

ending_data_kind!(NestedUnionset, Unionset);
ending_data_kind!(NestedGrainset, Grainset);
ending_data_kind!(NestedJoinset, Joinset);

struct Declaration {
    name: String,
    id: Option<EntityId>,
    temporal: Option<TemporalShape>,
}

struct RootShell {
    description: Option<String>,
    ai_context: Option<AiContext>,
    semantic_interface: SemanticInterface,
}

trait Composition {
    fn is_sugar_with_one_child(&self) -> bool;
    fn take_children(&mut self) -> Vec<Child>;
    fn place(&mut self, child: Child);
    fn holds(&self, data_kind: DataKindVariant) -> bool;
    fn sole_child_ending_data_kind(&self) -> Option<DataKindVariant>;
}

macro_rules! composition {
    ($body:ty, $($field:ident => $variant:ident),+) => {
        impl Composition for $body {
            fn is_sugar_with_one_child(&self) -> bool {
                self.minimum_children() == 1
            }

            fn take_children(&mut self) -> Vec<Child> {
                let mut taken = Vec::new();
                $(taken.extend(mem::take(&mut self.$field).into_iter().map(Child::$variant));)+
                taken
            }

            fn place(&mut self, child: Child) {
                match child {
                    $(Child::$variant(child) => self.$field.push(child),)+
                    _ => unreachable!("a composition places only the kinds it holds"),
                }
            }

            fn holds(&self, data_kind: DataKindVariant) -> bool {
                matches!(data_kind, $(DataKindVariant::$variant)|+)
            }

            fn sole_child_ending_data_kind(&self) -> Option<DataKindVariant> {
                if !self.is_sugar_with_one_child() {
                    return None;
                }
                let mut kinds = Vec::new();
                $(kinds.extend(self.$field.iter().map(EndingDataKind::ending_data_kind));)+
                match kinds.as_slice() {
                    [only] => Some(*only),
                    _ => None,
                }
            }
        }
    };
}

composition!(
    GrainsetBody,
    datasets => Dataset,
    unionsets => Unionset,
    joinsets => Joinset
);
composition!(
    UnionsetBody,
    datasets => Dataset,
    grainsets => Grainset,
    joinsets => Joinset
);
composition!(
    JoinsetBody,
    datasets => Dataset,
    grainsets => Grainset,
    unionsets => Unionset
);

macro_rules! with_base {
    ($child:expr, |$base:ident| $body:expr) => {
        match $child {
            Child::Dataset(child) => {
                let $base = &mut child.body.base;
                $body
            }
            Child::Unionset(child) => {
                let $base = &mut child.body.base;
                $body
            }
            Child::Grainset(child) => {
                let $base = &mut child.body.base;
                $body
            }
            Child::Joinset(child) => {
                let $base = &mut child.body.base;
                $body
            }
        }
    };
}

impl Child {
    fn composition(&mut self) -> Option<&mut dyn Composition> {
        match self {
            Self::Dataset(_) => None,
            Self::Unionset(child) => Some(&mut child.body),
            Self::Grainset(child) => Some(&mut child.body),
            Self::Joinset(child) => Some(&mut child.body),
        }
    }

    fn take_declaration(&mut self) -> Declaration {
        with_base!(self, |base| Declaration {
            name: mem::take(&mut base.name),
            id: base.id.take(),
            temporal: base.extras.temporal.take(),
        })
    }

    fn adopt(&mut self, declaration: Declaration) {
        with_base!(self, |base| {
            base.name = declaration.name;
            base.id = declaration.id;
            base.extras.temporal = base.extras.temporal.take().or(declaration.temporal);
        });
    }

    fn ending_data_kind(&self) -> DataKindVariant {
        match self {
            Self::Dataset(child) => child.ending_data_kind(),
            Self::Unionset(child) => child.ending_data_kind(),
            Self::Grainset(child) => child.ending_data_kind(),
            Self::Joinset(child) => child.ending_data_kind(),
        }
    }

    fn composition_ref(&self) -> Option<&dyn Composition> {
        match self {
            Self::Dataset(_) => None,
            Self::Unionset(child) => Some(&child.body),
            Self::Grainset(child) => Some(&child.body),
            Self::Joinset(child) => Some(&child.body),
        }
    }

    fn desugared(mut self, parent: Option<&dyn Composition>) -> Self {
        let Some(composition) = self.composition() else {
            return self;
        };
        let is_sugar = composition.is_sugar_with_one_child();
        let taken = composition.take_children();
        let collapses = is_sugar
            && taken.len() == 1
            && parent.is_none_or(|parent| parent.holds(taken[0].ending_data_kind()));
        let below = if collapses {
            parent
        } else {
            self.composition_ref()
        };
        let mut children: Vec<Self> = taken
            .into_iter()
            .map(|child| child.desugared(below))
            .collect();
        if collapses && children.len() == 1 {
            let mut only = children.remove(0);
            only.adopt(self.take_declaration());
            return only;
        }
        if let Some(composition) = self.composition() {
            for child in children {
                composition.place(child);
            }
        }
        self
    }
}

macro_rules! root_of {
    ($ty:ident, $child:expr, $shell:expr) => {{
        let mut root = $ty::builder(String::new()).build();
        root.body = $child.body;
        root.description = $shell.description;
        root.ai_context = $shell.ai_context;
        root.semantic_interface = $shell.semantic_interface;
        root
    }};
}

macro_rules! desugar_roots {
    ($model:ident, $map:ident, $variant:ident, $nested:ident) => {
        for (key, root) in mem::take(&mut $model.$map).into_inner() {
            let shell = RootShell {
                description: root.description,
                ai_context: root.ai_context,
                semantic_interface: root.semantic_interface,
            };
            let mut nested = $nested::builder(String::new()).build();
            nested.body = root.body;
            match Child::$variant(nested).desugared(None) {
                Child::Dataset(child) => {
                    $model.datasets.insert(key, root_of!(Dataset, child, shell));
                }
                Child::Unionset(child) => {
                    $model
                        .unionsets
                        .insert(key, root_of!(Unionset, child, shell));
                }
                Child::Grainset(child) => {
                    $model
                        .grainsets
                        .insert(key, root_of!(Grainset, child, shell));
                }
                Child::Joinset(child) => {
                    $model.joinsets.insert(key, root_of!(Joinset, child, shell));
                }
            }
        }
    };
}

pub(crate) fn desugar(mut model: SemanticModel) -> SemanticModel {
    desugar_roots!(model, unionsets, Unionset, NestedUnionset);
    desugar_roots!(model, grainsets, Grainset, NestedGrainset);
    desugar_roots!(model, joinsets, Joinset, NestedJoinset);
    model
}
