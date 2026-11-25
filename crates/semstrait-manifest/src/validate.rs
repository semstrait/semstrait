use std::collections::{BTreeMap, BTreeSet};

use semstrait_common::diagnostic::{Diagnostic, Diagnostics, Severity};
use semstrait_common::EntityId;
use semstrait_sql::expr::Expr;

use crate::compile::assemble::{
    condition_handles, detect_operand_cycle, originators_agree_with_operands,
    originators_within_reservation, ORIGINATORS_RULE, ORIGINATOR_AGREEMENT_RULE,
};
use crate::compile::catalog::partitions::{folder_key_values, partition_folder};
use crate::compile::qualifiers::misplaced_columns;
use crate::compile::relationships::unserved_condition_leaves;
use crate::compile::seal::orphan_expressions;
use crate::error::{CompileError, ReferenceSite, ReferenceTargetKind};
use crate::manifest::bitmap::SemanticBitmask;
use crate::manifest::coverage::Coverage;
use crate::manifest::expression::{ExprRoot, Link, LinkStep, ManifestExpressions, PhysicalExprId};
use crate::manifest::expression_layer::ExprLayer;
use crate::manifest::final_values::{final_value_layer, layer_of_final_value, ChildPools};
use crate::manifest::physical_source::{SchemaProvenance, StorageTarget, Uri};
use crate::manifest::semantic_definition::SemanticDefinition;
use crate::manifest::semantic_source::{RoleDefinition, Selector, SemanticSource, SourceRole};
use crate::manifest::serving::Serving;
use crate::manifest::source_alias::SourceAliases;
use crate::manifest::top::{SemanticManifest, MANIFEST_DIALECT};
use semstrait_ir::{JoinsetJoinType, MultisetOp, SemanticRole};

const GRAINSET_OWNS_UNMAPPED_METRICS_RULE: &str =
    "a Grainset owns a signature only for a Metric that no child maps and that is not row-level";

const GRAINSET_OWN_COLUMNS_RULE: &str = "a Grainset's own signature reads, as a column under its \
                                         own alias, only a name that one of its children serves; a \
                                         Metric that the Grainset computes is read through a link";

const GRAINSET_MASK_AGREEMENT_RULE: &str = "a Grainset's semantic_bitmask must set the \
                                            bit_positions of its listed signatures' definitions, \
                                            and no bit that neither these nor its grains' \
                                            semantic_bitmasks set";

const DATASET_SOURCE_RULE: &str = "a Dataset's SemanticSignature trees must name the \
                                  Dataset's physical source among their consuming \
                                  physical sources";

const LAYER_RULE: &str = "ManifestExpression.layer must equal the layer of its tree, each \
                          link counted as the layer of the entry it links";

const CROSS_CONDITION_RULE: &str = "Joinset.on_expr_id is absent iff Joinset.join_type is cross";

const JOINSET_CONDITION_LINK_RULE: &str = "a Joinset source's condition and its relationship \
                                           filter hold no link: each reads the operands' output \
                                           fields";
const COMPOSED_SIGNATURE_LINK_RULE: &str = "a composition's Dimension or Measure signature's \
                                            expression entry holds no link: it reads its \
                                            operand's output";
const COMPOSED_FILTER_LINK_RULE: &str = "a Joinset's or a Grainset's signature filter entry \
                                         holds no link: a Joinset's holds a copy for each \
                                         operand, and a Grainset's links no Dimension";
const JOINSET_NAMED_FILTER_LINK_RULE: &str = "a Joinset's named Filter signature's expression \
                                              entry holds no link: it holds a copy for each \
                                              operand";

const LINK_TARGET_RULE: &str = "a link in a signature's entry names the expression entry of \
                                a signature that the same semantic source lists";

const LINK_CYCLE_RULE: &str = "no entry of the expression store is reached again by its own \
                               links";

const NORMAL_FORM_RULE: &str = "every URI a path source or a selector stores is in normal \
                                form, and a base path names a folder";
const FIXED_ORDER_RULE: &str = "a listing and a Uris selector are sets sorted by URI";
const FORMAT_RULE: &str = "a path source is read as Parquet, and a table source is an \
                           Iceberg table";
const SELECTOR_RULE: &str = "a Table selector reads a table source, and a Prefix, Pattern or \
                             Uris selector a path source";
const NAMED_OBJECT_RULE: &str = "each URI of a Uris selector is an object of its physical \
                                 source's listing";
const BASE_PATH_RULE: &str = "every object a path source lists lies beneath its base path, and \
                              a Pattern selector's glob names objects directly under it";
const DIRECT_OBJECTS_RULE: &str = "objects directly under a base path are in no partition, and a \
                                   stored partition_def holds at least one";
const PARTITION_ORDER_RULE: &str = "partitions are sorted by partition folder, the parent folder \
                                    of the objects of each";
const PARTITION_LISTING_RULE: &str = "a partition's listing is a set sorted by URI of objects \
                                      of its path source's listing, each in the partition folder";
const KEY_VALUES_RULE: &str = "a partition's key_values are its folder's keys and values, in \
                               folder order, each key once and neither side empty";
const KEY_SEQUENCE_RULE: &str = "every partition of a path source has one key sequence";
const READ_ONCE_RULE: &str = "the sources of one Dataset have distinct base paths, and no \
                              object is selected by two of them";

fn is_sorted_set<T: PartialOrd>(items: &[T]) -> bool {
    items.is_sorted_by(|a, b| a < b)
}

fn link_free_rule(
    source: SourceRole,
    marker: ExprRoot,
    role: Option<SemanticRole>,
) -> Option<&'static str> {
    use ExprRoot::{JoinsetFilter, JoinsetOn, SignatureExpression, SignatureFilter};
    use SemanticRole::{Dimension, Filter, Measure, Metric};
    use SourceRole::{Dataset, Grainset, Joinset, Unionset};
    match (source, marker, role) {
        (_, JoinsetOn | JoinsetFilter, _) => Some(JOINSET_CONDITION_LINK_RULE),
        (Joinset | Unionset | Grainset, SignatureExpression, Some(Dimension | Measure)) => {
            Some(COMPOSED_SIGNATURE_LINK_RULE)
        }
        (Joinset, SignatureExpression, Some(Filter)) => Some(JOINSET_NAMED_FILTER_LINK_RULE),
        (Joinset | Grainset, SignatureFilter, _) => Some(COMPOSED_FILTER_LINK_RULE),
        (Dataset, SignatureExpression, Some(Dimension | Measure | Metric | Filter) | None)
        | (
            Dataset | Unionset,
            SignatureFilter,
            Some(Dimension | Measure | Metric | Filter) | None,
        )
        | (Unionset | Grainset, SignatureExpression, Some(Filter))
        | (Joinset | Unionset | Grainset, SignatureExpression, Some(Metric) | None) => None,
        other => unreachable!("semstrait-ir added a SemanticRole variant: {other:?}"),
    }
}

fn cycle_closing_links(store: &ManifestExpressions) -> usize {
    (store.walk_links(store.physical.keys().copied(), |entry| entry.expr.links()))
        .filter(|step| matches!(step, LinkStep::Closes))
        .count()
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum FaultReporting {
    #[default]
    EveryFault,
    FirstFault,
}

struct Halt;

struct Walk<'a> {
    reporting: FaultReporting,
    faults: Diagnostics<CompileError>,
    occupied: &'a BTreeSet<u32>,
}

impl<'a> Walk<'a> {
    fn new(reporting: FaultReporting, occupied: &'a BTreeSet<u32>) -> Self {
        Self {
            reporting,
            faults: Vec::new(),
            occupied,
        }
    }

    fn report(&mut self, fault: CompileError) -> Result<(), Halt> {
        self.faults.push(Diagnostic::new(fault));
        match self.reporting {
            FaultReporting::EveryFault => Ok(()),
            FaultReporting::FirstFault => Err(Halt),
        }
    }

    fn finish(self) -> Result<Diagnostics<CompileError>, Diagnostics<CompileError>> {
        if self
            .faults
            .iter()
            .any(|fault| fault.severity == Severity::Error)
        {
            Err(self.faults)
        } else {
            Ok(self.faults)
        }
    }
}

fn dangling(
    from: ReferenceSite,
    to_kind: ReferenceTargetKind,
    target_id: impl Into<String>,
) -> CompileError {
    CompileError::DanglingReference {
        from,
        to_kind,
        target_id: target_id.into(),
    }
}

fn structural(site: ReferenceSite, rule: &'static str) -> CompileError {
    CompileError::StructuralViolation { site, rule }
}

impl SemanticManifest {
    pub fn validate(&self) -> Result<Diagnostics<CompileError>, Diagnostics<CompileError>> {
        self.validate_with(FaultReporting::EveryFault)
    }

    pub fn validate_with(
        &self,
        reporting: FaultReporting,
    ) -> Result<Diagnostics<CompileError>, Diagnostics<CompileError>> {
        let occupied: BTreeSet<u32> = self
            .semantic_definitions
            .entries
            .values()
            .map(|def| def.bit_position)
            .collect();

        let mut walk = Walk::new(reporting, &occupied);
        let _ = self.traverse(&mut walk);
        walk.finish()
    }

    fn traverse(&self, walk: &mut Walk<'_>) -> Result<(), Halt> {
        if self.dialect != MANIFEST_DIALECT {
            let dialect = self.dialect.to_string();
            walk.report(CompileError::ManifestDialect { dialect })?;
        }
        self.check_id_equals_key(walk)?;
        Self::check_bit_position_dense(self.semantic_definitions.entries.len(), walk)?;
        self.check_global_id_uniqueness(walk)?;

        let aliases = SourceAliases::derive(
            self.semantic_sources
                .iter()
                .map(|(id, entry)| (id, entry.role_definition.source_role())),
            self.sources.keys(),
        );
        self.check_signatures(walk)?;
        self.check_semantic_sources(&aliases, walk)?;
        self.check_keysets(walk)?;
        self.check_sources(&aliases, walk)?;
        self.check_read_once(walk)?;
        self.check_reduction_carry_over(walk)?;
        self.check_reduction_leaf_signatures(walk)?;
        self.check_expression_pool(walk)?;
        self.check_links(walk)?;
        self.check_column_qualifiers(&aliases, walk)?;
        self.check_condition_leaves(walk)
    }

    fn check_id_equals_key(&self, walk: &mut Walk<'_>) -> Result<(), Halt> {
        const RULE: &str = "entity id must equal its map key";

        for (key, def) in &self.semantic_definitions.entries {
            if &def.id != key {
                walk.report(structural(ReferenceSite::Semantics(key.clone()), RULE))?;
            }
        }
        for (key, signature) in &self.semantic_signatures {
            if &signature.id != key {
                walk.report(structural(ReferenceSite::Signature(key.clone()), RULE))?;
            }
        }
        for (key, entry) in &self.semantic_sources {
            if &entry.id != key {
                walk.report(structural(ReferenceSite::SemanticSource(key.clone()), RULE))?;
            }
        }
        for (key, source) in &self.sources {
            if &source.id != key {
                walk.report(structural(ReferenceSite::Source(key.clone()), RULE))?;
            }
        }
        for (key, keyset) in &self.keysets {
            if &keyset.id != key {
                walk.report(structural(ReferenceSite::Keyset(key.clone()), RULE))?;
            }
        }
        Ok(())
    }

    fn check_bit_position_dense(registry_len: usize, walk: &mut Walk<'_>) -> Result<(), Halt> {
        let dense: BTreeSet<u32> = (0..u32::try_from(registry_len).unwrap_or(u32::MAX)).collect();
        if *walk.occupied != dense {
            walk.report(structural(
                ReferenceSite::Bitmap,
                "SemanticDefinition.bit_position values must be exactly 0..n over the registry",
            ))?;
        }
        Ok(())
    }

    // TODO: check that each id is canonical UUID text once compile() mints UUIDv5 ids;
    // only global uniqueness and id == key are checked now.
    fn check_global_id_uniqueness(&self, walk: &mut Walk<'_>) -> Result<(), Halt> {
        const RULE: &str = "entity ids must be globally unique across collections";
        let mut seen: BTreeSet<&str> = BTreeSet::new();

        seen.insert(self.model_id.as_str());
        for def in self.semantic_definitions.entries.values() {
            if !seen.insert(def.id.as_str()) {
                walk.report(structural(ReferenceSite::Semantics(def.id.clone()), RULE))?;
            }
        }
        for signature in self.semantic_signatures.values() {
            if !seen.insert(signature.id.as_str()) {
                walk.report(structural(
                    ReferenceSite::Signature(signature.id.clone()),
                    RULE,
                ))?;
            }
        }
        for entry in self.semantic_sources.values() {
            if !seen.insert(entry.id.as_str()) {
                walk.report(structural(
                    ReferenceSite::SemanticSource(entry.id.clone()),
                    RULE,
                ))?;
            }
        }
        for source in self.sources.values() {
            if !seen.insert(source.id.as_str()) {
                walk.report(structural(ReferenceSite::Source(source.id.clone()), RULE))?;
            }
        }
        for keyset in self.keysets.values() {
            if !seen.insert(keyset.id.as_str()) {
                walk.report(structural(ReferenceSite::Keyset(keyset.id.clone()), RULE))?;
            }
        }
        Ok(())
    }

    fn check_signatures(&self, walk: &mut Walk<'_>) -> Result<(), Halt> {
        const OWNERSHIP_RULE: &str =
            "every SemanticSignature must be listed by exactly one semantic_sources entry";
        const AGREEMENT_RULE: &str = "a reading source's semantic_bitmask must set exactly the \
                                      bit_positions of its listed signatures' definitions";
        const ONE_PER_BIT_RULE: &str = "a reading source must list exactly one SemanticSignature \
                                        per set bit of its semantic_bitmask";
        const FILTER_ROW_LEVEL_RULE: &str = "a SemanticSignature's filter entry must be \
             row-level: its layer, with each link counted, is scalar";
        const FILTER_DISJOINT_RULE: &str =
            "a SemanticSignature's filter entry must not be its expression entry";
        const FILTER_ROLE_RULE: &str =
            "only a Measure's or a Metric's SemanticSignature may carry a filter";

        let mut owner_counts: BTreeMap<&EntityId, usize> = BTreeMap::new();
        for entry in self.semantic_sources.values() {
            for signature_id in &entry.semantic_signature_ids {
                *owner_counts.entry(signature_id).or_insert(0) += 1;
            }
        }

        for signature in self.semantic_signatures.values() {
            let site = || ReferenceSite::Signature(signature.id.clone());
            if !self
                .semantic_definitions
                .entries
                .contains_key(&signature.definition_id)
            {
                walk.report(dangling(
                    site(),
                    ReferenceTargetKind::Semantics,
                    signature.definition_id.clone(),
                ))?;
            }
            if self.expressions.get(&signature.expression_id).is_none() {
                walk.report(dangling(
                    site(),
                    ReferenceTargetKind::Expressions,
                    signature.expression_id.to_string(),
                ))?;
            }
            if let Some(filter) = signature.filter_expr_id {
                if self.expressions.get(&filter).is_none() {
                    walk.report(dangling(
                        site(),
                        ReferenceTargetKind::Expressions,
                        filter.to_string(),
                    ))?;
                }
                let layer = (self.expressions.get(&filter))
                    .and_then(|entry| self.expressions.layer_of(entry.expr.expr()));
                if layer.is_some_and(|layer| layer != ExprLayer::Scalar) {
                    walk.report(structural(site(), FILTER_ROW_LEVEL_RULE))?;
                }
                if filter == signature.expression_id {
                    walk.report(structural(site(), FILTER_DISJOINT_RULE))?;
                }
                if let Some(definition) = self
                    .semantic_definitions
                    .entries
                    .get(&signature.definition_id)
                {
                    if !matches!(
                        definition.role_definition.role(),
                        SemanticRole::Measure | SemanticRole::Metric
                    ) {
                        walk.report(structural(site(), FILTER_ROLE_RULE))?;
                    }
                }
            }
            for keyset_id in &signature.keysets {
                if !self.keysets.contains_key(keyset_id) {
                    walk.report(dangling(
                        site(),
                        ReferenceTargetKind::Keysets,
                        keyset_id.clone(),
                    ))?;
                }
            }
            let owners = owner_counts.get(&signature.id).copied().unwrap_or(0);
            if owners != 1 {
                walk.report(structural(site(), OWNERSHIP_RULE))?;
            }
        }

        for entry in self.semantic_sources.values() {
            let site = || ReferenceSite::SemanticSource(entry.id.clone());
            let mut listed: BTreeMap<u32, usize> = BTreeMap::new();
            for signature_id in &entry.semantic_signature_ids {
                let Some(signature) = self.semantic_signatures.get(signature_id) else {
                    walk.report(dangling(
                        site(),
                        ReferenceTargetKind::Signatures,
                        signature_id.clone(),
                    ))?;
                    continue;
                };
                if let Some(definition) = self
                    .semantic_definitions
                    .entries
                    .get(&signature.definition_id)
                {
                    *listed.entry(definition.bit_position).or_insert(0) += 1;
                }
                if let RoleDefinition::Dataset { source_id, .. } = &entry.role_definition {
                    if self.sources.contains_key(source_id) {
                        for root in [Some(signature.expression_id), signature.filter_expr_id]
                            .into_iter()
                            .flatten()
                        {
                            if let Some(pooled) = self.expressions.get(&root) {
                                if !pooled.physical_sources.contains(source_id) {
                                    walk.report(structural(site(), DATASET_SOURCE_RULE))?;
                                }
                            }
                        }
                    }
                }
            }
            let set_bits: BTreeSet<u32> = self
                .semantic_definitions
                .entries
                .values()
                .filter(|definition| entry.semantic_bitmask.contains(definition.bit_position))
                .map(|definition| definition.bit_position)
                .collect();
            let listed_bits: BTreeSet<u32> = listed.keys().copied().collect();
            match &entry.role_definition {
                RoleDefinition::Dataset { .. }
                | RoleDefinition::Joinset { .. }
                | RoleDefinition::Unionset { .. } => {
                    if listed_bits != set_bits {
                        walk.report(structural(site(), AGREEMENT_RULE))?;
                    }
                    if listed.values().any(|count| *count > 1) {
                        walk.report(structural(site(), ONE_PER_BIT_RULE))?;
                    }
                }
                RoleDefinition::Grainset {
                    grains: children, ..
                } => {
                    if !self.grainset_owns_only_unmapped_metrics(entry, children) {
                        walk.report(structural(site(), GRAINSET_OWNS_UNMAPPED_METRICS_RULE))?;
                    }
                    if listed.values().any(|count| *count > 1) {
                        walk.report(structural(site(), ONE_PER_BIT_RULE))?;
                    }
                    if !self.grainset_reads_only_served_columns(entry, children) {
                        walk.report(structural(site(), GRAINSET_OWN_COLUMNS_RULE))?;
                    }
                    let backed = |bit: &u32| {
                        listed_bits.contains(bit) || self.some_grain_sets(children, *bit)
                    };
                    if !listed_bits.is_subset(&set_bits) || !set_bits.iter().all(backed) {
                        walk.report(structural(site(), GRAINSET_MASK_AGREEMENT_RULE))?;
                    }
                }
            }
        }
        Ok(())
    }

    fn check_semantic_sources(
        &self,
        aliases: &SourceAliases,
        walk: &mut Walk<'_>,
    ) -> Result<(), Halt> {
        if let Some(cycle) = detect_operand_cycle(&self.semantic_sources) {
            walk.report(CompileError::CyclicSemanticSource { cycle })?;
        }
        for ((id, entry), expected) in self.semantic_sources.iter().zip(aliases.semantic.values()) {
            if &entry.alias != expected {
                walk.report(CompileError::SourceAliasMismatch {
                    semantic_source: id.clone(),
                    expected: expected.clone(),
                    found: entry.alias.clone(),
                })?;
            }
            Self::check_bitmask(&entry.semantic_bitmask, &entry.id, walk)?;

            if let Some(shape) = entry.temporal.as_ref() {
                self.check_temporal_refs(shape, &entry.id, walk)?;
            }

            self.check_temporal_agreement(entry, walk)?;
            self.check_role_definition(entry, walk)?;
            self.check_originators(entry, walk)?;
        }
        for error in crate::compile::temporal::distinct_grains(
            &self.semantic_sources,
            &self.semantic_definitions,
        ) {
            walk.report(error)?;
        }
        Ok(())
    }

    fn check_role_definition(
        &self,
        entry: &SemanticSource,
        walk: &mut Walk<'_>,
    ) -> Result<(), Halt> {
        let site = || ReferenceSite::SemanticSource(entry.id.clone());
        match &entry.role_definition {
            RoleDefinition::Dataset {
                source_id,
                selector,
            } => match self.sources.get(source_id) {
                Some(source) => {
                    Self::check_selector(selector, &source.storage.target, &site, walk)?;
                }
                None => {
                    walk.report(dangling(site(), ReferenceTargetKind::Sources, source_id))?;
                }
            },
            RoleDefinition::Unionset { branches, mode, .. } => {
                let (minimum, rule) = match mode {
                    MultisetOp::UnionAll => (
                        1,
                        "Unionset.branches must have at least 1 operand (len >= 1)",
                    ),
                    _ => (
                        2,
                        "Unionset.branches must have at least 2 operands (len >= 2)",
                    ),
                };
                if branches.len() < minimum {
                    walk.report(structural(site(), rule))?;
                }
            }
            RoleDefinition::Grainset { grains, .. } => {
                if grains.is_empty() {
                    walk.report(structural(
                        site(),
                        "Grainset.grains must have at least 1 operand (len >= 1)",
                    ))?;
                }
            }
            RoleDefinition::Joinset {
                on_expr_id,
                filter_expr_id,
                join_type,
                cardinality,
                cross_filter,
                ..
            } => {
                if let Err(error) = crate::compile::relationships::guard_cardinality_pair(
                    entry.name.as_str(),
                    *cardinality,
                    *cross_filter,
                ) {
                    walk.report(error)?;
                }
                if on_expr_id.is_none() != matches!(join_type, JoinsetJoinType::Cross) {
                    walk.report(structural(site(), CROSS_CONDITION_RULE))?;
                }
                for id in on_expr_id.iter().chain(filter_expr_id) {
                    if !self.expressions.physical.contains_key(id) {
                        walk.report(dangling(
                            site(),
                            ReferenceTargetKind::Expressions,
                            id.to_string(),
                        ))?;
                    }
                }
            }
        }
        for operand in entry.role_definition.operands() {
            if !self.semantic_sources.contains_key(operand) {
                walk.report(dangling(
                    site(),
                    ReferenceTargetKind::SemanticSources,
                    operand,
                ))?;
            }
        }
        Ok(())
    }

    fn check_selector(
        selector: &Selector,
        target: &StorageTarget,
        site: &impl Fn() -> ReferenceSite,
        walk: &mut Walk<'_>,
    ) -> Result<(), Halt> {
        if !selector.reads(target) {
            walk.report(structural(site(), SELECTOR_RULE))?;
        }
        match selector {
            Selector::Pattern(glob) => {
                if !glob.is_normal() {
                    walk.report(structural(site(), NORMAL_FORM_RULE))?;
                }
                if let StorageTarget::Path { base_path, .. } = target {
                    if (glob.beneath(base_path)).is_none_or(|object| object.contains('/')) {
                        walk.report(structural(site(), BASE_PATH_RULE))?;
                    }
                }
            }
            Selector::Uris(named) => {
                if !named.iter().all(Uri::is_normal) {
                    walk.report(structural(site(), NORMAL_FORM_RULE))?;
                }
                if !is_sorted_set(named) {
                    walk.report(structural(site(), FIXED_ORDER_RULE))?;
                }
                if let StorageTarget::Path { listings, .. } = target {
                    let listed: BTreeSet<&Uri> = listings.iter().collect();
                    if !named.iter().all(|uri| listed.contains(uri)) {
                        walk.report(structural(site(), NAMED_OBJECT_RULE))?;
                    }
                }
            }
            Selector::Prefix | Selector::Table => {}
        }
        Ok(())
    }

    fn check_originators(&self, entry: &SemanticSource, walk: &mut Walk<'_>) -> Result<(), Halt> {
        let site = || ReferenceSite::SemanticSource(entry.id.clone());
        for originator in &entry.originators {
            if !self.semantic_sources.contains_key(originator) {
                walk.report(dangling(
                    site(),
                    ReferenceTargetKind::SemanticSources,
                    originator,
                ))?;
            }
        }
        if !originators_within_reservation(entry) {
            walk.report(structural(site(), ORIGINATORS_RULE))?;
        }
        if !originators_agree_with_operands(entry, &self.semantic_sources) {
            walk.report(structural(site(), ORIGINATOR_AGREEMENT_RULE))?;
        }
        Ok(())
    }

    fn check_temporal_refs(
        &self,
        shape: &semstrait_ir::TemporalShape,
        owner: &str,
        walk: &mut Walk<'_>,
    ) -> Result<(), Halt> {
        use semstrait_ir::TemporalType;
        let refs: Vec<&str> = match &shape.temporal_type {
            TemporalType::Timeseries { occurred_at } => vec![occurred_at.as_str()],
            TemporalType::Events { event_time } => vec![event_time.as_str()],
            TemporalType::Snapshot { snapshotted_at } => vec![snapshotted_at.as_str()],
            TemporalType::Scd {
                valid_from,
                valid_to,
                ..
            } => vec![valid_from.as_str(), valid_to.as_str()],
            other => unreachable!("semstrait-ir added a TemporalType variant: {other:?}"),
        };
        for r in refs {
            if !self.semantic_definitions.entries.contains_key(r) {
                walk.report(dangling(
                    ReferenceSite::SemanticSource(owner.to_owned()),
                    ReferenceTargetKind::Semantics,
                    r,
                ))?;
            }
        }
        Ok(())
    }

    fn check_temporal_agreement(
        &self,
        entry: &SemanticSource,
        walk: &mut Walk<'_>,
    ) -> Result<(), Halt> {
        if let Err(error) = crate::compile::temporal::composed_shape(entry, &self.semantic_sources)
        {
            walk.report(error)?;
        }
        Ok(())
    }

    fn check_keysets(&self, walk: &mut Walk<'_>) -> Result<(), Halt> {
        for source in self.semantic_sources.values() {
            self.check_grainset_keysets(source, walk)?;
            for signature_id in &source.semantic_signature_ids {
                let Some(signature) = self.semantic_signatures.get(signature_id) else {
                    continue;
                };
                let Some(definition) = self
                    .semantic_definitions
                    .entries
                    .get(&signature.definition_id)
                else {
                    continue;
                };
                for keyset_id in &signature.keysets {
                    self.check_keyset_member(
                        &signature.definition_id,
                        keyset_id,
                        source,
                        definition,
                        walk,
                    )?;
                }
            }
        }
        Ok(())
    }

    fn check_keyset_member(
        &self,
        definition_id: &EntityId,
        keyset_id: &EntityId,
        source: &SemanticSource,
        definition: &crate::manifest::semantic_definition::SemanticDefinition,
        walk: &mut Walk<'_>,
    ) -> Result<(), Halt> {
        let site = || ReferenceSite::Keyset(keyset_id.clone());
        if definition.role() != SemanticRole::Dimension {
            walk.report(structural(
                site(),
                "a keyset member must be a semantic of role Dimension",
            ))?;
        }
        if !self
            .semantic_definitions
            .mask_of(std::iter::once(definition_id))
            .is_subset_of(&source.semantic_bitmask)
        {
            walk.report(structural(
                site(),
                "a keyset member must be served by the semantic source attributing it",
            ))?;
        }
        Ok(())
    }

    fn check_grainset_keysets(
        &self,
        source: &SemanticSource,
        walk: &mut Walk<'_>,
    ) -> Result<(), Halt> {
        let RoleDefinition::Grainset { keysets, .. } = &source.role_definition else {
            return Ok(());
        };
        let site = || ReferenceSite::SemanticSource(source.id.clone());
        for keyset_id in keysets {
            if !self.keysets.contains_key(keyset_id) {
                walk.report(dangling(
                    site(),
                    ReferenceTargetKind::Keysets,
                    keyset_id.clone(),
                ))?;
            }
        }
        if !is_sorted_set(keysets) {
            walk.report(structural(
                site(),
                "a Grainset's keyset attribution must be sorted and deduplicated",
            ))?;
        }
        Ok(())
    }

    fn check_sources(&self, aliases: &SourceAliases, walk: &mut Walk<'_>) -> Result<(), Halt> {
        for (source, expected) in self.sources.values().zip(aliases.physical.values()) {
            if &source.alias != expected {
                walk.report(CompileError::PhysicalSourceAliasMismatch {
                    physical_source: source.id.clone(),
                    expected: expected.clone(),
                    found: source.alias.clone(),
                })?;
            }
            if let SchemaProvenance::CatalogDerived { schema, .. } = &source.schema_provenance {
                for column in &schema.columns {
                    if column.data_type.is_none() {
                        walk.report(CompileError::UntypedCatalogColumn {
                            physical_source_id: source.id.clone(),
                            column: column.name.clone(),
                        })?;
                    }
                }
            }
            let site = || ReferenceSite::Source(source.id.clone());
            if !source.storage.reads_a_supported_format() {
                walk.report(structural(site(), FORMAT_RULE))?;
            }
            if let StorageTarget::Path {
                base_path,
                listings,
                partition_def,
            } = &source.storage.target
            {
                let folder = base_path.is_normal() && base_path.as_str().ends_with('/');
                if !folder || !listings.iter().all(Uri::is_normal) {
                    walk.report(structural(site(), NORMAL_FORM_RULE))?;
                }
                if !is_sorted_set(listings) {
                    walk.report(structural(site(), FIXED_ORDER_RULE))?;
                }
                if !listings.iter().all(|uri| uri.beneath(base_path).is_some()) {
                    walk.report(structural(site(), BASE_PATH_RULE))?;
                }
                let partitions = partition_def
                    .as_ref()
                    .map_or(&[][..], |def| &def.partitions);
                let folders: Vec<Option<&str>> = (partitions.iter())
                    .map(|partition| {
                        (partition.listing.first())
                            .and_then(|object| partition_folder(object, base_path))
                    })
                    .collect();
                if (partition_def.is_some() && partitions.is_empty()) || folders.contains(&Some(""))
                {
                    walk.report(structural(site(), DIRECT_OBJECTS_RULE))?;
                }
                let listed: BTreeSet<&Uri> = listings.iter().collect();
                if !(partitions.iter().zip(&folders)).all(|(partition, folder)| {
                    folder.is_some()
                        && is_sorted_set(&partition.listing)
                        && (partition.listing.iter()).all(|object| {
                            listed.contains(object)
                                && partition_folder(object, base_path) == *folder
                        })
                }) {
                    walk.report(structural(site(), PARTITION_LISTING_RULE))?;
                }
                if !is_sorted_set(&folders) {
                    walk.report(structural(site(), PARTITION_ORDER_RULE))?;
                }
                if !(partitions.iter().zip(&folders)).all(|(partition, folder)| {
                    folder
                        .and_then(|folder| folder_key_values(folder).ok())
                        .as_ref()
                        == Some(&partition.key_values)
                }) {
                    walk.report(structural(site(), KEY_VALUES_RULE))?;
                }
                let sequences: BTreeSet<Vec<&str>> = (partitions.iter())
                    .map(|partition| {
                        (partition.key_values.iter())
                            .map(|held| held.key.as_str())
                            .collect()
                    })
                    .collect();
                if sequences.len() > 1 {
                    walk.report(structural(site(), KEY_SEQUENCE_RULE))?;
                }
            }
        }
        Ok(())
    }

    fn keyset_names_of(&self, data_kind_id: &str) -> BTreeSet<&str> {
        let mut names = BTreeSet::new();
        let Some(source) = self.semantic_sources.get(data_kind_id) else {
            return names;
        };
        for signature_id in &source.semantic_signature_ids {
            let Some(signature) = self.semantic_signatures.get(signature_id) else {
                continue;
            };
            for keyset_id in &signature.keysets {
                if let Some(keyset) = self.keysets.get(keyset_id) {
                    names.insert(keyset.name.as_str());
                }
            }
        }
        names
    }

    fn check_reduction_carry_over(&self, walk: &mut Walk<'_>) -> Result<(), Halt> {
        const RULE: &str = "a reduction's Unionset must carry every keyset name its leaves carry";

        for entry in self.semantic_sources.values() {
            let RoleDefinition::Unionset { branches, .. } = &entry.role_definition else {
                continue;
            };
            let carried = self.keyset_names_of(&entry.id);
            for operand_id in branches {
                let Some(operand) = self.semantic_sources.get(operand_id) else {
                    continue;
                };
                let is_reduction_leaf =
                    matches!(operand.role_definition, RoleDefinition::Dataset { .. })
                        && !operand.public
                        && operand.originators == [entry.id.clone()];
                if !is_reduction_leaf {
                    continue;
                }
                for name in self.keyset_names_of(operand_id) {
                    if !carried.contains(name) {
                        walk.report(structural(
                            ReferenceSite::SemanticSource(entry.id.clone()),
                            RULE,
                        ))?;
                    }
                }
            }
        }
        Ok(())
    }

    fn check_reduction_leaf_signatures(&self, walk: &mut Walk<'_>) -> Result<(), Halt> {
        for entry in self.semantic_sources.values() {
            let RoleDefinition::Unionset {
                branches,
                null_fill: false,
                ..
            } = &entry.role_definition
            else {
                continue;
            };
            for definition in self.semantic_definitions.entries.values() {
                let unserved = |branch: &EntityId| {
                    matches!(self.coverage_of(&definition.id, branch), Coverage::NullFill)
                };
                let leaf = (branches.iter().find(|branch| unserved(branch)))
                    .filter(|_| branches.iter().any(|branch| !unserved(branch)));
                if let Some(leaf) = leaf {
                    walk.report(CompileError::ReductionLeafMissingSignature {
                        semantic_source: entry.id.clone(),
                        leaf: leaf.clone(),
                        semantics: definition.name.as_str().to_owned(),
                    })?;
                }
            }
        }
        Ok(())
    }

    fn check_read_once(&self, walk: &mut Walk<'_>) -> Result<(), Halt> {
        for entry in self.semantic_sources.values() {
            let RoleDefinition::Unionset {
                branches,
                null_fill: false,
                ..
            } = &entry.role_definition
            else {
                continue;
            };
            let mut base_paths: BTreeSet<&Uri> = BTreeSet::new();
            let mut objects: BTreeSet<&Uri> = BTreeSet::new();
            let mut read_twice = false;
            for branch in branches {
                let Some(RoleDefinition::Dataset {
                    source_id,
                    selector,
                }) = (self.semantic_sources.get(branch)).map(|leaf| &leaf.role_definition)
                else {
                    continue;
                };
                let Some(StorageTarget::Path {
                    base_path,
                    listings,
                    ..
                }) = (self.sources.get(source_id)).map(|source| &source.storage.target)
                else {
                    continue;
                };
                read_twice |= !base_paths.insert(base_path);
                for object in listings {
                    if selector.selects(base_path, object) {
                        read_twice |= !objects.insert(object);
                    }
                }
            }
            if read_twice {
                walk.report(structural(
                    ReferenceSite::SemanticSource(entry.id.clone()),
                    READ_ONCE_RULE,
                ))?;
            }
        }
        Ok(())
    }

    fn check_expression_pool(&self, walk: &mut Walk<'_>) -> Result<(), Halt> {
        const ONE_ROOT_RULE: &str = "an entry of the expression store must be named by one \
                                    root: one signature's expression or filter, or one Joinset \
                                    source's condition or relationship filter";

        let mut roots: BTreeMap<PhysicalExprId, usize> = BTreeMap::new();
        for id in self.expression_roots() {
            *roots.entry(id).or_default() += 1;
        }

        let final_value_owners = self.final_value_owners();
        for (key, entry) in &self.expressions.physical {
            if &entry.id != key {
                walk.report(structural(
                    ReferenceSite::ExpressionPool,
                    "ManifestExpression.id must equal its map key",
                ))?;
            }
            if roots.get(key).is_some_and(|&named| named > 1) {
                walk.report(structural(ReferenceSite::ExpressionPool, ONE_ROOT_RULE))?;
            }
            let layer = match final_value_owners.get(key) {
                Some(owner) => self.layer_reading_final_values(entry.expr.expr(), *owner),
                None => self.expressions.layer_of(entry.expr.expr()),
            };
            if layer.is_some_and(|layer| layer != entry.layer) {
                walk.report(structural(ReferenceSite::ExpressionPool, LAYER_RULE))?;
            }
            for node in entry.expr.nodes() {
                if let Expr::Function(call) = node {
                    if let Err(error) = call.check_options() {
                        walk.report(CompileError::UndeclaredOption {
                            expr_id: *key,
                            error,
                        })?;
                    }
                }
            }
        }

        for expr_id in orphan_expressions(&self.expressions, self.expression_roots()) {
            walk.report(CompileError::OrphanExpression { expr_id })?;
        }
        Ok(())
    }

    fn check_links(&self, walk: &mut Walk<'_>) -> Result<(), Halt> {
        let links = (self.expressions.physical.values()).flat_map(|entry| entry.expr.links());
        for link in links {
            if self.expressions.get(&link).is_none() {
                walk.report(dangling(
                    ReferenceSite::ExpressionPool,
                    ReferenceTargetKind::Expressions,
                    link.to_string(),
                ))?;
            }
        }
        for source in self.semantic_sources.values() {
            self.check_link_places(source, walk)?;
        }
        for _ in 0..cycle_closing_links(&self.expressions) {
            walk.report(structural(ReferenceSite::ExpressionPool, LINK_CYCLE_RULE))?;
        }
        Ok(())
    }

    fn check_link_places(&self, source: &SemanticSource, walk: &mut Walk<'_>) -> Result<(), Halt> {
        let role = source.role_definition.source_role();
        let signatures: Vec<_> = self.signatures_of(source).collect();
        let listed: BTreeSet<PhysicalExprId> = signatures
            .iter()
            .map(|signature| signature.expression_id)
            .collect();
        let conditions = (condition_handles(source))
            .zip([ExprRoot::JoinsetOn, ExprRoot::JoinsetFilter])
            .map(|(root, marker)| (root, marker, None));
        let signature_entries = signatures.iter().flat_map(|&signature| {
            let expression = (signature.expression_id, ExprRoot::SignatureExpression);
            let filter =
                (signature.filter_expr_id).map(|filter| (filter, ExprRoot::SignatureFilter));
            (std::iter::once(expression).chain(filter))
                .map(move |(root, marker)| (root, marker, Some(signature)))
        });
        for (root, marker, signature) in conditions.chain(signature_entries) {
            let site = || match signature {
                Some(signature) => ReferenceSite::Signature(signature.id.clone()),
                None => ReferenceSite::SemanticSource(source.id.clone()),
            };
            let semantics = signature
                .and_then(|signature| self.semantic_definitions.get(&signature.definition_id))
                .map(SemanticDefinition::role);
            let mut links = self.expressions.links_of(&root);
            match link_free_rule(role, marker, semantics) {
                Some(rule) if links.next().is_some() => {
                    walk.report(structural(site(), rule))?;
                }
                Some(_) => {}
                None => {
                    let misplaced = links.filter(|link| {
                        self.expressions.get(link).is_some() && !listed.contains(link)
                    });
                    for _ in misplaced {
                        walk.report(structural(site(), LINK_TARGET_RULE))?;
                    }
                }
            }
        }
        Ok(())
    }

    fn grainset_owns_only_unmapped_metrics(
        &self,
        entry: &SemanticSource,
        children: &[EntityId],
    ) -> bool {
        self.signatures_of(entry).count() == entry.semantic_signature_ids.len()
            && self.signatures_of(entry).all(|signature| {
                let definition = self.semantic_definitions.get(&signature.definition_id);
                let stored =
                    (self.expressions.get(&signature.expression_id)).map(|tree| tree.layer);
                definition.is_some_and(|definition| {
                    definition.role() == SemanticRole::Metric
                        && !self.some_grain_sets(children, definition.bit_position)
                        && layer_of_final_value(&definition.role_definition, stored)
                            != ExprLayer::Scalar
                })
            })
    }

    fn grainset_reads_only_served_columns(
        &self,
        entry: &SemanticSource,
        children: &[EntityId],
    ) -> bool {
        self.signatures_of(entry).all(|signature| {
            [Some(signature.expression_id), signature.filter_expr_id]
                .into_iter()
                .flatten()
                .flat_map(|root| self.expressions.columns(&root))
                .all(|(_, column)| match column.parts() {
                    [qualifier, name] if entry.alias.as_str() == qualifier => self
                        .semantic_definitions
                        .entries
                        .values()
                        .any(|definition| {
                            definition.name.as_str() == name.as_str()
                                && self.some_grain_sets(children, definition.bit_position)
                        }),
                    _ => true,
                })
        })
    }

    fn some_grain_sets(&self, children: &[EntityId], bit: u32) -> bool {
        (children.iter())
            .filter_map(|child| self.semantic_sources.get(child))
            .any(|child| child.semantic_bitmask.contains(bit))
    }

    fn final_value_owners(&self) -> BTreeMap<PhysicalExprId, (&str, &[EntityId])> {
        let grainsets =
            (self.semantic_sources.values()).filter_map(|source| match &source.role_definition {
                RoleDefinition::Grainset {
                    grains: children, ..
                } => Some((source, children)),
                _ => None,
            });
        grainsets
            .flat_map(|(source, children)| {
                self.signatures_of(source).flat_map(move |signature| {
                    let roots = [Some(signature.expression_id), signature.filter_expr_id];
                    let owner = (source.alias.as_str(), children.as_slice());
                    roots.into_iter().flatten().map(move |root| (root, owner))
                })
            })
            .collect()
    }

    fn layer_reading_final_values(
        &self,
        expr: &Expr<Link>,
        (alias, children): (&str, &[EntityId]),
    ) -> Option<ExprLayer> {
        let pools = ChildPools {
            serving: Serving::new(&self.semantic_sources, &self.semantic_signatures),
            expressions: &self.expressions,
        };
        self.expressions.layer_of_reading(expr, &|column| {
            final_value_layer(alias, column, |name| {
                let definition = self.semantic_definitions.by_name(name)?;
                Some(pools.final_layer(children, definition))
            })
        })
    }

    fn expression_roots(&self) -> impl Iterator<Item = PhysicalExprId> + '_ {
        self.semantic_signatures
            .values()
            .flat_map(|signature| {
                std::iter::once(signature.expression_id).chain(signature.filter_expr_id)
            })
            .chain(self.semantic_sources.values().flat_map(condition_handles))
    }

    fn check_column_qualifiers(
        &self,
        aliases: &SourceAliases,
        walk: &mut Walk<'_>,
    ) -> Result<(), Halt> {
        for entry in self.semantic_sources.values() {
            for fault in misplaced_columns(
                entry,
                &self.semantic_sources,
                &self.semantic_signatures,
                aliases,
                &self.expressions,
            ) {
                walk.report(fault)?;
            }
        }
        Ok(())
    }

    fn check_condition_leaves(&self, walk: &mut Walk<'_>) -> Result<(), Halt> {
        for fault in unserved_condition_leaves(
            &self.semantic_sources,
            &self.semantic_signatures,
            self.semantic_definitions.entries.values(),
            &self.expressions,
        ) {
            walk.report(fault)?;
        }
        Ok(())
    }

    fn check_bitmask(mask: &SemanticBitmask, owner: &str, walk: &mut Walk<'_>) -> Result<(), Halt> {
        if !mask.is_canonical() {
            walk.report(structural(
                ReferenceSite::Bitmask(owner.to_owned()),
                "SemanticBitmask must be canonical (no trailing all-zero words)",
            ))?;
        }
        for pos in mask.set_positions() {
            if !walk.occupied.contains(&pos) {
                walk.report(dangling(
                    ReferenceSite::Bitmask(owner.to_owned()),
                    ReferenceTargetKind::Semantics,
                    format!("bit_position {pos}"),
                ))?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::time::{Duration, UNIX_EPOCH};

    use crate::manifest::semantic_source::Selector;
    use semstrait_common::{DataKindName, Granularity, SemanticsName};
    use semstrait_ir::{
        Cardinality, CrossFilter, Integrity, JoinType, MultisetOp, PhysicalExpr, TemporalShape,
        TemporalType,
    };
    use semstrait_model::Agg;
    use semstrait_sql::expr::{Expr, ExprOperationExt, Leaf, Literal, NamedReference};

    use crate::compile::expressions::lower::wrap_aggregate;
    use crate::identity::{mint_expr_id, mint_signature_id};
    use crate::manifest::expression::{ExprRoot, Link, ManifestExpression, StoredExpr};

    use crate::manifest::physical_source::{SourceColumn, SourceSchema};

    use crate::compile::catalog::table_ref_from_fqn as table_ref;

    use crate::manifest::bitmap::SemanticBitmap;
    use crate::manifest::keyset::Keyset;
    use crate::manifest::physical_source::{
        PhysicalSource, SchemaId, SchemaProvenance, StorageProperties, StorageTarget,
    };
    use crate::manifest::semantic_definition::{SemanticDefinition, SemanticRoleDefinition};
    use crate::manifest::source_alias::SourceAlias;
    use crate::{ExprLayer, SemanticManifestMetadata};

    const BIT_AMOUNT: u32 = 0;
    const BIT_KEY_A: u32 = 1;
    const BIT_KEY_B: u32 = 2;
    const BIT_OCCURRED: u32 = 3;
    const BIT_REGION: u32 = 4;

    fn metadata() -> SemanticManifestMetadata {
        SemanticManifestMetadata::new("0.1.0", None, BTreeMap::new())
    }

    fn provenance() -> crate::manifest::top::ManifestProvenance {
        crate::manifest::top::ManifestProvenance::new(
            UNIX_EPOCH + Duration::from_secs(1_700_000_000),
            UNIX_EPOCH + Duration::from_secs(1_800_000_000),
        )
    }

    fn def(id: &str, role: SemanticRole) -> SemanticDefinition {
        SemanticDefinition::new(
            id,
            SemanticsName::from(id),
            None,
            SemanticRoleDefinition::plain(role),
        )
    }

    fn source(
        id: &str,
        name: &str,
        mask: SemanticBitmask,
        role_definition: RoleDefinition,
    ) -> SemanticSource {
        SemanticSource::new(
            id,
            DataKindName::from(name),
            SourceAlias::from(name),
            true,
            mask,
            None,
            role_definition,
            Vec::new(),
        )
    }

    fn sign(
        m: &mut SemanticManifest,
        owner: &mut SemanticSource,
        semantic: &str,
        expr: &PhysicalExpr,
    ) {
        let id = crate::identity::mint_signature_id(&owner.id, semantic);
        let root = m
            .expressions
            .emit(expr, &id, ExprRoot::SignatureExpression, consumers(owner))
            .expect("emit");
        m.semantic_signatures.insert(
            id.clone(),
            crate::manifest::SemanticSignature::new(id.clone(), semantic, root),
        );
        owner.semantic_signature_ids.push(id);
        owner.semantic_signature_ids.sort();
    }

    fn consumers(owner: &SemanticSource) -> Vec<EntityId> {
        match &owner.role_definition {
            RoleDefinition::Dataset { source_id, .. } => vec![source_id.clone()],
            _ => Vec::new(),
        }
    }

    fn qualified(alias: &str, name: &str) -> PhysicalExpr {
        Expr::Column(NamedReference::qualified(alias, name).expect("an alias and a name"))
    }

    fn expr_id(raw: u128) -> PhysicalExprId {
        PhysicalExprId::new(uuid::Uuid::from_u128(raw))
    }

    fn valid_manifest() -> SemanticManifest {
        let mut m = SemanticManifest::new(1, provenance(), [0u8; 32], "model.root", metadata());

        m.semantic_definitions = SemanticBitmap::freeze([
            def("sem.region", SemanticRole::Dimension),
            def("sem.amount", SemanticRole::Measure),
            def("sem.key.a", SemanticRole::Dimension),
            def("sem.key.b", SemanticRole::Dimension),
            def("sem.occurred", SemanticRole::Dimension),
        ])
        .expect("five entries fit the bit width");

        let src = PhysicalSource::new(
            "src.orders",
            "orders",
            SourceAlias::from("ps1"),
            StorageProperties::bare(StorageTarget::table(table_ref("warehouse.orders"))),
            SchemaProvenance::model_fed(SourceSchema {
                columns: ["amount", "region", "key_a", "key_b", "occurred"]
                    .into_iter()
                    .map(|name| SourceColumn {
                        name: name.to_owned(),
                        data_type: None,
                        nullable: None,
                    })
                    .collect(),
            }),
            SchemaId::from("fixture-schema"),
        );
        m.sources.insert(src.id.clone(), src);

        let customers_src = PhysicalSource::new(
            "src.customers",
            "customers",
            SourceAlias::from("ps0"),
            StorageProperties::bare(StorageTarget::table(table_ref("warehouse.customers"))),
            SchemaProvenance::model_fed(SourceSchema {
                columns: ["key_a", "key_b"]
                    .into_iter()
                    .map(|name| SourceColumn {
                        name: name.to_owned(),
                        data_type: None,
                        nullable: None,
                    })
                    .collect(),
            }),
            SchemaId::from("fixture-schema"),
        );
        m.sources.insert(customers_src.id.clone(), customers_src);

        let mut orders = source(
            "dk.orders",
            "orders",
            SemanticBitmask::from_positions([
                BIT_REGION,
                BIT_AMOUNT,
                BIT_KEY_A,
                BIT_KEY_B,
                BIT_OCCURRED,
            ]),
            RoleDefinition::dataset("src.orders", Selector::Table),
        );
        orders.temporal = Some(TemporalShape::new(
            TemporalType::timeseries("sem.occurred"),
            Some(Granularity::Day),
        ));
        for (semantic, expr) in [
            ("sem.region", qualified("ps1", "region")),
            ("sem.amount", qualified("ps1", "amount")),
            ("sem.key.a", qualified("ps1", "key_a")),
            ("sem.key.b", qualified("ps1", "key_b")),
            ("sem.occurred", qualified("ps1", "region")),
        ] {
            sign(&mut m, &mut orders, semantic, &expr);
        }
        m.semantic_sources.insert(orders.id.clone(), orders);

        let mut customers = source(
            "dk.customers",
            "customers",
            SemanticBitmask::from_positions([BIT_KEY_A, BIT_KEY_B]),
            RoleDefinition::dataset("src.customers", Selector::Table),
        );
        for (semantic, expr) in [
            ("sem.key.a", qualified("ps0", "key_a")),
            ("sem.key.b", qualified("ps0", "key_b")),
        ] {
            sign(&mut m, &mut customers, semantic, &expr);
        }
        m.semantic_sources.insert(customers.id.clone(), customers);

        for branch in ["a", "b"] {
            let mut nested = source(
                &format!("dk.union.{branch}"),
                &format!("union.{branch}"),
                SemanticBitmask::from_positions([BIT_REGION]),
                RoleDefinition::dataset("src.orders", Selector::Table),
            );
            nested.public = false;
            nested.originators = vec!["dk.union".to_owned()];
            sign(
                &mut m,
                &mut nested,
                "sem.region",
                &qualified("ps1", "region"),
            );
            m.semantic_sources.insert(nested.id.clone(), nested);
        }
        let mut union = source(
            "dk.union",
            "union",
            SemanticBitmask::from_positions([BIT_REGION]),
            RoleDefinition::unionset(
                MultisetOp::UnionAll,
                vec!["dk.union.a".to_owned(), "dk.union.b".to_owned()],
                true,
            ),
        );
        let region = qualified("ssu4", "region");
        sign(&mut m, &mut union, "sem.region", &region);
        m.semantic_sources.insert(union.id.clone(), union);

        let grain = source(
            "dk.grain",
            "grain",
            SemanticBitmask::from_positions([BIT_REGION, BIT_AMOUNT, BIT_KEY_A, BIT_OCCURRED]),
            RoleDefinition::grainset(
                vec!["dk.orders".to_owned(), "dk.customers".to_owned()],
                Vec::new(),
            ),
        );
        m.semantic_sources.insert(grain.id.clone(), grain);

        let join_condition = m
            .expressions
            .emit(
                &qualified("ssd3", "sem.key.a"),
                "dk.join",
                ExprRoot::JoinsetOn,
                Vec::<String>::new(),
            )
            .expect("emit");
        let mut join = source(
            "dk.join",
            "join",
            SemanticBitmask::from_positions([
                BIT_REGION,
                BIT_AMOUNT,
                BIT_KEY_A,
                BIT_KEY_B,
                BIT_OCCURRED,
            ]),
            RoleDefinition::Joinset {
                from: "dk.orders".to_owned(),
                to: "dk.customers".to_owned(),
                on_expr_id: Some(join_condition),
                filter_expr_id: None,
                join_type: JoinsetJoinType::Join(JoinType::Inner),
                cardinality: Cardinality::ManyToOne,
                integrity: Integrity::Assumed,
                cross_filter: CrossFilter::None,
                description: None,
                ai_context: None,
            },
        );
        for (semantic, expr) in [
            ("sem.region", qualified("ssd3", "region")),
            ("sem.amount", qualified("ssd3", "amount")),
            ("sem.key.a", qualified("ssd3", "key_a")),
            ("sem.key.b", qualified("ssd3", "key_b")),
            ("sem.occurred", qualified("ssd3", "region")),
        ] {
            sign(&mut m, &mut join, semantic, &expr);
        }
        m.semantic_sources.insert(join.id.clone(), join);

        for keyset in [
            Keyset::new("dk.orders", "natural"),
            Keyset::new("dk.customers", "natural"),
            Keyset::new("dk.orders", "surrogate"),
        ] {
            m.keysets.insert(keyset.id.clone(), keyset);
        }
        attribute(&mut m, "sem.key.a", "dk.orders", &[natural_id()]);
        attribute(
            &mut m,
            "sem.key.a",
            "dk.customers",
            &[customers_natural_id()],
        );
        attribute(&mut m, "sem.key.b", "dk.orders", &[surrogate_id()]);

        for (id, alias) in [
            ("dk.customers", "ssd0"),
            ("dk.grain", "ssg1"),
            ("dk.join", "ssj2"),
            ("dk.orders", "ssd3"),
            ("dk.union", "ssu4"),
            ("dk.union.a", "ssd5"),
            ("dk.union.b", "ssd6"),
        ] {
            m.semantic_sources
                .get_mut(id)
                .expect("the fixture holds every source it aliases")
                .alias = SourceAlias::from(alias);
        }
        m
    }

    fn attribute(m: &mut SemanticManifest, semantic: &str, data_kind: &str, keysets: &[String]) {
        let signature_id = crate::identity::mint_signature_id(data_kind, semantic);
        m.semantic_signatures
            .get_mut(&signature_id)
            .unwrap_or_else(|| panic!("no signature for ({data_kind}, {semantic})"))
            .keysets = keysets.to_vec();
    }

    fn natural_id() -> String {
        crate::identity::mint_keyset_id("dk.orders", "natural")
    }

    fn customers_natural_id() -> String {
        crate::identity::mint_keyset_id("dk.customers", "natural")
    }

    fn surrogate_id() -> String {
        crate::identity::mint_keyset_id("dk.orders", "surrogate")
    }

    #[test]
    fn pristine_manifest_validates() {
        assert_valid(&valid_manifest());
    }

    type Faults = Result<Diagnostics<CompileError>, Diagnostics<CompileError>>;

    fn faults(result: Faults) -> Vec<CompileError> {
        match result {
            Err(diags) => diags.into_iter().map(|d| d.kind).collect(),
            Ok(warnings) => panic!("expected a refusal, got Ok({warnings:?})"),
        }
    }

    fn assert_dangling(result: Faults, kind: ReferenceTargetKind, target: &str) {
        let reported = faults(result);
        assert!(
            reported.iter().any(|fault| matches!(
                fault,
                CompileError::DanglingReference { to_kind, target_id, .. }
                    if *to_kind == kind && target_id == target
            )),
            "expected DanglingReference to {kind:?} ({target}) among {reported:?}"
        );
    }

    fn assert_structural(result: Faults, rule: &str) {
        let reported = faults(result);
        assert!(
            reported.iter().any(|fault| matches!(
                fault,
                CompileError::StructuralViolation { rule: r, .. } if *r == rule
            )),
            "expected StructuralViolation ({rule}) among {reported:?}"
        );
    }

    fn assert_valid(manifest: &SemanticManifest) {
        match manifest.validate() {
            Ok(warnings) => assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}"),
            Err(faults) => panic!("expected a valid manifest, got {faults:?}"),
        }
    }

    fn two_independent_faults() -> SemanticManifest {
        let mut m = valid_manifest();
        let signature_id = crate::identity::mint_signature_id("dk.orders", "sem.amount");
        m.semantic_signatures
            .get_mut(&signature_id)
            .unwrap()
            .expression_id = expr_id(999);
        m.expressions
            .physical
            .values_mut()
            .next()
            .expect("the fixture pool is non-empty")
            .layer = ExprLayer::Aggregate;
        m
    }

    #[test]
    fn the_default_pass_reports_every_independent_fault() {
        let reported = faults(two_independent_faults().validate());

        assert!(
            reported.iter().any(|fault| matches!(
                fault,
                CompileError::DanglingReference { target_id, .. } if target_id == "00000000-0000-0000-0000-0000000003e7"
            )),
            "the dangling reference must be reported: {reported:?}"
        );
        assert!(
            reported.iter().any(|fault| matches!(
                fault,
                CompileError::StructuralViolation { rule, .. } if *rule == LAYER_RULE
            )),
            "the structural violation must be reported too: {reported:?}"
        );
    }

    #[test]
    fn fail_fast_returns_the_same_type_carrying_one_fault() {
        let m = two_independent_faults();

        let accumulated = faults(m.validate());
        assert!(
            accumulated.len() > 1,
            "precondition: the accumulating pass finds more than one fault"
        );

        let short_circuited = faults(m.validate_with(FaultReporting::FirstFault));
        assert_eq!(
            short_circuited.len(),
            1,
            "fail-fast carries exactly one element: {short_circuited:?}"
        );
        assert!(
            accumulated.contains(&short_circuited[0]),
            "the short-circuited fault must be one the full pass also reports"
        );
    }

    #[test]
    fn the_accepted_arm_carries_an_empty_diagnostic_batch() {
        assert!(valid_manifest()
            .validate()
            .expect("pristine fixture")
            .is_empty());
    }

    #[test]
    fn a_signature_keyset_naming_an_absent_roster_entry_is_dangling() {
        let mut m = valid_manifest();
        let signature_id = crate::identity::mint_signature_id("dk.orders", "sem.key.a");
        m.semantic_signatures
            .get_mut(&signature_id)
            .unwrap()
            .keysets = vec!["ks.missing".to_owned()];
        assert_dangling(m.validate(), ReferenceTargetKind::Keysets, "ks.missing");
    }

    #[test]
    fn a_dataset_arms_reference_resolves_in_physical_sources() {
        let mut m = valid_manifest();
        m.semantic_sources
            .get_mut("dk.orders")
            .expect("the fixture holds the leaf")
            .role_definition = RoleDefinition::dataset("src.missing", Selector::Table);
        assert_dangling(m.validate(), ReferenceTargetKind::Sources, "src.missing");
    }

    #[test]
    fn a_nested_entry_is_checked_by_the_same_walk_with_no_recursion() {
        let mut m = valid_manifest();
        m.semantic_sources
            .get_mut("dk.union.a")
            .expect("a nested branch is an ordinary entry")
            .role_definition = RoleDefinition::dataset("src.missing", Selector::Table);
        assert_dangling(m.validate(), ReferenceTargetKind::Sources, "src.missing");
    }

    #[test]
    fn an_operand_naming_an_absent_entry_is_dangling() {
        for role in [
            RoleDefinition::unionset(
                MultisetOp::UnionAll,
                vec!["dk.union.a".to_owned(), "dk.missing".to_owned()],
                true,
            ),
            RoleDefinition::grainset(
                vec!["dk.orders".to_owned(), "dk.missing".to_owned()],
                Vec::new(),
            ),
        ] {
            let mut m = valid_manifest();
            m.semantic_sources
                .get_mut("dk.union")
                .expect("the fixture holds the composer")
                .role_definition = role;
            assert_dangling(
                m.validate(),
                ReferenceTargetKind::SemanticSources,
                "dk.missing",
            );
        }

        let mut m = valid_manifest();
        if let RoleDefinition::Joinset { to, .. } = &mut m
            .semantic_sources
            .get_mut("dk.join")
            .expect("the fixture holds the Joinset")
            .role_definition
        {
            *to = "dk.missing".to_owned();
        } else {
            unreachable!("the fixture's `dk.join` is a Joinset");
        }
        assert_dangling(
            m.validate(),
            ReferenceTargetKind::SemanticSources,
            "dk.missing",
        );
    }

    #[test]
    fn a_cycle_over_the_operand_edges_is_refused() {
        let mut m = valid_manifest();
        m.semantic_sources
            .get_mut("dk.union.a")
            .expect("a nested branch is an ordinary entry")
            .role_definition =
            RoleDefinition::unionset(MultisetOp::UnionAll, vec!["dk.union".to_owned()], true);
        let reported = faults(m.validate());
        assert!(
            reported.iter().any(|fault| matches!(
                fault,
                CompileError::CyclicSemanticSource { cycle }
                    if cycle.contains(&"dk.union".to_owned())
                        && cycle.contains(&"dk.union.a".to_owned())
            )),
            "expected CyclicSemanticSource naming both units among {reported:?}"
        );
    }

    #[test]
    fn a_dataset_arm_pointing_at_its_own_id_is_not_a_cycle() {
        let mut m = valid_manifest();
        let mut clash = m.sources["src.orders"].clone();
        clash.id = "src.orders".to_owned();
        let mut renamed = clash.clone();
        renamed.id = "dk.customers".to_owned();
        m.sources.insert(renamed.id.clone(), renamed);
        m.semantic_sources
            .get_mut("dk.customers")
            .expect("the fixture holds the leaf")
            .role_definition = RoleDefinition::dataset("dk.customers", Selector::Table);
        restamp(&mut m, "dk.customers");
        assert!(
            !faults(m.validate())
                .iter()
                .any(|fault| matches!(fault, CompileError::CyclicSemanticSource { .. })),
            "a physical-source reference must not be read as an operand edge"
        );
    }

    fn restamp(m: &mut SemanticManifest, data_kind: &str) {
        let owner = &m.semantic_sources[data_kind];
        let stamp: BTreeSet<EntityId> = consumers(owner).into_iter().collect();
        for signature_id in &owner.semantic_signature_ids {
            let root = m.semantic_signatures[signature_id].expression_id;
            m.expressions
                .physical
                .get_mut(&root)
                .expect("each fixture signature names its entry")
                .physical_sources = stamp.clone();
        }
    }

    #[test]
    fn the_two_originator_reservations_are_restated_here() {
        let mut m = valid_manifest();
        m.semantic_sources
            .get_mut("dk.union.a")
            .expect("a nested branch is an ordinary entry")
            .originators = vec!["dk.union".to_owned(), "dk.grain".to_owned()];
        assert_structural(m.validate(), ORIGINATORS_RULE);

        let mut m = valid_manifest();
        m.semantic_sources
            .get_mut("dk.union")
            .expect("the fixture holds the composer")
            .role_definition =
            RoleDefinition::unionset(MultisetOp::UnionAll, vec!["dk.union.b".to_owned(); 2], true);
        assert_structural(m.validate(), ORIGINATOR_AGREEMENT_RULE);

        let mut m = valid_manifest();
        m.semantic_sources
            .get_mut("dk.union.a")
            .expect("a nested branch is an ordinary entry")
            .originators = vec!["dk.missing".to_owned()];
        assert_dangling(
            m.validate(),
            ReferenceTargetKind::SemanticSources,
            "dk.missing",
        );
    }

    #[test]
    fn a_joinsets_pooled_condition_must_resolve() {
        for absent in [true, false] {
            let mut m = valid_manifest();
            if let RoleDefinition::Joinset {
                on_expr_id,
                filter_expr_id,
                ..
            } = &mut m
                .semantic_sources
                .get_mut("dk.join")
                .expect("the fixture holds the Joinset")
                .role_definition
            {
                if absent {
                    *on_expr_id = Some(expr_id(777));
                } else {
                    *filter_expr_id = Some(expr_id(777));
                }
            } else {
                unreachable!("the fixture's `dk.join` is a Joinset");
            }
            assert_dangling(
                m.validate(),
                ReferenceTargetKind::Expressions,
                "00000000-0000-0000-0000-000000000309",
            );
        }
    }

    #[test]
    fn a_joinsets_condition_is_absent_iff_it_is_a_cross_join() {
        for cross in [false, true] {
            let mut m = valid_manifest();
            let RoleDefinition::Joinset {
                on_expr_id,
                join_type,
                ..
            } = &mut m
                .semantic_sources
                .get_mut("dk.join")
                .expect("the fixture holds the Joinset")
                .role_definition
            else {
                unreachable!("the fixture's `dk.join` is a Joinset");
            };
            if cross {
                *join_type = JoinsetJoinType::Cross;
            } else {
                *on_expr_id = None;
            }
            assert_structural(m.validate(), CROSS_CONDITION_RULE);
        }
    }

    #[test]
    fn row13_temporal_identifying_dim_dangling() {
        let mut m = valid_manifest();
        m.semantic_sources.get_mut("dk.orders").unwrap().temporal = Some(TemporalShape::new(
            TemporalType::timeseries("sem.missing"),
            Some(Granularity::Day),
        ));
        assert_dangling(m.validate(), ReferenceTargetKind::Semantics, "sem.missing");
    }

    #[test]
    fn a_composer_may_carry_a_derived_temporal_shape() {
        let mut m = valid_manifest();
        m.semantic_sources.get_mut("dk.union").unwrap().temporal = Some(TemporalShape::new(
            TemporalType::timeseries("sem.occurred"),
            Some(Granularity::Day),
        ));
        assert_valid(&m);

        m.semantic_sources.get_mut("dk.union").unwrap().temporal = Some(TemporalShape::new(
            TemporalType::timeseries("sem.missing"),
            Some(Granularity::Day),
        ));
        assert_dangling(m.validate(), ReferenceTargetKind::Semantics, "sem.missing");
    }

    #[test]
    fn a_stack_over_two_clocks_is_reported_on_a_decoded_artifact() {
        let mut m = valid_manifest();
        m.semantic_sources.get_mut("dk.union.a").unwrap().temporal = Some(TemporalShape::new(
            TemporalType::events("sem.occurred"),
            Some(Granularity::Day),
        ));
        m.semantic_sources.get_mut("dk.union.b").unwrap().temporal = Some(TemporalShape::new(
            TemporalType::events("sem.region"),
            Some(Granularity::Day),
        ));
        let faults = m.validate().expect_err("two axes under one stack");
        assert!(
            faults.iter().any(|fault| matches!(
                fault.kind,
                CompileError::ComposedTemporalDisagreement { ref composite, ref shapes }
                    if composite == "dk.union" && shapes.len() == 2
            )),
            "expected the disagreement, naming the stack and both claims, got {faults:?}"
        );
    }

    #[test]
    fn the_agreement_restatement_admits_what_compile_produces() {
        let mut m = valid_manifest();
        m.semantic_sources.get_mut("dk.union").unwrap().temporal = Some(TemporalShape::new(
            TemporalType::events("sem.occurred"),
            None,
        ));
        assert_valid(&m);

        m.semantic_sources.get_mut("dk.union.a").unwrap().temporal = Some(TemporalShape::new(
            TemporalType::events("sem.occurred"),
            Some(Granularity::Day),
        ));
        m.semantic_sources.get_mut("dk.union.b").unwrap().temporal = Some(TemporalShape::new(
            TemporalType::events("sem.occurred"),
            Some(Granularity::Month),
        ));
        assert_valid(&m);
    }

    #[test]
    fn row15_bitmask_out_of_range_bit() {
        let mut m = valid_manifest();
        m.semantic_sources
            .get_mut("dk.customers")
            .unwrap()
            .semantic_bitmask = SemanticBitmask::from_positions([BIT_KEY_B, 9]);
        assert_dangling(
            m.validate(),
            ReferenceTargetKind::Semantics,
            "bit_position 9",
        );
    }

    #[test]
    fn a_gapped_bit_position_set_is_structural() {
        const RULE: &str =
            "SemanticDefinition.bit_position values must be exactly 0..n over the registry";

        let mut m = valid_manifest();
        m.semantic_definitions
            .entries
            .get_mut("sem.region")
            .expect("the fixture registers region")
            .bit_position = 9;
        assert_structural(m.validate(), RULE);

        let mut m = valid_manifest();
        m.semantic_definitions
            .entries
            .get_mut("sem.region")
            .expect("the fixture registers region")
            .bit_position = BIT_AMOUNT;
        assert_structural(m.validate(), RULE);
    }

    #[test]
    fn row17_id_not_equal_to_key() {
        let mut m = valid_manifest();
        let entry = m.semantic_sources.remove("dk.customers").unwrap();
        m.semantic_sources.insert("dk.rekeyed".to_string(), entry);
        assert_structural(m.validate(), "entity id must equal its map key");
    }

    #[test]
    fn row17_source_id_not_equal_to_key_reports_source_site() {
        let mut m = valid_manifest();
        let src = m.sources.remove("src.orders").unwrap();
        m.sources.insert("src.rekeyed".to_string(), src);
        assert!(
            faults(m.validate()).iter().any(|fault| matches!(
                fault,
                CompileError::StructuralViolation {
                    site: ReferenceSite::Source(id),
                    rule,
                } if id == "src.rekeyed" && *rule == "entity id must equal its map key"
            )),
            "expected the fault to locate at the Source site"
        );
    }

    #[test]
    fn a_composition_below_its_minimum_operands_is_structural() {
        let mut m = valid_manifest();
        m.semantic_sources
            .get_mut("dk.union")
            .expect("the fixture holds the Unionset")
            .role_definition =
            RoleDefinition::unionset(MultisetOp::UnionAll, vec!["dk.union.a".to_owned()], true);
        let spare = m
            .semantic_sources
            .get_mut("dk.union.b")
            .expect("the fixture holds the second branch");
        spare.originators.clear();
        spare.public = true;
        assert_valid(&m);

        let mut m = valid_manifest();
        m.semantic_sources
            .get_mut("dk.union")
            .expect("the fixture holds the Unionset")
            .role_definition = RoleDefinition::unionset(
            MultisetOp::UnionDistinct,
            vec!["dk.union.a".to_owned()],
            true,
        );
        assert_structural(
            m.validate(),
            "Unionset.branches must have at least 2 operands (len >= 2)",
        );

        let mut m = valid_manifest();
        m.semantic_sources
            .get_mut("dk.grain")
            .expect("the fixture holds the Grainset")
            .role_definition = RoleDefinition::grainset(vec!["dk.orders".to_owned()], Vec::new());
        assert_valid(&m);
    }

    #[test]
    fn structural5_non_canonical_bitmask() {
        let mut m = valid_manifest();
        m.semantic_sources
            .get_mut("dk.customers")
            .unwrap()
            .semantic_bitmask = SemanticBitmask { words: vec![1, 0] };
        assert_structural(
            m.validate(),
            "SemanticBitmask must be canonical (no trailing all-zero words)",
        );
    }

    #[test]
    fn structural6_global_id_collision() {
        let mut m = valid_manifest();
        let mut clash = PhysicalSource::new(
            "dk.orders",
            "x",
            SourceAlias::from("ps0"),
            StorageProperties::bare(StorageTarget::table(table_ref("warehouse.x"))),
            SchemaProvenance::model_fed(SourceSchema {
                columns: Vec::new(),
            }),
            SchemaId::from("fixture-schema"),
        );
        clash.id = "dk.orders".into();
        m.sources.insert("dk.orders".to_string(), clash);
        assert_structural(
            m.validate(),
            "entity ids must be globally unique across collections",
        );
    }

    #[test]
    fn a_set_bit_with_no_signature_is_a_violation() {
        let mut m = valid_manifest();
        let dropped = m
            .semantic_sources
            .get_mut("dk.orders")
            .expect("the fixture holds `dk.orders`")
            .semantic_signature_ids
            .pop()
            .expect("the leaf owns signatures");
        assert!(m.semantic_signatures.contains_key(&dropped));
        assert_structural(
            m.validate(),
            "a reading source's semantic_bitmask must set exactly the bit_positions of its \
             listed signatures' definitions",
        );
    }

    #[test]
    fn a_grainset_listing_a_signature_is_structural() {
        let mut m = valid_manifest();
        let mut grain = m
            .semantic_sources
            .get_mut("dk.grain")
            .expect("the fixture holds `dk.grain`")
            .clone();
        let region = qualified("grain", "region");
        sign(&mut m, &mut grain, "sem.region", &region);
        m.semantic_sources.insert(grain.id.clone(), grain);
        assert_structural(m.validate(), GRAINSET_OWNS_UNMAPPED_METRICS_RULE);
    }

    #[test]
    fn a_grainset_listing_a_signature_reports_only_the_grainset_rule() {
        let mut m = valid_manifest();
        let mut grain = m
            .semantic_sources
            .get_mut("dk.grain")
            .expect("the fixture holds `dk.grain`")
            .clone();
        let region = qualified("grain", "region");
        sign(&mut m, &mut grain, "sem.region", &region);
        m.semantic_sources.insert(grain.id.clone(), grain);

        let reported = faults(m.validate_with(FaultReporting::FirstFault));
        assert_eq!(
            reported.len(),
            1,
            "exactly one fault must be reported: {reported:?}"
        );
        assert!(
            matches!(
                &reported[0],
                CompileError::StructuralViolation { rule, .. }
                    if *rule == GRAINSET_OWNS_UNMAPPED_METRICS_RULE
            ),
            "the reported fault must be GRAINSET_OWNS_UNMAPPED_METRICS_RULE, not AGREEMENT_RULE: {reported:?}"
        );
    }

    const GRAIN_METRIC: &str = "sem.zratio";

    fn grain_metric_over_final_values(m: &mut SemanticManifest) -> EntityId {
        let amount = NamedReference::qualified("ssg1", "sem.amount").expect("an alias and a name");
        grain_metric_reading(m, amount, None)
    }

    fn own_final_value_layer(column: &NamedReference) -> ExprLayer {
        match column.parts() {
            [alias, name] if alias.as_str() == "ssg1" && name.as_str() == "sem.amount" => {
                ExprLayer::Aggregate
            }
            _ => ExprLayer::Scalar,
        }
    }

    fn grain_metric_reading(
        m: &mut SemanticManifest,
        column: NamedReference,
        filter: Option<NamedReference>,
    ) -> EntityId {
        let own = NamedReference::qualified("ssg1", "sem.amount").expect("an alias and a name");
        grain_metric_of(m, Expr::Column(own) / Expr::Column(column), filter)
    }

    fn grain_metric_of(
        m: &mut SemanticManifest,
        tree: Expr<Link>,
        filter: Option<NamedReference>,
    ) -> EntityId {
        register(m, SemanticRole::Metric, &[GRAIN_METRIC]);
        let id = mint_signature_id("dk.grain", GRAIN_METRIC);
        let root = m
            .expressions
            .emit_linked_reading(
                &tree,
                &id,
                ExprRoot::SignatureExpression,
                Vec::<EntityId>::new(),
                &own_final_value_layer,
            )
            .expect("emit");
        let mut signature = crate::manifest::SemanticSignature::new(id.clone(), GRAIN_METRIC, root);
        if let Some(filter) = filter {
            let predicate = m
                .expressions
                .emit_linked_reading(
                    &Expr::Column(filter),
                    &id,
                    ExprRoot::SignatureFilter,
                    Vec::<EntityId>::new(),
                    &own_final_value_layer,
                )
                .expect("emit");
            signature = signature.with_filter(predicate);
        }
        m.semantic_signatures.insert(id.clone(), signature);
        let bit = m.semantic_definitions.entries[GRAIN_METRIC].bit_position;
        let grain = m
            .semantic_sources
            .get_mut("dk.grain")
            .expect("the fixture holds `dk.grain`");
        grain.semantic_bitmask = SemanticBitmask::from_positions(
            (grain.semantic_bitmask.set_positions().into_iter()).chain([bit]),
        );
        grain.semantic_signature_ids.push(id.clone());
        id
    }

    #[test]
    fn a_grainset_listing_the_signature_of_a_metric_no_child_maps_validates() {
        let mut m = valid_manifest();
        grain_metric_over_final_values(&mut m);
        assert_valid(&m);
    }

    #[test]
    fn a_grainset_listing_two_signatures_for_one_metric_is_structural() {
        let mut m = valid_manifest();
        grain_metric_over_final_values(&mut m);
        let amount = || {
            Expr::Column(
                NamedReference::qualified("ssg1", "sem.amount").expect("an alias and a name"),
            )
        };
        let twin_id = mint_signature_id("dk.grain", "twin");
        let root = m
            .expressions
            .emit_linked_reading(
                &(amount() * amount()),
                &twin_id,
                ExprRoot::SignatureExpression,
                Vec::<EntityId>::new(),
                &own_final_value_layer,
            )
            .expect("emit");
        let twin = crate::manifest::SemanticSignature::new(twin_id.clone(), GRAIN_METRIC, root);
        m.semantic_sources
            .get_mut("dk.grain")
            .expect("the fixture holds `dk.grain`")
            .semantic_signature_ids
            .push(twin_id.clone());
        m.semantic_signatures.insert(twin_id, twin);
        assert_eq!(
            faults(m.validate()),
            [structural(
                ReferenceSite::SemanticSource("dk.grain".to_owned()),
                "a reading source must list exactly one SemanticSignature per set bit of its \
                 semantic_bitmask",
            )]
        );
    }

    fn self_referencing_grain_metric(m: &mut SemanticManifest) {
        let own = |name: &str| {
            Expr::Column(NamedReference::qualified("ssg1", name).expect("an alias and a name"))
        };
        let id = grain_metric_of(m, own(GRAIN_METRIC) + own("sem.amount"), None);
        let root = m.semantic_signatures[&id].expression_id;
        assert_eq!(
            m.expressions.physical[&root].layer,
            ExprLayer::PostAggregate
        );
    }

    #[test]
    fn a_grainset_metric_reading_itself_as_a_column_is_structural() {
        let mut m = valid_manifest();
        self_referencing_grain_metric(&mut m);
        assert_structural(m.validate(), GRAINSET_OWN_COLUMNS_RULE);
    }

    #[test]
    fn a_grainset_child_metric_reading_itself_as_a_column_is_structural() {
        let mut m = valid_manifest();
        self_referencing_grain_metric(&mut m);
        let mut parent = source(
            "dk.wrap",
            "wrap",
            m.semantic_sources["dk.grain"].semantic_bitmask.clone(),
            RoleDefinition::grainset(vec!["dk.grain".to_owned()], Vec::new()),
        );
        parent.alias = SourceAlias::from("ssg7");
        m.semantic_sources.insert(parent.id.clone(), parent);
        assert_structural(m.validate(), GRAINSET_OWN_COLUMNS_RULE);
    }

    #[test]
    fn a_grainset_metric_reading_a_definition_no_child_serves_is_structural() {
        let mut m = valid_manifest();
        register(&mut m, SemanticRole::Measure, &["sem.zorphan"]);
        let orphan = NamedReference::qualified("ssg1", "sem.zorphan").expect("an alias and a name");
        grain_metric_reading(&mut m, orphan, None);
        assert_structural(m.validate(), GRAINSET_OWN_COLUMNS_RULE);
    }

    fn assert_misplaced_in_the_grainset(
        m: &SemanticManifest,
        expression: PhysicalExprId,
        column: NamedReference,
    ) {
        let misplaced = CompileError::ColumnQualifierMismatch {
            expression,
            column,
            expected: vec![SourceAlias::from("ssg1")],
        };
        let reported = faults(m.validate());
        assert!(
            !reported.is_empty() && reported.iter().all(|fault| *fault == misplaced),
            "expected only {misplaced:?}, got {reported:?}"
        );
    }

    #[test]
    fn a_grainset_metric_reading_another_sources_column_is_misplaced() {
        let mut m = valid_manifest();
        let foreign = NamedReference::qualified("ssu4", "sem.amount").expect("an alias and a name");
        let id = grain_metric_reading(&mut m, foreign.clone(), None);
        let root = m.semantic_signatures[&id].expression_id;
        assert_misplaced_in_the_grainset(&m, root, foreign);
    }

    #[test]
    fn a_grainset_metric_reading_a_one_part_column_is_misplaced() {
        let mut m = valid_manifest();
        let unqualified = NamedReference::unqualified("sem.amount").expect("a name");
        let id = grain_metric_reading(&mut m, unqualified.clone(), None);
        let root = m.semantic_signatures[&id].expression_id;
        assert_misplaced_in_the_grainset(&m, root, unqualified);
    }

    #[test]
    fn a_grainset_metric_filter_reading_another_sources_column_is_misplaced() {
        let mut m = valid_manifest();
        let own = NamedReference::qualified("ssg1", "sem.amount").expect("an alias and a name");
        let foreign = NamedReference::qualified("ssu4", "sem.region").expect("an alias and a name");
        let id = grain_metric_reading(&mut m, own, Some(foreign.clone()));
        let filter = (m.semantic_signatures[&id].filter_expr_id).expect("the signature's filter");
        assert_misplaced_in_the_grainset(&m, filter, foreign);
    }

    fn row_level_grain_metric(m: &mut SemanticManifest) {
        let region = Expr::Column(
            NamedReference::qualified("ssg1", "sem.region").expect("an alias and a name"),
        );
        let id = grain_metric_of(m, region.clone() / region, None);
        let root = m.semantic_signatures[&id].expression_id;
        assert_eq!(m.expressions.physical[&root].layer, ExprLayer::Scalar);
    }

    #[test]
    fn a_grainset_listing_the_signature_of_a_row_level_metric_is_structural() {
        let mut m = valid_manifest();
        row_level_grain_metric(&mut m);
        assert_eq!(
            faults(m.validate()),
            [structural(
                ReferenceSite::SemanticSource("dk.grain".to_owned()),
                GRAINSET_OWNS_UNMAPPED_METRICS_RULE,
            )]
        );
    }

    #[test]
    fn a_grainset_child_listing_the_signature_of_a_row_level_metric_is_structural() {
        let mut m = valid_manifest();
        row_level_grain_metric(&mut m);
        let mut parent = source(
            "dk.wrap",
            "wrap",
            m.semantic_sources["dk.grain"].semantic_bitmask.clone(),
            RoleDefinition::grainset(vec!["dk.grain".to_owned()], Vec::new()),
        );
        parent.alias = SourceAlias::from("ssg7");
        m.semantic_sources.insert(parent.id.clone(), parent);
        assert_eq!(
            faults(m.validate()),
            [structural(
                ReferenceSite::SemanticSource("dk.grain".to_owned()),
                GRAINSET_OWNS_UNMAPPED_METRICS_RULE,
            )]
        );
    }

    #[test]
    fn a_grainset_listing_the_signature_of_a_measure_is_structural() {
        let mut m = valid_manifest();
        expose(
            &mut m,
            "dk.grain",
            "sem.amount",
            &qualified("ssg1", "amount"),
        );
        assert_structural(m.validate(), GRAINSET_OWNS_UNMAPPED_METRICS_RULE);
    }

    #[test]
    fn a_grainset_listing_the_signature_of_a_metric_a_child_maps_is_structural() {
        let mut m = valid_manifest();
        grain_metric_over_final_values(&mut m);
        let bit = m.semantic_definitions.entries[GRAIN_METRIC].bit_position;
        let child = m
            .semantic_sources
            .get_mut("dk.orders")
            .expect("the fixture holds `dk.orders`");
        child.semantic_bitmask = SemanticBitmask::from_positions(
            (child.semantic_bitmask.set_positions().into_iter()).chain([bit]),
        );
        assert!(faults(m.validate()).contains(&structural(
            ReferenceSite::SemanticSource("dk.grain".to_owned()),
            GRAINSET_OWNS_UNMAPPED_METRICS_RULE,
        )));
    }

    fn mask_bit(m: &mut SemanticManifest, id: &str, bit: u32, set: bool) {
        let entry = (m.semantic_sources.get_mut(id)).expect("the fixture holds the source");
        let held = (entry.semantic_bitmask.set_positions().into_iter()).filter(|held| *held != bit);
        entry.semantic_bitmask = match set {
            true => SemanticBitmask::from_positions(held.chain([bit])),
            false => SemanticBitmask::from_positions(held),
        };
    }

    #[test]
    fn a_grainset_whose_mask_clears_the_bit_of_a_metric_it_computes_is_structural() {
        let mut m = valid_manifest();
        grain_metric_over_final_values(&mut m);
        let bit = m.semantic_definitions.entries[GRAIN_METRIC].bit_position;
        mask_bit(&mut m, "dk.grain", bit, false);
        assert_eq!(
            faults(m.validate()),
            [structural(
                ReferenceSite::SemanticSource("dk.grain".to_owned()),
                GRAINSET_MASK_AGREEMENT_RULE,
            )]
        );
    }

    #[test]
    fn a_grainset_whose_mask_sets_a_bit_no_grain_sets_and_no_signature_lists_is_structural() {
        let mut m = valid_manifest();
        register(&mut m, SemanticRole::Metric, &[GRAIN_METRIC]);
        let bit = m.semantic_definitions.entries[GRAIN_METRIC].bit_position;
        mask_bit(&mut m, "dk.grain", bit, true);
        assert_eq!(
            faults(m.validate()),
            [structural(
                ReferenceSite::SemanticSource("dk.grain".to_owned()),
                GRAINSET_MASK_AGREEMENT_RULE,
            )]
        );
    }

    #[test]
    fn a_grainset_child_whose_mask_clears_the_bit_of_a_metric_it_computes_is_structural() {
        let mut m = valid_manifest();
        grain_metric_over_final_values(&mut m);
        let mut parent = source(
            "dk.wrap",
            "wrap",
            m.semantic_sources["dk.grain"].semantic_bitmask.clone(),
            RoleDefinition::grainset(vec!["dk.grain".to_owned()], Vec::new()),
        );
        parent.alias = SourceAlias::from("ssg7");
        m.semantic_sources.insert(parent.id.clone(), parent);
        assert_valid(&m);

        let bit = m.semantic_definitions.entries[GRAIN_METRIC].bit_position;
        mask_bit(&mut m, "dk.grain", bit, false);
        assert_eq!(
            faults(m.validate()),
            [
                structural(
                    ReferenceSite::SemanticSource("dk.grain".to_owned()),
                    GRAINSET_MASK_AGREEMENT_RULE,
                ),
                structural(
                    ReferenceSite::SemanticSource("dk.wrap".to_owned()),
                    GRAINSET_MASK_AGREEMENT_RULE,
                ),
            ]
        );
    }

    #[test]
    fn a_grainset_signatures_layer_counts_its_own_aggregate_columns_as_aggregate() {
        let mut m = valid_manifest();
        let id = grain_metric_over_final_values(&mut m);
        let root = m.semantic_signatures[&id].expression_id;
        assert_eq!(
            m.expressions.physical[&root].layer,
            ExprLayer::PostAggregate
        );
        m.expressions
            .physical
            .get_mut(&root)
            .expect("the entry")
            .layer = ExprLayer::Scalar;
        assert!(
            faults(m.validate()).contains(&structural(ReferenceSite::ExpressionPool, LAYER_RULE))
        );
    }

    #[test]
    fn a_composers_set_bit_with_no_signature_is_a_violation() {
        let mut m = valid_manifest();
        m.semantic_sources
            .get_mut("dk.union")
            .expect("the fixture holds `dk.union`")
            .semantic_signature_ids
            .clear();
        assert_structural(
            m.validate(),
            "a reading source's semantic_bitmask must set exactly the bit_positions of its \
             listed signatures' definitions",
        );
    }

    #[test]
    fn a_composers_signature_whose_bit_is_unset_is_a_violation() {
        let mut m = valid_manifest();
        m.semantic_sources
            .get_mut("dk.join")
            .expect("the fixture holds `dk.join`")
            .semantic_bitmask =
            SemanticBitmask::from_positions([BIT_AMOUNT, BIT_KEY_A, BIT_KEY_B, BIT_OCCURRED]);
        assert_structural(
            m.validate(),
            "a reading source's semantic_bitmask must set exactly the bit_positions of its \
             listed signatures' definitions",
        );
    }

    #[test]
    fn two_signatures_for_one_definition_on_one_source_is_a_violation() {
        let mut m = valid_manifest();
        let mut orders = m.semantic_sources["dk.orders"].clone();
        let id: EntityId = "sig.rival.region".into();
        let root = m
            .expressions
            .emit(
                &qualified("ps1", "region_again"),
                &id,
                ExprRoot::SignatureExpression,
                consumers(&orders),
            )
            .expect("emit");
        m.semantic_signatures.insert(
            id.clone(),
            crate::manifest::SemanticSignature::new(id.clone(), "sem.region", root),
        );
        orders.semantic_signature_ids.push(id);
        orders.semantic_signature_ids.sort();
        m.semantic_sources.insert(orders.id.clone(), orders);

        let reported = faults(m.validate_with(FaultReporting::FirstFault));
        assert_eq!(
            reported.len(),
            1,
            "the duplicate is one fault, and the mask still agrees: {reported:?}"
        );
        assert_structural(
            m.validate(),
            "a reading source must list exactly one SemanticSignature per set bit of its \
             semantic_bitmask",
        );
    }

    #[test]
    fn a_signature_listed_by_two_sources_is_a_violation() {
        let mut m = valid_manifest();
        let borrowed = m.semantic_sources["dk.orders"].semantic_signature_ids[0].clone();
        m.semantic_sources
            .get_mut("dk.customers")
            .expect("the fixture holds `dk.customers`")
            .semantic_signature_ids
            .push(borrowed);
        assert_structural(
            m.validate(),
            "every SemanticSignature must be listed by exactly one semantic_sources entry",
        );
    }

    #[test]
    fn a_signature_naming_no_definition_is_dangling() {
        let mut m = valid_manifest();
        let id = m.semantic_sources["dk.orders"].semantic_signature_ids[0].clone();
        m.semantic_signatures
            .get_mut(&id)
            .expect("the listed signature is held")
            .definition_id = "sem.nope".to_owned();
        assert_dangling(m.validate(), ReferenceTargetKind::Semantics, "sem.nope");
    }

    #[test]
    fn a_signature_naming_no_pooled_expression_is_dangling() {
        let mut m = valid_manifest();
        let id = m.semantic_sources["dk.orders"].semantic_signature_ids[0].clone();
        m.semantic_signatures
            .get_mut(&id)
            .expect("the listed signature is held")
            .expression_id = expr_id(0xDEAD_BEEF);
        assert_dangling(
            m.validate(),
            ReferenceTargetKind::Expressions,
            "00000000-0000-0000-0000-0000deadbeef",
        );
    }

    #[test]
    fn a_signature_filter_whose_layer_is_not_scalar_is_refused() {
        let id = crate::identity::mint_signature_id("dk.orders", "sem.amount");
        let filter_id = crate::identity::mint_expr_id(&id, ExprRoot::SignatureFilter);
        let stored_filter = |m: &mut SemanticManifest, expr: Expr<Link>| {
            let layer = m
                .expressions
                .layer_of(&expr)
                .expect("each link names an entry");
            m.expressions.physical.insert(
                filter_id,
                ManifestExpression {
                    id: filter_id,
                    expr: StoredExpr::from(expr),
                    layer,
                    physical_sources: BTreeSet::from(["src.orders".to_owned()]),
                },
            );
            m.semantic_signatures
                .get_mut(&id)
                .expect("the measure's signature is held")
                .filter_expr_id = Some(filter_id);
        };

        let mut aggregate_in_the_tree = valid_manifest();
        stored_filter(
            &mut aggregate_in_the_tree,
            sum(Expr::Column(
                NamedReference::qualified("ps1", "amount").expect("an alias and a name"),
            )),
        );
        let mut link_to_a_measure = valid_manifest();
        let measure = link_to_a_measure.semantic_signatures[&id].expression_id;
        link_to_a_measure.expressions.physical.remove(&measure);
        link_to_a_measure
            .expressions
            .emit(
                &sum(qualified("ps1", "amount")),
                &id,
                ExprRoot::SignatureExpression,
                ["src.orders".to_owned()],
            )
            .expect("the measure's marker is free again");
        stored_filter(&mut link_to_a_measure, Expr::Leaf(Link(measure)));

        for m in [aggregate_in_the_tree, link_to_a_measure] {
            assert_structural(
                m.validate(),
                "a SemanticSignature's filter entry must be row-level: its layer, with each \
                 link counted, is scalar",
            );
        }
    }

    #[test]
    fn a_signature_filter_that_is_its_expression_entry_is_refused() {
        let mut m = valid_manifest();
        let id = crate::identity::mint_signature_id("dk.orders", "sem.amount");
        let signature = m
            .semantic_signatures
            .get_mut(&id)
            .expect("the measure's signature is held");
        signature.filter_expr_id = Some(signature.expression_id);
        assert_structural(
            m.validate(),
            "a SemanticSignature's filter entry must not be its expression entry",
        );
    }

    #[test]
    fn a_dataset_signature_realised_by_another_datasets_tree_is_a_violation() {
        let mut m = valid_manifest();
        let id = crate::identity::mint_signature_id("dk.customers", "sem.key.a");
        let previous = m.semantic_signatures[&id].expression_id;
        m.expressions.physical.remove(&previous);
        let root = m
            .expressions
            .emit(
                &qualified("ps0", "key_a"),
                &id,
                ExprRoot::SignatureExpression,
                ["src.orders".to_owned()],
            )
            .expect("emit");
        m.semantic_signatures
            .get_mut(&id)
            .expect("the fixture holds the second leaf's signature")
            .expression_id = root;
        let reported = faults(m.validate());
        assert!(
            reported.iter().all(|fault| matches!(
                fault,
                CompileError::StructuralViolation { rule, .. } if *rule == DATASET_SOURCE_RULE
            )),
            "the wrong stamp must be the only fault, got {reported:?}"
        );
        assert_structural(m.validate(), DATASET_SOURCE_RULE);
    }

    #[test]
    fn a_dataset_signature_tree_naming_no_consuming_source_is_a_violation() {
        let mut m = valid_manifest();
        let id = crate::identity::mint_signature_id("dk.orders", "sem.region");
        let previous = m.semantic_signatures[&id].expression_id;
        m.expressions.physical.remove(&previous);
        let root = m
            .expressions
            .emit(
                &qualified("ps1", "region"),
                &id,
                ExprRoot::SignatureExpression,
                Vec::<String>::new(),
            )
            .expect("emit");
        m.semantic_signatures
            .get_mut(&id)
            .expect("the fixture holds the signature")
            .expression_id = root;
        assert_structural(m.validate(), DATASET_SOURCE_RULE);
    }

    #[test]
    fn a_dataset_signature_filter_naming_another_datasets_source_is_a_violation() {
        let mut m = valid_manifest();
        let id = crate::identity::mint_signature_id("dk.orders", "sem.amount");
        let root = m
            .expressions
            .emit(
                &qualified("ps1", "region"),
                &id,
                ExprRoot::SignatureFilter,
                ["src.customers".to_owned()],
            )
            .expect("the filter marker is free under this signature");
        m.semantic_signatures
            .get_mut(&id)
            .expect("the measure's signature is held")
            .filter_expr_id = Some(root);
        assert_structural(m.validate(), DATASET_SOURCE_RULE);
    }

    #[test]
    fn a_dimension_signature_carrying_a_filter_is_a_violation() {
        let mut m = valid_manifest();
        let id = crate::identity::mint_signature_id("dk.orders", "sem.region");
        let signature = m
            .semantic_signatures
            .get_mut(&id)
            .expect("the dimension's signature is held");
        signature.filter_expr_id = Some(signature.expression_id);
        assert_structural(
            m.validate(),
            "only a Measure's or a Metric's SemanticSignature may carry a filter",
        );
        assert_structural(
            m.validate(),
            "a SemanticSignature's filter entry must not be its expression entry",
        );
    }

    #[test]
    fn an_entry_whose_layer_is_not_its_trees_is_refused() {
        let mut m = valid_manifest();
        let entry = m
            .expressions
            .physical
            .values_mut()
            .next()
            .expect("the fixture pool is non-empty");
        entry.layer = ExprLayer::Aggregate;
        assert_structural(m.validate(), LAYER_RULE);
    }

    #[test]
    fn pooled_expression_id_must_equal_its_key() {
        let mut m = valid_manifest();
        let entry = m
            .expressions
            .physical
            .values_mut()
            .next()
            .expect("the fixture pool is non-empty");
        entry.id = expr_id(999);
        assert_structural(m.validate(), "ManifestExpression.id must equal its map key");
    }

    fn sum<L: Leaf>(inner: Expr<L>) -> Expr<L> {
        let agg = Agg::of_name("sum").expect("a catalog aggregate");
        wrap_aggregate(inner, (agg.function(), agg.distinct()), "test").expect("sum wraps")
    }

    #[test]
    fn an_entry_that_two_signatures_name_is_refused() {
        let mut m = valid_manifest();
        let amount = crate::identity::mint_signature_id("dk.orders", "sem.amount");
        let shared = m.semantic_signatures[&amount].expression_id;
        let sharer = crate::identity::mint_signature_id("dk.orders", "sem.region");
        m.semantic_signatures
            .get_mut(&sharer)
            .expect("the fixture holds the signature")
            .expression_id = shared;
        assert_structural(
            m.validate(),
            "an entry of the expression store must be named by one root: one signature's \
             expression or filter, or one Joinset source's condition or relationship \
             filter",
        );
    }

    #[test]
    fn an_entry_that_no_signature_and_no_joinset_source_names_is_an_orphan() {
        let mut m = valid_manifest();
        let orphan = m
            .expressions
            .emit(
                &qualified("ps1", "region"),
                "dk.nobody",
                ExprRoot::SignatureExpression,
                Vec::<String>::new(),
            )
            .expect("emit");
        assert_eq!(
            faults(m.validate()),
            [CompileError::OrphanExpression { expr_id: orphan }]
        );
    }

    #[test]
    fn a_nested_calls_undeclared_option_is_refused() {
        let mut m = valid_manifest();
        let id = crate::identity::mint_signature_id("dk.orders", "sem.region");
        let root = m.semantic_signatures[&id].expression_id;
        let region = r#"{"type":"column","content":["ps1","region"]}"#;
        let tree: StoredExpr = serde_json::from_str(&format!(
            r#"{{"type":"scalar","content":{{"name":"upper","args":[{{"type":"scalar","content":{{"name":"concat","options":{{"null_policy":"IGNORE_NULLS"}},"args":[{region},{region}]}}}}]}}}}"#
        ))
        .expect("reading reads any option");
        m.expressions
            .physical
            .get_mut(&root)
            .expect("the fixture's entry")
            .expr = tree;
        assert_eq!(
            faults(m.validate()),
            [CompileError::UndeclaredOption {
                expr_id: root,
                error: semstrait_sql::expr::CallError::UnknownOption {
                    function: "concat",
                    option: "null_policy".to_owned(),
                },
            }]
        );
    }

    fn link(id: PhysicalExprId) -> Expr<Link> {
        Expr::Leaf(Link(id))
    }

    fn expression_of(data_kind: &str, semantic: &str) -> PhysicalExprId {
        mint_expr_id(
            &mint_signature_id(data_kind, semantic),
            ExprRoot::SignatureExpression,
        )
    }

    fn put(
        m: &mut SemanticManifest,
        owner: &str,
        root: ExprRoot,
        expr: Expr<Link>,
    ) -> PhysicalExprId {
        let id = mint_expr_id(owner, root);
        let layer = m.expressions.layer_of(&expr).unwrap_or(ExprLayer::Scalar);
        let physical_sources = (m.expressions.physical.remove(&id))
            .map(|entry| entry.physical_sources)
            .unwrap_or_default();
        m.expressions.physical.insert(
            id,
            ManifestExpression {
                id,
                expr: StoredExpr::from(expr),
                layer,
                physical_sources,
            },
        );
        id
    }

    fn put_expression(
        m: &mut SemanticManifest,
        data_kind: &str,
        semantic: &str,
        expr: Expr<Link>,
    ) -> EntityId {
        let signature = mint_signature_id(data_kind, semantic);
        put(m, &signature, ExprRoot::SignatureExpression, expr);
        signature
    }

    #[test]
    fn a_link_that_names_no_entry_is_dangling() {
        let mut m = valid_manifest();
        put_expression(&mut m, "dk.orders", "sem.region", link(expr_id(0xBAD)));
        assert_eq!(
            faults(m.validate()),
            [CompileError::DanglingReference {
                from: ReferenceSite::ExpressionPool,
                to_kind: ReferenceTargetKind::Expressions,
                target_id: "00000000-0000-0000-0000-000000000bad".to_owned(),
            }]
        );
    }

    #[test]
    fn a_link_in_a_joinset_condition_or_relationship_filter_is_refused() {
        let mut m = valid_manifest();
        let region = expression_of("dk.join", "sem.region");
        put(&mut m, "dk.join", ExprRoot::JoinsetOn, link(region));
        let relationship_filter = put(&mut m, "dk.join", ExprRoot::JoinsetFilter, link(region));
        let RoleDefinition::Joinset { filter_expr_id, .. } = &mut m
            .semantic_sources
            .get_mut("dk.join")
            .expect("the fixture holds the Joinset")
            .role_definition
        else {
            unreachable!("the fixture's `dk.join` is a Joinset");
        };
        *filter_expr_id = Some(relationship_filter);
        let joinset = ReferenceSite::SemanticSource("dk.join".to_owned());
        assert_eq!(
            faults(m.validate()),
            [
                structural(joinset.clone(), JOINSET_CONDITION_LINK_RULE),
                structural(joinset, JOINSET_CONDITION_LINK_RULE),
            ]
        );
    }

    fn expose(
        m: &mut SemanticManifest,
        data_kind: &str,
        semantic: &str,
        expr: &PhysicalExpr,
    ) -> EntityId {
        let mut owner =
            (m.semantic_sources.remove(data_kind)).expect("the fixture holds the source");
        let bit = m.semantic_definitions.entries[semantic].bit_position;
        owner.semantic_bitmask = SemanticBitmask::from_positions(
            owner
                .semantic_bitmask
                .set_positions()
                .into_iter()
                .chain([bit]),
        );
        sign(m, &mut owner, semantic, expr);
        m.semantic_sources.insert(owner.id.clone(), owner);
        mint_signature_id(data_kind, semantic)
    }

    #[test]
    fn a_link_in_a_compositions_dimension_or_measure_signature_is_refused() {
        let mut m = valid_manifest();
        expose(&mut m, "dk.union", "sem.key.a", &qualified("ssu4", "key_a"));
        let key = link(expression_of("dk.union", "sem.key.a"));
        let union_region = put_expression(&mut m, "dk.union", "sem.region", key);
        let region = link(expression_of("dk.join", "sem.region"));
        let join_amount = put_expression(&mut m, "dk.join", "sem.amount", region);
        assert_eq!(
            faults(m.validate()),
            [
                structural(
                    ReferenceSite::Signature(join_amount),
                    COMPOSED_SIGNATURE_LINK_RULE
                ),
                structural(
                    ReferenceSite::Signature(union_region),
                    COMPOSED_SIGNATURE_LINK_RULE
                ),
            ]
        );
    }

    fn put_filter(m: &mut SemanticManifest, signature: &EntityId, expr: Expr<Link>) {
        let filter = put(m, signature, ExprRoot::SignatureFilter, expr);
        (m.semantic_signatures.get_mut(signature))
            .expect("the fixture holds the signature")
            .filter_expr_id = Some(filter);
    }

    #[test]
    fn a_link_in_a_joinset_or_grainset_signature_filter_is_refused() {
        let mut m = valid_manifest();
        let join_amount = mint_signature_id("dk.join", "sem.amount");
        let region = expression_of("dk.join", "sem.region");
        put_filter(&mut m, &join_amount, link(region));
        let grain_amount = expose(
            &mut m,
            "dk.grain",
            "sem.amount",
            &qualified("ssg1", "amount"),
        );
        let amount = expression_of("dk.grain", "sem.amount");
        put_filter(&mut m, &grain_amount, link(amount));
        assert_eq!(
            faults(m.validate()),
            [
                structural(
                    ReferenceSite::SemanticSource("dk.grain".to_owned()),
                    GRAINSET_OWNS_UNMAPPED_METRICS_RULE,
                ),
                structural(
                    ReferenceSite::SemanticSource("dk.grain".to_owned()),
                    GRAINSET_OWN_COLUMNS_RULE,
                ),
                structural(
                    ReferenceSite::Signature(grain_amount),
                    COMPOSED_FILTER_LINK_RULE
                ),
                structural(
                    ReferenceSite::Signature(join_amount),
                    COMPOSED_FILTER_LINK_RULE
                ),
            ]
        );
    }

    fn register(m: &mut SemanticManifest, role: SemanticRole, ids: &[&str]) {
        let registered = m.semantic_definitions.entries.values().cloned();
        let added = ids.iter().map(|id| def(id, role));
        m.semantic_definitions = SemanticBitmap::freeze(registered.chain(added))
            .expect("the registry fits the bit width");
    }

    #[test]
    fn a_link_in_a_joinset_named_filters_body_is_refused() {
        let mut m = valid_manifest();
        register(&mut m, SemanticRole::Filter, &["sem.zfilter"]);
        expose(
            &mut m,
            "dk.join",
            "sem.zfilter",
            &qualified("ssd3", "region"),
        );
        let region = link(expression_of("dk.join", "sem.region"));
        let named = put_expression(&mut m, "dk.join", "sem.zfilter", region);
        assert_eq!(
            faults(m.validate()),
            [structural(
                ReferenceSite::Signature(named),
                JOINSET_NAMED_FILTER_LINK_RULE
            )]
        );
    }

    #[test]
    fn a_link_to_another_sources_entry_or_to_a_filter_entry_is_refused() {
        let mut m = valid_manifest();
        let other = link(expression_of("dk.customers", "sem.key.a"));
        let key = put_expression(&mut m, "dk.orders", "sem.key.a", other);
        assert_eq!(
            faults(m.validate()),
            [structural(ReferenceSite::Signature(key), LINK_TARGET_RULE)]
        );

        let mut m = valid_manifest();
        let amount = mint_signature_id("dk.orders", "sem.amount");
        let filter = m
            .expressions
            .emit(
                &qualified("ps1", "region"),
                &amount,
                ExprRoot::SignatureFilter,
                ["src.orders".to_owned()],
            )
            .expect("the filter marker is free under this signature");
        (m.semantic_signatures.get_mut(&amount))
            .expect("the measure's signature is held")
            .filter_expr_id = Some(filter);
        let region = put_expression(&mut m, "dk.orders", "sem.region", link(filter));
        assert_eq!(
            faults(m.validate()),
            [structural(
                ReferenceSite::Signature(region),
                LINK_TARGET_RULE
            )]
        );
    }

    #[test]
    fn a_joinset_metrics_link_to_the_joinsets_own_signature_validates() {
        let mut m = valid_manifest();
        register(&mut m, SemanticRole::Metric, &["sem.zmetric"]);
        expose(
            &mut m,
            "dk.join",
            "sem.zmetric",
            &qualified("ssd3", "amount"),
        );
        let amount = link(expression_of("dk.join", "sem.amount"));
        let doubled = amount * Expr::Literal(Literal::BigInt(2));
        put_expression(&mut m, "dk.join", "sem.zmetric", doubled);
        assert_valid(&m);
    }

    #[test]
    fn a_unionset_measures_filter_link_to_its_own_dimension_validates() {
        let mut m = valid_manifest();
        let amount = expose(
            &mut m,
            "dk.union",
            "sem.amount",
            &qualified("ssu4", "amount"),
        );
        let region = expression_of("dk.union", "sem.region");
        let in_eu = link(region).eq(Expr::Literal(Literal::String("eu".to_owned())));
        put_filter(&mut m, &amount, in_eu);
        assert_valid(&m);
    }

    #[test]
    fn a_cycle_of_links_is_refused() {
        let mut m = valid_manifest();
        for (semantic, linked) in [
            ("sem.region", "sem.key.a"),
            ("sem.key.a", "sem.region"),
            ("sem.key.b", "sem.key.b"),
        ] {
            let linked = link(expression_of("dk.orders", linked));
            put_expression(&mut m, "dk.orders", semantic, linked);
        }
        assert_eq!(
            faults(m.validate()),
            [
                structural(ReferenceSite::ExpressionPool, LINK_CYCLE_RULE),
                structural(ReferenceSite::ExpressionPool, LINK_CYCLE_RULE),
            ]
        );

        let mut m = valid_manifest();
        let chain: Vec<String> = (0..4096).map(|at| format!("sem.zchain.{at:04}")).collect();
        let chain: Vec<&str> = chain.iter().map(String::as_str).collect();
        register(&mut m, SemanticRole::Dimension, &chain);
        for semantic in &chain {
            expose(&mut m, "dk.orders", semantic, &qualified("ps1", "region"));
        }
        for pair in chain.windows(2) {
            let next = link(expression_of("dk.orders", pair[1]));
            put_expression(&mut m, "dk.orders", pair[0], next);
        }
        assert_valid(&m);
    }

    #[test]
    fn a_decoded_many_to_many_joinset_refuses_a_directional_cross_filter() {
        for direction in [CrossFilter::Left, CrossFilter::Right] {
            let mut m = valid_manifest();
            let RoleDefinition::Joinset {
                cardinality,
                cross_filter,
                ..
            } = &mut m
                .semantic_sources
                .get_mut("dk.join")
                .expect("the fixture holds a Joinset")
                .role_definition
            else {
                unreachable!("the fixture's `dk.join` is a Joinset");
            };
            *cardinality = Cardinality::ManyToMany;
            *cross_filter = direction;
            let faults = m.validate().expect_err("the resolved pair is refused");
            assert!(
                faults.iter().any(|fault| matches!(
                    fault.kind,
                    CompileError::RelationshipManyToManyCrossFilterDirectional { .. }
                )),
                "expected the dedicated variant, got {faults:?}"
            );
        }
    }

    #[test]
    fn a_model_fed_arm_with_no_observed_types_validates() {
        let mut m = valid_manifest();
        let source = m.sources.get_mut("src.orders").expect("the fixture source");
        source.schema_provenance = SchemaProvenance::model_fed(SourceSchema {
            columns: vec![
                SourceColumn {
                    name: "amount".to_owned(),
                    data_type: None,
                    nullable: None,
                },
                SourceColumn {
                    name: "region".to_owned(),
                    data_type: None,
                    nullable: None,
                },
            ],
        });
        assert_valid(&m);
    }

    #[test]
    fn keyset_pristine_roster_validates_and_one_name_is_one_entity_per_source() {
        let m = valid_manifest();
        assert_valid(&m);
        assert_eq!(m.keysets.len(), 3);
        assert_eq!(m.keysets[&natural_id()].name.as_str(), "natural");
        assert_eq!(m.keysets[&customers_natural_id()].name.as_str(), "natural");
        assert_ne!(natural_id(), customers_natural_id());
    }

    #[test]
    fn keyset_check1_id_not_equal_to_key() {
        let mut m = valid_manifest();
        let keyset = m.keysets.remove(&natural_id()).expect("pristine keyset");
        m.keysets.insert("keyset.rekeyed".to_string(), keyset);
        assert!(
            faults(m.validate()).iter().any(|fault| matches!(
                fault,
                CompileError::StructuralViolation {
                    site: ReferenceSite::Keyset(id),
                    rule,
                } if id == "keyset.rekeyed" && *rule == "entity id must equal its map key"
            )),
            "expected the fault to locate at the Keyset site"
        );
    }

    #[test]
    fn keyset_check4_an_attributed_member_must_have_role_dimension() {
        let mut m = valid_manifest();
        attribute(&mut m, "sem.amount", "dk.orders", &[natural_id()]);
        assert_structural(
            m.validate(),
            "a keyset member must be a semantic of role Dimension",
        );
    }

    #[test]
    fn keyset_check7_an_attributed_member_must_be_served_by_its_source() {
        let mut m = valid_manifest();
        m.semantic_sources
            .get_mut("dk.orders")
            .expect("the fixture holds the orders leaf")
            .semantic_bitmask =
            SemanticBitmask::from_positions([BIT_REGION, BIT_AMOUNT, BIT_KEY_A, BIT_OCCURRED]);
        assert_structural(
            m.validate(),
            "a keyset member must be served by the semantic source attributing it",
        );
    }

    #[test]
    fn keyset_an_attribution_naming_no_roster_entry_is_dangling() {
        let mut m = valid_manifest();
        attribute(&mut m, "sem.key.a", "dk.orders", &["ks.missing".to_owned()]);
        assert_dangling(m.validate(), ReferenceTargetKind::Keysets, "ks.missing");
    }

    fn attribute_on_grainset(m: &mut SemanticManifest, name: &str) -> String {
        let keyset = Keyset::new("dk.grain", name);
        let keyset_id = keyset.id.clone();
        m.keysets.insert(keyset_id.clone(), keyset);
        set_grainset_keysets(m, vec![keyset_id.clone()]);
        keyset_id
    }

    fn set_grainset_keysets(m: &mut SemanticManifest, keysets: Vec<String>) {
        let grain = m
            .semantic_sources
            .get_mut("dk.grain")
            .expect("the fixture holds the Grainset");
        let RoleDefinition::Grainset { grains, .. } = &grain.role_definition else {
            panic!("the fixture's `dk.grain` is a Grainset");
        };
        grain.role_definition = RoleDefinition::grainset(grains.clone(), keysets);
    }

    #[test]
    fn a_grainset_with_no_signature_and_its_own_keyset_validates() {
        let mut m = valid_manifest();
        attribute_on_grainset(&mut m, "natural");
        assert!(m.semantic_sources["dk.grain"]
            .semantic_signature_ids
            .is_empty());
        assert_valid(&m);
    }

    #[test]
    fn a_grainset_keyset_naming_no_roster_entry_is_dangling() {
        let mut m = valid_manifest();
        set_grainset_keysets(&mut m, vec!["ks.missing".to_owned()]);
        assert_dangling(m.validate(), ReferenceTargetKind::Keysets, "ks.missing");
    }

    #[test]
    fn a_grainset_keyset_list_must_be_sorted_and_deduplicated() {
        let mut m = valid_manifest();
        let id = attribute_on_grainset(&mut m, "natural");
        set_grainset_keysets(&mut m, vec![id.clone(), id]);
        assert_structural(
            m.validate(),
            "a Grainset's keyset attribution must be sorted and deduplicated",
        );
    }

    #[test]
    fn keyset_check9_id_participates_in_global_uniqueness() {
        let mut m = valid_manifest();
        let mut clash = Keyset::new("dk.orders", "natural");
        clash.id = "dk.orders".to_owned();
        m.keysets.clear();
        m.keysets.insert("dk.orders".to_string(), clash);
        assert_structural(
            m.validate(),
            "entity ids must be globally unique across collections",
        );
    }

    fn sign_region(m: &mut SemanticManifest, data_kind: &str) -> String {
        let signature_id = crate::identity::mint_signature_id(data_kind, "sem.region");
        if !m.semantic_signatures.contains_key(&signature_id) {
            let stamp = consumers(&m.semantic_sources[data_kind]);
            let region_expr = m
                .expressions
                .emit(
                    &Expr::Column(NamedReference::unqualified("region").expect("a name")),
                    &signature_id,
                    ExprRoot::SignatureExpression,
                    stamp,
                )
                .expect("emit");
            m.semantic_signatures.insert(
                signature_id.clone(),
                crate::manifest::SemanticSignature::new(
                    signature_id.clone(),
                    "sem.region",
                    region_expr,
                ),
            );
        }
        let owner = m
            .semantic_sources
            .get_mut(data_kind)
            .unwrap_or_else(|| panic!("no semantic source `{data_kind}`"));
        if !owner.semantic_signature_ids.contains(&signature_id) {
            owner.semantic_signature_ids.push(signature_id.clone());
            owner.semantic_signature_ids.sort();
        }
        signature_id
    }

    fn carry_keyset(m: &mut SemanticManifest, data_kind: &str, name: &str) {
        let keyset = Keyset::new(data_kind, name);
        let keyset_id = keyset.id.clone();
        m.keysets.insert(keyset_id.clone(), keyset);

        let signature_id = sign_region(m, data_kind);
        m.semantic_signatures
            .get_mut(&signature_id)
            .expect("just written")
            .keysets = vec![keyset_id];
    }

    #[test]
    fn a_reduction_leaf_carrying_a_keyset_its_unionset_does_not_is_structural() {
        let mut m = valid_manifest();
        carry_keyset(&mut m, "dk.union.a", "natural");
        assert_structural(
            m.validate(),
            "a reduction's Unionset must carry every keyset name its leaves carry",
        );
    }

    #[test]
    fn a_reduction_whose_unionset_carries_the_leaf_keyset_name_validates() {
        let mut m = valid_manifest();
        carry_keyset(&mut m, "dk.union.a", "natural");
        carry_keyset(&mut m, "dk.union.b", "natural");
        carry_keyset(&mut m, "dk.union", "natural");
        assert_ne!(
            crate::identity::mint_keyset_id("dk.union.a", "natural"),
            crate::identity::mint_keyset_id("dk.union", "natural"),
            "one authored name is one entity per source (ADR 0048)"
        );
        assert_valid(&m);
    }

    #[test]
    fn a_unionset_carrying_more_than_its_leaves_is_not_a_carry_over_failure() {
        let mut m = valid_manifest();
        carry_keyset(&mut m, "dk.union", "natural");
        assert_valid(&m);
    }

    #[test]
    fn an_implicit_unionsets_reduction_leaf_missing_a_signature_is_refused() {
        let mut m = valid_manifest();
        m.semantic_sources
            .get_mut("dk.union")
            .expect("the fixture holds `dk.union`")
            .role_definition = RoleDefinition::unionset(
            MultisetOp::UnionAll,
            vec!["dk.union.a".to_owned(), "dk.union.b".to_owned()],
            false,
        );

        let signature_id = crate::identity::mint_signature_id("dk.union.a", "sem.region");
        let leaf = m
            .semantic_sources
            .get_mut("dk.union.a")
            .expect("the fixture holds `dk.union.a`");
        leaf.semantic_signature_ids.retain(|id| id != &signature_id);
        leaf.semantic_bitmask = SemanticBitmask::from_positions([]);
        let dropped = m
            .semantic_signatures
            .remove(&signature_id)
            .expect("the fixture holds the signature");
        m.expressions.physical.remove(&dropped.expression_id);

        let reported = faults(m.validate());
        assert!(
            reported.iter().any(|fault| matches!(
                fault,
                CompileError::ReductionLeafMissingSignature { semantic_source, leaf, semantics }
                    if semantic_source == "dk.union"
                        && leaf == "dk.union.a"
                        && semantics == "sem.region"
            )),
            "expected ReductionLeafMissingSignature among {reported:?}"
        );
    }

    #[test]
    fn a_decoded_artifact_with_a_colliding_id_is_refused() {
        let mut m = valid_manifest();
        let mut clash = Keyset::new("dk.orders", "natural");
        clash.id = "dk.orders".to_owned();
        m.keysets.clear();
        m.keysets.insert("dk.orders".to_string(), clash);

        let json = serde_json::to_string(&m).expect("the artifact serializes");
        let decoded: SemanticManifest = serde_json::from_str(&json).expect("and decodes");
        assert_structural(
            decoded.validate(),
            "entity ids must be globally unique across collections",
        );
    }

    #[test]
    fn semantics_id_not_equal_to_key_reports_semantics_site() {
        let mut m = valid_manifest();
        let def = m.semantic_definitions.entries.remove("sem.region").unwrap();
        m.semantic_definitions
            .entries
            .insert("sem.rekeyed".to_string(), def);
        assert!(
            faults(m.validate()).iter().any(|fault| matches!(
                fault,
                CompileError::StructuralViolation {
                    site: ReferenceSite::Semantics(id),
                    rule,
                } if id == "sem.rekeyed" && *rule == "entity id must equal its map key"
            )),
            "expected the fault to locate at the Semantics site"
        );
    }
}
