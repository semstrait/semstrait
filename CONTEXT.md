# semstrait

This is the domain glossary for semstrait: the one place that fixes what each
word in the project means. It exists because the same idea has accumulated
several spellings across specs and code, and a word chosen loosely here
propagates straight into type names, diagnostics, and authoring surfaces.

## Language

### Artifacts and stages

**Semantic model**:
The typed form of an author's YAML modeling document, still carrying unresolved
names and authoring-level expressions. It is the input to compilation, never an
input to planning.
_Avoid_: "the model" bare for the type, "resolved model".

**Semantic manifest**:
The sealed, engine-agnostic artifact in which every name, source, and expression
has been resolved and denormalized per semantic source, so planning reads it by
id and needs no other lookup or I/O. An expression may hold a link, resolved at
compile, to another entry on its own semantic source (see Physical expression).
_Avoid_: "compiled model", "metadata store".

**Model fingerprint**:
The content digest of a semantic model — the operational meaning of "this model
changed". It covers every authored field, the document's name among them, and
excludes only the document's own identity. Renaming therefore moves it; what the
exclusion buys is that a model re-authored under a different document identity,
with otherwise identical content, fingerprints the same — so the store addresses
meaning rather than documents.
_Avoid_: "model version", "checksum", "model id"; also "a rename leaves it
unmoved", which is false — a rename moves it.

**Compilation identity**:
The identity of one compilation of one model content, derived from the model
fingerprint. It changes whenever the content changes, which is what distinguishes
it from the identity of the *document*, which survives content changes.
_Avoid_: "manifest id", "model id" for the document's own identity.

**Entity id**:
The technical key of a named entity — one untyped lane of canonical UUID text,
durable across compiles and the handle every stored artifact addresses. The
entity's **name** is its natural key: what a definition states and what every
reference resolves. The id is layered on top for storage and cross-artifact
addressing, never for authoring, and an author may write one or leave it to be
minted.
_Avoid_: "primary key", "uuid" for the concept, "entity key"; and do not read an
id as authoring surface — a reference names the entity, never its id.

**Manifest version**:
The version of the persisted manifest structure. It is moved deliberately when
the artifact's shape changes, never by content and never by the clock, and it is
the only version dimension a cached derivative may key on.
_Avoid_: "manifest epoch", "manifest hash", "schema version".

**Semantic graph**:
The in-memory structure built from a manifest at setup, which rebuilds the
composed surfaces and join skeletons the manifest deliberately does not persist.
_Avoid_: "manifest graph", "resolved model".

**Semantic plan**:
The engine-neutral plan produced for one request against a manifest, expressed
purely in canonical plan-node and expression vocabulary.
_Avoid_: "logical plan", "query plan".

**Canonical vocabulary**:
The closed set of plan-node and expression variants a semantic plan is built
from. Each plan node is semstrait's own implementation of one Substrait relation
at a pinned reference release: it has that relation's slots, with their meaning,
and admits only the values semstrait writes. A mirrored slot, value or node
enters with the code that writes it or by an ADR; a field of semstrait's own,
and a move of the reference release, only by an ADR. The expression variants
are semstrait's own and grow only by an ADR. It is engine-free: every target
engine and every format release is served by mapping onto it, never by widening
it.
_Avoid_: "catalog", which is the physical catalog; "plan catalog", "node roster";
reading `#[non_exhaustive]` as a licence to grow it.

**Node-common block**:
The fields every plan node composes beside its own: its anchor, the identity
unique within its plan; its output schema; its trace; and the optional
format-common fields, the emit form and the hint. A present optional field is
the planner's instruction to use it; an absent one changes no result.
_Avoid_: "RelCommon" (the format's name for its wire image), "meta" (the retired
name), "header".

**Emit form**:
Which of a node's natural output fields the node outputs, and in what order. It
selects and orders output fields that exist and computes none; computing an
output field is a projection's. It keeps every field that its own node computes
(a projection's expressions; an aggregate's groupings and aggregates), because
only the output schema holds a computed field's name; a node above leaves such a
field out by its own emit form.
_Avoid_: "emit" alone, which names the SQL-shaped form of adapt.

**Output schema**:
The ordered output fields a plan node gives the node that reads it (see Output
field). Each name is one identifier: a semantic output uses the definition name
with the qualifier its reader uses (bare at the root), and a planner-created
output field uses a reserved name. A Dataset's physical read, its empty read
included (ADR 0073), uses physical column names; the empty read of a Unionset
keeps the union's output names.
_Avoid_: "schema" alone, which is a physical source's; cutting a qualified name to
bind it.

**Output field**:
One named, typed element of a plan node's output schema, which the node above
reads as its input. It is named for a semantics, with the column qualifier that
its reader's leaf carries (bare at the root), or with a reserved name where the
planner creates it; a Dataset's physical read names its output fields by their
physical columns (see Output schema). A join condition, a
relationship filter and a composed projection read output fields.
_Avoid_: "column" for it, which is physical; "output column".

**Format adapter**:
The adapt-stage component that lowers a semantic plan to one wire form — a
Substrait plan today, SQL text later. Its inputs are the plan and the engine
rules it was built with, and nothing else: no manifest, no catalog, no engine
state.
_Avoid_: "serializer", "emitter" for the Substrait one; giving it the manifest.

**Producer**:
Substrait's own name for the side that writes a plan; the Substrait format
adapter is a producer, and the engine that reads its plan a *consumer*.
_Avoid_: "exporter", "converter".

**Engine adapter**:
The per-engine rewrite rules a format adapter is parameterised by —
semantics-preserving rewrites of plan nodes and expressions, function spellings,
type encodings — with identity and faithful-format defaults, so an engine is a
rules implementation and never a second walk of the plan; it adapts a plan to
its engine and never refuses one. It is built for one release of its engine, and
a rule names its releases where it holds for only some. Executing a plan on an
engine is beneath this layer and is the engine's own.
_Avoid_: "dialect" for these rules: a dialect is an engine's or a format's
declared vocabulary, which the rules read; "backend"; "refusal" or "capability" for an
engine's shortfall — it is a rewrite.

**Dialect**:
One engine's or one format's own vocabulary of functions, operation nodes and
data types, as semstrait declares it. DataFusion's, Substrait's and Iceberg's
are the dialects today: DataFusion's declares functions and operation nodes, and
Substrait's and Iceberg's declare data types only. Substrait's is a vocabulary
that several engines implement, each with its own extensions. Iceberg's is the
Iceberg table format's, in which an Iceberg catalog reports a column's **native
type**, and compile reads a native type through it where the canonical names and
aliases give it no type. It names a vocabulary, not the engine, the format, the
catalog provider or an adapter. A dialect that has releases is read at one,
written `name=release`, as `datafusion=53`. Its declarations are its reference
release's, except where an entry names other releases that differ. DataFusion's
reference release is 55, and its oldest is 53; Substrait's and Iceberg's have
no release.
_Avoid_: "dialect" for the engine adapter's rules or for SQL text; reading
Substrait's dialect as a Substrait dialect file, which states what one engine
supports.

**Format artifact**:
A format adapter's output — a Substrait plan or SQL text — and the first place
format-specific representation is legal. A dialect may declare a format's names
for a type, and only a format artifact writes them. It carries the request's output names
and no semantic vocabulary — no roles, no definitions, no trace; trace
appears only in the explain copy of a plan, as an extension a consumer may
ignore. A format's reuse of a role name (Substrait's `AggregateRel.Measure` is
an aggregate computation) names compute, never the role.
_Avoid_: "engine artifact" (a format is shared by engines), "query string",
"output".

**Resolution pipeline**:
The fixed progression from authoring YAML to a format artifact: parse, validate,
compile, plan, adapt. A stage may be near-identity but is never skipped or
reordered.
_Avoid_: naming "resolve", "emit" or "optimize" as stages; emit is the
SQL-shaped form of adapt, and optimizing a plan is the consuming engine's.

**Canonical layer**:
The engine- and provider-agnostic middle of the system, between YAML authoring
and engine emission, in which no SQL text may appear: SQL text is a format
artifact, and only a format adapter writes it. On native types and provider
identity the rule is a narrower one than a prohibition: the canonical layer never
*reads* a native type or a provider identity, though it may *record* either as
non-load-bearing provenance. A catalog's metadata enters at compile's catalog stage,
on the layer's edge: it reads a referenced column's **native type**, through the
canonical names and aliases, then the dialect that the provider names, and gives the
column its canonical type. A dialect names a vocabulary, not the provider.
_Avoid_: "core", "IR layer"; reading the native-type clause as an outright ban —
recording one as provenance is admitted, and the catalog stage translates one at the
layer's edge; only reading one inside the layer to decide something is forbidden;
reading the SQL clause as a ban on SQL's expression structure: it bans SQL text.

**Request**:
The query-time statement of what a caller wants — the selections to return, the
request predicate and named filters to apply, the ordering and row window, and
optionally a public semantic source to answer from — expressed only in semantic
names. It states intent and never placement: where a predicate lands relative to
aggregation is the planner's reading of its leaves, not the caller's instruction.
_Avoid_: "query"; "session" for the settings carrier — see Request context.

**Request context**:
A separate carrier of settings under which a Request is planned; today
`timezone` (IANA name, default UTC), used when bucketing timestamp-typed temporal
dimensions to a requested granularity, and `week_start` (Monday or Sunday, default
Monday), used when bucketing to the week granularity.
_Avoid_: "session".

**Selection**:
One entry of a request's `select`: a semantics name, untyped by default with its
role read from the manifest, or typed with a modifier — today only a requested
granularity on a dimension. Selections are ordered and fix the output fields, each
named for its semantics.
_Avoid_: "field", "column", "projection item".

**Request predicate**:
The one caller-authored predicate a request carries, written over semantics
names, as distinct from a named filter, which the model authored and a request
only activates. It may demand semantics the request does not project.
_Avoid_: "raw filter", "where clause", "condition", "user filter".

**Requested granularity**:
The granularity a request attaches to a selected temporal dimension, asking for
that dimension bucketed to it. Only the requested granularity on a source's time
axis is a routing and rollup input; a request selecting no temporal dimension has
none. With the selected dimensions it forms the request's grain.
_Avoid_: "requested grain", the older name; "query grain", "output grain", a
top-level "grain".

**Demanded set**:
The semantics a request requires its answering source to serve — every
selection, every leaf of the request predicate, and every activated named filter
— read as one mask over the semantic definitions. It is what a candidate is
tested against.
_Avoid_: "requested fields", "projection" — it includes what is filtered but not
projected.

**Advisory**:
A non-fatal notice returned beside a plan when the planner proceeded on a
judgement the caller may want to revisit — a composition known to be lossy, or a
path ambiguity resolved by name order. It names the warn-and-proceed posture, as
opposed to refusal.
_Avoid_: "warning" (a severity), "diagnostic" (the carrier); reading it as
lossy-only — lossiness was its first instance, not its definition.

**Advisory system**:
The consumer that collects advisories and a model's smells, and analyzes them
for a caller or an author. It reads what the planner returns and never shapes an
artifact.
_Avoid_: "advisory" for it: an advisory is one notice.

**Trace**:
The planning facts and decisions that a plan node holds for explain. The
canonical layer records it and never reads it.
_Avoid_: "annotation", "metadata", "hint".

### Data kinds

**DataKind**:
One named, queryable modeling unit: either a leaf bound to physical data or a
Complex data kind over other data kinds. It is what a request is ultimately
answered from.
_Avoid_: "entity", "table", "model object", and bare "kind" — "kind" is
overloaded and must always be qualified.

**Dataset**:
The only leaf data kind, and the only one that names physical storage: a named
semantic surface with no child data kinds of its own. It names one storage entry,
which may expand to several physical sources — one-to-many as authored, while
each Dataset the manifest holds names exactly one, so the expansion is composed
above the leaves rather than folded inside one (ADR 0049). A Dataset may itself
be a constituent of a Complex kind, and each appearance layers above it rather
than altering it. Its temporal shape and granularity are authored or absent,
never inferred.
_Avoid_: "table", "source", "fact"; "binding" for what it owns — see Binding;
saying it reaches physical data through *semantic* expressions, which do not
survive compile — see Physical expression; "Simple DataKind" is the structural
name for the same thing, and Dataset is canonical in authoring prose.

**Complex DataKind**:
A data kind that composes child data kinds instead of reaching physical data
itself. A Joinset and a Unionset of mode `unique` need two or more children, and
are refused with `ComplexDataKindInsufficientChildren` otherwise; a `unique`
Unionset removes duplicate rows of its branch and so does not equal it. A
Grainset with one child and a Unionset of mode `all` with one branch are *Sugar*.
A Joinset and a Unionset are reading surfaces: each owns one semantic signature
per name it serves. A Grainset owns a signature only for each Metric that it
computes (ADRs 0065, 0084).
_Avoid_: "composer", "composite", "virtual dataset"; reading a Grainset as a
reading surface.

**Unionset**:
The Complex data kind that stacks children row-wise into one surface, supplying
typed nulls for fields a child lacks. Its children agree on temporal shape,
where agreeing includes both being absent, so the requirement says nothing about
time on children that classify none; their granularities may differ, and then it
carries none; as a constituent of another composition it may declare one, and
it carries one (declared or derived) or is refused where it publishes a temporal shape or, as a child of a Grainset, carries a time axis, while a child with no granularity takes no
part in that; the Unionset may
declare that shape itself, as one more party to the agreement rather than as an
override. What it can serve is the union of its
children's coverage; what identifies its rows is authored inside a keyset
boundary, since a vertical stack guarantees no determinant of its own and
therefore derives no keyset.
_Avoid_: "union view", "merge", "append kind"; reading its keysets as the
intersection of its children's — ADR 0039's rule, which ADR 0048 replaced with
the authored-inside-a-boundary route.

**Grainset**:
The Complex data kind that groups two or more children holding the same subject. A Grainset with one child is Sugar. No two have the same grain and the same coverage mask, so a request can be routed to the child that
serves it, or answered by joining the children that serve it together. Children may share a granularity. All of its children that have a time axis share one temporal shape kind. A child with no time axis may stand beside them, and a Grainset may have no time axis. A child with no time axis serves only a demand that reads no time axis. A
composed child carries the granularity declared on it, else the one that every
constituent below it, at every depth, that carries one agrees on, the
derivation stopping at a constituent that declares one, and a composed child that carries a time axis and no granularity is refused; a Grainset that is a constituent may declare a granularity
that every child serves, and a root Grainset, which declares none, carries the granularity that its children agree on, and none where they differ. A Metric that one child maps whole is served by that child alone. A Metric that no child maps whole is computed by the Grainset where its children can compute it together, and the Grainset owns a semantic signature only for each such Metric (ADR 0084); its coverage mask is kept for routing, and its authored
keysets ride its own `keysets` field on the semantic source (ADR 0065).
_Avoid_: "rollup set", "aggregate table set", "granularity set"; reading the shared
shape as what makes its children's keysets agree — that follows from the keyset
rules and this data kind's own specification, not from the temporal axis; "grain"
for a child — see Grain; "routing unit" or "effective routing unit" for the child
a request is routed to; "grain group", the retired fold of children at one
granularity into an implicit union.

**Joinset**:
The Complex data kind that composes constituents horizontally over declared
relationships. Its specification is a set of constituents and the relationships
among them, and every legal composition of that set is queryable — the whole of
it under the authored name, each smaller one privately.
_Avoid_: "join view", "join model"; describing it as traversal outward from an
anchor, which nothing authors.

**Nested data kind**:
A data kind written inline inside a Complex parent rather than referenced by
name. Whether it declares a semantic interface depends on the authoring
direction; it is never independently queryable and nothing may reference it.
_Avoid_: "inline kind", "child kind" as the concept name; do not define it as
"structure only", which describes one direction.

**Authoring direction**:
Which end of a composition declares the semantic interface, chosen per root data
kind and never mixed within one. Interface-first has the root declare it and
each constituent map into it; constituent-first has each constituent declare its
own and the root's derived from them. It is a fact about the authoring document
and does not survive compilation.
_Avoid_: "authoring mode" ("mode" is a Unionset field), "top-down" /
"bottom-up", "style".

**Semantic source**:
The manifest's carrier for one data kind — authored, nested and
compile-synthesized alike — held as a single entry in one flat, id-addressed
collection, and, except a reduction leaf, identified by a name unique among
them; a reduction leaf's name is only a label (see Reduction leaf).
_Avoid_: "manifest data kind", "compiled data kind"; never bare "source", which
is Physical source.

**Public semantic source**:
A semantic source a request may name directly in its `from`. Every other
semantic source is private and exists only as a resolution path element.
_Avoid_: "exposed", "queryable kind", "entry point" as the concept name.

**Originator**:
The nearest authored data kind whose specification produced a given semantic
source, recorded on that source. It states provenance, which is a different edge
from the operand references that state composition.
_Avoid_: "parent", "owner", "origin".

**Source role**:
What a semantic source is and does in planning, in structure and in behaviour: it
reads one physical source (Dataset), joins two operands (Joinset), stacks
branches (Unionset), or picks one child or joins its children (Grainset). It chooses the one Strategy
that plans the source, and so fixes how the source's rows are formed and which
actions are open to it, such as a bypass or a pruned operand. A
compile-synthesized semantic source has one too.
_Avoid_: bare "role" where a semantics' role (Dimension, Measure, Metric,
Filter) could be meant; "body" and "composition body", ADR 0073's older words —
the model's authoring forms are not the source role; "authored data kind".

### Semantics

**Semantics**:
One author-declared, globally named element of a data kind's queryable
vocabulary — a dimension, measure, metric, or filter. The countable singular
is deliberate: one name is always one logical semantics.
_Avoid_: "field", "attribute", "column"; never use bare "semantics" for the
ordinary meaning-of-behavior sense; a key is not one — identity is an attribute
of a dimension site, not an element of the vocabulary.

**Occurrence**:
One authored appearance of a semantics name. All occurrences are symmetric
contributors to the same logical semantics, not competing definitions.
_Avoid_: "duplicate", "redefinition", "override".

**Site**:
The one place a data kind's semantic interface exposes a named semantics —
either an inline declaration or a reference to the shared pool. It is where
facts local to that data kind are stated, so a field present only at a site and
absent from the pool is a local attribution rather than an override of anything.
_Avoid_: "entry", "slot", "usage"; do not use it for an appearance of a name
inside an expression, which is an occurrence.

**Shape lock**:
The rule that a semantics' identity-bearing fields are identical at every
occurrence of its name, with disagreement a compile-time conflict rather than a
negotiation. Only how its value is obtained may vary per data kind.
_Avoid_: confusing this with temporal shape; shape-lock concerns semantics
identity only.

**Semantic definition**:
The global, request-facing contract of one semantics — its name, its data type
(absent for a named filter), its role with that role's shape-locked families (a
dimension's category; a measure's aggregation and optional additivity; a metric's
optional aggregation; a filter's none), its bit position,
and its description and AI context. It owns no expression and no keysets: it says
what a name means and never how a value is obtained. It is what a request names,
and nothing is ever planned against it directly — what a plan reads is the
semantic signature some semantic source owns for it.
_Avoid_: bare "global definition"; giving it an expression handle of any kind, or
keysets; reading it as the thing a planner computes from.

**Semantic interface**:
The set of semantics a data kind exposes as its consumer-facing contract. It is
authored on one side of a composition and derived on the other, according to the
authoring direction.
_Avoid_: "schema", bare "surface"; do not restrict it to top-level data kinds.

**Dimension**:
A semantics that qualifies rows — an axis for grouping or filtering — rather than
something aggregated.
_Avoid_: "attribute", "column", "group-by field".

**Measure**:
A semantics that aggregates an underlying quantity, pairing an aggregate function
of the function catalog with the expression it aggregates.
_Avoid_: "metric" — measure and metric are distinct here and must never be
swapped; reading an engine's or format's use of the word (Substrait's
`AggregateRel.Measure`) as this role — no semantics reach a format artifact.

**Metric**:
A semantics defined as a formula over the aggregates of measures and other
metrics, computed at the grain of the demand that finishes it. A metric with no
aggregation of its own, whose formula reads a dimension or a column that the
demand does not group by, takes the sum of its formula over its formula parts.
A metric with an aggregation of its own over a formula that reads a measure or
a metric aggregates the formula's values at its formula grain. In the simplest
case the formula reads no measure and no metric: it is then a row-level
expression, planned as a measure where it declares an aggregation, and as a
dimension where it declares none.
_Avoid_: "derived measure", "calculated field", "KPI".

**Compute frame**:
The frame kind that a frame takes where its demand holds a Metric whose stored tree reads the frame's own final values (a Metric that a Grainset computes). It is outside the closed set of combines, and no Strategy returns it. It reads one frame for each formula group (ADR 0084, rule 8).

**Formula group**:
A subtree of a Compute frame, planned as a frame of its own: the subtree of a Metric with its own `agg:` whose formula grain is finer than the demand's grain, or of a Metric that takes the parts, which two Metrics may share; or the group of the other values that the demand and the formulas read (ADR 0084, rule 8). It has a path of its own, `<grainset>#group<n>`.

**Part key**:
For a Metric with no `agg:` whose formula reads a Dimension that the request does not group by, the columns that its step 1 groups by beside the demand: one for each `case` that selects, which holds the position of its result, and one for each piece, which holds its value. The row-level projection gives them, from each Dimension's value before bucketing; a Unionset branch that gives step 1's aggregates computes them on its own rows (ADR 0074, *The parts* and *A Unionset*). On a Grainset, each child that gives a value that the formula reads computes the key on its own rows, the children join `Full` on the demand and the key, the formula runs on each part, and the parts sum to the demand (ADR 0084, rule 8).

**Key dimension**:
A dimension a data kind states as participating in identifying its rows, by
marking the site where the data kind exposes it. The claim is per data kind and
per keyset, never a property of the dimension itself: the same dimension may be
a key dimension for one data kind and an ordinary axis for another.
_Avoid_: "key" as a standalone thing, "primary key column", "identifier";
identity is an attribute of a site, not a semantics of its own.

**Keyset**:
A named group of a data kind's dimensions that advises how its rows are
identified. Uniqueness over the group is assumed, never asserted, enforced, or
verified against data. It constrains nothing about data, but it is load-bearing
input to derived join bases, to a Grainset's implicit cross-grain composition,
to a source's grain, to fan-out deduplication and to a mode-`unique`
Unionset's distinct, so a wrong keyset yields a wrong plan rather than a
rejected model.
_Avoid_: "primary key", "unique key", "constraint", "identifier"; also the key
domain of a map — a `BTreeMap`'s "keyset" is unrelated to this term, and the
`semstrait-manifest` load invariants use the word in that sense.

**Guaranteed keyset**:
A keyset the manifest holds, whether authored or derived at compile — the
minimum identification a semantic source offers. It is the only kind a
composition or a basis derivation may read.
_Avoid_: "static keyset", "compiled keyset", "offline keyset".

**Runtime keyset**:
A guaranteed keyset extended by identification a request itself establishes. It
serves the advisory system only and never shapes an artifact.
_Avoid_: "promoted keyset", "dynamic keyset", "query keyset"; do not read it as a
second kind of declaration.

**Keyset boundary**:
The ceiling on what an author may declare as a composite's keyset, computed from
its children's keysets and closed downward, so every subset of a boundary
element is admissible. It permits and rejects; it derives nothing.
_Avoid_: "allowed keyset", "keyset constraint", "candidate keyset"; do not
conflate it with a derived keyset.

**Filter**:
A named, reusable predicate on a data kind's interface, applied only when a
request names it. It is distinct from an aggregation-level condition, which
narrows which rows feed one aggregation and cannot be named by a request.
_Avoid_: "predicate", "where clause", "segment"; always-on filters do not exist.

**Additivity**:
Whether an aggregation may be computed in parts and recombined without loss —
fully, along every dimension but one, or not at all. It governs pre-aggregation and
rollup safety, and an authored declaration may narrow it but never relax it. It has
two inputs that stay apart: the aggregate function's additivity says whether its
partial aggregates combine again, and a Measure's declared additivity narrows it. A
**Semi-additive strategy** keeps the rows that the function then combines, except
along its `by`, and `non` combines nothing again.
_Avoid_: "aggregability", "summability"; prefer the additive / semi-additive /
non-additive roster.

**Semi-additive strategy**:
How a semi-additive aggregate gives one value across the values of the
Dimension it is ordered by, inside each combination of its axes: latest or
earliest.
_Avoid_: bare "strategy", which is the per-role planning logic — see Strategy;
"resolution", which is the resolver's.

**Bit position**:
The dense zero-based index a semantics occupies in one manifest's semantic
bitmap, assigned in canonical `EntityId` order and meaningful only inside the
manifest that declares it. Coverage masks address semantics by bit position;
`EntityId` is the durable handle.
_Avoid_: treating it as an identifier, persisting it outside its manifest, or
carrying a mask across compiles.

**Bitmap index**:
A lookup structure a consumer builds when it wants to move between a semantics'
name and its bit position, in either direction — name to bit to build a demanded
set, bit to name to render a diagnostic. It is derived from the semantic
definitions every time it is wanted, so it holds no authority and can never
disagree with them, and it is never part of the artifact.
_Avoid_: confusing it with **bit position**, which compile assigns and the
artifact carries; persisting it, or treating it as a second place to look.

### Physical binding

**Binding**:
A retired term. It named a single join-point entity between one leaf data kind's
semantic surface and the physical data underneath it, and there is no such
entity: a semantics reaches physical data through its expression, and the
expression names every physical source that consumes it. Its former second
clause — that complex data kinds own none — is false for the same reason, since
a joinset's join and filter conditions compile to expressions of their own.
_Avoid_: "binding" for any manifest entity or collection, and "SemanticBinding",
a retired spelling; name the expression, the semantic mapping, or the physical
source instead.

**Physical source**:
One resolved physical target read as a single unit — a catalog table, or a
**base path** read in one format. It corresponds to one relation on the engine
side, and one authored storage entry may expand to many. Many semantic sources
may read one, each through its own **selector**. What is stored identifies it,
never the Datasets that read it; its name is a label derived from its location
for people to read, never its natural key, and two physical sources may share
one name; a column reference names it by its source alias. It is reached two
ways, and which way is
recorded on the artifact rather than inferred: derived from a catalog where one
is provided, which observes the schema; or fed by the model where no metadata
provider exists, in which case its columns are the ones the expressions naming it
reference, and a column that a semantics maps directly takes its **Read type**. Its column **set** is
complete and authoritative on both routes; a column's canonical type and
nullability are **optional**, and absent means "this compile observed neither" —
never "untyped", and never a licence to default. On the model-fed route a
column's nullability is always absent, and an absent type also means that no
semantics maps the column directly. A provider's **native type**
spelling is kept beside the canonical list as provenance, and it is what makes an
observed absence legible: compile's catalog stage reads the spelling where it
enters, to give the column its canonical type, and nothing in the canonical layer
reads the kept copy. Catalog-specific
specialization sits beneath it, in the catalog provider layer, not beside it.
_Avoid_: bare "source", "path", "table" as the general word;
"ResolvedPhysicalSource", a retired spelling; "engine-specific" for its
specialization — engine means the compute target, and specialization here is per
catalog; "typed from the referencing definition" for a column that an expression reads, a
retired derivation; reading an
absent type as `String`, as `Unknown`, or as a signal of which route was taken —
the route is the recorded arm, not a count of absences; reading its name as an
identifier, or as a table name an engine registers; reading a selector as part
of it.

**Native type**:
A catalog provider's own spelling of a column's type, as the catalog reports it:
Iceberg's `timestamptz` or `fixed[16]`, or a nested type's JSON object. The
physical source keeps it verbatim beside the column's canonical type, as
provenance. Compile's catalog stage reads it where it enters, to give the column
its canonical type: through the canonical names and aliases, then through the
dialect that the provider names, and nothing in the canonical layer reads the kept
copy. A referenced column whose native type maps to no type fails the compile.
_Avoid_: "engine type": a catalog is not an engine; reading a canonical default
into a bare native type; reading a dialect's declaration as a native type: a
native type is what a catalog reported for one column, which a dialect's
declarations read.

**Base path**:
The deepest literal folder a path physical source reads. Every object the source
reads sits beneath it, and one base path read in one format is one physical
source, however many semantic sources read it.
_Avoid_: "glob root", "root", "table path".

**Selector**:
How a semantic source picks what it reads from its physical source: the whole
folder, one glob, a named set of files, or the whole table. It belongs to the
semantic source, so two semantic sources may read one physical source through
different selectors.
_Avoid_: "filter", which is a semantic filter's word; "pattern" as the general
word, since a glob is one kind of selector; reading it as part of a physical
source's identity.

**Listing**:
The objects compile found under a physical source: every object that any
semantic source over it selects. It is what a read without partitions is
planned from; a glob or a folder is only a shortcut to it. A read with
discovered partitions is planned from its base path, and the engine lists it
again, so the engine's set can differ from the listing.
_Avoid_: "contents", and "files" as the general word.

**Partition**:
One `key=value` folder sequence beneath a path physical source's base path,
together with the objects in it. Its values reach a query as partition columns.
_Avoid_: "partition dimension" for the dimension it feeds; "partition index",
which in Substrait numbers a split of work, not a partition of data.

**Partition column**:
A column that the engine gives from a partition's key rather than from the
files, where a semantic source reads a path physical source by its whole folder.
Any semantics mapped to it reads it like any other column.
_Avoid_: "key column", "virtual column".

**Column**:
The physical representation of a semantics — one named element of a physical
source's schema, and the reference naming it inside a lowered expression. A
column enters the language in exactly one place, a semantic mapping, which is
where a semantics is tied to physical form; it is read below the projection that
turns physical columns into a semantic source's interface, and above that
projection a consumer names semantics only. So a composition introduces no
physical form of its own: what a plan node above that projection gives are
output fields, not columns (see Output field).
_Avoid_: "column" for a semantics, a dimension, a measure or a filter — those are
the queryable vocabulary and a column is how one of them is stored; "field" for
either side of that distinction, or "column" for an output field; reading a
qualified reference in a composition's tree as a second kind of column: it names
an output field, and qualification says which semantic source supplies it and
nothing more.

**Semantic mapping**:
The recipe stating how each named semantics obtains its value from a source — a
column, a literal, an expression, or extracted source metadata. Those four are
authoring vocabulary: in a manifest each occurrence lands as one semantic
signature on its source, and the four collapse into the leaves of that
signature's tree.
_Avoid_: "ColumnMapping" and its resolved spelling; semantic mapping is
canonical. In manifest context, naming a "mapping value" variant type at all.

**Semantic signature**:
One concrete implementation of a semantic definition on one semantic source. It
owns exactly one expression entry, whose tree is complete and fully physical once
each link in it is followed — its root included where the definition's
role is a measure — an optional row-level filter entry for a measure or metric,
and that source's keyset attribution. Its id is
minted from the pair `(semantic source, semantic definition)`, it is listed by
exactly one owning source, and it carries no source reference of its own: the
owner's list is the one authority for ownership. Exactly one exists per set bit
of a reading source's coverage mask, and none for an unset bit. A Grainset owns one
only for each Metric that it computes.
_Avoid_: "override", "binding", "realisation row"; putting a source id on the
signature; reading a link to another signature's entry as a share of that entry:
the linked entry stays its own signature's.

**Semantic override**:
A retired term. It named the per `(semantics, data kind)` row that carried an
*optional* expression handle plus that pair's keysets, so a semantics could reach
physical data with no row at all and a consumer resolved through a constituent
instead. A semantic signature replaces it and is not a synonym: its tree is
mandatory, it is owned by one semantic source rather than keyed by a pair, and
there is exactly one per set bit of that source's mask.
_Avoid_: "semantic_overrides", "override row", "SemanticOverride"; describing a
signature as an override of anything.

**Coverage**:
The per-contributor statement of *how* a semantics is obtained — natively
present, derived by expression, or null-filled. It is read against a
composition, never off a contributor alone: null fill is the absence of any
realization, so it says "null-filled" only where the composition above
advertises the semantics, and says nothing at all elsewhere. So name the
composition as well as the contributor, and say whether the contributor is a
physical source or a constituent.
_Avoid_: "availability", "completeness"; "supplied from metadata" as a fourth
outcome — a metadata-supplied value is an ordinary literal, indistinguishable
from an authored one, and counts as derived.

**Coverage mask**:
The bitmask of one semantic source that states which semantics the source can
serve: a set bit asserts that this source, exactly as it stands, can serve that
semantics in a query without further composition. It is a serveability
statement, so null fill contributes a bit, and a set bit does not say that real
data exists. It holds one bit per semantics, by *Bit position*, and the planner
tests it to find the sources that can serve a request.
_Avoid_: "coverage" alone, and reading it as a mask of *Coverage* outcomes
(native, derived, null-filled) — it carries none of them; reading a set bit as
"real data exists here" (ADR 0047).

**NULL fill**:
The coverage outcome in which a contributor genuinely lacks a semantics and
supplies a typed NULL constant instead of being disqualified.
_Avoid_: "missing column", "null padding"; the NULL columns an outer join
produces are a separate phenomenon.

**Metadata dimension**:
A dimension whose value comes from where a row was read — a segment of the
physical source's location — rather than from row data, and which therefore
legitimately differs per source.
_Avoid_: "virtual column", "partition dimension".

### Composition

**Composition**:
Presenting several data kinds as one queryable surface — vertically by stacking
equivalent shapes, horizontally by joining complementary ones. It is what makes a
request answerable when no single data kind holds everything asked for.

**Composed surface**:
The unified queryable interface a composition publishes: the merged semantics —
one field per name, same-named ones always fusing, since one name is one logical
semantics — plus per-field provenance and per-constituent coverage.
_Avoid_: "merged schema", "virtual table", and using "semantic interface" for it;
constituent-qualified field names, which do not exist.

**Constituent**:
A data kind participating in a composition and retaining its own identity and
coverage inside the composed surface.
_Avoid_: using "member", "branch", "child", and "level" interchangeably — those
are per-variant words for the same role, and constituent is the
composition-level canonical one.

**Operand**:
One of the semantic sources a composition reads directly — a side of a Joinset,
a branch of a Unionset, or a child that a Grainset drills across. It may itself
be a composition. A Joinset's projection
reads what its operand outputs; a Unionset's reads the stack of what its branches
output, under the Unionset's own qualifier, and names no branch. A join condition
and a relationship filter read a constituent's output field instead, which the
operand passes up, and that constituent may sit more than one level below the
operand.
_Avoid_: "immediate constituent", "child", "input" as the concept name; reading
it as the constituent a join condition's output field belongs to; reading a
Unionset's output field as one branch's.

**Field provenance**:
The record, per composed-surface field, of which constituents own its value — one
alone, several in agreement, some by NULL fill, or none because it is derived
above the composition.
_Avoid_: "lineage"; provenance is ownership, coverage is availability.

**Explicit composition**:
A composition an author declared and named, owning its members, traversal,
filters, and keys. Contrast implicit composition, which the compiler enumerates
from declared relationships and identifies by a derived name, and which an
explicit declaration must differ from structurally to be worth declaring.
_Avoid_: bare "implicit", which separately labels synthesized data kinds, derived
join paths, and enumerated compositions; reading its identity as content-derived.

**Implicit Unionset**:
A Unionset the compiler synthesizes rather than the author declaring one, from a
leaf whose storage entry expands to several physical sources: one Dataset per
source, stacked. It exists for a multi-source leaf only; a leaf over one physical
source stays a leaf and keeps the authored name. Every branch carries the leaf's
single semantic mapping, so it is a narrowing of the authored form — the same
semantics on each branch, and no null fill. It holds the authored name, so it
holds the keysets the author declared under that name, by carry-over.
_Avoid_: "auto-union", "synthetic union"; a Grainset never folds its children
into one, so those Unionsets are authored; reading its keysets as derived, or as
absent for want of an author.

**Reduction leaf**:
One branch of an implicit Unionset: the Dataset the compiler makes for one of
the physical sources a Dataset reads, when it reads more than one. Its Dataset
and its physical source identify it. Its name joins their names and is only a
label, which two leaves may share. It is private, since nobody wrote its name.
_Avoid_: reading its name as its natural key, or as its physical source's name;
"sub-dataset".

**Carry-over**:
What an author states on a data kind reaching the several semantic sources
compile makes from it, unchanged. It arises where compile divides one authored
data kind rather than composing separate ones, so nothing is inferred and no
claim is widened: the authored name, and every claim made under it, travels to
the semantic source that name resolves to.
_Avoid_: "propagation", "inheritance" as the concept name; do not read it as a
third route to a keyset beside a keyset boundary and derivation — it is the
boundary route, with the author of the divided data kind as the actor.

**Relationship**:
A declared, named, bidirectional association between two data kinds, stated in
semantics names and carrying its join basis, cardinality, integrity, preserved
side, and filter direction. It is a declaration, not an operation.
_Avoid_: "join" for the declaration, "association", "foreign key".

**Join basis**:
What a relationship's equi-terms are formed from: an ordered list of dimension
names, each naming one semantics that both sides expose, so one name yields one
equality between the two sides. A basis may also be absent by derivation or
absent by declaration.
_Avoid_: "join key" for the whole list, "key pair", "join columns"; the basis is
stated in semantics names, never in physical columns.

**Join key spec**:
Which of the three states a relationship's basis is in — derived from what the
two sides share, declared as names, or deliberately none. All three are stated
under `on:`: `on: auto` (or `on:` omitted), `on: [names]`, and `on: cross`. The
third is the spelling for a join with no dimension-name basis, and pairing it
with a filter is how a non-equi condition is stated.
_Avoid_: reading "no basis" as "unconditional", and treating an absent
declaration and a declared absence as one state.

**Cardinality**:
The author's assertion of how many rows on one side of a relationship may match
the other. It is trusted planning metadata, never inferred and never verified
against data — though the resolved value is checked for internal consistency. It
is omitted when and only when the basis is `on: cross`, where the join's own
shape settles it; an omitted assertion resolves to many-to-many with no preserved
side and no filter propagation, a cross join. Stating it stays legal beside
`on: cross`, which is how the single-row broadcast is declared honestly.
_Avoid_: "multiplicity"; do not conflate it with key uniqueness; do not read the
omitted case as an assertion — it is the absence of one.

**Integrity**:
The author's claim that a relationship's composition is consistent — that
composing over it loses rows from neither side. It is trusted planning metadata
on the same terms as cardinality, and it decides which smaller compositions of a
Joinset exist at all: a constituent may stand apart from a relationship only
where that relationship carries the claim, because otherwise its filtering is
what the author asked for. A claim may be asserted outright or held to
confirmation against a catalog; with no catalog the authored claim stands, and a
catalog that disagrees warns rather than rejects.
_Avoid_: reading it as referential integrity, which speaks about one side only;
"constraint", "enforcement", "validation" — nothing is checked against data; and
reading its absence as a claim, since a relationship that states nothing is
claimed consistent by default.

**Join type**:
The operational join flavour of a traversal — inner, left, right, full, or
cross. The first four derive from the relationship's preserved side; cross
derives from the basis being declared absent, carries no preserved side, and has
no join condition. A relationship may separately filter the rows the traversal
produces; that does not become part of its join type or condition. In a
semantic plan a cross traversal is the cartesian product of its two inputs, with
any relationship filter applied to that product.
_Avoid_: reading cross as an encoding of some other type over an always-true
condition; reading "no condition" as "the traversal result cannot be filtered".

**Join condition**:
The expression used while matching rows from the two join inputs — SQL `ON`
semantics. It is what a declared join basis becomes once resolved. A cross join
has no condition and says so through its join type. A relationship filter
instead applies to the joined result and is not part of the join condition.
_Avoid_: "join basis" for the condition — the basis is what the author declared
in semantics names, the condition is what the join tests; "residual" for the
relationship filter; stating an absent condition as a tautology.

**Scope-local relationship shadow**:
A relationship declared inside a Joinset that carries a root-level name and
replaces it for that Joinset only. It is a full independent declaration and the
sole sanctioned way to give one composition different join semantics.
_Avoid_: "override", "per-hop override", "join-type override".

**Hop**:
One relationship applied to two constituents, forming one composition. A hop is
not a step within a larger unit: each one is a semantic source of its own, and a
longer composition is a hop whose left constituent is another hop's result.
Direction belongs to the relationship, since a relationship is stated from one
side to the other.
_Avoid_: "edge", "leg", "join step"; reading a hop as an element of a path, or as
something a unit contains rather than is; "directionality" named a retired
relationship-level restriction. An **anchor** — a single driving constituent that
a Joinset is traversed outward from — is not part of the language: nothing
authors one, and a composition has two constituents rather than a root.

**Path ambiguity**:
The condition where two or more candidates for one request tie on structural
complexity and granularity fit. It is never tie-broken silently: among public candidates the planner
takes the first in name order; inside a resolution scope the scope root comes
first and name order decides only among the remaining tied alternatives. Either way the tie is reported to the caller as an advisory — a public tie is the
caller's to resolve with `from`, an in-scope tie the author's to resolve in the
model. A tie in a Grainset's child rank is reported as an advisory in the same way; it is inside the resolution scope, so it is the author's to resolve in the model.
_Avoid_: reading it as a refusal — it is a warn-and-proceed posture; "tie" as
the concept name; "cover exactly", since a candidate serves the demanded set and
may serve more.

**Name order**:
The order of names (of children, candidates or semantics), used as the last
tie-break. A pick of one candidate, public or inside a resolution scope, or of
one Grainset child that name order decides is reported as an advisory (*Path
ambiguity*). The order of a drill-across's joins and
the choice of a *Read type* use name order with no advisory. For the children of a
composition it is the order of the child's name, not the order of declaration:
the planner's read model sorts every semantic source's operands by name before
any Strategy reads them.
_Avoid_: "declaration order", which the manifest keeps and no tie-break reads.

**Fan-out**:
The row multiplication that occurs when a traversal's declared cardinality lets
one row match many, so an aggregate above the join would count a row of the
fanned side more than once. The plan undoes it by letting the aggregate read one
row per row identity of the fanned operand.
_Avoid_: "row explosion"; reading it as an accepted consequence of the
declaration; the expansion of one storage entry into several sources
is multi-source expansion, not fan-out.

### Time

**Temporal shape**:
The classification of how a data kind's rows record time — a regularly sampled
timeseries, a stream of sparse events, periodic full-state snapshots, or
slowly-changing history kept as validity ranges. It gates rollup legality and
time-anchored matching, is declared on a leaf, and is derived for a Complex kind
from its constituents'. Its absence says the data kind publishes no time axis,
and never says why.
_Avoid_: "historization mode", "temporal type", bare "shape"; "composer" for what
derives one — see Complex DataKind.

**Time axis**:
The temporal dimension that a data kind's temporal shape names as the time of
its rows (`occurred_at`, `event_time`, `snapshotted_at`), or the two dimensions
of an SCD validity range (`valid_from`, `valid_to`).
_Avoid_: "time spine" — in other semantic layers a time spine is a table with
one row per date.

**Grain**:
The level of detail of a set of rows: the dimensions at which one row stands for
one combination of values, together with the granularity of each of them that has
one. A request's grain is its selected dimensions with their requested
granularity; a source's grain is at any of its guaranteed keysets, or at all its
dimensions where it has none, with its own granularity.
_Avoid_: "grain level", "resolution"; reading grain as the time axis alone —
where older text gives a data kind "a grain" on the time axis, it means its
granularity; "grain" for a Grainset's child — older text (ADRs 0064–0074, the
manifest's `grains`) uses it so, and the child is a child.

**Granularity**:
How coarse the values of one dimension are, drawn from a totally ordered roster.
Today only a temporal dimension has one, from minute to year. Coarseness
expresses selection rank, not arithmetic divisibility.
_Avoid_: "grain" for this one coordinate of a grain; "resolution".

**Cadence**:
The source-side interval at which a timeseries or snapshot data kind produces
rows, fixing the finest granularity it can answer.
_Avoid_: using cadence as a synonym for granularity.

**Rollup**:
Coarsening a surface's granularity on its time axis by bucketing and
re-aggregating. Its legality depends on granularity order, temporal shape, and
additivity.
_Avoid_: "coarsening"; "rollup" for every re-aggregation — see Re-aggregation.

**As-of anchoring**:
Resolving a history- or capture-bearing surface to the single row valid at a
given reference instant, instead of filtering rows by a time axis. The
temporally matched join is one mechanism for it, not the concept.
_Avoid_: "as-of join" for the semantics.

**Cascade**:
The rule by which a Complex kind's effective temporal shape is reconciled with
its constituents'. Downward, a parent may supply a shape to constituents that
declare none, and may declare its own only where it agrees with theirs. Upward, a
vertical stack and a grain-routed group both require agreement, while a
horizontal composition takes the highest shape rank instead. Leaf-only
configuration such as catalog and storage never cascades, and neither does a
granularity downward — though a granularity the constituents agree on is derived
upward. No constituent's entry takes its parent's granularity; the planner
reads the granularity of a composition that is a constituent of another
composition, declared or derived, for every constituent below it, down to a
constituent that declares its own (ADR 0074).
_Avoid_: "inheritance", "propagation"; "composer".

**Shape rank**:
The total order on temporal shapes by how much each multiplies rows for one
entity — validity ranges lowest, then a regular sample, then a periodic capture,
then individual occurrences. A horizontal composition takes the highest rank
among the constituents that carry a shape; a constituent carrying none abstains,
and equal ranks over different time axes yield no shape at all where the
composition declares none. A horizontal composition may declare its own shape.
The declared shape must equal a shape at the highest rank, unless no constituent
carries a shape, and the composition then carries the declared shape. It is a total
order so that the result does not depend on how a composition is bracketed.
_Avoid_: "precedence", "priority", "dominance"; reading a rank as a claim about
row counts in any particular model.

### Expressions

**Semantic expression**:
The transient authoring form of an expression, whose leaves name declared
semantics. It exists only between parse and compile and never reaches a planner
or an adapter.
_Avoid_: dropping the "unresolved" qualifier when the distinction from the
lowered form matters.

**Physical expression**:
The lowered, engine-portable form of an expression, which references only columns,
output fields and, in a manifest, other entries, holds literals, and, in a plan
only, plan-bound placeholders.
This is the form a manifest persists, with no placeholder: one entry per
expression of the semantic model, a semantic signature's expression or filter or a Joinset
source's condition or relationship filter, each holding its whole tree and owned by that one
signature or Joinset source. In the tree, a reference to another signature on the
same semantic source is a link, that signature's expression id, outside a filter
on a Joinset, which holds a copy for each operand (ADR 0077), and every other
part is inlined. Each id is minted from the owner and the root marker, never
from content, and is written as UUID text. The expression is attached one of
two ways, and every column reference is qualified by a source alias: attached
to the physical sources that consume it, each names a column under its physical
source's alias; attached to a composition, each names an output field under the
alias of the semantic source that supplies it (see Column qualifier).
_Avoid_: "resolved SQL", "compiled expression"; reading a qualified reference as
a second kind of expression, or what it names as a second kind of column — it
names an output field in a composition's tree, and the attachment decides
whether a qualifier names a physical source or a semantic source; reading two
identical trees as one shared entry; "node" for an entry, which holds a whole
tree.

**Source alias**:
The short, engine-safe handle a semantic source or a physical source is known
by, which a qualified reference holds as its first part to say which semantic
source or physical source supplies it. Compile makes it from its index in the
manifest: `ss`, the
source role's letter and the index for a semantic source (ADR 0078); `ps` and
the index for a physical source (ADR 0085). It is meaningful only inside the
manifest that declares it.
It exists because a name is not an identifier an engine accepts unchanged: names carry punctuation
and grow with composition depth, and engines truncate. That each alias names one
semantic source or one physical source is a precondition of routing by qualifier
(ADR 0075), which the index gives (ADR 0078).
_Avoid_: "correlation name", "table alias"; treating it as a second identity,
since it resolves to what its id does; persisting it outside its manifest.

**Column qualifier**:
The first part of a qualified reference, naming what supplies it: the physical
source in a Dataset's tree, and the semantic source that supplies an output field
above it. In a join condition it names the constituent
whose semantics the condition reads. In a Joinset's projection it names the
operand whose output the projection reads, and that operand may itself be a
composition. In a Unionset's projection it names the Unionset itself, because
the branches are stacked into one output below it (ADR 0065) — see Output field and
Operand. Above a Grainset's drill-across it names the Grainset itself, because
its children's final aggregates are joined into one output below it (ADR 0084).
Without it a join condition compares one name with itself, since one
name is one semantics on both sides, so both sides of an equality reach the same
name before qualification.
_Avoid_: cutting a qualified name's parts to bind it, since a Dataset's leaf
binds to the scan by its last part, the physical column (ADR 0071), and every
other leaf binds by its parts together as `<first>.<last>`;
reading it as qualifying a semantics, since it says only which semantic source
or physical source supplies the reference.

**Sugar**:
Authoring surface with no canonical form of its own, eliminated at compile time —
either by folding metadata-determined subtrees or by lowering an accessor (a
request for a value relative to another row of the same series) into its
canonical windowed shape, or by replacing a one-child composition (a Grainset
with one child, a Unionset of mode `all` with one branch) with its child, at every
depth. The child takes the composition's name and id; for `extras.temporal` the
child keeps its own shape where it has one, and the composition's fills a child
that has none. Where the composition is a root, the child also takes the
composition's description, AI context and semantic interface. A composition is not
replaced where its parent cannot hold its child's data kind (ADR 0026): it stays as
authored, and replacement continues below it.
_Avoid_: calling parse-time tag normalization sugar; "window function syntax"
for accessors, which authors never write directly.

**Fold**:
The partial evaluation of an expression over its literals, by its declarations'
rules and the canonical CAST's, for a closed list of parts (ADR 0089, rule 21): each part on the list that
its literals decide becomes its value, a literal or the branch that a CASE's
literal conditions choose, and every other part stays as it is, for the engine. A
part joins the list only when a rule of the planner or of compile reads its value.
_Avoid_: "simplify" or "optimize" for it: a fold evaluates literals and changes no
meaning.

**Untyped literal**:
A literal that a model writes with no type tag, such as `42` or `lit: acme`. It
holds its parse default until compile gives it the type of its context.
_Avoid_: "untagged literal".

**Expression layer**:
The classification of where an expression's result lands relative to
aggregation — row-level, produced by aggregation, or computed over already
aggregated inputs. It follows from structure alone; in a manifest entry, a link
counts as the layer of the entry that it links.
_Avoid_: bare "layer", which also names the canonical vocabulary split and the
dependency stack.

**Operation node**:
A node of the expression tree that is not a column, a literal, a leaf or a
function call: an operator such as `+`, `=` or `AND`, or one of CASE, CAST, IN,
BETWEEN, LIKE and IS NULL. It is its own node, never a call of a function.
_Avoid_: "function" for it; "operator" alone where a plan node could be meant.

**Canonical data type**:
The engine-agnostic roster of types a model may declare: the scalar types and the
array. A type takes SQL's name for it where SQL has a name with the same meaning.
It is deliberately spelled in no single engine's dialect. A model may write a
type by an alias, another spelling of the same type, and the manifest keeps the
canonical name. A dialect may
name a type in its own words and take only some of its values. A format adapter
writes it in its format's own names at emission, and the canonical layer never
coerces implicitly.
_Avoid_: "logical type", "semantic type" used loosely; never conflate it with a
**native type** or an engine's own type; reading a dialect declaration as a second type: it names the
canonical one and narrows its values; `byte`, `short`, `long` or `float` as a
type's name: they are the aliases of `tinyint`, `smallint`, `bigint` and `real`, in
that order; `list` for the array.

**Type family**:
A named set of canonical data types: number, string, boolean, binary, date and
timestamp, time, interval, year-month interval, day-time interval, uuid, and an
array family for each of them, by the array's element. Each canonical data type
is in one family. An
operation node other than CASE and CAST checks that its operands are in the
family its declaration names, and a function's parameter may name a family.
Inside a family, the declared widening order names the bigger type.
_Avoid_: "type class" for it, the retired name; reading a family as a coercion:
the engine converts the operands, and semstrait only checks and names the
result; reading a family as the cast rule: which casts are legal is a table of
its own.

**Type system**:
The rules that give a result type from the types of the arguments, where a
return rule names one: the sum, product, quotient and remainder of two
decimals, the bigger of two decimals, and the result of `sum`, `avg`, `median`,
`ceil`, `floor` and `round`. A rule reads a type's limits, such as the greatest
decimal precision or an integer's digits, from that type's declaration. The
canonical type system holds the result types that semstrait wants, and a
dialect's type system overrides it where the dialect's rules differ, as
Substrait's does for `avg`. A manifest is typed by the canonical one, and a
Substrait plan by Substrait's.
_Avoid_: reading it as a conversion of the operands: the engine converts them,
and the type system only names the result; reading it as the data types and
their limits, which Substrait calls its type system: those are the canonical
data types and their declarations; reading it as how an engine holds a value
inside, such as DataFusion's Arrow types; reading a dialect's type system as
the types of a manifest.

**Canonical function**:
A function in the function catalog. Its identity is stable and independent of
any engine's spelling.
_Avoid_: "builtin", "UDF", "SQL function".

**Function catalog**:
The canonical declarations of every function that semstrait knows, each with its
dialect declarations. A call can name only a function in it, and a function
enters it by its declaration, not by an extension.
_Avoid_: bare "catalog", which is the physical catalog; "registry", which is the
code that finds a function by its name; "builtins"; "adapter-extended function":
a function is in the catalog or not at all.

**Canonical declaration**:
What semstrait declares once, for every engine, about a function, an operation
or a data type. For a function or an operation, it is its name and the other
spellings an author may write, its generic shape (its arguments or operands, and
its result type), when its result is NULL, and the options that change its
behaviour; for a function, also its **volatility** and the clauses a call may
write. For a data type, it is its name, the other spellings an author may write,
and the values its precision, scale and length take. A
function is canonical when it is declared with one.
_Avoid_: "max spec", "function spec"; reading it as a claim that every engine
has the function in this form.

**Dialect declaration**:
What semstrait declares about a function, an operation or a data type as one
dialect has it: the same parts as a canonical declaration, under the dialect's
own name and with no other spelling. For a function or an operation, it is
written only where the dialect's differs in name, arguments, NULL behaviour, the
clauses a call may write, or result type; for an operation, also where only the
values of an option differ. A DataFusion declaration gives the canonical result
type, because the plan that DataFusion runs holds Substrait's types. For a data
type, it is written only where the dialect's type differs in name or in the
values it takes, or, for a dialect whose spellings compile reads, where its bare
spelling takes a value that the dialect fixes. Where the dialect has no such
function, operation or data type, the dialect's entry says so; for a function,
the engine's rules then write the call from other functions. A function's
holds for every release of its dialect, or only for the releases that it
names, where they differ from the dialect's reference release. It holds no
rewrite.
_Avoid_: "engine function", "override", "mapping".

**Function type**:
Whether a function is scalar, aggregate or window. A call of a function is a
call of its function type, and the type fixes which call clauses that call can
hold; each function's declaration says which of them a call may write, and a
call holds no other.
_Avoid_: "function category", which is the value domain: string, math,
temporal.

**Volatility**:
What a function's declaration says about when its call can give a new value. An
Immutable call gives the same value for the same arguments in every request. A
Stable call gives the same value for the same arguments within one request, as
`current_date` does. A Volatile call can give a new value each time it is
evaluated, as `random()` does. A semantics whose tree calls a Volatile function
still has one value on each row in one request. Every reader of the semantics on
that row reads that value.
_Avoid_: "deterministic" and "non-deterministic", which do not tell Immutable
from Stable; "constant" for Stable; a volatility on an operation, which has none
(ADR 0092).

**Call clause**:
SQL syntax that a call writes next to its arguments: DISTINCT, FILTER, ORDER BY,
WITHIN GROUP, IGNORE NULLS or RESPECT NULLS, and OVER with its partition, order
and window frame. A declaration says, for each clause, whether a call may write
it.
_Avoid_: "option"; "modifier".

**Function option**:
A behaviour of a function or an operation, chosen from a closed set, that has no
SQL syntax on the call, for example how `concat` treats a NULL argument. A declaration names
each option's values and its canonical value.
_Avoid_: "clause"; "flag".

**Parameter**:
A typed placeholder left in a canonical expression for a value knowable only once
a request exists. It is never authored and must be bound before a format
artifact is produced.
_Avoid_: "bind variable", "query parameter".

### Planning and routing

**Resolver**:
The kind-agnostic part of planning that runs before any frame: it binds a
request's names to bits, fixes its scope root, and orders the candidates inside
the resolution scope. It chooses where a request is answered from and builds no
plan; the scope root's demand comes from its result.
_Avoid_: "router", "selector"; using it for the per-kind logic — see Strategy;
"step" for it — a step builds one plan node.

**Field-first resolution**:
Choosing a request's scope root among the public candidates when the request
names no `from`, by locating who serves the demanded set. It infers what `from`
would have named and nothing more: once the root is fixed, planning proceeds
exactly as if the request had named it.
_Avoid_: "auto-join" — it composes nothing; "implicit from" as the concept name,
though that is what it infers.

**Candidate**:
A semantic source that can answer a request as it stands: it serves the whole
demanded set and admits the requested granularity on its time axis. Field-first
resolution chooses the scope root among public candidates; a Strategy chooses
within the resolution scope among the candidates there.
_Avoid_: "match", "hit"; reading "not a candidate" as "pruned" — a
non-candidate is still executed wherever a candidate composition has it as an
operand.

**Scope root**:
The public semantic source a request is answered through — named by `from`, or
inferred by field-first resolution. A request has exactly one.
_Avoid_: "entry point", "anchor", "target", "base".

**Resolution scope**:
The semantic sources a request may be answered from once its scope root is
fixed: the root together with the private units it originated. It is never
widened past the root; which unit inside it is used is the root's Strategy's
decision.
_Avoid_: "search space", which is the public sources before a root is fixed;
"entry point"; "anchor".

**Bypass**:
A Strategy answering a request from a private unit of the resolution scope — its
stand-in — instead of the scope root itself, because that unit is a candidate of
lower structural complexity. Which units may stand in for the root is a fact of
the root's source role — a Joinset's sub-compositions and a Grainset's children
may, a Unionset's branches never may.
_Avoid_: "skip"; "pruning" — the stand-in is read, and what the plan leaves out
is pruned; "shortcut".

**Structural complexity**:
The request-free, statistics-free part of how candidates are ranked: first by
how many composed semantic sources the candidate is built from — itself and
everything beneath it through its operands, a Dataset counting none — then by
how many semantics it exposes, fewer being lower. The resolver ranks by its
first term, then by granularity fit, then by its second term.
_Avoid_: "cost", which implies statistics the planner never reads; "weight";
counting only direct operands, which would score a deep chain the same as one
join.

**Granularity fit**:
The order in which sources that serve a request rank by the granularity of
their time axis. First comes a source whose time axis the request buckets, then
one whose time axis the request selects without a granularity, then one whose
time axis the request does not select, or that has no time axis. Inside each group the coarsest comes
first, except where the request selects the time axis without a granularity:
there the finest comes first. A source with no granularity holds its time axis
at its own values, which are finer than any granularity, and a public Grainset
ranks at the granularity of what its routing answers with. A Grainset picks its child by the *Child rank*, which uses this order. A child with no time axis ranks coarser than Year. The resolver ranks public candidates by this order after their
count of compositions.
_Avoid_: "grain fit", since it compares granularity only, and a grain's
Dimensions are the candidate test's; "pick" for the order, since a pick chooses one;
"cost".

**Child rank**:
The order in which a Grainset ranks its children that serve a demanded name. A
Grainset sends each name to the child ranked first for it alone, and it joins
the children of a drill-across in this order. The rank has two forms, both the
one `ChildFit`. The pick of a child for a name uses fewest Dimensions, then
*Granularity fit*, then *Name order*. The joins of a drill-across use
*Granularity fit* first, then fewest Dimensions, then name order, so the first
child is at the requested granularity, or the closest to it. The pick uses the
form of the joins where the demand selects the time axis without a granularity
(finest first). Where a
child's grain equals the demand's grain, the child rank picks that child. A tie
in the pick of a child for a name that name order decides is reported as an
advisory (*Path ambiguity*). The order of a drill-across's joins raises none
(*Name order*).
_Avoid_: "granularity fit" for the whole rank, since the fit is one of its three
parts; the resolver's rank for public candidates, which ranks by *Structural
complexity* first.

**Strategy**:
The per-role planning logic that decides, for one frame and its demand, who
answers — the semantic source itself or a stand-in — which operands are read and
which names each serves, and how their rows combine. Its Strategy rules make
that decision; it builds no plan nodes, and it runs no shared rule. The source
role of the frame's semantic source chooses it, and exactly one exists per source
role.
_Avoid_: "handler", "lowering pass"; "horizontal strategy"; "resolver", which is
the kind-agnostic part before it — see Resolver; choosing it by authored data
kind, since a compile-synthesized semantic source is planned by the same rule.

**Strategy rule**:
A planning rule that belongs to one Strategy and is part of its decision — for
example the Unionset's pruning of a branch that a predicate excludes, the
Joinset's pruning of an operand, or the Grainset's pick of a child.
_Avoid_: bare "rule" in planning, since compile and validation have rules too;
"horizontal strategy".

**Shared rule**:
A planning rule that applies to every frame, whatever its source role: the
delivery of each operand, the re-aggregation of partial aggregates, where a
Metric's formula is applied, bucketing to the requested granularity, and where
each predicate is applied. The driver runs the shared rules as one fixed list for
every frame; a shared rule reads how rows combine, never which source role
combined them.
_Avoid_: bare "rule" in planning; "vertical strategy", "delivery rules".

**Frame**:
One semantic source's part in planning one request: its demand, its Strategy's
decision, what the shared rules fixed for it, and the frames of the operands it
reads. Frames form the frame tree from the scope root, built before any plan node
exists; each frame becomes its own subtree of plan nodes. A frame is named by its
frame path — the operands from the scope root down to it — and never by its
semantic source, which can stand at two paths in one plan.
_Avoid_: "frame" alone for a window frame — write "window frame"; "decision
tree", ADR 0073's older name for the frame tree; "unit"; "lineage" for the
frame tree; "memo group" — a frame holds one decision and no cost.

**Demand**:
What a frame's parent requires of it: the names it must serve, each with its
delivery; the requested granularity; the semantic sources that may stand in for
it; and the predicates placed on its rows. The scope root's demand comes from
the resolver; an operand's demand comes from its parent's decision and the
shared rules.
_Avoid_: "demanded set" for it, which is the whole request's; "slice of a
request"; "requirements".

**Delivery**:
The form in which an operand gives one name to its parent: rows, partial
aggregates that the parent combines again, or final aggregates. Partial
aggregates are an intermediate step: grouped by the values before bucketing, and
combined again by the parent. Final aggregates are finished at the request's
grain, bucketed, and nothing aggregates them again. The parent's demand fixes
the delivery, through the shared rules.
_Avoid_: "form" alone, which suggests Emit form; "phase" — no plan node carries
an aggregation phase; "state", "mode".

**Combine**:
How the operands that a frame reads give it one input: leaf (a Dataset over its
physical read), join, cross, stack (the union of branches), pass-through (one
input unchanged — a stand-in's, or the one operand a Joinset keeps), or
drill-across (the final aggregates of a Grainset's children, each at the grain of
the frame's demand, joined on its grouping). The values are a closed set, and the
driver builds each through steps, in ADR 0073's build order: no node for a
pass-through, and for a drill-across one join, or one cross, for each child
after the first.
_Avoid_: "merge"; "composition" for it — the composition is the model's, the
combine is the plan's.

**Driver**:
The planner's loop over the frame tree. In the decide pass, top-down from the
scope root, it calls each frame's Strategy and runs the shared rules; in the
build pass, bottom-up, it turns each frame into plan nodes through steps, in one
fixed order.
_Avoid_: "lower" or "lowering" for the build pass — lowering is compile's;
"emit", the adapter's; "optimize"; the retired cross-grain sense ("the routing
unit covering the largest share of a request"), with "primary/secondary" and
"base/extension".

**Step**:
A builder of one plan node: from values that are already resolved, it writes the
node and its node-common block. It decides nothing and reads nothing from the
manifest.
_Avoid_: "operator", the plan-node vocabulary's; "pass"; "stage", the
resolution pipeline's; "hop" and "join step", a composition's.

**Physical read**:
The read below every Dataset frame: the reader copied from the physical source,
the columns read, and their types. It is not a data kind and not a frame; the
Dataset's Strategy creates it.
_Avoid_: "physical frame", "physical source frame", "scan frame"; "scan" for it —
a scan is the plan node it builds.

**Read type**:
The type that a physical read reads a column as where no catalog observed the
column's type: the widest declared type among the semantics that map the column
directly, else the first in name order, and a count's type only where only counts
map it. Compile writes it into the source's schema; it is not an observation.
_Avoid_: "observed type", which only a catalog gives; "read type" for the kind of
read, which is the scan's read identity (ADR 0067).

**Pruned**:
A part of a frame's composition that a rule removes from the plan, so that it has
no plan nodes: a Unionset branch that a placed predicate excludes, a Unionset
branch whose children cannot serve the demand, a Joinset operand that the
Joinset does not read, or a Grainset child that the Grainset does not pick. The frame may read the part's signature or names to decide; for a Unionset branch whose children cannot serve the demand, it reads the children's coverage masks and guaranteed keysets.
_Avoid_: "pruned" for an error or a refusal; "skipped", "elided", "left out" as
other names for it; "pruned" for the columns a read does not read, or for the
partitions an engine skips.

**Re-aggregation**:
Aggregating again over values that are already aggregates: partial aggregates
combined above a union or a join, or a coarser granularity on the time axis (see
Rollup). Additivity decides whether it is faithful.
_Avoid_: "refold", "merge"; "decomposable" as a test beside additivity.

**Formula push-down**:
The planner's rewrite of a metric's formula into one sum over the rows of its
frame, where the formula reads a dimension or a column that the finishing demand
does not group by. It applies only to a fixed pattern of terms over sums and
counts. In ADR 0074, "the push-down" is its short form after first use.
_Avoid_: "linear rewrite", "value push-down", a bare "push-down". A predicate is
placed (ADR 0071), and an aggregate below a join is pre-aggregation.

**Formula part**:
One of the sets of rows into which a group of a metric's demand splits, where
the metric has no aggregation and its formula reads a dimension or a column
that the demand does not group by: the rows that select the same results of the formula's cases
and hold the same values of the row-level expressions that those results read. The
metric's value is the sum of the formula's values over its parts.
_Avoid_: "branch", which is a Unionset's; "bucket", which is a granularity's.

**Formula grain**:
The grain at which a metric with an aggregation of its own, whose formula reads
a measure or a metric, computes its formula before it aggregates the formula's
values: the grain of the demand that
finishes it, with each dimension and column that the formula reads outside an
aggregate, at the value before bucketing.
_Avoid_: "inner grain" as a declared property: the model declares none, and the
formula grain is read from the formula.

**Formula node**:
One aggregate call of a measure under one condition. A metric's formula reads
the call from the branches of the Unionset that finishes the metric. That
Unionset fixes the condition: the whole filter of the call in the metric's tree, that is,
the measure's `filter:` at its site in that Unionset and the `filter:` of the
metric and of each metric that it links on the way to the measure, joined by
`and`. Equal calls, of one metric or of two, are one formula node. Each composed
branch that maps the measure gives the node in one output field, under the
node's condition joined by `and` after the branch's own filter for the measure.
The condition that the branch's demand holds for the measure decides only the
field: the measure's own field where that condition equals the node's and the
branch gives partial aggregates, else a field of its own under a reserved name.
Where the branch is a Unionset, a Grainset or a pass-through, each frame that it
reads and that maps the measure gets the node unchanged, in the same way. A
composed branch that NULL-fills the measure gives no field, and the Unionset
writes the typed NULL of the call; for a branch that gives rows, the Unionset's
own aggregate step computes the node.
_Avoid_: "aggregate node" for it, which names one aggregate call of a formula's
tree; `AggregateNode`, a plan node; "node" alone, which suggests a plan node or
a node of the expression tree.

**Disjoint branches**:
Unionset branches that a selected Dimension separates: on every branch it folds
to a different non-NULL literal, through a composed branch's operands as ADR
0073 folds a predicate to prune a branch, and the values stay different after
bucketing, so no group spans two branches.
_Avoid_: "partitions", "shards".

**Row identity**:
The columns by which the plan keeps one row per row of a fanned operand before
an aggregate reads it: the Dimensions of the operand's first guaranteed keyset,
or every Dimension it exposes where it has none, which is an inference.
_Avoid_: "dedupe key", "primary key"; "symmetric aggregates", Looker's
mechanism, which gets the same answer by another computation.

**Marker**:
A window output field that the planner adds below an aggregate step. An
aggregate reads only the rows where each of its markers is 1: one row for each
row identity, or the rows that a semi-additive strategy keeps.
_Avoid_: "flag"; "window frame"; "filter", which is the model's named predicate.

**Lossy re-aggregation**:
The condition where the rows an aggregate reads are already aggregates and the
grain comparison says that a request re-aggregates them, so a non-additive aggregate
over them is unfaithful to a single pass over the underlying data. It raises an
advisory rather than a refusal.
