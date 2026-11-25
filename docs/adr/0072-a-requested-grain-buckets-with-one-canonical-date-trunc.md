---
number: 72
title: A requested grain buckets with one canonical `date_trunc`, in the request's timezone
status: accepted
date: 2026-09-29
source:
- CONTEXT.md
- docs/adr/0021-portability-floor-no-implicit-coercion.md
- docs/adr/0022-engine-independent-canonical-vocabulary.md
- docs/adr/0062-a-request-states-intent-never-placement.md
- docs/adr/0063-the-planner-takes-a-request-context-not-a-session.md
- docs/adr/0066-the-plan-vocabulary-mirrors-substrait-v0-103-1-relations.md
- docs/adr/0068-adapters-are-layered-format-first.md
- docs/adr/0071-a-predicate-is-applied-where-its-semantics-carry-their-meaning.md
- Substrait `extensions/functions_datetime.yaml` at v0.85.0 (vendored in crate `substrait` 0.63.0) and at v0.103.1
- crates `datafusion-substrait` 55.0.0 (`logical_plan/consumer/expr/function_arguments.rs`, `scalar_function.rs`) and `datafusion-functions` 55.0.0 (`datetime/date_trunc.rs`, `datetime/to_local_time.rs`)
- the human's rulings of 2026-10-17
links:
- target: 21
  kind: amends
- target: 22
  kind: amends
- target: 63
  kind: amends
- target: 68
  kind: amends
- target: 74
  kind: amendedby
- target: 80
  kind: amendedby
- target: 87
  kind: amendedby
- target: 91
  kind: amendedby
---

# A requested grain buckets with one canonical `date_trunc`, in the request's timezone

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0074](0074-an-aggregate-sits-where-grain-and-additivity-allow-and-moves-up-otherwise.md): "requested grain" is read as the requested granularity, and "grain" for the temporal roster means the granularity.
> - [ADR 0080](0080-only-a-frame-that-gives-final-aggregates-buckets-and-every-filter-reads-the-value-before-bucketing.md): in *The function*, "the subtree that groups the temporal item" is the frame whose own aggregate step gives final aggregates; a frame that gives rows or partial aggregates gives the temporal item at its value before bucketing; the rest of 0072 stands (0080's *Only a frame that gives final aggregates buckets*).
> - [ADR 0087](0087-the-function-catalog-declares-the-common-sql-functions-that-datafusion-can-run.md): in *Admission under ADR 0021*, "The floor stands for every other function and argument", and in its *Amends* item on ADR 0021, "The intersection floor stands for every other function and argument", become: every other function is in the function catalog by ADR 0087's rule 1, which admits a function that the DataFusion rules write in DataFusion's own functions. The rest of 0072 stands.
> - [ADR 0091](0091-a-function-has-a-canonical-declaration-and-a-dialect-declaration-where-the-engine-differs-and-a-call-is-a-call-of-its-function-s-type.md): in its own *Amends* item on 0068, "the plan holds only canonical names: engine identity stays in the rules" becomes: the plan holds only canonical names: engine identity stays in the rules and in its dialect's declarations. The rest of 0072 stands.

## Context and Problem Statement

A requested granularity on a selected temporal Dimension (ADR 0062) asks for
that Dimension bucketed to the granularity. Before this ADR nothing could write
the bucket: no canonical function truncates a date or a timestamp, and the
model's `date_trunc` form parses but compile refuses it as `UnknownFunction`,
because the function registry has no entry for it.

## Considered Options

- One canonical `date_trunc` (chosen).
- **The producer writes `date_trunc` by name.** DataFusion would read it as it
  is, but `date_trunc` is no Substrait function, and ADR 0068 has the producer
  write Substrait's spellings and leave an engine's needs to its rules.
- **No bucketing in MVP**, a requested granularity coarser than the source's
  refused. It leaves ADR 0062's requested granularity and ADR 0063's timezone
  without a reader.
- **Outputting the UTC instant of the local bucket start.** The April bucket of
  the Stockholm example below would read `2026-03-31 22:00`, a March date.
- **A separate type for a clock value.** The plan has no reader that needs to
  tell the two meanings apart (see *What a bucketed timestamp means*). It
  becomes needed when something reads a bucket as an instant, or tells apart
  the two hours of a repeated local hour.
- **Grouping a repeated local hour by its instant.** The answer would hold two
  rows with the same clock value, which a caller cannot tell apart without the
  instant in the output.
- **Refusing `minute` and `hour` in a timezone with daylight saving.** It
  refuses every hourly bucket of the year for one hour.
- **An engine's own call in the plan**, as a plan form or as a name in the
  canonical function list. ADR 0068 keeps engine identity in the rules, and an
  engine never widens the canonical vocabulary.
- **A walk that tracks what a rule created** and offers only the offered node's
  own inputs. It changes the walk for every rule, where the function hook is
  enough.
- **Any weekday as `week_start`.** Substrait's `round_calendar` names week
  origins for Monday and Sunday only. Any other day needs a shift in the producer
  as well as in every engine's rules, for a week start no caller has asked for.
- **A timezone-free timestamp read as local clock time.** The request timezone
  would then change nothing, and ADR 0063 reads it for exactly this.
- **A fixed Monday** and **a week start stored in the model**. The first cannot
  be changed per caller. The second needs a manifest carrier for a setting that
  ADR 0063's context already has room for.

## Decision Outcome

This ADR admits one function, says what it means, and
says where each layer writes it.

### The function

`date_trunc` joins the canonical function vocabulary (ADR 0066: the expression
vocabulary grows only by ADR). Its arguments:

| Argument | Form | Present | Default |
|---|---|---|---|
| `granularity` | a literal from the `Granularity` roster: `minute`, `hour`, `day`, `week`, `month`, `quarter`, `year` | always | |
| `value` | the Dimension's value, `Date` or `Timestamp` | always | |
| `timezone` | an IANA name literal | for a `Timestamp` only | `'UTC'` |
| `week_start` | a weekday literal, Monday or Sunday | for the `week` grain only | `'monday'` |

The defaults are the declaration's (ADR 0087, rule 14). The planner writes the
request's timezone and week start (ADR 0063); a semantic model's `date_trunc` takes the
defaults where it leaves them out (ADR 0089, rule 4).

The result type is the type of `value`: a `Date` stays a `Date`, a `Timestamp`
stays a `Timestamp`. Nothing is coerced (ADR 0021).

The planner writes it at the subtree that groups the temporal item, as part of the
projection before that subtree's aggregate step. A request predicate reads the
value before bucketing (ADR 0071).

### What a bucketed timestamp means

ADR 0022 keeps timestamps timezone-free. This ADR gives the timestamp a source
supplies its meaning: **it holds a UTC instant.** Bucketing reads that instant
on the request timezone's clock, truncates there, and outputs the bucket's start
as a **clock value** on that clock, still timezone-free. The output is a label,
not an instant.

Example: `2026-03-31 23:30`, grain `month`, timezone `Europe/Stockholm` (UTC+2
on that date). The clock value in Stockholm is `2026-04-01 01:30`, and the
bucket is `2026-04-01 00:00` — April, the month the row belongs to in Stockholm,
labelled as April. The label is midnight in Stockholm; read as a UTC instant it
would be two hours late.

So one timezone-free `Timestamp` type has two meanings: an instant where a
source supplies it, a clock value where bucketing outputs it. The two never meet
in the plan. A request predicate reads the value before bucketing (ADR 0071),
and a bucket is only grouped by and output: nothing converts it or buckets it
again.

When the clocks go back, one local hour happens twice, and both fall in one
bucket. In `America/New_York` on 2026-11-01, the source values `05:30` and
`06:30`, UTC instants, both read `01:30` on the local clock. At the `hour` grain
both rows fall in the bucket `01:00`, and its Measures read both hours; at the
`minute` grain each repeated minute does the same. When the clocks go forward,
the skipped hour has no bucket. The `day` grain and coarser are unaffected. This
follows from a bucket being a clock value, and the DataFusion rules give the
same buckets: DataFusion's `date_trunc` keeps the two hours apart, and
`to_local_time` gives both `01:00`.

A `Date` takes no timezone. Under the default timezone, UTC, the conversion is
the identity.

### The week start is a request setting

`RequestContext` gains **`week_start`**, **Monday or Sunday**, **default
Monday**. It is read at one site: bucketing to the `week` grain. It sits beside
`timezone` because both are settings under which a request is planned, and a
cached plan keys on the context already (ADR 0063).

The two values are the two week starts Substrait's `round_calendar` can name
(below). Another weekday is a new value in the roster, which breaks no caller.

### Where each layer writes it

- **The plan** carries canonical `date_trunc` with the arguments above.
- **The Substrait producer** writes Substrait's own functions with their official
  spelling (ADR 0068). For a `Timestamp`:

  ```text
  local_timestamp(
    round_calendar(assume_timezone(value, 'UTC'), FLOOR, unit, origin, multiple, timezone),
    timezone)
  ```

  `assume_timezone` reads the value as the UTC instant it holds. `round_calendar`
  floors it on the request timezone's clock and returns the instant of the
  bucket's start. `local_timestamp` turns that instant into its clock value. For a
  `Date` the producer writes `round_calendar` alone. The grain gives the rest:

  | Grain | `unit` | `multiple` | `origin` |
  |---|---|---|---|
  | `minute` | `MINUTE` | 1 | `HOUR` |
  | `hour` | `HOUR` | 1 | `DAY` |
  | `day` | `DAY` | 1 | `MONTH` |
  | `week` | `WEEK` | 1 | `ISO_WEEK` for Monday, `US_WEEK` for Sunday |
  | `month` | `MONTH` | 1 | `YEAR` |
  | `quarter` | `MONTH` | 3 | `YEAR` |
  | `year` | `YEAR` | 1 | `YEAR` |

  Substrait has no quarter unit. The `MONDAY_WEEK` and `SUNDAY_WEEK` origins are
  not used: they count weeks from the first Monday or Sunday of January, so the
  days before it would start a week on 1 January. ISO and US weeks run whole
  across the turn of the year.

  `minute` and `hour` apply to a `Timestamp` only: Substrait's `Date` form takes
  `DAY` and coarser units. The grain says which granularity the value is turned
  into; the timezone and the week start only change how that function behaves.
  Whether a grain finer than the value's own may be asked for is ADR 0062's grain
  rule, not this ADR's.

  For a `Date`, the overload also takes an `origin` of type `date`, which
  Substrait does not describe. The producer writes an anchor that agrees with the
  enum origin, so either reading of the argument gives the same buckets:
  `1970-01-01`, the start of a year, a quarter, a month and a day, for `day`,
  `month`, `quarter` and `year`; for `week`, `1970-01-05`, a Monday, or
  `1970-01-04`, a Sunday. The producer's tests check the buckets; they do not
  choose the anchor.
- **The DataFusion rules** write the canonical `date_trunc` call in DataFusion's
  own functions when the producer lowers it (ADR 0068, as amended below).
  DataFusion 55's Substrait consumer cannot read what the producer writes: it
  resolves a function by name, knows none of `round_calendar`, `assume_timezone`
  and `local_timestamp`, and refuses an enum argument. So the rules' function
  hook returns the whole call in Substrait, built around the arguments the
  producer has already lowered. No rule sees what the hook writes, so
  DataFusion's own `date_trunc` is never read as the canonical one. DataFusion
  resolves `date_trunc('<grain>', value)` and `to_local_time(value)` by name.
  `date_trunc` truncates on the clock of its timestamp's zone, starts a week on
  Monday, and returns a `Timestamp` for a `Date`. So the rules write the zone in
  two steps: they tag the value as UTC (a cast to `precision_timestamp_tz`,
  which DataFusion reads as zone `UTC`), then move it to the request timezone
  with `arrow_cast`, the one route to a named zone. A single `arrow_cast` from
  the timezone-free value would read it as local time in the target zone. The
  rules then write `to_local_time` for `local_timestamp`, and cast a `Date`
  result back to `Date`.

  For a Sunday week start the rules shift twice: the value one local calendar day
  forward before `date_trunc`, and the result one local calendar day back after
  it, both before `to_local_time` and before the cast back to `Date`. `date_trunc`
  returns a Monday. Shifting only the value labels the bucket with that Monday;
  shifting only the result puts a Sunday into the week before.

**Admission under ADR 0021.** `date_trunc(grain, value)` is native in DataFusion,
DuckDB and Spark, so it passes ADR 0021's floor. The `timezone` and `week_start`
arguments do not: they are the calendar spec, not the grain, and each engine's
rules write them in that engine's own functions (ADR 0068), which is not a
trivial renaming. This ADR amends ADR 0021 for exactly that case: a
canonical function may take a calendar-spec argument when every engine's rules
write it in that engine's own functions. The floor stands for every other
function and argument.

### Amends

- **0063:** `RequestContext` carries `week_start`, Monday or Sunday, default
  Monday, beside `timezone`, read only when bucketing to the `week` grain; the
  timezone is no longer its only field.
- **0022:** a timezone-free `Timestamp` that a source supplies holds a UTC
  instant, and a bucketed one is a clock value in the request timezone. The type
  roster is unchanged.
- **0021:** a canonical function may take a calendar-spec argument — the
  request's timezone or week start — that no engine takes natively, when every
  engine's rules write it in that engine's own functions (ADR 0068). The
  intersection floor stands for every other function and argument.
- **0068:** a rewrite stays in the canonical vocabulary, and no rule rewrites a
  node the replacement creates. The format-detail hook for a canonical function
  widens: besides the Substrait name and URI, it may return the whole call,
  written in the engine's own functions around the arguments the producer has
  already lowered. The producer writes it as returned and offers it to no rule.
  So the walk still ends by construction, and the plan holds only canonical
  names: engine identity stays in the rules.

### Consequences

- `date_trunc` gains a declaration in the function catalog (ADR 0091, rule 3).
  The semantic model's own `date_trunc` form compiles against it, with the
  declared defaults (ADR 0089, rule 4), and so does its `extract` form (ADR 0087,
  rule 14).
- A Sunday week start and a timezone other than UTC are written by every
  engine's rules, not plan changes.
- In the DataFusion call, the values between the two casts and `to_local_time`
  carry a timezone. They exist only in the Substrait the function hook writes,
  so ADR 0022's type roster needs no type for them.
- At the `minute` and `hour` grains, a repeated local hour is one bucket
  (above). A caller that needs its two hours apart needs the rejected type for a
  clock value.
- Substrait does not describe the `Date` overload's `origin` date. The anchors
  above hold under either reading of it.

## More Information

Decided 2026-09-29, and on 2026-10-17 for the
defaults of `date_trunc`.
