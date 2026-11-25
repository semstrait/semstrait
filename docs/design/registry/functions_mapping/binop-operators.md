---
id: registry-functions-mapping-binop
status: ratified
version: 1.0
impl: drift
---

# BinaryOp Operators — Comparison, Arithmetic, Logical

This leaf specifies BinaryOp per-engine emission for comparison operators (Eq/NotEq/Lt/LtEq/Gt/GtEq), arithmetic operators (Add/Subtract/Multiply/Divide/SafeDivide/Mod) including per-engine result-type reality, and logical operators (And/Or/Not/IsNull/IsNotNull).

## 4. Comparison Operators

All comparison operators are `BinaryOp` variants per `14 §3.3` (`BinaryOpKind::{Eq, NotEq, Lt, LtEq, Gt, GtEq}`) — rendered by the Dialect layer's `BinaryOp` emission path. Return type always `Boolean`.

| Canonical | `BinaryOpKind` | SQL | DataFusion | Tier | DuckDB | Tier | Spark | Tier | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `equal` | `Eq` | `a = b` | `a = b` | Name-only | `a = b` | Name-only | `a = b` | Name-only | Universal. |
| `not_equal` | `NotEq` | `a <> b` / `a != b` | `a <> b` | Name-only | `a <> b` / `a != b` | Name-only | `a <> b` / `a != b` | Name-only | All three accept both spellings; canonical emission uses `<>` (ANSI). |
| `lt` | `Lt` | `a < b` | `a < b` | Name-only | `a < b` | Name-only | `a < b` | Name-only | Universal. |
| `lte` | `LtEq` | `a <= b` | `a <= b` | Name-only | `a <= b` | Name-only | `a <= b` | Name-only | Universal. |
| `gt` | `Gt` | `a > b` | `a > b` | Name-only | `a > b` | Name-only | `a > b` | Name-only | Universal. |
| `gte` | `GtEq` | `a >= b` | `a >= b` | Name-only | `a >= b` | Name-only | `a >= b` | Name-only | Universal. |

Portability summary: **Universal** across all six.

Per `14 §5.4`, semstrait does NOT validate operand-type compatibility for comparisons — each engine raises its own diagnostics at execution time if e.g. `Integer < String` is attempted. Cross-engine comparability rules (`Integer` vs `Double`, `String` vs `Date`) are engine-native; see §5 below for arithmetic promotion reference tables.

---

## 5. Arithmetic Operators — per-engine reality (fills the `14a §5` gap)

Arithmetic operators are `BinaryOp` variants (`BinaryOpKind::{Add, Subtract, Multiply, Divide, SafeDivide, Mod}`). Per `14a §5` / Q11, there is no canonical promotion lattice — semstrait delegates result-type derivation to the engine. This section documents the **per-engine observed behavior** so authors writing `data_type:` declarations at a Semantics boundary (and adapter implementers writing reconciliation Casts per `19 §3.7`) can predict each engine's natural result type.

### 5.1 Canonical → engine rendering

All arithmetic `BinaryOp`s render as native infix SQL operators across all three engines:

| Canonical | `BinaryOpKind` | SQL | DataFusion | Tier | DuckDB | Tier | Spark | Tier | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `add` | `Add` | `a + b` | `a + b` | Name-only | `a + b` | Name-only | `a + b` | Name-only | Universal. |
| `subtract` | `Subtract` | `a - b` | `a - b` | Name-only | `a - b` | Name-only | `a - b` | Name-only | Universal. |
| `multiply` | `Multiply` | `a * b` | `a * b` | Name-only | `a * b` | Name-only | `a * b` | Name-only | Universal. |
| `divide` | `Divide` | `a / b` | `a / b` | Name-only | `a / b` | Name-only | `a / b` | Name-only | Universal in syntax; result-type differs per engine — see §5.2. |
| `safe_divide` | `SafeDivide` | `a / NULLIF(b, 0)` | `a / NULLIF(b, 0)` 🟡 | Structural | `a / NULLIF(b, 0)` 🟡 | Structural | `a / NULLIF(b, 0)` / `try_divide(a, b)` (Spark 3.3+) 🟡 | Structural | Canonical posture: emit `NULLIF` wrap universally. Spark 3.3+ adapter MAY switch to `try_divide`. See Q-FUNCS-MAP-017. |
| `mod` | `Mod` | `a % b` | `a % b` 🟡 | Name-only | `a % b` / `mod(a, b)` | Name-only | `a % b` / `mod(a, b)` / `pmod(a, b)` | Name-only | Canonical form is the `%` operator per `14a §4.3` / `14 §3.3`. `mod(a, b)` as a function call is adapter-extended. |

⚠️ `Divide` result-type divergence across engines — see §5.2.

### 5.2 Per-engine result types — integer arithmetic

⚠️ These tables record observed per-engine behavior; all rows 🟡 pending `TD-FUNCS-MAPPING-BINOP-EMPIRICAL` (Q-FUNCS-MAP-012).

**Mixed-integer promotion** (`Add` / `Subtract` / `Multiply`):

| Left | Right | DataFusion | DuckDB | Spark | Notes |
|---|---|---|---|---|---|
| `Integer` | `Integer` | `Integer` | `Integer` | `Integer` | Convergent. Overflow is engine-per-engine; none raises a compile-time error. |
| `Integer` | `Long` | `Long` | `Long` | `Long` | Convergent widen-to-larger. |
| `Integer` | `Double` | `Double` | `Double` | `Double` | Convergent widen-to-float. |
| `Long` | `Long` | `Long` | `Long` | `Long` | Convergent. |
| `Long` | `Double` | `Double` | `Double` | `Double` | Convergent. |
| `Float` | `Double` | `Double` 🟡 | `Double` 🟡 | `Double` 🟡 | Convergent widen to double-precision. |

**Integer division** (`Divide`):

| Left | Right | DataFusion | DuckDB | Spark | Notes |
|---|---|---|---|---|---|
| `Integer` | `Integer` | `Integer` (truncating) 🟡 | `Double` (promotes) 🟡 | `Double` (promotes) 🟡 | ⚠️ **Divergent.** Authors declaring `data_type: Integer` on a quotient on DuckDB / Spark receive an implicit narrowing (`EXPR_W_CAST_NARROW`). See §16.1 and Q-FUNCS-MAP-016. |
| `Long` | `Long` | `Long` (truncating) 🟡 | `Double` 🟡 | `Double` 🟡 | ⚠️ **Divergent.** Same pattern. |
| `Integer` | `Double` | `Double` | `Double` | `Double` | Convergent. |
| `Double` | `Double` | `Double` | `Double` | `Double` | Convergent. |

### 5.3 Per-engine result types — `Decimal(p, s)` arithmetic

Decimal result-type rules are genuinely per-engine and are the widest divergence-surface in the three-engine trio. Rows documented from engine reference docs (DuckDB 1.1.x, Spark 3.5.x ANSI mode, DataFusion 40.x+) and are 🟡 until the `TD-FUNCS-MAPPING-BINOP-EMPIRICAL` test harness lands.

**Addition / Subtraction** (`Decimal(p1, s1) + Decimal(p2, s2)`):

| Engine | Result precision | Result scale | Notes |
|---|---|---|---|
| DataFusion | `max(p1 - s1, p2 - s2) + max(s1, s2) + 1` | `max(s1, s2)` | 🟡 follows Arrow's decimal-arithmetic kernel rules; capped at precision 38 (Decimal128). |
| DuckDB | `max(p1 - s1, p2 - s2) + max(s1, s2) + 1` | `max(s1, s2)` | 🟡 follows SQL standard width-1-widening rule. |
| Spark | `max(p1 - s1, p2 - s2) + max(s1, s2) + 1` | `max(s1, s2)` | 🟡 documented in Spark `DecimalPrecision` analyzer. ANSI-mode overflow raises; non-ANSI wraps. |

Convergent at the precision/scale formula, divergent on overflow handling.

**Multiplication** (`Decimal(p1, s1) * Decimal(p2, s2)`):

| Engine | Result precision | Result scale | Notes |
|---|---|---|---|
| DataFusion | `p1 + p2 + 1` (capped 38) | `s1 + s2` | 🟡 Arrow kernel. |
| DuckDB | `p1 + p2 + 1` (capped 38) | `s1 + s2` | 🟡 SQL-standard form. |
| Spark | `p1 + p2 + 1` (capped 38) | `s1 + s2` | 🟡 Spark `DecimalPrecision`. |

Convergent.

**Division** (`Decimal(p1, s1) / Decimal(p2, s2)`):

| Engine | Result precision | Result scale | Notes |
|---|---|---|---|
| DataFusion | `p1 - s1 + s2 + max(6, s1 + p2 + 1)` (capped 38) 🟡 | `max(6, s1 + p2 + 1)` 🟡 | Follows Spark's formula per DataFusion analyzer docs. |
| DuckDB | `p1 + s2` (capped 38) 🟡 | `s1` 🟡 | ⚠️ **Divergent** — DuckDB does not inflate scale the way Spark does. |
| Spark | `p1 - s1 + s2 + max(6, s1 + p2 + 1)` (capped 38) 🟡 | `max(6, s1 + p2 + 1)` 🟡 | Canonical reference for DF's formula. |

⚠️ DuckDB vs Spark/DF Decimal-division divergence — a Semantics declaring `data_type: Decimal(p, s)` over a decimal division will need different reconciliation Casts depending on the target adapter. Tracked as `TD-FUNCS-MAPPING-DECIMAL-DIV`.

### 5.4 Interval arithmetic

| Expression | DataFusion | DuckDB | Spark | Notes |
|---|---|---|---|---|
| `Date + Interval` | native | native | native (requires SQL-typed interval literal) 🟡 | Per `registry/types_mapping.md §3.2`. Spark `CalendarInterval` rendering caveats apply. |
| `Timestamp + Interval` | native | native | native 🟡 | Same caveats. |
| `Interval + Interval` | native (within same subclass) 🟡 | native | native 🟡 | DF splits intervals into three Arrow subclasses — cross-subclass addition may require promotion to `IntervalMonthDayNano`. |
| `Date - Date` | `Interval` 🟡 | `Integer` (days) 🟡 | `Integer` (days) 🟡 | ⚠️ **Divergent return type.** See `TD-FUNCS-MAPPING-DATE-SUB-DATE`. |

Authors SHOULD declare `data_type:` explicitly on Semantics consuming interval arithmetic results.

---

## 6. Logical Operators

`BinaryOp` variants `And` / `Or` plus the unary / predicate variants `Not`, `IsNull`, `IsNotNull` from `14 §3.3`. Return type always `Boolean`.

| Canonical | IR Repr | DataFusion | Tier | DuckDB | Tier | Spark | Tier | Notes |
|---|---|---|---|---|---|---|---|---|
| `and` | `BinaryOp::And` | `a AND b` | Name-only | `a AND b` | Name-only | `a AND b` | Name-only | Universal. |
| `or` | `BinaryOp::Or` | `a OR b` | Name-only | `a OR b` | Name-only | `a OR b` | Name-only | Universal. |
| `not` | `Expr::Not` | `NOT expr` | Name-only | `NOT expr` | Name-only | `NOT expr` | Name-only | Universal. |
| `is_null` | `Expr::IsNull` | `expr IS NULL` | Name-only | `expr IS NULL` | Name-only | `expr IS NULL` | Name-only | Universal. Rendered by Dialect-layer dedicated path. |
| `is_not_null` | `Expr::IsNotNull` | `expr IS NOT NULL` | Name-only | `expr IS NOT NULL` | Name-only | `expr IS NOT NULL` | Name-only | Universal. Rendered by Dialect-layer dedicated path. |

Portability summary: **Universal** across all five.

---

## Testing Strategy

BinaryOp behavioral gates:

- **Comparison operators universal**: All six (=, <>, <, <=, >, >=) emit correctly on all engines.
- **Arithmetic per-engine result type**: Integer÷Integer → DataFusion=Integer (truncating), DuckDB/Spark=Double (promoting). Decimal÷Decimal → DataFusion/Spark use Spark formula, DuckDB diverges. Date-Date → DataFusion=Interval, DuckDB/Spark=Integer.
- **SafeDivide structural rewrite**: `SafeDivide` → `a / NULLIF(b, 0)` on all engines (universal structural).
- **mod → % operator**: `Mod` BinaryOpKind → `%` infix operator (universal Name-only).
- **Logical operators universal**: AND, OR, NOT, IS NULL, IS NOT NULL emit correctly on all engines.
- **Interval arithmetic**: Date+Interval, Timestamp+Interval, Interval+Interval native on all engines (per types_mapping.md).

See [../../testing/50_behavioral_contract_gates.md](../../testing/50_behavioral_contract_gates.md) for the shared behavioral harness.
