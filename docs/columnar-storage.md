# Columnar Storage

How Parquet reading/writing and scan pushdown work.
See `docs/architecture.md`'s "Columnar storage (Parquet)" design
decisions for the reasoning behind the choices below; this document is
the "what actually
happens" companion, the way `docs/shuffle.md` is for shuffle and
`docs/execution-model.md` is for the DAG/Stage/Task machinery.

## Scope: columnar storage, row-based engine

Every physical operator (`ScanExec`, `FilterExec`, `ProjectExec`,
`HashAggregateExec`, `HashJoinExec`, `SortExec`) still operates on
`Record = dict[str, Any]`, one row at a time. What is genuinely new and
genuinely columnar is
confined to `storage/parquet.py`: pyarrow reads a Parquet file with real
column pruning (an unrequested column's data pages are never decoded)
and real, row-group-level predicate pushdown (a row group whose own
min/max statistics prove it cannot match a pushed filter is skipped
without reading its data). A Parquet-backed partition's `records_fn`
converts the matching rows to plain `Record` dicts
(`RecordBatch.to_pylist()`) right at the partition boundary; nothing
downstream of that point knows or cares that the data came from a
columnar source. Vectorized execution (Filter/Project operating on Arrow
batches without ever materializing Records) is explicit future work, not
attempted here.

## Reading: row-group-granular partitioning

`ParquetDataSource.read()` opens the target (a single `.parquet` file or
a directory of them, whatever `pyarrow.dataset.dataset()` accepts) and
enumerates every row group across every file via `pyarrow.dataset`'s
fragment API (`get_fragments()`, then `split_by_row_group()` on each).
Row groups, not rows, are the unit assigned to `num_partitions` buckets,
contiguous chunking, matching CSV's row-range chunking's spirit but at
the file format's own natural physical unit. Each partition's
`records_fn` is `functools.partial(_read_fragments, its_own_row_groups,
columns, pa_filter)`: a plain module-level function, picklable, bound to
plain, already-picklable arguments (verified directly against pyarrow
20, including a round trip through a real `ProcessPoolExecutor`, before
relying on it: both `ParquetFileFragment` and `pyarrow.dataset.
Expression` pickle and unpickle correctly, with no need to reconstruct
either from a `(path, row_group_index)` pair inside the worker the way a
plain file path is re-opened for CSV).

`PartitionMetadata.row_count` reflects the *pre-filter* row count (summed
from each assigned row group's own footer-metadata `num_rows`, cheap,
no data read), matching every other source's convention (CSV's
`row_count` is also pre-filter, since CSV does not filter at the source
at all). The actual, post-filter row count is only known once a
partition is actually read.

## Real column pruning and predicate pushdown, and why they stop being
"real" past the row-level operators that always remain

`columns`/`filter` passed to `ParquetDataSource.read()` go straight to
`fragment.to_table(columns=, filter=)` per fragment; the column
decoding and row-group-statistics-based skipping are pyarrow's own,
already-tested behavior, not reimplemented here. What this module adds
is `translate_predicate()`: a best-effort translator from a MiniSpark
`Expression` tree into a `pyarrow.dataset.Expression`.

Translation rules, and why each one is the way it is:

* **Comparisons** (`>`, `>=`, `<`, `<=`, `==`, `!=`) translate when the
  left operand is a `Column` (`pyarrow.dataset.field(name)`) and the
  right is either another `Column` or a non-`None` `Literal`. The
  concrete operator classes' own `.op` (already `operator.gt`/`operator.
  eq`/etc, see `expressions/binary.py`) is reused directly:
  `pyarrow.dataset.Expression` overloads the same dunder methods
  (`__gt__`, `__eq__`, ...) that `operator.gt`/`operator.eq` call, so
  `expr.op(left, right)` produces the right pyarrow expression without a
  second operator-to-symbol mapping.
* **`And`** may push just one side, a safe superset: whichever side does
  not translate (e.g. it involves arithmetic) is simply left out of what
  gets pushed, and the row-level `FilterExec`, which always stays in the
  physical plan regardless of what was pushed underneath it, still
  checks the full original condition. **`Or`** must translate both sides
  or neither: pushing only one side of an "or" could wrongly exclude
  rows the untranslated side would have kept, an over-exclusion a
  superset-only optimization must never produce.
* **`IsNull`/`IsNotNull`** translate to `field.is_null()`/`field.
  is_valid()` when their operand is a `Column`.
* **A `None`-valued `Literal` is never translated**, not even to a
  pyarrow null scalar. pyarrow's comparison operators implement SQL's
  three-valued NULL logic (`x == null` never matches, even when `x` is
  itself null); MiniSpark's row engine evaluates `==`/`!=` as plain
  Python equality, where `None == None` is `True`. Pushing `col("x") ==
  None` the way pyarrow would evaluate it could wrongly exclude rows the
  row-level Filter would have kept, exactly the over-exclusion mistake
  every other rule here is designed to avoid. Verified directly:
  `tests/unit/test_parquet_predicate_translation.py`'s
  `test_none_literal_comparison_does_not_translate`.
* **Arithmetic** (`Add`/`Subtract`/`Multiply`/`Divide`) inside a
  predicate is never translated: pushing `(a + b) > 5` down is possible
  in principle via `pyarrow.compute` but is not implemented, out of
  scope here. A predicate
  containing arithmetic simply is not pushed (or, inside an `And`, just
  that conjunct is not pushed); it is still evaluated correctly by the
  row-level Filter.

**A related, accepted inconsistency, not a bug fixed here:** a
comparison against a column whose *actual value* is null (not a `None`
literal) makes pyarrow's pushed filter exclude that row, silently. The
row-based engine, reached without pushdown (e.g. a CSV source, which
never pushes filters), would instead raise `TypeError` evaluating the
same comparison (`None > 18` is not orderable in Python), a
pre-existing limitation of row-at-a-time expression evaluation, not
something Parquet pushdown introduces.
Fixing it would mean giving every comparison operator real, general
null-handling semantics, a broader change than "predicate pushdown"
needs; documented here rather than silently left for someone to
rediscover.

**Column pruning is always at least as wide as every row-level operator
downstream needs.** `physical/planner.py`'s scan-pushdown pass (see
below) computes the exact set of columns a `Project`/`Filter` chain
reaching a `Scan` requires and unions in whatever the accumulated filter
expression itself references, before calling `.read()`; a source that
honors `columns` is never asked to drop something a `FilterExec`/
`ProjectExec` still above it will need.

## The scan-pushdown pass (why this belongs to `physical/planner.py`, not the optimizer)

`optimizer/rules.py`'s `ProjectionPruning` and `PredicatePushdown`
already compute, at the *logical plan shape* level and
without touching data, which columns are needed and how far a filter can
safely move toward a `Scan`. What they cannot do is act on that
information: a rule is held to "never touch data," and re-reading a
Parquet file with a narrower column set or a pushed filter is real I/O.
`physical/planner.py`'s `_pushdown_scan_reads()` is where that
information becomes an actual, smaller read, run once per `plan_physical
()` call (not once per recursive translation step, an earlier version of
this function had that bug: calling it inside the same recursive
function used to translate `Aggregate`/`Join`/`Sort`'s children meant a
`Filter`/`Scan` chain sitting under one of those would be re-read a
second, wasted time; fixed by splitting `plan_physical()` into a public
entry point that runs the pass once and an internal `_translate()` that
recurses without repeating it), before any physical translation happens.

The walk that decides what to push:

* At a **`Scan`** with a `source` (a `logical.nodes.ScanSource`, see
  below) and at least one pending hint, re-read it: `scan.source.
  read(columns=, filter=)`, replacing this Scan's original, unpruned
  `Dataset` with a freshly (and, for a source that honors the hints,
  more narrowly) read one, while keeping the same `source_name`/
  `source` reference.
* At a **`Filter`**, its condition is combined (`AND`) with whatever
  filter expression was already accumulated, and its own referenced
  columns are unioned into whatever column set was already required;
  both are carried further down, including through more `Filter`s.
* At a **`Project`**, the columns it actually needs are recomputed fresh
  from its own expressions (`referenced_columns()`, which already walks
  into `Alias`/computed sub-expressions, not just plain `Column`s) and
  carried down as the new column hint. The accumulated filter expression
  is deliberately **not** carried past a `Project`, even a plain-column
  one: a `Filter`'s condition is tied to whichever namespace was valid
  where it was written, and a `Project` is exactly a namespace boundary
  (output names need not match input names). In practice, this means
  filter pushdown to Parquet only ever fires along a chain where a
  `Filter` sits directly on a `Scan` (or on another `Filter` that
  ultimately sits on a `Scan`), never past a `Project`, which is
  exactly the shape `PredicatePushdown`'s own logical rule already
  arranges whenever pushing a filter below a `Project` is namespace-
  safe; a `Filter` that rule could not push that far down is, correctly,
  one this pass does not push to storage either.
* At an **`Aggregate`**, a **`Join`**, or a **`Sort`**, both hints reset
  to nothing before recursing into each child: whatever was needed above
  does not carry meaning inside a fundamentally different computation
  (an aggregate's own group-by/aggregate columns, a join's own key
  columns, ...), but the walk still recurses into every child, so a
  `Scan` nested arbitrarily deep still gets its own, independent
  pushdown opportunity.

The row-level `FilterExec`/`ProjectExec` are **always** built normally on
top of whatever `Scan` results, whether or not pushdown actually
narrowed anything underneath: pushdown here is exclusively an
optimization, never a substitute for the physical plan's own row-level
correctness (see `storage/datasource.py`'s `DataSource.read()`
docstring).

## `ScanSource`: how `Scan` reaches a `DataSource` without `logical/` importing `storage/`

Without pushdown, `Scan` would hold only an already-`.read()` `Dataset`:
the actual read would happen once, eagerly, at DataFrame-construction
time (`session.read.csv(path)` calls `.read()` immediately), long before
the optimizer has computed anything pushdown could use. Making pushdown
real needs `Scan` to be able to trigger a *second*, better-informed read
once those hints are known. But `logical/` never imports `storage/` (a
deliberate boundary: the logical-plan layer only knows the data model a
source produces, not the storage layer's I/O code), and pushdown does
not change that.

The fix is a structural `Protocol`, defined in `logical/nodes.py` itself:

```python
class ScanSource(Protocol):
    def read(self, columns: list[str] | None = None, filter: Expression | None = None) -> Dataset: ...
```

`storage.datasource.DataSource` satisfies this purely by having a
matching `read` method, no inheritance, no registration, and no import
from `logical/` to `storage/` anywhere. `Scan.source: ScanSource | None`
defaults to `None`: a hand-built `Scan(dataset, name)`, exactly the way
tests that predate pushdown already construct one, keeps working
unchanged, and pushdown simply never applies to it.

## Writing

`DataFrame.write.parquet(path)` (`api/writer.py`, `storage/parquet.py`'s
`write_parquet_dataset()`) runs the DataFrame's plan now (via the same
`DataFrame._collect_dataset()` seam `collect()`/`count()`/`checkpoint()`
use) and writes one `.parquet` file per partition
(`part-00000.parquet`, `part-00001.parquet`, ...) under `path`,
mirroring how a real distributed write produces one file per partition
rather than a single file needing a coordinating merge. Unlike a shuffle
block or a checkpoint file, this is not a streaming, row-at-a-time
writer: each partition is fully materialized into one `pyarrow.Table`
(`pyarrow.Table.from_pylist`) before `pyarrow.parquet.write_table()`
writes it, since Parquet's format is not append-a-row-at-a-time the way
a pickled block file is.

## Type mapping

MiniSpark's `core/types.py` has a small, closed set (`INT`, `FLOAT`,
`STRING`, `BOOL`, `NULL`); `storage/parquet.py` maps to/from pyarrow's
equivalents (`pa.int64()`, `pa.float64()`, `pa.string()`, `pa.bool_()`,
`pa.null()`). Reading a column whose Arrow type does not map to one of
these (`date32`, `timestamp`, `decimal128`, nested/list/struct types)
raises a clear `ValueError` naming the unsupported type, rather than
silently coercing or dropping data: there is no representation for those
types in this codebase yet, matching `core/types.py`'s own documented
scope.

## Optional dependency isolation

`pyarrow` is declared as the `columnar` optional extra
(`pip install minispark[columnar]`, see `pyproject.toml`), not a core
dependency: `import minispark.api.session` (to call `.csv()`, say) must
never require it to be installed. `storage/parquet.py` is the only
module that imports pyarrow at all, and `api/session.py`'s
`DataFrameReader.parquet()` and `api/writer.py`'s `DataFrameWriter.
parquet()` both import it lazily, inside the method body, not at module
top, so merely importing those modules never pulls pyarrow in. Every
Parquet-specific test file (`tests/unit/test_parquet_source.py`,
`tests/unit/test_parquet_predicate_translation.py`,
`tests/integration/test_parquet_e2e.py`) starts with `pytest.
importorskip("pyarrow")`, so the core test suite still passes in full in
an environment without the `columnar` extra installed; it is installed
in this project's own development environment, so these tests do run
and are verified here, not merely written and hoped to work.

## What this is not

No vectorized execution past the Parquet read boundary (see "Scope",
above). No date/timestamp/decimal/nested type support. No size- or
row-count-aware row-group-to-partition balancing, contiguous chunking
can still produce skewed partitions from a file with very unevenly
sized row groups. No target-file-size control or small-file coalescing
on write. No arithmetic-expression pushdown. No fine-grained (Spark's
`InSet`, `StartsWith`, etc.) predicate translation beyond comparisons,
boolean connectives, and null checks.
