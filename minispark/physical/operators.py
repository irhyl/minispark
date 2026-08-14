"""Physical operator execution: turns a PhysicalPlan into a Dataset.

Deliberately mirrors execution/executor.py's per-node logic almost exactly
for Scan/Filter/Project: there is only one way to execute those, so "the
physical strategy" and "the naive logical interpretation" happen to
compute the same thing. HashAggregateExec, HashJoinExec, and
ShuffleReadExec have no equivalent in execution/executor.py at all (the
naive executor only knows Scan/Filter/Project): grouping and joining
fundamentally need a shuffle to be correct across partitions, which is
exactly what Milestone 3's single-process, single-stage naive executor
cannot do.
"""

from __future__ import annotations

import functools
import heapq
import itertools
import sys
from collections.abc import Iterator

from minispark.core.dataset import Dataset
from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.record import Record
from minispark.expressions.aggregate import AggregateFunction
from minispark.expressions.base import Alias, Expression
from minispark.logical.nodes import output_name
from minispark.physical.plan import (
    FilterExec,
    HashAggregateExec,
    HashJoinExec,
    PhysicalPlan,
    ProjectExec,
    ScanExec,
    ShuffleReadExec,
    SortExec,
)
from minispark.physical.spill import (
    cleanup_spill_dir,
    make_spill_dir,
    read_spill_file,
    write_spill_file,
)
from minispark.shuffle.partitioner import HashPartitioner
from minispark.shuffle.reader import (
    MissingShuffleDataError,
    ShuffleChecksumError,
    read_shuffle_blocks,
)
from minispark.shuffle.writer import ShuffleBlockMeta

# Fixed fan-out for HashAggregateExec's grace-hash spill (see
# `_execute_hash_aggregate_partition`). Not derived from MemoryConfig or
# EngineConfig, same reasoning as `physical/plan.py`'s NEVER_SPILL:
# physical/ does not import config/, and this constant only controls how
# finely a spilled hash table is split, not whether spilling happens at
# all (that is `plan.spill_threshold_bytes`).
_AGGREGATE_SPILL_BUCKETS = 32

# Keyed by source stage_id: a Task whose plan reads from more than one
# prior stage (a HashJoinExec-rooted stage reads from two) needs each
# ShuffleReadExec leaf to find only its own stage's blocks, not the
# other leaf's. See execution/scheduler.py for how this dict is built.
ShuffleBlocksByStage = dict[int, list[ShuffleBlockMeta]]


def execute(plan: PhysicalPlan) -> Dataset:
    if isinstance(plan, ScanExec):
        return plan.dataset
    if isinstance(plan, FilterExec):
        return _execute_filter(plan)
    if isinstance(plan, ProjectExec):
        return _execute_project(plan)
    raise NotImplementedError(
        f"No physical operator implemented for {type(plan).__name__}"
    )


def _execute_filter(plan: FilterExec) -> Dataset:
    child_dataset = execute(plan.child)
    condition = plan.condition

    def filtered_partition(parent: Partition) -> Partition:
        def records_fn() -> Iterator[Record]:
            for record in parent:
                if condition.evaluate(record):
                    yield record

        return Partition(parent.partition_id, parent.schema, records_fn, PartitionMetadata())

    new_partitions = [filtered_partition(p) for p in child_dataset.partitions()]
    return Dataset(child_dataset.schema, new_partitions)


def _execute_project(plan: ProjectExec) -> Dataset:
    child_dataset = execute(plan.child)
    columns = plan.columns
    output_schema = plan.schema
    output_names = output_schema.field_names()

    def projected_partition(parent: Partition) -> Partition:
        def records_fn() -> Iterator[Record]:
            for record in parent:
                yield {
                    name: expr.evaluate(record)
                    for name, expr in zip(output_names, columns, strict=True)
                }

        return Partition(
            parent.partition_id,
            output_schema,
            records_fn,
            PartitionMetadata(row_count=parent.row_count()),
        )

    new_partitions = [projected_partition(p) for p in child_dataset.partitions()]
    return Dataset(output_schema, new_partitions)


def execute_partition(
    plan: PhysicalPlan,
    partition_id: int,
    shuffle_blocks: ShuffleBlocksByStage | None = None,
) -> Partition:
    """Execute physical operators for exactly one partition.

    This, not `execute()`, is what a Task actually runs (see
    execution/tasks.py, execution/worker.py). A Task should ship "run this
    plan for partition i" to a worker, not "run this plan for every
    partition and send all of it back": only one partition's rows need to
    cross the process boundary as a result. `plan` itself is safe to send
    whole (every physical node and expression is plain, picklable data,
    and Milestone 3's fix to storage/memory.py and storage/csv.py made
    Dataset/Partition picklable too); `execute_partition` picks out a
    single source partition at each `ScanExec` leaf instead of reading
    `plan.dataset` in full.

    `shuffle_blocks` is only meaningful when `plan` contains one or more
    `ShuffleReadExec` leaves (a stage that reads a prior stage's shuffle
    output; a `HashJoinExec`-rooted stage can have two, one per side): the
    caller (execution/worker.py) already knows exactly which blocks
    belong to this partition for each upstream stage (filtered driver-side
    by `shuffle/manager.py`'s `ShuffleManager.blocks_for()`) and passes
    that fixed, per-stage mapping in, rather than this function looking
    anything up itself.
    """
    if isinstance(plan, ScanExec):
        return plan.dataset.partition(partition_id)
    if isinstance(plan, FilterExec):
        return _execute_filter_partition(plan, partition_id, shuffle_blocks)
    if isinstance(plan, ProjectExec):
        return _execute_project_partition(plan, partition_id, shuffle_blocks)
    if isinstance(plan, HashAggregateExec):
        return _execute_hash_aggregate_partition(plan, partition_id, shuffle_blocks)
    if isinstance(plan, HashJoinExec):
        return _execute_hash_join_partition(plan, partition_id, shuffle_blocks)
    if isinstance(plan, SortExec):
        return _execute_sort_partition(plan, partition_id, shuffle_blocks)
    if isinstance(plan, ShuffleReadExec):
        return _execute_shuffle_read_partition(plan, partition_id, shuffle_blocks)
    raise NotImplementedError(
        f"No per-partition physical operator implemented for {type(plan).__name__}"
    )


def _execute_filter_partition(
    plan: FilterExec, partition_id: int, shuffle_blocks: ShuffleBlocksByStage | None
) -> Partition:
    parent = execute_partition(plan.child, partition_id, shuffle_blocks)
    condition = plan.condition

    def records_fn() -> Iterator[Record]:
        for record in parent:
            if condition.evaluate(record):
                yield record

    return Partition(parent.partition_id, parent.schema, records_fn, PartitionMetadata())


def _execute_project_partition(
    plan: ProjectExec, partition_id: int, shuffle_blocks: ShuffleBlocksByStage | None
) -> Partition:
    parent = execute_partition(plan.child, partition_id, shuffle_blocks)
    columns = plan.columns
    output_schema = plan.schema
    output_names = output_schema.field_names()

    def records_fn() -> Iterator[Record]:
        for record in parent:
            yield {
                name: expr.evaluate(record)
                for name, expr in zip(output_names, columns, strict=True)
            }

    return Partition(
        parent.partition_id,
        output_schema,
        records_fn,
        PartitionMetadata(row_count=parent.row_count()),
    )


def _unwrap_aggregate(expr: Expression) -> AggregateFunction:
    inner = expr.child if isinstance(expr, Alias) else expr
    assert isinstance(inner, AggregateFunction)
    return inner


def _execute_hash_aggregate_partition(
    plan: HashAggregateExec, partition_id: int, shuffle_blocks: ShuffleBlocksByStage | None
) -> Partition:
    """Group this one partition's rows by `plan.group_by` and combine
    `plan.aggregates` per group.

    Unlike Filter/Project, this cannot stream row-by-row: a group's final
    state is only known once every row for that key (within this
    partition) has been seen, so the `groups` table below is built up
    front, not lazily inside `records_fn`. This is the same reason a real
    hash-based group-by needs a hash table sized to the *distinct key*
    count, not the row count.

    Milestone 9 spilling: `groups_bytes` tracks the current in-memory
    table's estimated size (`_estimate_group_bytes`, incrementally updated
    as each key's state is replaced, not recomputed from scratch per row).
    When it crosses `plan.spill_threshold_bytes`, the whole `groups` table
    is partitioned by `HashPartitioner(_AGGREGATE_SPILL_BUCKETS)` into
    buckets and each non-empty bucket is written as one spill file
    (`_spill_groups`), then `groups` is cleared and accumulation resumes.
    A key spilled once and seen again later simply restarts from
    `initialize()`/incoming state as if it were new; the two partial
    states for that key are reconciled later, in the merge phase, via
    `AggregateFunction.merge()`, exactly the same function that already
    reconciles states from different source partitions after a shuffle.
    This is a grace-hash join's spilling strategy applied to group-by
    instead of join.

    If no spill ever happens, behavior is byte-for-byte what it was
    before this milestone: `groups` is finalized directly, eagerly, and
    `records_fn` just replays it.

    If spilling did happen, the still-in-memory remainder is kept as-is
    (not itself spilled, saving a round-trip) and `records_fn` merges one
    bucket at a time: for each of the `_AGGREGATE_SPILL_BUCKETS` buckets,
    it seeds a small dict from the remainder's matching keys, folds in
    every spill file written for that bucket (across every spill round)
    via `merge()`, finalizes, and yields, before moving to the next
    bucket. This bounds memory during the merge phase too, to one
    bucket's distinct-key set at a time, not the whole partition's.
    `plan.spill_threshold_bytes` only bounds the accumulation phase; the
    number of buckets (not size-based) bounds the merge phase, so a
    single bucket holding a disproportionate share of distinct keys
    (skew) is not itself protected against here, a known, documented gap
    left for a future milestone. See `docs/spilling.md`.
    """
    parent = execute_partition(plan.child, partition_id, shuffle_blocks)
    group_by = plan.group_by
    aggregates = [_unwrap_aggregate(a) for a in plan.aggregates]
    group_names = [output_name(g) for g in group_by]
    threshold = plan.spill_threshold_bytes
    partitioner = HashPartitioner(_AGGREGATE_SPILL_BUCKETS)

    groups: dict[tuple, list] = {}
    groups_bytes = 0
    spill_dir: str | None = None
    spill_paths: list[list[str]] = [[] for _ in range(_AGGREGATE_SPILL_BUCKETS)]
    round_num = 0

    for record in parent:
        key = tuple(g.evaluate(record) for g in group_by)
        existing = groups.get(key)
        if plan.is_partial:
            state = existing if existing is not None else [agg.initialize() for agg in aggregates]
            new_state = [agg.update(s, record) for agg, s in zip(aggregates, state, strict=True)]
        else:
            incoming = [record[f"__agg_state_{i}"] for i in range(len(aggregates))]
            new_state = (
                incoming
                if existing is None
                else [
                    agg.merge(s, inc)
                    for agg, s, inc in zip(aggregates, existing, incoming, strict=True)
                ]
            )
        if existing is not None:
            groups_bytes -= _estimate_group_bytes(key, existing)
        groups_bytes += _estimate_group_bytes(key, new_state)
        groups[key] = new_state

        if groups_bytes >= threshold:
            if spill_dir is None:
                spill_dir = make_spill_dir("minispark-aggregate-spill-")
            _spill_groups(groups, spill_dir, spill_paths, partitioner, round_num)
            round_num += 1
            groups = {}
            groups_bytes = 0

    output_schema = plan.schema
    agg_output_names = [output_name(a) for a in plan.aggregates]

    if spill_dir is None:

        def records_fn() -> Iterator[Record]:
            for key, state in groups.items():
                row: Record = dict(zip(group_names, key, strict=True))
                if plan.is_partial:
                    for i, s in enumerate(state):
                        row[f"__agg_state_{i}"] = s
                else:
                    for name, agg, s in zip(agg_output_names, aggregates, state, strict=True):
                        row[name] = agg.finalize(s)
                yield row

        return Partition(
            partition_id, output_schema, records_fn, PartitionMetadata(row_count=len(groups))
        )

    remainder = groups

    def spilled_records_fn() -> Iterator[Record]:
        try:
            remainder_by_bucket: list[dict[tuple, list]] = [
                {} for _ in range(_AGGREGATE_SPILL_BUCKETS)
            ]
            for key, state in remainder.items():
                remainder_by_bucket[partitioner.partition_for(key)][key] = state

            for bucket in range(_AGGREGATE_SPILL_BUCKETS):
                merged = remainder_by_bucket[bucket]
                for path in spill_paths[bucket]:
                    for key, state in read_spill_file(path):
                        existing = merged.get(key)
                        merged[key] = (
                            state
                            if existing is None
                            else [
                                agg.merge(s, inc)
                                for agg, s, inc in zip(aggregates, existing, state, strict=True)
                            ]
                        )
                for key, state in merged.items():
                    row: Record = dict(zip(group_names, key, strict=True))
                    if plan.is_partial:
                        for i, s in enumerate(state):
                            row[f"__agg_state_{i}"] = s
                    else:
                        for name, agg, s in zip(agg_output_names, aggregates, state, strict=True):
                            row[name] = agg.finalize(s)
                    yield row
        finally:
            cleanup_spill_dir(spill_dir)

    return Partition(partition_id, output_schema, spilled_records_fn, PartitionMetadata())


def _spill_groups(
    groups: dict[tuple, list],
    spill_dir: str,
    spill_paths: list[list[str]],
    partitioner: HashPartitioner,
    round_num: int,
) -> None:
    """Partition `groups` by key hash and write each non-empty bucket as
    one spill file, appending its path to that bucket's entry in
    `spill_paths` (mutated in place). `round_num` only distinguishes file
    names across repeated calls in the same spill directory; nothing
    reads it back out, the merge phase groups by bucket and does not care
    which round a file came from."""
    by_bucket: dict[int, list[tuple[tuple, list]]] = {}
    for key, state in groups.items():
        by_bucket.setdefault(partitioner.partition_for(key), []).append((key, state))
    for bucket, items in by_bucket.items():
        path = write_spill_file(spill_dir, f"round{round_num}_bucket{bucket}.pkl", items)
        spill_paths[bucket].append(path)


def _execute_shuffle_read_partition(
    plan: ShuffleReadExec, partition_id: int, shuffle_blocks: ShuffleBlocksByStage | None
) -> Partition:
    if shuffle_blocks is None or plan.from_stage_id not in shuffle_blocks:
        raise ValueError(
            f"ShuffleReadExec for stage {plan.from_stage_id} partition {partition_id} "
            "was executed with no blocks for that stage; the caller must supply a "
            "shuffle_blocks[stage_id] entry for every ShuffleReadExec in the plan "
            "(see execution/worker.py, execution/scheduler.py)."
        )
    blocks = shuffle_blocks[plan.from_stage_id]
    try:
        records = list(read_shuffle_blocks(blocks))
    except (FileNotFoundError, ShuffleChecksumError) as exc:
        # A block file this partition needs is gone or corrupted: retrying
        # this same read can never succeed, since the data simply is not
        # there any more. Re-raised as a distinct error type so
        # execution/worker.py and execution/scheduler.py can tell "the
        # upstream stage needs to be recomputed" apart from an ordinary,
        # possibly-transient task failure.
        raise MissingShuffleDataError(plan.from_stage_id, partition_id, str(exc)) from exc
    return Partition(
        partition_id,
        plan.schema,
        functools.partial(iter, records),
        PartitionMetadata(row_count=len(records)),
    )


def _execute_hash_join_partition(
    plan: HashJoinExec, partition_id: int, shuffle_blocks: ShuffleBlocksByStage | None
) -> Partition:
    """Build a hash table on `left`'s rows (keyed by `left_keys`), then
    probe it with `right`'s rows (keyed by `right_keys`), for this one
    partition. By the time this runs, `left` and `right` already produce
    exactly the rows this partition's join needs to see, whether that is
    because both sides were shuffle-partitioned by the same key (shuffle
    hash join) or because `right` is a full broadcast copy read
    identically by every partition (broadcast join); see
    physical/planner.py for which case built this node's children.

    Building on `left` (rather than choosing the smaller side) is a fixed
    choice, not a cost-based one: nothing here has a reliable, cheap
    byte-size estimate for either side to choose from (see
    optimizer/statistics.py's documented caveats and logical/nodes.py's
    `Join` docstring on why broadcast side selection is an explicit hint,
    not automatic).
    """
    left_partition = execute_partition(plan.left, partition_id, shuffle_blocks)
    right_partition = execute_partition(plan.right, partition_id, shuffle_blocks)
    left_keys = plan.left_keys
    right_keys = plan.right_keys
    on_set = set(plan.on)

    table: dict[tuple, list[Record]] = {}
    for record in left_partition:
        key = tuple(e.evaluate(record) for e in left_keys)
        table.setdefault(key, []).append(record)

    right_rows = list(right_partition)
    output_schema = plan.schema

    def records_fn() -> Iterator[Record]:
        for right_row in right_rows:
            key = tuple(e.evaluate(right_row) for e in right_keys)
            for left_row in table.get(key, ()):
                merged = dict(left_row)
                for name, value in right_row.items():
                    if name not in on_set:
                        merged[name] = value
                yield merged

    return Partition(partition_id, output_schema, records_fn, PartitionMetadata())


def _execute_sort_partition(
    plan: SortExec, partition_id: int, shuffle_blocks: ShuffleBlocksByStage | None
) -> Partition:
    """Sort this one partition's rows by `plan.sort_exprs`/`plan.ascending`.

    Like HashAggregateExec, this cannot stream in the general case: the
    whole partition must be seen before any row's final position is
    known. Rows accumulate into an in-memory buffer; `_sort_rows()` (used
    both here and to sort each spilled run) achieves multi-key, mixed
    ascending/descending order with repeated stable sorts, last key
    first (Python's `sort` is guaranteed stable, so an earlier pass's
    relative order survives for rows that tie on a later, higher-
    priority key): a single `sorted(key=...)` call sorting on a tuple of
    keys cannot vary direction per key without either negating values
    (which breaks for non-numeric types like strings) or a custom
    comparator (removed from Python 3's `sorted`).

    Nulls always sort last, regardless of ascending/descending: a
    deliberate, documented simplification rather than implementing
    per-column NULLS FIRST/LAST placement.

    Milestone 9 spilling: if the buffer's estimated size (`sys.
    getsizeof`-summed, the same heuristic and "not an exact byte count"
    caveat as `execution/worker.py`'s `_estimate_bytes`) crosses `plan.
    spill_threshold_bytes`, the buffer is sorted with `_sort_rows()` and
    written to a spill file (`physical/spill.py`) as one sorted run, then
    cleared; this can repeat any number of times. If no spill ever
    happens, behavior is byte-for-byte what it was before this milestone
    (sort the one in-memory list, return it eagerly).

    If spilling did happen, the final, still-in-memory remainder is
    itself sorted into one more run, and every run (spilled files plus
    the final in-memory one) is merged with `heapq.merge()`, streaming
    its output lazily rather than materializing it, a real (if
    incidental) benefit of the spilling path over the non-spilling one.
    Every record is tagged with a strictly increasing `seq` as it is
    first consumed from `parent`, carried through buffering/spilling as
    `(seq, record)`, specifically so the merge can break a full tie (on
    every sort key) by original arrival order, exactly matching what a
    single stable, non-spilling sort already does for free: `heapq.
    merge()` only preserves order *within* one already-sorted input and
    resolves cross-input ties by *input position in the runs list*, not
    by any property of the records themselves, so without `seq` as an
    explicit final tie-breaker in `_composite_sort_key()`, two tied rows
    that happened to land in different spill chunks could come out in a
    different relative order than the same query would produce without
    spilling, an internal, invisible-to-the-plan performance knob
    silently changing an observable result. Caught by testing (`tests/
    unit/test_sort_physical_plan.py` compares spilling and non-spilling
    output on the same data directly), not inspection: an earlier version
    without the `seq` tie-breaker passed every test that did not
    specifically construct enough full ties to expose it. See
    `docs/spilling.md`.
    """
    parent = execute_partition(plan.child, partition_id, shuffle_blocks)
    threshold = plan.spill_threshold_bytes
    seq_counter = itertools.count()

    buffer: list[tuple[int, Record]] = []
    buffer_bytes = 0
    total_rows = 0
    spill_dir: str | None = None
    spill_paths: list[str] = []

    for record in parent:
        buffer.append((next(seq_counter), record))
        buffer_bytes += _estimate_record_bytes(record)
        total_rows += 1
        if buffer_bytes >= threshold:
            if spill_dir is None:
                spill_dir = make_spill_dir("minispark-sort-spill-")
            sorted_chunk = _sort_rows(buffer, plan.sort_exprs, plan.ascending)
            spill_paths.append(
                write_spill_file(spill_dir, f"run_{len(spill_paths)}.pkl", sorted_chunk)
            )
            buffer = []
            buffer_bytes = 0

    if not spill_paths:
        sorted_rows = _sort_rows(buffer, plan.sort_exprs, plan.ascending)
        return Partition(
            partition_id,
            plan.schema,
            functools.partial(iter, (record for _, record in sorted_rows)),
            PartitionMetadata(row_count=len(sorted_rows)),
        )

    final_run = _sort_rows(buffer, plan.sort_exprs, plan.ascending)
    key_fn = functools.partial(_composite_sort_key, plan.sort_exprs, plan.ascending)

    def records_fn() -> Iterator[Record]:
        try:
            runs = [iter(final_run), *(read_spill_file(p) for p in spill_paths)]
            for _, record in heapq.merge(*runs, key=key_fn):
                yield record
        finally:
            cleanup_spill_dir(spill_dir)

    return Partition(
        partition_id, plan.schema, records_fn, PartitionMetadata(row_count=total_rows)
    )


def _sort_rows(
    rows: list[tuple[int, Record]], sort_exprs: list[Expression], ascending: list[bool]
) -> list[tuple[int, Record]]:
    """Sort `(seq, record)` pairs (a new list; `rows` itself is sorted in
    place and returned, matching `list.sort()`'s own contract) via
    repeated stable passes, last key first. See `_execute_sort_partition`'s
    docstring for why this technique, not a single composite key, is
    used here, and why `seq` rides along even though it plays no part in
    *this* function's own comparisons (Python's stable sort already
    preserves `rows`' incoming relative order for a full tie, which is
    exactly `seq` order, since `rows` is always built in arrival order)."""
    for expr, asc in reversed(list(zip(sort_exprs, ascending, strict=True))):
        key_fn = functools.partial(_null_last_sort_key, expr, asc)
        rows.sort(key=lambda pair: key_fn(pair[1]), reverse=not asc)
    return rows


def _null_last_sort_key(expr: Expression, ascending: bool, record: Record) -> tuple[bool, object]:
    value = expr.evaluate(record)
    is_null = value is None
    # `rows.sort(..., reverse=not ascending)` flips the *whole* key tuple,
    # including whichever boolean marks a null, not just the value part.
    # Pre-flipping the sentinel here (rather than always using `is_null`)
    # is what keeps nulls sorting last in the final output for a
    # descending column too, not only for an ascending one.
    null_sentinel = is_null if ascending else not is_null
    return (null_sentinel, value if not is_null else 0)


class _Desc:
    """Wraps a value so tuple/heapq comparison treats it as descending,
    without negating the value itself (which breaks for non-numeric
    types like strings, see `_execute_sort_partition`'s docstring).
    Delegates to the wrapped value's own `<`/`==`, just flipped; works
    for any orderable type the wrapped value itself supports."""

    __slots__ = ("value",)

    def __init__(self, value: object):
        self.value = value

    def __lt__(self, other: _Desc) -> bool:
        return other.value < self.value  # type: ignore[operator]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Desc) and self.value == other.value


def _composite_sort_key(
    sort_exprs: list[Expression], ascending: list[bool], pair: tuple[int, Record]
) -> tuple:
    """One comparable key per `(seq, record)` pair capturing the full
    multi-key, mixed-direction, null-last order, plus `seq` as a final
    tie-breaker, in a single tuple, for `heapq.merge()`'s `key=` (which
    needs one monotonic key comparable across every run being merged,
    unlike `_sort_rows()`'s repeated-pass technique, which only needs to
    compare within a single list at a time). Sorting by the sort-key
    portion of this tuple is mathematically equivalent to `_sort_rows()`'s
    "stable-sort last key first" for the same reason lexicographic tuple
    comparison is always equivalent to that technique for multi-key sort;
    appending `seq` last reproduces stable-sort's tie-breaking (original
    arrival order) explicitly, since `heapq.merge()`, unlike a single
    `list.sort()`, has no other way to know which of two equal-keyed rows
    from *different* runs arrived first. `tests/unit/
    test_sort_physical_plan.py` checks this produces orderings identical
    to `_sort_rows()`, including for full ties, directly, not just
    assumes it.
    """
    seq, record = pair
    parts: list[object] = []
    for expr, asc in zip(sort_exprs, ascending, strict=True):
        value = expr.evaluate(record)
        is_null = value is None
        placeholder = 0 if is_null else value
        parts.append(is_null)
        parts.append(placeholder if asc else _Desc(placeholder))
    parts.append(seq)
    return tuple(parts)


def _estimate_record_bytes(record: Record) -> int:
    """Rough, Python-object-overhead-inclusive estimate, not an on-disk
    byte count. Same heuristic and same caveat as `execution/worker.py`'s
    `_estimate_bytes` and `optimizer/statistics.py`'s `compute_statistics`."""
    return sum(sys.getsizeof(v) for v in record.values())


def _estimate_group_bytes(key: tuple, state: list) -> int:
    """Rough estimate of one group's contribution to the in-memory hash
    table's size (key tuple plus aggregate state list), same `sys.
    getsizeof`-summed heuristic as `_estimate_record_bytes`. Shallow, not
    recursive: a state holding a large nested container (no built-in
    aggregate in expressions/aggregate.py does; a hypothetical
    `collect_list` would) is under-counted, same caveat as elsewhere."""
    return sum(sys.getsizeof(v) for v in key) + sum(sys.getsizeof(v) for v in state)
