# Distributed readiness

What would have to change for MiniSpark's workers to run on separate
machines instead of separate local processes, and what already would
not. This is an analysis, not an implementation: nothing in this
document adds networking, RPC, or any new configuration surface, and
this project forbids claiming distributed execution when the system is
only multiprocessing. `EngineConfig.master` (`config/config.py`) still
only parses `"local"` / `"local[N]"` and raises `ValueError` for
anything else after reading this document; that has not changed.

The project's stated goal is narrower than "add distributed execution
later": the same architecture should extend to multiple machines later
without a rewrite. This document is the evidence for whether that goal
is actually true today, checked against the real code, not asserted
from the design alone.

## Already transport-agnostic

These pieces would not need to change in shape if a worker became a
separate machine instead of a separate local process; they were already
built around the constraint that a worker cannot share memory with the
driver.

**`Task`, `TaskResult`, and every `PhysicalPlan` node are already
picklable, and this is exercised for real today, not just asserted.**
`execution/tasks.py`'s `Task` carries a whole `PhysicalPlan` (shared
across every task in a stage) plus `partition_id`, `shuffle_blocks`
(keyed by source `stage_id`), and enough to run independently of
anything else the driver holds. `local[N]` with `N > 1` already sends a
`Task` across a real OS process boundary via `pickle` (Windows `spawn`,
not `fork`, re-imports the whole package fresh in every worker, an even
stricter test of this than `fork` would be, see `docs/benchmarks.md`'s
"local[1] vs local[N]" section). Sending a `Task` to a different machine
over a socket instead of a different local process via
`ProcessPoolExecutor` is a change in *transport*, not in what gets
serialized: the payload shape is already exactly what it would need to
be.

`TaskResult` carries a failure as a formatted string (`error: str |
None`), never as an exception object: exception instances are not
reliably picklable/reconstructable (they can hold unpicklable state,
e.g. a file handle), a message string always round-trips. This was
already the right design for crossing a process boundary and remains
the right design for crossing a network boundary; nothing here assumes
the sender and receiver share a process or a machine.

**`execute_task` is already a plain, importable module-level function,
not a method on a stateful object**, specifically so it stays picklable
and callable from anywhere it gets sent (`execution/worker.py`'s own
docstring names this directly as the seam that lets the worker API later
become a remote process). A remote worker process, whatever transport
delivers the `Task` to it, would still call this exact same function.

**`LocalScheduler` already dispatches tasks through one injectable seam,
`run_task: RunTaskFn | None`** (`execution/scheduler.py`), defaulting to
`execute_task` but replaceable, currently used by tests to avoid real
subprocess cost or to inject deterministic failures, not by anything
that talks over a network. This is the one existing extension point
close to where a remote dispatch mechanism would need to attach, though
see "Not already there" below for what it does not cover.

**The shuffle block format is already a self-contained, checksummed
unit, not a bare file handle.** `ShuffleBlockMeta` (`shuffle/writer.py`)
carries `stage_id`, `source_task_id`, `target_partition`, `record_count`,
`byte_length`, and an MD5 checksum of the block's own bytes, verified on
every read (`shuffle/reader.py`) and raising a distinct
`ShuffleChecksumError` on mismatch. A block being fetched over a network
instead of opened from local disk would still want exactly this: a
self-describing unit whose integrity is checked independently of how it
arrived, not a bare byte stream trusted by construction.

**Lineage-based recovery is already indifferent to *why* data went
missing, only *that* it did.** `_try_recover_missing_shuffle`
(`execution/scheduler.py`) triggers on `TaskResult.missing_shuffle_
stage_id`, set whenever `_execute_shuffle_read_partition` catches either
`FileNotFoundError` or `ShuffleChecksumError` and raises
`MissingShuffleDataError`. The recovery itself is pure `stage_id -> look
up the stage -> re-run its tasks from scratch`, nothing in it inspects
what failed or where. A network fetch failure or an unreachable worker
would need to become a third way to raise `MissingShuffleDataError` (see
"Not already there" below), but once raised, the existing recovery path
handles it exactly as it handles a locally deleted file today: no
new recovery logic, only a new trigger.

## Not already there

These are the parts that currently assume one machine, a shared local
filesystem, and a set of processes the driver already knows how to
reach directly.

**Task dispatch is `ProcessPoolExecutor`, called directly inline, not
behind an abstraction.** `LocalScheduler._run_batch` (`execution/
scheduler.py`) does `with ProcessPoolExecutor(max_workers=self.
num_workers) as pool: pool.map(self._run_task, ...)` when `num_workers >
1`. The injectable `run_task` seam controls *what runs*, not *how it is
sent to a worker*: there is no `WorkerPool`/`ClusterClient`/similar
interface a remote implementation could satisfy alongside the local one,
because there is currently only one thing that needs to satisfy
anything. Introducing that interface now, with no second implementation
to prove it right, would be exactly the kind of speculative abstraction
this project avoids elsewhere (see `optimizer/statistics.py`'s
statistics computed but not yet consulted by any rule, or `config/
config.py` deferring YAML config loading until a real consumer needs
it): the honest state is that this seam does not exist yet, not that it
exists in a form nothing uses.

**Every scratch directory assumes a shared local filesystem all workers
can read directly.** The shuffle scratch directory (`ShuffleManager.
root_dir`, `shuffle/manager.py`, one `tempfile.mkdtemp()`), spill files
(`physical/spill.py`'s `make_spill_dir`, explicitly documented as
"same-process, same-machine scratch"), and shuffle block reads
themselves (`shuffle/reader.py`'s `open(block.path, "rb")`) all assume
whichever process needs a file can open it directly from local disk.
Real multi-machine execution needs each worker's shuffle output to be
*fetched* by whichever worker consumes it, not opened from a path that
happens to be meaningless off the machine that wrote it. The existing
checksum-verified block format (above) is the right unit to fetch; the
fetch mechanism itself does not exist.

**There is no worker identity anywhere in the codebase.** No
`hostname`, `worker_id`, `address`, or `endpoint` concept exists: a task
result comes back from `ProcessPoolExecutor.map` with no record of which
OS process ran it, let alone which machine. A driver that needs to route
a task to a specific remote worker, or know which workers are currently
reachable, has nothing to build on yet; this is genuinely new surface,
not an existing seam waiting to be widened.

**Pickle is used trusting that every process unpickling MiniSpark data
is a MiniSpark process on the same codebase, willing to unpickle
MiniSpark data** (`shuffle/writer.py`'s own comment makes this trust
boundary explicit). That is a reasonable assumption for a
`ProcessPoolExecutor` on one machine, spawned by this exact process from
this exact codebase. It stops being reasonable the moment a "worker" is
a separate machine reachable over a network: unpickling arbitrary bytes
received over a socket, from a process that is not provably this same
trusted deployment, is a known unsafe pattern (arbitrary code execution
during deserialization), not a hypothetical one. A real remote-worker
design would need either a safer wire format for anything crossing an
actual network boundary, or an authentication/allowlist scheme strong
enough to make "every sender is trusted" true again. Nothing here
proposes which; both are real, non-trivial design decisions this
document is not making.

**`_try_recover_missing_shuffle` recomputes a whole stage, with no
notion of a partially-available machine.** It already does not care
whether one block or every block for a stage went missing (see "Already
transport-agnostic," above); it also has no way to know that, say, nine
of ten remote workers are still reachable and only one block truly needs
recomputing versus a scenario where recomputing is pointless because the
whole cluster is unreachable. Coarse-grained, stage-level recompute
already exists (`docs/execution-model.md`'s "Lineage-based
recomputation") and generalizes to "some remote worker's output became
unreachable" without new *recovery* logic, but a real deployment would
still want a way to distinguish "recompute is worth attempting" from
"nothing is reachable, fail fast," which nothing here provides.

## What a real distributed version would add, mapped to what exists

| Concern | Exists today | Would need to be added |
|---|---|---|
| Task wire format | `Task`/`TaskResult`, already picklable | A transport (gRPC/HTTP/sockets are the allowed choices) to move the same payload between machines instead of processes |
| Task dispatch | `run_task` seam, `ProcessPoolExecutor` inline | A dispatch interface with more than one implementation, satisfied by both the existing local path and a new remote one |
| Shuffle data movement | Checksummed, self-describing block format | A fetch-over-network read path, keeping the same checksum verification, replacing direct local `open()` |
| Worker addressing | Nothing | Worker identity, discovery, and routing: real new surface |
| Failure detection | `MissingShuffleDataError` from a local file read failing | A network/unreachable-worker failure also raising it, so the existing stage-level recovery applies unchanged |
| Serialization trust | Pickle, trusted same-machine/same-codebase processes | A safer wire format or an authentication scheme for anything crossing an actual network boundary |
| Configuration | `EngineConfig.master` parses only `"local"` / `"local[N]"` | A cluster/remote master syntax, deliberately not added here |

## What this is not

Not an implementation, not a partial implementation, not a "readiness"
interface layer added speculatively ahead of a second, real consumer.
`EngineConfig.master` still raises on anything but `"local[N]"`;
`LocalScheduler` still only dispatches through `ProcessPoolExecutor` or
sequentially in-process; nothing anywhere opens a socket or makes a
network call. This document is the map of which parts of the existing
design already satisfy the "extend without a rewrite" goal (a real,
checked claim, not an aspiration) and which parts are genuinely unbuilt,
so that whoever eventually builds remote-worker support starts from an
accurate accounting instead of rediscovering the same seams from
scratch.
