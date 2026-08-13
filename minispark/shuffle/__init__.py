"""Shuffle: moving data between partitions so a wide-dependency operator
(group by, eventually join/sort) can see every row for a given key in one
place.

`partitioner.py` decides *which* target partition a record's key belongs
to. `writer.py` and `reader.py` are the disk-backed mechanics of getting
records there and back: `execution/scheduler.py` runs an upstream stage's
tasks, each of which partitions its rows with a `Partitioner` and writes
one block per target partition via `writer.py`, then runs a downstream
stage's tasks, each of which reads every block written for its target
partition via `reader.py`. `manager.py`'s `ShuffleManager` is the
driver-side bookkeeper: it owns the shuffle's scratch directory and knows
which blocks exist for which target partition, so it can hand a
downstream task exactly the block list that task needs to read, without
that task needing to query anything at read time.

Everything here writes to local disk (a temp directory, cleaned up when
the query finishes), per the build spec's constraint that a shuffle
partition must never need to fit in memory as a whole. There is no
cross-machine shuffle: this is real multiprocessing on one machine, not a
distributed shuffle service.
"""
