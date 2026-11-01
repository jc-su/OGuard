# Index-valued intermediates are a normal query path

Recorded 2026-09-27. The executable audit and GPU replay are in
[`index_intermediates.py`](index_intermediates.py), and the complete
record is
`records/e2_index_intermediates_2026-09-27.json` (`../records/e2_index_intermediates_2026-09-27.json`, not included in this archive).

This experiment asks whether OpGate's range tracking between operators solves a path that
real analytical plans use. It has two parts with deliberately separate claims.

## E2a: all 22 pinned TPC-H implementations

The audit reads DataKernelBench at commit
`43c361d4ed789d1feaffbcee750e5b256590571a`. This is the same independent,
public artifact pinned by the generated-kernel replay. It checks every validated cuDF/TorchPlan for
TPC-H Q1 through Q22 and the corresponding SF10 physical-plan JSON. Every
source and plan file is recorded by its path. Historical result files retain
the metadata captured by their original runs.

For each query, the audit requires a reviewed producer-to-consumer witness in
the source and records both line numbers. Examples include:

- `torch.unique(..., return_inverse=True)` producing group IDs that a later
  `scatter_add_` consumes;
- `argsort` or `searchsorted` producing row indices that later tensor indexing
  consumes;
- a predicate producing a boolean selection vector that a later cuDF or Torch
  selection consumes.

All 22 plans have a witness: 10 group-ID paths, 8 selection-vector paths, and
4 gather-map paths. Twenty-one carry the value in a named Python variable; Q9
materializes its boolean selection during eager evaluation of one indexing
expression.

The independent physical plans contain 94 join, order, or top-N nodes across
21 of 22 queries: 67 hash joins, 9 other joins, 13 orders, and 5 top-N nodes.
Q6's predicate is pushed into its scan in that plan, while its validated source
still materializes and consumes a selection vector. These JSON plans establish
relational shape; they are DuckDB plans and are not presented as libcudf launch
traces.

## E2b: the concrete libcudf representation

On cuDF/pylibcudf 26.8.1 and the RTX A6000, the replay executes five current
GPU API paths:

| Producer | Consumer | Observed index values | Result |
| --- | --- | --- | --- |
| `sorting.sorted_order` | `copying.gather(DONT_CHECK)` | `[1, 2, 0]` | `[1, 2, 3]` |
| `sorting.top_k_order` | `copying.gather(DONT_CHECK)` | the two row IDs for tied maxima | `[4, 4]` |
| `stream_compaction.distinct_indices` | `copying.gather(DONT_CHECK)` | `[0, 1, 3, 4]` | `[4, 1, 2, 3]` |
| `join.inner_join` | two `copying.gather(DONT_CHECK)` calls | left and right join maps | matching `[1, 3]` columns |
| a cuDF predicate | `stream_compaction.apply_boolean_mask` | `[true, false, true]` | `[3, 2]` |

Every returned map satisfied the source table's row range and occupied a
device buffer distinct from its source. The four gather consumers were invoked
with libcudf's `DONT_CHECK` policy. Thus a producer's value range is a real
cross-operation safety fact: corrupt indices can make the consumer address
outside its input even when the consumer's own address calculation is
otherwise correct.

## Claim and limit

this experiment supports the claim that selection vectors, gather maps, join maps, and
group IDs are ordinary dataflow in current TPC-H GPU implementations, and that
current libcudf exposes concrete device index columns that later operations
consume without a bounds check. This motivates OpGate's index-output and
index-input roles and the catalog status carried between launches.

This experiment is a commit-pinned source audit plus representative API replay. It is not a
launch-level trace of all 22 queries, and it does not measure the catalog or
validation-kernel cost; those remain RQ3/RQ4 experiments after the executor and
catalog exist.

Run it after checking out the pinned DataKernelBench artifact:

```console
.venv/bin/python eval/motivation/index_intermediates.py all \
  --upstream /path/to/datakernelbench \
  --output eval/records/e2_index_intermediates_2026-09-27.json
```

