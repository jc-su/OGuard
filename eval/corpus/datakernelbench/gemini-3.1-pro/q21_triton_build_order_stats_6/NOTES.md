# q21_triton_build_order_stats_6

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/triton/full/queries/q21/round_02.py`, copied as `plan.py`: `build_order_stats` in `_KERNEL_SOURCE`, lines 22-53 of the plan file.

## Extraction edits

- `def build_order_stats(` is named `def kernel(`.
- The `@triton.autotune` decorator (lines 13-21) is replaced by the values of configurations 11 and 12 of its list: `CONSTANTS` and `NUM_WARPS`.
- The module states `TYPES`, `CONSTANTS`, and `NUM_WARPS`; the imports `triton` and `triton.language as tl` are the plan's.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`run_query` launches `_K.build_order_stats[grid](...)` with `grid = (triton.cdiv(num_lineitems, BLOCK_SIZE),)`; the plan does not check `num_lineitems > 0`.

This corpus kernel is the configuration `BLOCK_SIZE = 4096`, `num_warps = 8`. The autotuner lists this pair with `num_stages` 2 and 4; the corpus module does not state `num_stages`, which sets the pipelining of `tl.range` loops that the kernel does not use, and its PTX is the same for `num_stages` 2, 3 (the front end's default), and 4, so the two configurations are one corpus kernel.

## Lengths and ranges

- `l_orderkey`, `l_suppkey`, `l_receiptdate`, `l_commitdate`: columns of `lineitem` as int32, `num_lineitems` rows each (i32 at scale factor 10).
- `order_status_F`: `torch.zeros(max_orderkey + 1)` of uint8 with `max_orderkey = max(l_orderkey.max(), o_orderkey.max())`.
- `o_supp_min`, `o_supp_max`, `o_late_min`, `o_late_max`: `torch.full(max_orderkey + 1)` with 2147483647 or -1, filled by the plan and updated with `tl.atomic_min` and `tl.atomic_max`: readwrite.
- `l_orderkey`: `max_orderkey` is at least this column's maximum, and the TPC-H keys are positive: `index-in` into `order_status_F` (the four statistics have the same length). `l_suppkey` supplies the stored values: `read`.
