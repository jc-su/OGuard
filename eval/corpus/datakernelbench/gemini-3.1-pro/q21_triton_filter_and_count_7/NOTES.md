# q21_triton_filter_and_count_7

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/triton/full/queries/q21/round_02.py`, copied as `plan.py`: `filter_and_count` in `_KERNEL_SOURCE`, lines 66-108 of the plan file.

## Extraction edits

- `def filter_and_count(` is named `def kernel(`.
- The `@triton.autotune` decorator (lines 56-65) is replaced by the values of configurations 13 and 14 of its list: `CONSTANTS` and `NUM_WARPS`.
- The module states `TYPES`, `CONSTANTS`, and `NUM_WARPS`; the imports `triton` and `triton.language as tl` are the plan's.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`run_query` launches `_K.filter_and_count[grid](...)` with `grid = (triton.cdiv(num_lineitems, BLOCK_SIZE),)`, after `build_order_stats`.

This corpus kernel is the configuration `BLOCK_SIZE = 8192`, `num_warps = 4`. The autotuner lists this pair with `num_stages` 2 and 4; the corpus module does not state `num_stages`, which sets the pipelining of `tl.range` loops that the kernel does not use, and its PTX is the same for `num_stages` 2, 3 (the front end's default), and 4, so the two configurations are one corpus kernel.

## Lengths and ranges

- `l_orderkey`, `l_suppkey`, `l_receiptdate`, `l_commitdate`: columns of `lineitem` as int32, `num_lineitems` rows each (i32 at scale factor 10).
- The four statistics and `order_status_F`: `max_orderkey + 1` elements with `max_orderkey = max(l_orderkey.max(), o_orderkey.max())`.
- `target_supp_map`: `torch.full((max_suppkey + 1,), -1)` with `max_suppkey = max(l_suppkey.max(), s_suppkey.max())`, holding -1 or a dense index in `[0, num_supp)`.
- `counts`: `torch.zeros(len(supp_n))`, `num_supp = len(supp_n)` at least 1 (the plan returns earlier when there are none), zeroed and accumulated with `tl.atomic_add`: readwrite.
- `l_orderkey` and `l_suppkey`: `max_orderkey` and `max_suppkey` are at least these columns' maxima, and the TPC-H keys are positive: `index-in` into `order_status_F` and `target_supp_map`. `target_supp_map` holds -1, so it is not an index buffer: `read`.
