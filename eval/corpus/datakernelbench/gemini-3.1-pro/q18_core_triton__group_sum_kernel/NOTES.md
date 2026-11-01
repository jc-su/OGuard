# q18_core_triton__group_sum_kernel

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/triton/core/queries/q18/round_02.py`, copied as `plan.py`: `_group_sum_kernel` in `_KERNEL_SOURCE`, lines 15-31 of the plan file.

## Extraction edits

- `def _group_sum_kernel(` is named `def kernel(`.
- The module states `TYPES`, `CONSTANTS`, and `NUM_WARPS`; the imports `triton` and `triton.language as tl` are the plan's.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`_query_core` launches `_K._group_sum_kernel[grid](..., BLOCK_SIZE=4096, num_warps=8)` with `grid = (triton.cdiv(n, 4096),)`; it returns before the launch when `n == 0`.

The launch passes `BLOCK_SIZE = 4096` and `num_warps = 8`.

`triton.json` states the grid maxima (524288, 1, 1): `2^31 / BLOCK_SIZE` programs in x, where the 32-bit offset `pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` wraps, and 1 in y and z, which the kernel ignores.

## Lengths and ranges

- `orderkey_ptr`, `quantity_ptr`: the `l_orderkey` and `l_quantity` columns of `lineitem`, `n_elements = l_orderkey.numel()` rows each, at least 1 at the launch (i32 at scale factor 10).
- `sums_ptr`: `torch.zeros(max_key + 1)` with `max_key = int(l_orderkey.max())`, zeroed by the plan and accumulated with `tl.atomic_add`: readwrite.
- `orderkey_ptr`: the plan sizes `sums_ptr` by this column's maximum, and the TPC-H order keys are positive: `index-in` into `sums_ptr`.
