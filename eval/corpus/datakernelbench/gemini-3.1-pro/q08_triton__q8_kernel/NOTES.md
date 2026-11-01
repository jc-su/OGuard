# q08_triton__q8_kernel

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/triton/full/queries/q08/round_03.py`, copied as `plan.py`: `_q8_kernel` in `_KERNEL_SOURCE`, lines 14-37 of the plan file.

## Extraction edits

- `def _q8_kernel(` is named `def kernel(`.
- The module states `TYPES`, `CONSTANTS`, and `NUM_WARPS`; the imports `triton` and `triton.language as tl` are the plan's.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`run_query` launches `_K._q8_kernel[grid](..., BLOCK_SIZE=4096)` with `grid = (triton.cdiv(n_elements, 4096),)` and the default `num_warps` of 4; it returns before the launch when the join is empty.

The launch passes `BLOCK_SIZE = 4096` and `num_warps = 4`.

## Lengths and ranges

- `year_ptr`, `price_ptr`, `discount_ptr`, `is_target_ptr`: columns of the joined rows, `n_elements = year_t.numel()` rows each, at least 1 (i32 at scale factor 10).
- `sum_target_ptr`, `sum_total_ptr`: `torch.zeros(2500)`, zeroed and accumulated with `tl.atomic_add`: readwrite, 2500 elements.
- `year_ptr`: the year of `o_orderdate`, which the plan filters to 1995-01-01 .. 1996-12-31, so each value is 1995 or 1996: `index-in` into `sum_total_ptr`.
