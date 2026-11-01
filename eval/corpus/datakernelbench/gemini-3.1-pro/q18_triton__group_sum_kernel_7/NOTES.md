# q18_triton__group_sum_kernel_7

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/triton/full/queries/q18/round_03.py`, copied as `plan.py`: `_group_sum_kernel` in `_KERNEL_SOURCE`, lines 23-43 of the plan file.

## Extraction edits

- `def _group_sum_kernel(` is named `def kernel(`.
- The `@triton.autotune` decorator (lines 13-22) is replaced by the values of configurations 13 and 14 of its list: `CONSTANTS` and `NUM_WARPS`.
- The module states `TYPES`, `CONSTANTS`, and `NUM_WARPS`; the imports `triton` and `triton.language as tl` are the plan's.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`run_query` launches `_K._group_sum_kernel[grid](...)` with `grid = (triton.cdiv(n, BLOCK_SIZE),)`, when `n > 0`.

This corpus kernel is the configuration `BLOCK_SIZE = 8192`, `num_warps = 4`. The autotuner lists this pair with `num_stages` 2 and 4; the corpus module does not state `num_stages`, which sets the pipelining of `tl.range` loops that the kernel does not use, and its PTX is the same for `num_stages` 2, 3 (the front end's default), and 4, so the two configurations are one corpus kernel.

## Lengths and ranges

- `orderkey_ptr`, `quantity_ptr`: columns of `lineitem`, `n_elements = n` rows each, at least 1 (i32 at scale factor 10).
- `out_ptr`: `torch.zeros(max_key + 1)` with `max_key = l_orderkey.max()`, zeroed and accumulated with `tl.atomic_add`: readwrite.
- `orderkey_ptr`: the plan sizes `out_ptr` by this column's maximum, and the TPC-H keys are positive: `index-in` into `out_ptr`.
