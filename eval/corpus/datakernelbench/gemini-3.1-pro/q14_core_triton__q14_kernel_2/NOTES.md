# q14_core_triton__q14_kernel_2

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/triton/core/queries/q14/round_05.py`, copied as `plan.py`: `_q14_kernel` in `_KERNEL_SOURCE`, lines 25-57 of the plan file.

## Extraction edits

- `def _q14_kernel(` is named `def kernel(`.
- The `@triton.autotune` decorator (lines 14-23) is replaced by the values of configuration 2 of its list: `CONSTANTS` and `NUM_WARPS`.
- The module states `TYPES`, `CONSTANTS`, and `NUM_WARPS`; the imports `triton` and `triton.language as tl` are the plan's.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`_query_core` launches `_K._q14_kernel[grid](...)` with `grid = (triton.cdiv(n, BLOCK_SIZE),)`, when `n > 0`.

This corpus kernel is the configuration `BLOCK_SIZE = 2048`, `num_warps = 4`, which the autotuner lists with `num_stages = 3`; the corpus module does not state `num_stages`, which sets the pipelining of `tl.range` loops that the kernel does not use, and its PTX is the same for `num_stages` 3 and 3 (the front end's default).

`triton.json` states the grid maxima (1048576, 1, 1): `2^31 / BLOCK_SIZE` programs in x, where the 32-bit offset `pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` wraps, and 1 in y and z, which the kernel ignores.

## Lengths and ranges

- `price_ptr`, `discount_ptr`, `shipdate_ptr`, `promo_ptr`: columns of the join of `lineitem` with `part`, `n_elements = l_extendedprice.numel()` rows each, at least 1 at the launch (i32 at scale factor 10); `promo_ptr` is a bool tensor.
- `out_ptr`: `torch.zeros(2)`, zeroed and accumulated with `tl.atomic_add`: readwrite, 2 elements.
- The date bounds are any values.
