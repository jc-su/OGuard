# q09_core_triton__q9_kernel

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/triton/core/queries/q09/round_02.py`, copied as `plan.py`: `_q9_kernel` in `_KERNEL_SOURCE`, lines 14-28 of the plan file.

## Extraction edits

- `def _q9_kernel(` is named `def kernel(`.
- The module states `TYPES`, `CONSTANTS`, and `NUM_WARPS`; the imports `triton` and `triton.language as tl` are the plan's.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`_query_core` launches `_K._q9_kernel[grid](..., BLOCK_SIZE=4096, num_warps=8)` with `grid = (triton.cdiv(n, 4096),)`; it returns before the launch when `n == 0`.

The launch passes `BLOCK_SIZE = 4096` and `num_warps = 8`.

`triton.json` states the grid maxima (524288, 1, 1): `2^31 / BLOCK_SIZE` programs in x, where the 32-bit offset `pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` wraps, and 1 in y and z, which the kernel ignores.

## Lengths and ranges

- `price_ptr`, `discount_ptr`, `cost_ptr`, `qty_ptr`: columns of the joined frame, `n_elements = l_extendedprice.numel()` rows each, at least 1 at the launch (i32 at scale factor 10).
- `out_ptr`: `torch.empty_like(l_extendedprice)`, written by the kernel: write, `n_elements` elements.
