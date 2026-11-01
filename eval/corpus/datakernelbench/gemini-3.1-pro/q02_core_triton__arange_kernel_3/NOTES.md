# q02_core_triton__arange_kernel_3

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/triton/core/queries/q02/round_01.py`, copied as `plan.py`: `_arange_kernel` in `_KERNEL_SOURCE`, lines 22-26 of the plan file.

## Extraction edits

- `def _arange_kernel(` is named `def kernel(`.
- The `@triton.autotune` decorator (lines 13-20) is replaced by the values of configuration 3 of its list: `CONSTANTS` and `NUM_WARPS`.
- The module states `TYPES`, `CONSTANTS`, and `NUM_WARPS`; the imports `triton` and `triton.language as tl` are the plan's.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`_query_core` launches `_K._arange_kernel[grid](idx, k)` with `grid = (triton.cdiv(k, BLOCK_SIZE),)`, when `k > 0`.

This corpus kernel is the configuration `BLOCK_SIZE = 2048`, `num_warps = 4`; the configuration leaves `num_stages` at the default.

`triton.json` states the grid maxima (1048576, 1, 1): `2^31 / BLOCK_SIZE` programs in x, where the 32-bit offset `pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` wraps, and 1 in y and z, which the kernel ignores.

## Lengths and ranges

- `out_ptr`: `torch.empty(k)` with `k = min(limit, n)`, the rows to return, written by the kernel: write; `n_elements = k` is its length (size), at least 1 at the launch.
