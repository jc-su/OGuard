# q12_core_triton__q12_kernel_8

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/triton/core/queries/q12/round_01.py`, copied as `plan.py`: `_q12_kernel` in `_KERNEL_SOURCE`, lines 25-67 of the plan file.

## Extraction edits

- `def _q12_kernel(` is named `def kernel(`.
- The `@triton.autotune` decorator (lines 14-23) is replaced by the values of configurations 15 and 16 of its list: `CONSTANTS` and `NUM_WARPS`.
- The module states `TYPES`, `CONSTANTS`, and `NUM_WARPS`; the imports `triton` and `triton.language as tl` are the plan's.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`_query_core` launches `_K._q12_kernel[grid](...)` with `grid = (triton.cdiv(n, BLOCK_SIZE),)`, when `n > 0`.

This corpus kernel is the configuration `BLOCK_SIZE = 8192`, `num_warps = 8`. The autotuner lists this pair with `num_stages` 2 and 4; the corpus module does not state `num_stages`, which sets the pipelining of `tl.range` loops that the kernel does not use, and its PTX is the same for `num_stages` 2, 3 (the front end's default), and 4, so the two configurations are one corpus kernel.

`triton.json` states the grid maxima (262144, 1, 1): `2^31 / BLOCK_SIZE` programs in x, where the 32-bit offset `pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` wraps, and 1 in y and z, which the kernel ignores.

## Lengths and ranges

- `shipmode_ptr`, `priority_ptr`, `commit_ptr`, `receipt_ptr`, `ship_ptr`: the ship mode code (0 or 1), the priority flag (0 or 1), and the three dates of the join of the filtered `lineitem` with `orders`, `n_elements = l_shipmode_code.numel()` rows each, at least 1 at the launch (i32 at scale factor 10).
- `high_counts_ptr`, `low_counts_ptr`: `torch.zeros(2)`, zeroed and accumulated with `tl.atomic_add`: readwrite, 2 elements each.
- The date bounds are any values; the codes are compared only.
