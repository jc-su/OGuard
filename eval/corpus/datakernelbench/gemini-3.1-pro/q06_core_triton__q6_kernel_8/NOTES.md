# q06_core_triton__q6_kernel_8

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/triton/core/queries/q06/round_01.py`, copied as `plan.py`: `_q6_kernel` in `_KERNEL_SOURCE`, lines 25-56 of the plan file.

## Extraction edits

- `def _q6_kernel(` is named `def kernel(`.
- The `@triton.autotune` decorator (lines 14-23) is replaced by the values of configurations 15 and 16 of its list: `CONSTANTS` and `NUM_WARPS`.
- The module states `TYPES`, `CONSTANTS`, and `NUM_WARPS`; the imports `triton` and `triton.language as tl` are the plan's.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`_query_core` launches `_K._q6_kernel[grid](...)` with `grid = (triton.cdiv(n, BLOCK_SIZE),)`; the plan does not guard the launch, and `lineitem` has rows.

This corpus kernel is the configuration `BLOCK_SIZE = 8192`, `num_warps = 8`. The autotuner lists this pair with `num_stages` 2 and 4; the corpus module does not state `num_stages`, which sets the pipelining of `tl.range` loops that the kernel does not use, and its PTX is the same for `num_stages` 2, 3 (the front end's default), and 4, so the two configurations are one corpus kernel.

`triton.json` states the grid maxima (262144, 1, 1): `2^31 / BLOCK_SIZE` programs in x, where the 32-bit offset `pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` wraps, and 1 in y and z, which the kernel ignores.

## Lengths and ranges

- `price_ptr`, `discount_ptr`, `quantity_ptr`, `shipdate_ptr`: columns of `lineitem`, `n_elements = l_extendedprice.numel()` rows each, at least 1 (i32 at scale factor 10).
- `thresh_ptr`: `torch.tensor([discount_lo, discount_hi, quantity_hi])`, 3 elements.
- `out_ptr`: `torch.zeros(1)`, zeroed and accumulated with `tl.atomic_add`: readwrite.
- The date bounds are any values.
