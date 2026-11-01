# q06_triton__q6_kernel_3

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/triton/full/queries/q06/round_01.py`, copied as `plan.py`: `_q6_kernel` in `_KERNEL_SOURCE`, lines 24-58 of the plan file.

## Extraction edits

- `def _q6_kernel(` is named `def kernel(`.
- The `@triton.autotune` decorator (lines 14-23) is replaced by the values of configurations 5 and 6 of its list: `CONSTANTS` and `NUM_WARPS`.
- The module states `TYPES`, `CONSTANTS`, and `NUM_WARPS`; the imports `triton` and `triton.language as tl` are the plan's.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`run_query` launches `_K._q6_kernel[grid](...)` with `grid = (triton.cdiv(n, BLOCK_SIZE),)`; it returns before the launch when `n == 0`.

This corpus kernel is the configuration `BLOCK_SIZE = 2048`, `num_warps = 4`. The autotuner lists this pair with `num_stages` 2 and 4; the corpus module does not state `num_stages`, which sets the pipelining of `tl.range` loops that the kernel does not use, and its PTX is the same for `num_stages` 2, 3 (the front end's default), and 4, so the two configurations are one corpus kernel.

## Lengths and ranges

- `price_ptr`, `discount_ptr`, `quantity_ptr`, `shipdate_ptr`: columns of `lineitem`, `n_elements = l_extendedprice_t.numel()` rows each, at least 1 (i32 at scale factor 10).
- `thresh_ptr`: `torch.tensor([discount_lo, discount_hi, quantity_hi])`, 3 elements.
- `out_ptr`: `torch.zeros(1)`, zeroed and accumulated with `tl.atomic_add`: readwrite.
- The date bounds are any values.
