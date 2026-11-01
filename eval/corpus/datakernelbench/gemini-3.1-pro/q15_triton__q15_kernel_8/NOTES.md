# q15_triton__q15_kernel_8

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/triton/full/queries/q15/round_01.py`, copied as `plan.py`: `_q15_kernel` in `_KERNEL_SOURCE`, lines 24-58 of the plan file.

## Extraction edits

- `def _q15_kernel(` is named `def kernel(`.
- The `@triton.autotune` decorator (lines 14-23) is replaced by the values of configurations 15 and 16 of its list: `CONSTANTS` and `NUM_WARPS`.
- The module states `TYPES`, `CONSTANTS`, and `NUM_WARPS`; the imports `triton` and `triton.language as tl` are the plan's.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`_query_core` launches `_K._q15_kernel[grid](...)` with `grid = (triton.cdiv(n, BLOCK_SIZE),)`; it returns before the launch when `n == 0` or `supplier` is empty.

This corpus kernel is the configuration `BLOCK_SIZE = 8192`, `num_warps = 8`. The autotuner lists this pair with `num_stages` 2 and 4; the corpus module does not state `num_stages`, which sets the pipelining of `tl.range` loops that the kernel does not use, and its PTX is the same for `num_stages` 2, 3 (the front end's default), and 4, so the two configurations are one corpus kernel.

## Lengths and ranges

- `l_suppkey_ptr`, `l_extendedprice_ptr`, `l_discount_ptr`, `l_shipdate_ptr`: columns of `lineitem`, `n_elements = l_suppkey.numel()` rows each, at least 1 (i32 at scale factor 10).
- `revenue_ptr`, `present_ptr`: `torch.zeros(max_suppkey + 1)` with `max_suppkey = s_suppkey.max()` of `supplier`, zeroed and updated with `tl.atomic_add` and `tl.atomic_max`: readwrite.
- `l_suppkey_ptr` refers to `supplier` by a foreign key; the plan sizes the outputs by `supplier`, not by this column: `read`.
