# q15_core_triton__q15_kernel_6

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/triton/core/queries/q15/round_02.py`, copied as `plan.py`: `_q15_kernel` in `_KERNEL_SOURCE`, lines 25-60 of the plan file.

## Extraction edits

- `def _q15_kernel(` is named `def kernel(`.
- The `@triton.autotune` decorator (lines 14-23) is replaced by the values of configurations 11 and 12 of its list: `CONSTANTS` and `NUM_WARPS`.
- The module states `TYPES`, `CONSTANTS`, and `NUM_WARPS`; the imports `triton` and `triton.language as tl` are the plan's.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`_query_core` launches `_K._q15_kernel[grid](...)` with `grid = (triton.cdiv(n_elements, BLOCK_SIZE),)`; the plan does not guard the launch, and an empty `lineitem` gives an empty grid.

This corpus kernel is the configuration `BLOCK_SIZE = 4096`, `num_warps = 8`. The autotuner lists this pair with `num_stages` 2 and 4; the corpus module does not state `num_stages`, which sets the pipelining of `tl.range` loops that the kernel does not use, and its PTX is the same for `num_stages` 2, 3 (the front end's default), and 4, so the two configurations are one corpus kernel.

`triton.json` states the grid maxima (524288, 1, 1): `2^31 / BLOCK_SIZE` programs in x, where the 32-bit offset `pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` wraps, and 1 in y and z, which the kernel ignores.

## Lengths and ranges

- `l_suppkey_ptr`, `l_extendedprice_ptr`, `l_discount_ptr`, `l_shipdate_ptr`: columns of `lineitem`, `n_elements = l_suppkey.numel()` rows each, possibly none (i32 at scale factor 10).
- `rev_by_suppkey_ptr`, `counts_by_suppkey_ptr`: `torch.zeros(max_s + 1)` with `max_s = int(s_suppkey.max())` of `supplier` (0 when `supplier` is empty), zeroed and accumulated with `tl.atomic_add`: readwrite; `max_s` is the argument of their length (size).
- `l_suppkey_ptr` is checked against `max_s` by the kernel: `read`.
