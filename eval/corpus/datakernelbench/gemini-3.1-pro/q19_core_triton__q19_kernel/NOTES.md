# q19_core_triton__q19_kernel

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/triton/core/queries/q19/round_02.py`, copied as `plan.py`: `_q19_kernel` in `_KERNEL_SOURCE`, lines 14-68 of the plan file.

## Extraction edits

- `def _q19_kernel(` is named `def kernel(`.
- The module states `TYPES`, `CONSTANTS`, and `NUM_WARPS`; the imports `triton` and `triton.language as tl` are the plan's.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`_query_core` launches `_K._q19_kernel[grid](..., BLOCK_SIZE=4096, num_warps=8)` with `grid = (triton.cdiv(n, 4096),)`, when `n > 0`.

The launch passes `BLOCK_SIZE = 4096` and `num_warps = 8`.

`triton.json` states the grid maxima (524288, 1, 1): `2^31 / BLOCK_SIZE` programs in x, where the 32-bit offset `pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` wraps, and 1 in y and z, which the kernel ignores.

## Lengths and ranges

- `l_quantity_ptr`, `l_extendedprice_ptr`, `l_discount_ptr`, `p_brand_id_ptr`, `p_container_grp_ptr`, `p_size_ptr`: columns of the join of the filtered `lineitem` with the filtered `part`, `n_elements = l_quantity.numel()` rows each, at least 1 at the launch (i32 at scale factor 10).
- `out_ptr`: `torch.zeros(1)`, zeroed and accumulated with `tl.atomic_add`: readwrite.
- The brand and container codes are compared only.
