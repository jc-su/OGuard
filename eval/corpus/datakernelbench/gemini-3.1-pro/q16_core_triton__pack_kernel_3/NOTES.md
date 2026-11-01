# q16_core_triton__pack_kernel_3

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/triton/core/queries/q16/round_02.py`, copied as `plan.py`: `_pack_kernel` in `_KERNEL_SOURCE`, lines 25-39 of the plan file.

## Extraction edits

- `def _pack_kernel(` is named `def kernel(`.
- The `@triton.autotune` decorator (lines 15-23) is replaced by the values of configuration 3 of its list: `CONSTANTS` and `NUM_WARPS`.
- The module states `TYPES`, `CONSTANTS`, and `NUM_WARPS`; the imports `triton` and `triton.language as tl` are the plan's.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`_query_core` launches `_K._pack_kernel[grid_pack](...)` with `grid_pack = (triton.cdiv(n, BLOCK_SIZE),)`; it returns before the launch when `n == 0`.

This corpus kernel is the configuration `BLOCK_SIZE = 4096`, `num_warps = 8`; the configuration leaves `num_stages` at the default.

`triton.json` states the grid maxima (524288, 1, 1): `2^31 / BLOCK_SIZE` programs in x, where the 32-bit offset `pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` wraps, and 1 in y and z, which the kernel ignores.

## Lengths and ranges

- `group_ids_ptr`, `supp_ids_ptr`: the group id and `ps_suppkey` of the joined rows, `n_elements = group_ids.numel()` rows each, at least 1 at the launch (i32 at scale factor 10).
- `packed_ptr`: `torch.empty_like(group_ids)`, written by the kernel: write, `n_elements` elements.
