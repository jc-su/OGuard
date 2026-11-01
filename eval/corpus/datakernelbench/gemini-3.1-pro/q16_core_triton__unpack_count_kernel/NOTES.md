# q16_core_triton__unpack_count_kernel

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/triton/core/queries/q16/round_02.py`, copied as `plan.py`: `_unpack_count_kernel` in `_KERNEL_SOURCE`, lines 52-65 of the plan file.

## Extraction edits

- `def _unpack_count_kernel(` is named `def kernel(`.
- The `@triton.autotune` decorator (lines 41-50) is replaced by the values of configuration 1 of its list: `CONSTANTS` and `NUM_WARPS`.
- The module states `TYPES`, `CONSTANTS`, and `NUM_WARPS`; the imports `triton` and `triton.language as tl` are the plan's.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`_query_core` launches `_K._unpack_count_kernel[grid_count](...)` with `grid_count = (triton.cdiv(m, BLOCK_SIZE),)`, after `_pack_kernel` and `torch.unique`; it returns before both launches when `n == 0`.

This corpus kernel is the configuration `BLOCK_SIZE = 1024`, `num_warps = 4`; the configuration leaves `num_stages` at the default.

`triton.json` states the grid maxima (2097152, 1, 1): `2^31 / BLOCK_SIZE` programs in x, where the 32-bit offset `pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` wraps, and 1 in y and z, which the kernel ignores.

## Lengths and ranges

- `unique_packed_ptr`: `torch.unique(packed)`, the distinct packed keys, `n_elements = m` rows, at least 1 when the join has a row (i32 at scale factor 10).
- `counts_ptr`: `torch.zeros(num_groups)` with `num_groups = len(group_df)`, the distinct (brand, type, size) groups, zeroed by the plan and accumulated with `tl.atomic_add`: readwrite; `num_groups` is at least 1 when the join has a row.
- The high 32 bits of each packed key are a group id in `[0, num_groups)` by the packing, a relation the signature does not state: `unique_packed_ptr` is `read`.
