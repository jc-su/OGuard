# q16_triton__count_unique_sorted_kernel_5

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/triton/full/queries/q16/round_03.py`, copied as `plan.py`: `_count_unique_sorted_kernel` in `_KERNEL_SOURCE`, lines 24-55 of the plan file.

## Extraction edits

- `def _count_unique_sorted_kernel(` is named `def kernel(`.
- The `@triton.autotune` decorator (lines 15-23) is replaced by the values of configuration 5 of its list: `CONSTANTS` and `NUM_WARPS`.
- The module states `TYPES`, `CONSTANTS`, and `NUM_WARPS`; the imports `triton` and `triton.language as tl` are the plan's.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`_query_core` launches `_K._count_unique_sorted_kernel[grid](...)` with `grid = (triton.cdiv(num_elements, BLOCK_SIZE),)`; it returns before the launch when the join is empty.

This corpus kernel is the configuration `BLOCK_SIZE = 4096`, `num_warps = 4`; the configuration leaves `num_stages` at the default.

## Lengths and ranges

- `sorted_comp_ptr`: the sorted keys `group_ids * multiplier + supp_ids`, `num_elements` rows, at least 1 (i32 at scale factor 10).
- `counts_ptr`: `torch.zeros(num_groups)`, `num_groups = len(group_df)` at least 1, zeroed and accumulated with `tl.atomic_add`: readwrite.
- `multiplier = max_supp + 1`, at least 1.
