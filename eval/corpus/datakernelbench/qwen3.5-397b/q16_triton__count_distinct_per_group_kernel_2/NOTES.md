# q16_triton__count_distinct_per_group_kernel_2

The kernel `_count_distinct_per_group_kernel` of the Triton plan for q16 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/full/queries/q16/round_01.py`,
copied as `plan.py`), in configuration 2 of 8: `BLOCK_SIZE = 1024`, `num_warps = 8`.

## Extraction edits

- `_count_distinct_per_group_kernel` is renamed `kernel`.
- The `@triton.autotune` decorator is replaced by this configuration's values,
  `CONSTANTS` and `NUM_WARPS`. The autotuner lists this configuration with
  `num_stages` 2, 4; the front end does not take `num_stages`, and the kernel
  has no loop for the software pipeliner, so these give one corpus kernel.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `group_ptr: *i64`, `supp_ptr: *i64`, `out_ptr: *i64`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 87: `_K._count_distinct_per_group_kernel[grid](...)` in `_launch_count_distinct`, when `n > 0`, with `grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)` and the autotuner's `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- `group_ids` and `supp_ids` have `n = group_ids.numel()` rows, at least 1 at the launch.
- `_gid` is `range(len(group_df))` joined onto the rows, so each group id lies in `[0, num_groups)`: `group_ptr` is `index-in` `out_ptr`. `out` is `torch.zeros(num_groups)`, an atomic accumulator (`readwrite`).
- The rows are sorted by (group, supplier); the gate does not establish sortedness, and the kernel's accesses do not need it.
