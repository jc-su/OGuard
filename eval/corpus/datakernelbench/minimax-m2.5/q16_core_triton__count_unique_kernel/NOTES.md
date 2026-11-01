# q16_core_triton__count_unique_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/triton/core/queries/q16/round_06.py`, copied as `plan.py`.

## Extraction

- The kernel `_count_unique_kernel` is renamed `kernel`; its body and parameters are the plan's.
- The `@triton.autotune` decorator is replaced by the values of one configuration: configuration 1 of 6, `BLOCK_SIZE = 256` and `num_warps = 4`.
- Copied the module's imports `import triton` and `import triton.language as tl`.
- `TYPES` states the Triton type of each argument that is not a `tl.constexpr`, from the value that the plan passes.
- Left out the host code.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`_launch_unique_count` launches `_K._count_unique_kernel[grid]` with `grid = (triton.cdiv(num_unique, BLOCK_SIZE),)`, only when `num_unique > 0`; the autotuner picks one of its configurations at run time.

## Lengths and ranges

- `unique_group_ptr`: `n_elements` elements: `unique_groups`, the group id of each distinct (group, supplier) pair, which the plan assigns as `_gid = range(len(group_df))`, so every value lies in [0, num_groups): an index into `output_ptr`.
- `output_ptr`: `torch.zeros(num_groups, int64)` with `num_groups = len(group_df)`; the kernel adds to it with `tl.atomic_add`.
- `n_elements`: `num_unique = unique_groups.numel()`, below 2^31 at scale factor 10 (`i32`), the size symbol of `unique_group_ptr`; the plan launches only when it is positive, so its least value is 1.
- `num_groups`: `len(group_df)`, the size symbol of `output_ptr`; the plan returns before the launch when the join is empty, so its least value is 1.
