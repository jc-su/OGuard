# q16_triton__count_unique_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/triton/full/queries/q16/round_01.py`, copied as `plan.py`.

## Extraction

- The kernel `_count_unique_kernel` is renamed `kernel`; its body and parameters are the plan's.
- The `@triton.autotune` decorator is replaced by the values of one configuration: configuration 1 of 8, `BLOCK_SIZE = 1024` and `num_warps = 4`.
- The autotuner's configurations also vary `num_stages` (2 and 4), which the front end does not take; the configurations that differ only in `num_stages` give this one kernel.
- Copied the module's imports `import triton` and `import triton.language as tl`.
- `TYPES` states the Triton type of each argument that is not a `tl.constexpr`, from the value that the plan passes.
- Left out the host code.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`run_query` launches `_K._count_unique_kernel[grid](sorted_keys, counts, n_rows)` with `grid = (triton.cdiv(n_rows, BLOCK_SIZE),)`; the autotuner picks one of its configurations at run time.

## Lengths and ranges

- `sorted_keys_ptr`: `n_elements` elements: the sorted composite keys `group_id * 100001 + ps_suppkey`, one per joined row; the group id `key // 100001` is below `num_groups` by the plan, which the signature cannot state.
- `out_counts_ptr`: `torch.zeros(num_groups)`, one element per (brand, type, size) group; the kernel adds to it with `tl.atomic_add`.
- `n_elements`: `n_rows = sorted_keys.shape[0]`; the plan returns before the launch when the join is empty, so its least value is 1.
