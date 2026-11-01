# q11_core_triton__grouped_sum_kernel_7

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/triton/core/queries/q11/round_01.py`, copied as `plan.py`.

## Extraction

- The kernel `_grouped_sum_kernel` is renamed `kernel`; its body and parameters are the plan's.
- The `@triton.autotune` decorator is replaced by the values of one configuration: configuration 7 of 8, `BLOCK_SIZE = 8192` and `num_warps = 4`.
- The autotuner's configurations also vary `num_stages` (2 and 4), which the front end does not take; the configurations that differ only in `num_stages` give this one kernel.
- Copied the module's imports `import triton` and `import triton.language as tl`.
- `TYPES` states the Triton type of each argument that is not a `tl.constexpr`, from the value that the plan passes.
- Left out the host code.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`_launch_grouped_sum` launches `_K._grouped_sum_kernel[grid]` with `grid = (triton.cdiv(n, BLOCK_SIZE),)`; the autotuner picks one of its configurations at run time.

## Lengths and ranges

- `supplycost_ptr`: `n_elements` elements: `ps_supplycost` of the joined frame, `n = ps_supplycost.numel()` rows.
- `availqty_ptr`: `n_elements` elements: `ps_availqty` as float64, one element per joined row.
- `inverse_ptr`: `n_elements` elements: the inverse of `torch.unique(ps_partkey, return_inverse=True)` over the same rows, so every value lies in [0, num_groups): an index into `grouped_ptr`.
- `grouped_ptr`: `torch.zeros(num_groups)` with `num_groups = unique_keys.shape[0]`; the kernel adds to it with `tl.atomic_add`.
- `total_ptr`: `torch.zeros(1)`; the kernel adds to it with `tl.atomic_add`.
- `n_elements`: `n = ps_supplycost.numel()`, below 2^31 at scale factor 10 (`i32`), the size symbol of the row buffers.
- `num_groups`: `unique_keys.shape[0]`, the size symbol of `grouped_ptr`. A launch runs a program only when `n` is positive, and then `torch.unique` has a group, so its least value is 1.
