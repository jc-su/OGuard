# q22_triton__avg_kernel_8

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/triton/full/queries/q22/round_03.py`, copied as `plan.py`.

## Extraction

- The kernel `_avg_kernel` is renamed `kernel`; its body and parameters are the plan's.
- The `@triton.autotune` decorator is replaced by the values of one configuration: configuration 8 of 8, `BLOCK_SIZE = 8192` and `num_warps = 8`.
- The autotuner's configurations also vary `num_stages` (2 and 4), which the front end does not take; the configurations that differ only in `num_stages` give this one kernel.
- Copied the module's imports `import triton` and `import triton.language as tl`.
- `TYPES` states the Triton type of each argument that is not a `tl.constexpr`, from the value that the plan passes.
- Left out the host code.
- Left out the other kernels of the module: `_filter_groupby_kernel`.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`_compute_avg_gpu` launches `_K._avg_kernel[grid]` with `grid = (triton.cdiv(n, BLOCK_SIZE),)`; the autotuner picks one of its configurations at run time.

## Lengths and ranges

- `acctbal_ptr`: `n_elements` elements: `c_acctbal`, one element per customer.
- `cntrycode_ptr`: `n_elements` elements: one element per customer.
- `in_mask_ptr`: `n_elements` elements: one element per customer.
- `sum_ptr`: `torch.zeros(1)`; the kernel adds to it with `tl.atomic_add`.
- `count_ptr`: `torch.zeros(1, dtype=torch.int64)`; the kernel adds to it with `tl.atomic_add`.
- `n_elements`: `n = c_acctbal.numel()`, the size symbol of the row buffers.
