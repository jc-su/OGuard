# q22_triton__filter_groupby_kernel_4

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/triton/full/queries/q22/round_03.py`, copied as `plan.py`.

## Extraction

- The kernel `_filter_groupby_kernel` is renamed `kernel`; its body and parameters are the plan's.
- The `@triton.autotune` decorator is replaced by the values of one configuration: configuration 4 of 8, `BLOCK_SIZE = 2048` and `num_warps = 8`.
- The autotuner's configurations also vary `num_stages` (2 and 4), which the front end does not take; the configurations that differ only in `num_stages` give this one kernel.
- Copied the module's imports `import triton` and `import triton.language as tl`.
- `TYPES` states the Triton type of each argument that is not a `tl.constexpr`, from the value that the plan passes.
- Left out the host code.
- Left out the other kernels of the module: `_avg_kernel`.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`_filter_groupby_gpu` launches `_K._filter_groupby_kernel[grid]` with `grid = (triton.cdiv(n, BLOCK_SIZE),)`; the autotuner picks one of its configurations at run time.

## Lengths and ranges

- `acctbal_ptr`: `n_elements` elements: `c_acctbal`, one element per customer.
- `cntrycode_ptr`: `n_elements` elements: one element per customer.
- `in_mask_ptr`: `n_elements` elements: one element per customer.
- `has_order_ptr`: `n_elements` elements: one element per customer.
- `avg_ptr`: `torch.tensor([avg_val])`, 1 element.
- `numcust_ptr`: `torch.zeros(num_groups)` with `num_groups = 7`; the kernel adds to elements 0 to 6 with `tl.atomic_add`.
- `totacctbal_ptr`: `torch.zeros(num_groups)`, 7 elements, as `numcust_ptr`.
- `n_elements`: `n = c_acctbal.numel()`, the size symbol of the row buffers.
