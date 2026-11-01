# q02_triton__topk_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/triton/full/queries/q02/round_02.py`, copied as `plan.py`.

## Extraction

- The kernel `_topk_kernel` is renamed `kernel`; its body and parameters are the plan's.
- The `@triton.autotune` decorator is replaced by the values of one configuration: configuration 1 of 8, `BLOCK_SIZE = 1024` and `num_warps = 4`.
- The autotuner's configurations also vary `num_stages` (2 and 4), which the front end does not take; the configurations that differ only in `num_stages` give this one kernel.
- Copied the module's imports `import triton` and `import triton.language as tl`.
- `TYPES` states the Triton type of each argument that is not a `tl.constexpr`, from the value that the plan passes.
- Left out the host code.
- Left out the other kernels of the module: `_min_supplycost_kernel`, `_q2_filter_kernel`.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`_query_core` launches `_K._topk_kernel[grid]` with `grid = (triton.cdiv(n, BLOCK_SIZE),)` when `n > 0`; the autotuner picks one of its configurations at run time. The plan's other two kernels, `_min_supplycost_kernel` and `_q2_filter_kernel`, are not launched.

## Lengths and ranges

- `acctbal_ptr`: `n_elements` elements: `s_acctbal` of the final frame, `n = s_acctbal.shape[0]` rows.
- `n_name_ptr`: `n_elements` elements: `torch.zeros(n, dtype=torch.int64)`, a placeholder that the kernel does not access.
- `s_name_ptr`: `n_elements` elements: `torch.zeros(n, dtype=torch.int64)`, a placeholder that the kernel does not access.
- `partkey_ptr`: `n_elements` elements: `p_partkey` of the final frame, not accessed.
- `out_idx_ptr`: `n_elements` elements: `torch.empty(n, dtype=torch.int64)`.
- `n_elements`: `n = s_acctbal.shape[0]`; the plan launches only when it is positive.
- `k`: `min(limit, n)`; not used by the kernel.
