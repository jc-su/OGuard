# q22_core_triton__group_agg_kernel_6

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/triton/core/queries/q22/round_01.py`, copied as `plan.py`.

## Extraction

- The kernel `_group_agg_kernel` is renamed `kernel`; its body and parameters are the plan's.
- The `@triton.autotune` decorator is replaced by the values of one configuration: configuration 6 of 8, `BLOCK_SIZE = 4096` and `num_warps = 8`.
- The autotuner's configurations also vary `num_stages` (2 and 4), which the front end does not take; the configurations that differ only in `num_stages` give this one kernel.
- Copied the module's imports `import triton` and `import triton.language as tl`.
- `TYPES` states the Triton type of each argument that is not a `tl.constexpr`, from the value that the plan passes.
- Left out the host code.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`_query_core` launches `_K._group_agg_kernel[grid]` with `grid = (triton.cdiv(n, BLOCK_SIZE),)`; the autotuner picks one of its configurations at run time.

## Lengths and ranges

- `cntrycode_ptr`: `n_elements` elements: the country code of each customer, the first two characters of `c_phone` as int32, which the plan expects in [0, 100); the signature cannot state the range.
- `acctbal_ptr`: `n_elements` elements: `c_acctbal`, one element per customer row.
- `outer_mask_ptr`: `n_elements` elements: the outer filter of each customer row.
- `count_ptr`: `torch.zeros(CODE_MAX, int64)` with `CODE_MAX = 100`; the kernel adds to it with `tl.atomic_add` at the loaded codes.
- `sum_ptr`: `torch.zeros(CODE_MAX)`: 100 elements, as `count_ptr`.
- `n_elements`: `n = c_acctbal.numel()`, below 2^31 at scale factor 10 (`i32`), the size symbol of the row buffers.
