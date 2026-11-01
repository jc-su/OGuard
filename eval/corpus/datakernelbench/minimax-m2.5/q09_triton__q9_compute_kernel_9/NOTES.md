# q09_triton__q9_compute_kernel_9

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/triton/full/queries/q09/round_01.py`, copied as `plan.py`.

## Extraction

- The kernel `_q9_compute_kernel` is renamed `kernel`; its body and parameters are the plan's.
- The `@triton.autotune` decorator is replaced by the values of one configuration: configuration 9 of 12, `BLOCK_SIZE = 4096` and `num_warps = 16`.
- The autotuner's configurations also vary `num_stages` (2 and 4), which the front end does not take; the configurations that differ only in `num_stages` give this one kernel.
- Copied the module's imports `import triton` and `import triton.language as tl`.
- `TYPES` states the Triton type of each argument that is not a `tl.constexpr`, from the value that the plan passes.
- The plan loads the module `_KERNEL_SOURCE_V2`; the kernel is its `_q9_compute_kernel` (the first source string, which also defines a `_q9_compute_kernel`, is not loaded).
- Left out the host code.
- Left out the other kernels of the module: `_q9_fused_agg_kernel`.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`_launch_q9_compute` launches `_K._q9_compute_kernel[grid]` with `grid = (triton.cdiv(n, BLOCK_SIZE),)`; the autotuner picks one of its configurations at run time. The `_q9_fused_agg_kernel` of the same module is not launched.

## Lengths and ranges

- `price_ptr`: `n_elements` elements: `l_extendedprice` of the joined frame, `n = l_extendedprice.numel()` rows.
- `discount_ptr`: `n_elements` elements: one element per joined row.
- `supplycost_ptr`: `n_elements` elements: one element per joined row.
- `quantity_ptr`: `n_elements` elements: one element per joined row.
- `out_ptr`: `n_elements` elements: `torch.empty_like(l_extendedprice)`.
- `n_elements`: `n = l_extendedprice.numel()`, the size symbol of every buffer.
