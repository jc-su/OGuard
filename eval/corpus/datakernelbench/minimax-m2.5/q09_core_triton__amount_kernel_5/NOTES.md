# q09_core_triton__amount_kernel_5

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/triton/core/queries/q09/round_02.py`, copied as `plan.py`.

## Extraction

- The kernel `_amount_kernel` is renamed `kernel`; its body and parameters are the plan's.
- The `@triton.autotune` decorator is replaced by the values of one configuration: configuration 5 of 8, `BLOCK_SIZE = 4096` and `num_warps = 4`.
- The autotuner's configurations also vary `num_stages` (2 and 4), which the front end does not take; the configurations that differ only in `num_stages` give this one kernel.
- Copied the module's imports `import triton` and `import triton.language as tl`.
- `TYPES` states the Triton type of each argument that is not a `tl.constexpr`, from the value that the plan passes.
- Left out the host code.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`_launch_amount` launches `_K._amount_kernel[grid]` with `grid = (triton.cdiv(n, BLOCK_SIZE),)`; the autotuner picks one of its configurations at run time.

## Lengths and ranges

- `price_ptr`: `n_elements` elements: `l_extendedprice` of the joined frame, `n = l_extendedprice.numel()` rows.
- `discount_ptr`: `n_elements` elements: one element per joined row.
- `supplycost_ptr`: `n_elements` elements: one element per joined row.
- `quantity_ptr`: `n_elements` elements: one element per joined row.
- `out_ptr`: `n_elements` elements: `torch.empty_like(l_extendedprice)`.
- `n_elements`: `n = l_extendedprice.numel()`, below 2^31 at scale factor 10 (`i32`), the size symbol of every buffer.
