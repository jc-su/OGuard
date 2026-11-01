# q13_core_triton__count_orders_kernel_5

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/triton/core/queries/q13/round_06.py`, copied as `plan.py`.

## Extraction

- The kernel `_count_orders_kernel` is renamed `kernel`; its body and parameters are the plan's.
- The `@triton.autotune` decorator is replaced by the values of one configuration: configuration 5 of 8, `BLOCK_SIZE = 4096` and `num_warps = 4`.
- The autotuner's configurations also vary `num_stages` (2 and 4), which the front end does not take; the configurations that differ only in `num_stages` give this one kernel.
- Copied the module's imports `import triton` and `import triton.language as tl`.
- `TYPES` states the Triton type of each argument that is not a `tl.constexpr`, from the value that the plan passes.
- Left out the host code.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`_query_core` launches `_K._count_orders_kernel[grid]` with `grid = (triton.cdiv(n_orders, BLOCK_SIZE),)`; the autotuner picks one of its configurations at run time.

## Lengths and ranges

- `o_custkey_ptr`: `num_orders` elements: `o_custkey` of the orders whose comment does not match, one element per such order.
- `order_counts_ptr`: `torch.zeros(num_customers + 1, int64)` with `num_customers = int(df_customer['c_custkey'].max())`; the kernel adds to it with `tl.atomic_add` at the loaded keys.
- `num_orders`: `o_custkey_filtered.numel()`, below 2^31 at scale factor 10 (`i32`), the size symbol of `o_custkey_ptr`.
- `num_customers_plus_one`: `num_customers + 1`, the size symbol of `order_counts_ptr`, which the kernel does not read; `c_custkey.max()` raises on an empty table and TPC-H keys start at 1, so its least value is 2.
