# q13_core_triton__q13_count_orders_kernel

The kernel `_q13_count_orders_kernel` of the Triton plan for q13 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/core/queries/q13/round_04.py`,
copied as `plan.py`), `BLOCK_SIZE = 8192`, `num_warps = 4`.

## Extraction edits

- `_q13_count_orders_kernel` is renamed `kernel`.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `o_custkey_ptr: *i64`, `order_counts_ptr: *i64`, `n_orders: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 109: `_K._q13_count_orders_kernel[grid_count](..., BLOCK_SIZE=BLOCK_SIZE)` in `_query_core`, with `grid_count = (triton.cdiv(n_orders, 8192),)`, `BLOCK_SIZE = 8192`, and the default `num_warps = 4`.

## Lengths and ranges

- `o_custkey_filtered` is the `o_custkey` column of the orders whose comment does not match, `n_orders` rows.
- `order_counts` is `torch.zeros(num_customers + 1)` with `num_customers = max(c_custkey)` of the customer table, an atomic accumulator: `readwrite`. The kernel indexes it by `o_custkey`, whose range the plan takes from TPC-H's foreign key from orders to customer without establishing it, so `o_custkey_ptr` is `read`.
