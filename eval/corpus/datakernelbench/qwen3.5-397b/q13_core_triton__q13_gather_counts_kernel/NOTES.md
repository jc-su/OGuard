# q13_core_triton__q13_gather_counts_kernel

The kernel `_q13_gather_counts_kernel` of the Triton plan for q13 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/core/queries/q13/round_04.py`,
copied as `plan.py`), `BLOCK_SIZE = 8192`, `num_warps = 4`.

## Extraction edits

- `_q13_gather_counts_kernel` is renamed `kernel`.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `c_custkey_ptr: *i64`, `order_counts_ptr: *i64`, `c_count_ptr: *i64`, `n_customers: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 117: `_K._q13_gather_counts_kernel[grid_gather](..., BLOCK_SIZE=BLOCK_SIZE)` in `_query_core`, after `_q13_count_orders_kernel`, with `grid_gather = (triton.cdiv(n_customers, 8192),)`, `BLOCK_SIZE = 8192`, and the default `num_warps = 4`.

## Lengths and ranges

- `c_custkey` is the customer table's key column, `n_customers` rows; `c_count` is `torch.zeros(n_customers)`, written once per customer: `write`.
- `order_counts` has `num_customers + 1` elements with `num_customers = max(c_custkey)`, the counts of the previous kernel, read only here. The plan sizes it from the largest `c_custkey`, so `c_custkey_ptr` is `index-in` `order_counts_ptr` (customer keys are TPC-H's non-negative keys).
