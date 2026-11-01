# q22_core_triton__agg_kernel

The kernel `_agg_kernel` of the Triton plan for q22 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/core/queries/q22/round_01.py`,
copied as `plan.py`), in configuration 1 of 6: `BLOCK_SIZE = 1024`, `num_warps = 4`.

## Extraction edits

- `_agg_kernel` is renamed `kernel`.
- The `@triton.autotune` decorator is replaced by this configuration's values,
  `CONSTANTS` and `NUM_WARPS`. The autotuner lists this configuration with
  `num_stages` 2, 3; the front end does not take `num_stages`, and the kernel's only loop
  is a `tl.static_range`, which is unrolled,
  so these give one corpus kernel.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `custkey_ptr: *i64`, `acctbal_ptr: *fp64`, `cntrycode_ptr: *i32`, `in_mask_ptr: *i1`, `orders_ptr: *i64`, `n_orders: i32`, `avg_val: fp64`, `code_map_ptr: *i32`, `group_counts_ptr: *i64`, `group_sums_ptr: *fp64`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 188: `_K._agg_kernel[grid](...)` in `_query_core`, after `_avg_kernel`, with `grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)` and the autotuner's `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- The four customer columns have `n = c_custkey.numel()` rows; `in_cntrycode_mask` is a bool tensor.
- `sorted_okeys` is the sorted distinct `o_custkey`, `n_orders = sorted_okeys.numel()`; the gate does not establish sortedness, and the search's loads are masked by `mid < n_orders` whatever the order.
- `code_map` is `torch.full((100,), -1)` with the seven target codes mapped to `0..6`, 100 elements, indexed by `cntrycode` (the first two digits of `c_phone` as an integer), a range the plan does not check.
- `group_counts` and `group_sums` are `torch.zeros(7)`, atomic accumulators: `readwrite`.
- `avg_val` is a Python float (`fp64`).
