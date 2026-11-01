# q21_triton__q21_kernel_5

The kernel `_q21_kernel` of the Triton plan for q21 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/full/queries/q21/round_01.py`,
copied as `plan.py`), in configuration 5 of 6: `BLOCK_SIZE = 4096`, `num_warps = 4`.

## Extraction edits

- `_q21_kernel` is renamed `kernel`.
- The `@triton.autotune` decorator is replaced by this configuration's values,
  `CONSTANTS` and `NUM_WARPS`. The autotuner lists this configuration with
  `num_stages` 2, 4; the front end does not take `num_stages`, and the kernel
  has no loop for the software pipeliner, so these give one corpus kernel.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `l1_orderkey_ptr: *i64`, `l1_supp_id_ptr: *i32`, `agg_orderkey_ptr: *i64`, `agg_distinct_ptr: *i32`, `agg_late_distinct_ptr: *i32`, `counts_ptr: *i64`, `n_l1: i32`, `n_agg: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 205: `_K._q21_kernel[grid](...)` in `run_query`, when `n_l1 > 0 and n_agg > 0`, with `grid = lambda meta: (triton.cdiv(n_l1, meta['BLOCK_SIZE']),)` and the autotuner's `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- `l1_orderkey_t` and `l1_supp_id_t` have `n_l1` rows; the three aggregate columns of `agg` have `n_agg` rows; both are at least 1 at the launch.
- `supp_id` is `range(len(supp_n))` joined onto the rows, so each value lies in `[0, num_suppliers)`: `l1_supp_id_ptr` is `index-in` `counts_ptr`. `counts_t` is `torch.zeros(num_suppliers)`, an atomic accumulator (`readwrite`).
- `agg` is sorted by orderkey; the gate does not establish sortedness, and the binary search's loads are masked by `mid < n_agg` whatever the order.
