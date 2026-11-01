# q10_triton__q10_aggregate_kernel_6

The kernel `_q10_aggregate_kernel` of the Triton plan for q10 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/full/queries/q10/round_01.py`,
copied as `plan.py`), in configuration 6 of 8: `BLOCK_SIZE = 4096`, `num_warps = 8`.

## Extraction edits

- `_q10_aggregate_kernel` is renamed `kernel`.
- The `@triton.autotune` decorator is replaced by this configuration's values,
  `CONSTANTS` and `NUM_WARPS`. The autotuner lists this configuration with
  `num_stages` 2, 4; the front end does not take `num_stages`, and the kernel
  has no loop for the software pipeliner, so these give one corpus kernel.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `extendedprice_ptr: *fp64`, `discount_ptr: *fp64`, `inverse_ptr: *i64`, `agg_revenue_ptr: *fp64`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 105: `_K._q10_aggregate_kernel[grid](...)` in `_launch_aggregate`, with `grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)` and the autotuner's `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- The columns have `n = extendedprice.numel()` rows; `inverse` comes from `torch.unique(c_custkey, return_inverse=True)` over as many rows.
- `agg_revenue` is `torch.zeros(num_unique)`, an atomic accumulator (`readwrite`); the inverse indices lie in `[0, num_unique)`, so `inverse_ptr` is `index-in` `agg_revenue_ptr`.
