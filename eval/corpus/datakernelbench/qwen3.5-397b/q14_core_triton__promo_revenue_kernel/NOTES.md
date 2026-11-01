# q14_core_triton__promo_revenue_kernel

The kernel `_promo_revenue_kernel` of the Triton plan for q14 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/core/queries/q14/round_02.py`,
copied as `plan.py`), in configuration 1 of 8: `BLOCK_SIZE = 1024`, `num_warps = 4`.

## Extraction edits

- `_promo_revenue_kernel` is renamed `kernel`.
- The `@triton.autotune` decorator is replaced by this configuration's values,
  `CONSTANTS` and `NUM_WARPS`. The autotuner lists this configuration with
  `num_stages` 2, 4; the front end does not take `num_stages`, and the kernel
  has no loop for the software pipeliner,
  so these give one corpus kernel.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `extendedprice_ptr: *fp64`, `discount_ptr: *fp64`, `shipdate_ptr: *i32`, `promo_ptr: *i1`, `date_lo: i32`, `date_hi: i32`, `out_ptr: *fp64`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 131: `_K._promo_revenue_kernel[grid](...)` in `_launch_promo_revenue`, with `grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)` and the autotuner's `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- The four columns are columns of the joined lineitem and part table, of `n = l_extendedprice.numel()` rows; `p_type_is_promo` is a bool tensor.
- `out` is `torch.zeros(2)`, `[total, promo]`, an atomic accumulator: `readwrite`.
