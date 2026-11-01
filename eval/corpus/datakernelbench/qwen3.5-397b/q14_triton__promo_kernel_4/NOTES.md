# q14_triton__promo_kernel_4

The kernel `_promo_kernel` of the Triton plan for q14 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/full/queries/q14/round_01.py`,
copied as `plan.py`), in configuration 4 of 8: `BLOCK_SIZE = 2048`, `num_warps = 8`.

## Extraction edits

- `_promo_kernel` is renamed `kernel`.
- The `@triton.autotune` decorator is replaced by this configuration's values,
  `CONSTANTS` and `NUM_WARPS`. The autotuner lists this configuration with
  `num_stages` 2, 4; the front end does not take `num_stages`, and the kernel
  has no loop for the software pipeliner, so these give one corpus kernel.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `shipdate_ptr: *i32`, `price_ptr: *fp64`, `discount_ptr: *fp64`, `promo_ptr: *i1`, `out_total_ptr: *fp64`, `out_promo_ptr: *fp64`, `date_lo: i32`, `date_hi: i32`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 142: `_K._promo_kernel[grid](...)` in `run_query`, with `grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)` and the autotuner's `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- The four columns are columns of the joined table, of `n = l_extendedprice_t.numel()` rows; `p_type_is_promo_t` is a bool tensor.
- `out_total` and `out_promo` are `torch.zeros(1)`, atomic accumulators: `readwrite`.
