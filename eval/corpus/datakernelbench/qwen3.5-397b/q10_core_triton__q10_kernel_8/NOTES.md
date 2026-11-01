# q10_core_triton__q10_kernel_8

The kernel `_q10_kernel` of the Triton plan for q10 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/core/queries/q10/round_01.py`,
copied as `plan.py`), in configuration 8 of 8: `BLOCK_SIZE = 8192`, `num_warps = 8`.

## Extraction edits

- `_q10_kernel` is renamed `kernel`.
- The `@triton.autotune` decorator is replaced by this configuration's values,
  `CONSTANTS` and `NUM_WARPS`. The autotuner lists this configuration with
  `num_stages` 2, 4; the front end does not take `num_stages`, and the kernel
  has no loop for the software pipeliner,
  so these give one corpus kernel.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `extendedprice_ptr: *fp64`, `discount_ptr: *fp64`, `inverse_ptr: *i64`, `revenue_ptr: *fp64`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 93: `_K._q10_kernel[grid](...)` in `_launch_q10`, with `grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)` and the autotuner's `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- The three columns have `n = extendedprice.numel()` rows of the joined table; `inverse` is the inverse of `torch.unique(c_custkey)`, in `[0, num_unique)`: `index-in` `revenue_ptr`.
- `revenue` is `torch.zeros(num_unique)` with `num_unique` the number of distinct customer keys, an atomic accumulator: `readwrite`.
