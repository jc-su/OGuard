# q06_core_triton__q6_kernel_5

The kernel `_q6_kernel` of the Triton plan for q06 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/core/queries/q06/round_01.py`,
copied as `plan.py`), in configuration 5 of 8: `BLOCK_SIZE = 4096`, `num_warps = 4`.

## Extraction edits

- `_q6_kernel` is renamed `kernel`.
- The `@triton.autotune` decorator is replaced by this configuration's values,
  `CONSTANTS` and `NUM_WARPS`. The autotuner lists this configuration with
  `num_stages` 2, 4; the front end does not take `num_stages`, and the kernel
  has no loop for the software pipeliner,
  so these give one corpus kernel.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `price_ptr: *fp64`, `discount_ptr: *fp64`, `quantity_ptr: *fp64`, `shipdate_ptr: *i32`, `date_lo: i32`, `date_hi: i32`, `thresh_ptr: *fp64`, `out_ptr: *fp64`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 90: `_K._q6_kernel[grid](...)` in `_launch_q6`, with `grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)` and the autotuner's `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- The four columns are lineitem columns of `n = l_extendedprice.numel()` rows.
- `thresh` is `torch.tensor([discount_lo, discount_hi, quantity_hi])`, 3 elements.
- `out` is `torch.zeros(1)`, an atomic accumulator: `readwrite`.
