# q20_triton__q20_compare_kernel_8

The kernel `_q20_compare_kernel` of the Triton plan for q20 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/full/queries/q20/round_03.py`,
copied as `plan.py`), in configuration 8 of 8: `BLOCK_SIZE = 8192`, `num_warps = 8`.

## Extraction edits

- `_q20_compare_kernel` is renamed `kernel`.
- The `@triton.autotune` decorator is replaced by this configuration's values,
  `CONSTANTS` and `NUM_WARPS`. The autotuner lists this configuration with
  `num_stages` 2, 4; the front end does not take `num_stages`, and the kernel
  has no loop for the software pipeliner, so these give one corpus kernel.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `ps_availqty_ptr: *fp64`, `sum_qty_ptr: *fp64`, `out_mask_ptr: *i32`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 118: `_K._q20_compare_kernel[grid](...)` in `_query_core`, with `grid = lambda meta: (triton.cdiv(n_ps, meta['BLOCK_SIZE']),)` and the autotuner's `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- `ps_availqty` and `ps_sum_qty = sum_qty[ps_group]` have `n_ps = len(ps_availqty)` rows.
- `out_mask` is `torch.zeros(n_ps)`; the kernel stores every position below `n_elements` and reads none: `write`.
