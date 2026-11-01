# q22_triton__q22_avg_kernel_6

The kernel `_q22_avg_kernel` of the Triton plan for q22 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/full/queries/q22/round_01.py`,
copied as `plan.py`), in configuration 6 of 8: `BLOCK_SIZE = 4096`, `num_warps = 8`.

## Extraction edits

- `_q22_avg_kernel` is renamed `kernel`.
- The `@triton.autotune` decorator is replaced by this configuration's values,
  `CONSTANTS` and `NUM_WARPS`. The autotuner lists this configuration with
  `num_stages` 2, 4; the front end does not take `num_stages`, and the kernel
  has no loop for the software pipeliner, so these give one corpus kernel.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `acctbal_ptr: *fp64`, `cntrycode_ptr: *i32`, `in_mask_ptr: *i1`, `out_ptr: *fp64`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 169: `_K._q22_avg_kernel[grid](...)` in `_compute_avg_threshold`, with `grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)` and the autotuner's `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- The customer columns have `n = c_acctbal.numel()` rows; `in_mask` (and `has_order`) are bool tensors.
- `out` is `torch.zeros(2)`, an atomic accumulator: `readwrite`.
