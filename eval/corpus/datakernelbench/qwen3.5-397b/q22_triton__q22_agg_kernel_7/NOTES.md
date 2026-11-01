# q22_triton__q22_agg_kernel_7

The kernel `_q22_agg_kernel` of the Triton plan for q22 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/full/queries/q22/round_01.py`,
copied as `plan.py`), in configuration 7 of 8: `num_groups = 7`, `BLOCK_SIZE = 8192`, `num_warps = 4`.

## Extraction edits

- `_q22_agg_kernel` is renamed `kernel`.
- The `@triton.autotune` decorator is replaced by this configuration's values,
  `CONSTANTS` and `NUM_WARPS`. The autotuner lists this configuration with
  `num_stages` 2, 4; the front end does not take `num_stages`, and the kernel
  has no loop for the software pipeliner, so these give one corpus kernel.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `acctbal_ptr: *fp64`, `cntrycode_ptr: *i32`, `in_mask_ptr: *i1`, `has_order_ptr: *i1`, `avg_thresh_ptr: *fp64`, `out_sum_ptr: *fp64`, `out_count_ptr: *i64`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 203: `_K._q22_agg_kernel[grid](...)` in `_final_aggregation`, with `grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)`, the `tl.constexpr` `num_groups = 7`, and the autotuner's `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- The customer columns have `n = c_acctbal.numel()` rows; `in_mask` (and `has_order`) are bool tensors.
- `avg_thresh` is the zero-dimensional tensor `total_sum / total_count` (or `torch.tensor(0.0)`): one element.
- `out_sum` and `out_count` are `torch.zeros(7)`, atomic accumulators: `readwrite`.
