# q22_core_triton__avg_kernel_2

The kernel `_avg_kernel` of the Triton plan for q22 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/core/queries/q22/round_01.py`,
copied as `plan.py`), in configuration 2 of 6: `BLOCK_SIZE = 1024`, `num_warps = 8`.

## Extraction edits

- `_avg_kernel` is renamed `kernel`.
- The `@triton.autotune` decorator is replaced by this configuration's values,
  `CONSTANTS` and `NUM_WARPS`. The autotuner lists this configuration with
  `num_stages` 2, 3; the front end does not take `num_stages`, and the kernel
  has no loop for the software pipeliner,
  so these give one corpus kernel.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `acctbal_ptr: *fp64`, `in_mask_ptr: *i1`, `sum_ptr: *fp64`, `count_ptr: *i64`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 170: `_K._avg_kernel[grid](...)` in `_query_core`, with `grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)` and the autotuner's `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- The two columns are customer columns of `n = c_custkey.numel()` rows; `in_cntrycode_mask` is a bool tensor.
- `sum_out` and `count_out` are `torch.zeros(1)`, atomic accumulators: `readwrite`.
