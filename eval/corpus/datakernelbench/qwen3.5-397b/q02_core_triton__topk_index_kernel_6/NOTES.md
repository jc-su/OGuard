# q02_core_triton__topk_index_kernel_6

The kernel `_topk_index_kernel` of the Triton plan for q02 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/core/queries/q02/round_02.py`,
copied as `plan.py`), in configuration 6 of 8: `BLOCK_SIZE = 4096`, `num_warps = 8`.

## Extraction edits

- `_topk_index_kernel` is renamed `kernel`.
- The `@triton.autotune` decorator is replaced by this configuration's values,
  `CONSTANTS` and `NUM_WARPS`. The autotuner lists this configuration with
  `num_stages` 2, 4; the front end does not take `num_stages`, and the kernel
  has no loop for the software pipeliner,
  so these give one corpus kernel.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `out_ptr: *i64`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 84: `_K._topk_index_kernel[grid](out, n)` in `_launch_topk_index`, with `grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)` and the autotuner's `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- `out` is `torch.zeros(n)` with `n = k = min(limit, len(final))`, the number of indices to generate; `_query_core` returns before the launch when `k == 0`, so `n_elements` is at least 1.
