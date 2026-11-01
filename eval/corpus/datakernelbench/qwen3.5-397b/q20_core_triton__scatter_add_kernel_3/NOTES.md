# q20_core_triton__scatter_add_kernel_3

The kernel `_scatter_add_kernel` of the Triton plan for q20 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/core/queries/q20/round_01.py`,
copied as `plan.py`), in configuration 3 of 8: `BLOCK_SIZE = 2048`, `num_warps = 4`.

## Extraction edits

- `_scatter_add_kernel` is renamed `kernel`.
- The `@triton.autotune` decorator is replaced by this configuration's values,
  `CONSTANTS` and `NUM_WARPS`. The autotuner lists this configuration with
  `num_stages` 2, 4; the front end does not take `num_stages`, and the kernel
  has no loop for the software pipeliner,
  so these give one corpus kernel.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `out_ptr: *fp64`, `indices_ptr: *i64`, `values_ptr: *fp64`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 100: `_K._scatter_add_kernel[grid](out, indices, values, n)` in `_scatter_add_triton`, with `grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)` and the autotuner's `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- `indices` is `li_group`, the lineitem rows' part of the inverse of `torch.unique` over the concatenated (partkey, suppkey) pairs, so its values lie in `[0, num_groups)` with `num_groups = unique_pairs.shape[0]`: `index-in` `out_ptr`; `values` is `l_quantity` of the same `n = indices.numel()` rows.
- `out` is `torch.zeros(num_groups)`, an atomic accumulator: `readwrite`.
