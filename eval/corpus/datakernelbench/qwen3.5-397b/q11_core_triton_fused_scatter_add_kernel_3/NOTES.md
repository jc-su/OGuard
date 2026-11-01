# q11_core_triton_fused_scatter_add_kernel_3

The kernel `fused_scatter_add_kernel` of the Triton plan for q11 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/core/queries/q11/round_01.py`,
copied as `plan.py`), in configuration 3 of 8: `BLOCK_SIZE = 2048`, `num_warps = 4`.

## Extraction edits

- `fused_scatter_add_kernel` is renamed `kernel`.
- The `@triton.autotune` decorator is replaced by this configuration's values,
  `CONSTANTS` and `NUM_WARPS`. The autotuner lists this configuration with
  `num_stages` 2, 4; the front end does not take `num_stages`, and the kernel
  has no loop for the software pipeliner,
  so these give one corpus kernel.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `cost_ptr: *fp64`, `qty_ptr: *fp64`, `inverse_ptr: *i64`, `out_ptr: *fp64`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 82: `_K.fused_scatter_add_kernel[grid](...)` in `_query_core`, with `grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)` and the autotuner's `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- The three columns have `n_elements = ps_partkey.numel()` rows of the joined partsupp; `inverse` is the inverse of `torch.unique(ps_partkey)`, in `[0, n_unique)`: `index-in` `out_ptr`.
- `group_sums` is `torch.zeros(n_unique)` with `n_unique` the number of distinct part keys, an atomic accumulator: `readwrite`.
