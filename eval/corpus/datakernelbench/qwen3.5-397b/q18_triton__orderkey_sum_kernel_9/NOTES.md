# q18_triton__orderkey_sum_kernel_9

The kernel `_orderkey_sum_kernel` of the Triton plan for q18 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/full/queries/q18/round_04.py`,
copied as `plan.py`), in configuration 9 of 12: `BLOCK_SIZE = 8192`, `num_warps = 32`.

## Extraction edits

- `_orderkey_sum_kernel` is renamed `kernel`.
- The `@triton.autotune` decorator is replaced by this configuration's values,
  `CONSTANTS` and `NUM_WARPS`. The autotuner lists this configuration with
  `num_stages` 4, 5, 6; the front end does not take `num_stages`, and the kernel
  has no loop for the software pipeliner, so these give one corpus kernel.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `quantity_ptr: *fp64`, `inverse_ptr: *i64`, `sum_ptr: *fp64`, `n_elements: i32`, `n_unique: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 147: `_K._orderkey_sum_kernel[grid](...)` in `_query_core`, with `grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)` and the autotuner's `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- `l_quantity` and `inverse` have `n_elements = l_orderkey.numel()` rows; `inverse` comes from `torch.unique(l_orderkey, return_inverse=True)`.
- `sums` is `torch.zeros(n_unique)` with `n_unique = unique_keys.numel()`, passed as the kernel's `n_unique`; an atomic accumulator (`readwrite`). The inverse indices lie in `[0, n_unique)`: `inverse_ptr` is `index-in` `sum_ptr`.
