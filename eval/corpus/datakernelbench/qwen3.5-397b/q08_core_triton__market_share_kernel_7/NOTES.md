# q08_core_triton__market_share_kernel_7

The kernel `_market_share_kernel` of the Triton plan for q08 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/core/queries/q08/round_01.py`,
copied as `plan.py`), in configuration 7 of 8: `BLOCK_SIZE = 8192`, `num_warps = 4`.

## Extraction edits

- `_market_share_kernel` is renamed `kernel`.
- The `@triton.autotune` decorator is replaced by this configuration's values,
  `CONSTANTS` and `NUM_WARPS`. The autotuner lists this configuration with
  `num_stages` 2, 4; the front end does not take `num_stages`, and the kernel
  has no loop for the software pipeliner,
  so these give one corpus kernel.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `volume_ptr: *fp64`, `is_brazil_ptr: *i1`, `inv_ptr: *i64`, `brazil_ptr: *fp64`, `total_ptr: *fp64`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 104: `_K._market_share_kernel[grid](...)` in `_launch_market_share`, with `grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)` and the autotuner's `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- The three columns have `n = volume.numel()` rows of the joined table; `is_brazil` is a bool tensor.
- `inv` is the inverse of `torch.unique(o_year)`, in `[0, num_groups)` with `num_groups` the number of distinct years: `inv_ptr` is `index-in` `total_ptr` (`brazil_ptr` has the same length).
- `brazil_vol` and `total_vol` are `torch.zeros(num_groups)`, atomic accumulators: `readwrite`.
