# q05_core_triton__q5_kernel_5

The kernel `_q5_kernel` of the Triton plan for q05 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/core/queries/q05/round_01.py`,
copied as `plan.py`), in configuration 5 of 8: `BLOCK_SIZE = 4096`, `num_warps = 4`.

## Extraction edits

- `_q5_kernel` is renamed `kernel`.
- The `@triton.autotune` decorator is replaced by this configuration's values,
  `CONSTANTS` and `NUM_WARPS`. The autotuner lists this configuration with
  `num_stages` 2, 4; the front end does not take `num_stages`, and the kernel
  has no loop for the software pipeliner,
  so these give one corpus kernel.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `nation_id_ptr: *i64`, `extendedprice_ptr: *fp64`, `discount_ptr: *fp64`, `out_ptr: *fp64`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 81: `_K._q5_kernel[grid](...)` in `_query_core`, with `grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)` and the autotuner's `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- The three columns are columns of the joined table `final_df`, of `n = nation_id.numel()` rows.
- `out` is `torch.zeros(num_nations)`, an atomic accumulator: `readwrite`; `num_nations` is the number of distinct nations of the region.
- `nation_id` is `range(len(nations_unique))` joined onto the rows, so each value lies in `[0, num_nations)`: `nation_id_ptr` is `index-in` `out_ptr`.
