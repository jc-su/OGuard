# q05_triton__q5_kernel

The kernel `_q5_kernel` of the Triton plan for q05 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/full/queries/q05/round_04.py`,
copied as `plan.py`): `BLOCK_SIZE = 1024`, `num_warps = 4`.

## Extraction edits

- `_q5_kernel` is renamed `kernel`.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `nation_id_ptr: *i64`, `price_ptr: *fp64`, `discount_ptr: *fp64`, `out_ptr: *fp64`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 68: `_K._q5_kernel[grid](..., BLOCK_SIZE=BLOCK_SIZE)` in `_launch_q5`, with `grid = (triton.cdiv(n, 1024),)`, `BLOCK_SIZE = 1024`, and the default `num_warps = 4`.

## Lengths and ranges

- The three columns have `n = nation_ids.numel()` rows.
- `out` is `torch.zeros(num_nations)`, an atomic accumulator: `readwrite`.
- `nation_id` is `range(len(nations_unique))` joined onto the rows, so each value lies in `[0, num_nations)`: `nation_id_ptr` is `index-in` `out_ptr`.
