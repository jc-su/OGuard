# q09_core_triton__profit_kernel

The kernel `_profit_kernel` of the Triton plan for q09 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/core/queries/q09/round_03.py`,
copied as `plan.py`), `BLOCK_SIZE = 8192`, `num_warps = 4`.

## Extraction edits

- `_profit_kernel` is renamed `kernel`.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `extendedprice_ptr: *fp64`, `discount_ptr: *fp64`, `supplycost_ptr: *fp64`, `quantity_ptr: *fp64`, `out_ptr: *fp64`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 56: `_K._profit_kernel[grid](..., BLOCK_SIZE=8192)` in `_launch_profit_kernel`, with `grid = (triton.cdiv(n, 8192),)`, `BLOCK_SIZE = 8192`, and the default `num_warps = 4`.

## Lengths and ranges

- The four columns are columns of the joined table, of `n = l_extendedprice.numel()` rows; `out` is `torch.empty(n)`, written once per row: `write`.
