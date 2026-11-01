# q09_triton__q9_amount_kernel

The kernel `_q9_amount_kernel` of the Triton plan for q09 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/full/queries/q09/round_06.py`,
copied as `plan.py`): `BLOCK_SIZE = 16384`, `num_warps = 4`.

## Extraction edits

- `_q9_amount_kernel` is renamed `kernel`.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `ext_price_ptr: *fp64`, `discount_ptr: *fp64`, `supply_cost_ptr: *fp64`, `quantity_ptr: *fp64`, `out_ptr: *fp64`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 64: `_K._q9_amount_kernel[grid](...)` in `_query_core`, with `grid = (triton.cdiv(n, 16384),)`, the default `BLOCK_SIZE = 16384` of the kernel's signature, and the default `num_warps = 4`.

## Lengths and ranges

- The four columns have `n = l_extendedprice.numel()` rows.
- `out` is `torch.empty(n)`, which the kernel only writes: `write`.
