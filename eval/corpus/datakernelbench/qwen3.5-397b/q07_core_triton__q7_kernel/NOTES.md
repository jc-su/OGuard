# q07_core_triton__q7_kernel

The kernel `_q7_kernel` of the Triton plan for q07 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/core/queries/q07/round_03.py`,
copied as `plan.py`), `BLOCK_SIZE = 4096`, `num_warps = 4`.

## Extraction edits

- `_q7_kernel` is renamed `kernel`.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `supp_ptr: *i32`, `cust_ptr: *i32`, `shipdate_ptr: *i32`, `price_ptr: *fp64`, `discount_ptr: *fp64`, `date_lo: i32`, `date_hi: i32`, `out_ptr: *fp64`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 103: `_K._q7_kernel[grid](..., BLOCK_SIZE=BLOCK_SIZE)` in `_launch_q7`, with `grid = (triton.cdiv(n, 4096),)`, `BLOCK_SIZE = 4096`, and the default `num_warps = 4`.

## Lengths and ranges

- The five columns are columns of the joined table, of `n = l_extendedprice.numel()` rows.
- `supp_ptr` and `cust_ptr` are 0/1 flags (`(supp_nation == nation_b)` cast to int32); no argument role states that range.
- `out` is `torch.zeros(8)`, an atomic accumulator: `readwrite`.
