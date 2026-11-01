# q03_core_triton__filter_kernel

The kernel `_filter_kernel` of the Triton plan for q03 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/core/queries/q03/round_02.py`,
copied as `plan.py`), `BLOCK_SIZE = 4096`, `num_warps = 4`.

## Extraction edits

- `_filter_kernel` is renamed `kernel`.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `l_shipdate_ptr: *i32`, `o_orderdate_ptr: *i32`, `shipdate_lo: i32`, `orderdate_hi: i32`, `out_valid_ptr: *i8`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 67: `_K._filter_kernel[grid](..., BLOCK_SIZE=BLOCK_SIZE)` in `_query_core`, with `grid = (triton.cdiv(n, 4096),)`, `BLOCK_SIZE = 4096`, and the default `num_warps = 4`.

## Lengths and ranges

- The two date columns are columns of the joined table `col`, of `n = l_orderkey.numel()` rows; `valid_mask` is `torch.empty(n, dtype=torch.int8)`, written once per row: `write`.
