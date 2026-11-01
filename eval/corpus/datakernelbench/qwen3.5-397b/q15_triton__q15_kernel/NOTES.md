# q15_triton__q15_kernel

The kernel `_q15_kernel` of the Triton plan for q15 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/full/queries/q15/round_08.py`,
copied as `plan.py`): `BLOCK_SIZE = 8192`, `num_warps = 4`.

## Extraction edits

- `_q15_kernel` is renamed `kernel`.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `shipdate_ptr: *i32`, `price_ptr: *fp64`, `disc_ptr: *fp64`, `supp_ptr: *i64`, `totals_ptr: *fp64`, `date_lo: i32`, `date_hi: i32`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 108: `_K._q15_kernel[grid](..., BLOCK_SIZE=BLOCK_SIZE)` in `run_query`, with `grid = (triton.cdiv(n_elements, 8192),)`, `BLOCK_SIZE = 8192`, and the default `num_warps = 4`.

## Lengths and ranges

- The four columns are lineitem columns of `n_elements = l_suppkey_t.numel()` rows.
- `totals` is `torch.zeros(n_sup + 1)` with `n_sup` the rows of supplier: `n_suppliers + 1` elements, an atomic accumulator (`readwrite`).
- The plan indexes `totals` by `l_suppkey`, relying on TPC-H's supplier keys being `1..n_sup` (its comment says so); it does not establish that range, so `supp_ptr` takes the role `read`.
