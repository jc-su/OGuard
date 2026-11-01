# q15_core_triton__q15_kernel_7

The kernel `_q15_kernel` of the Triton plan for q15 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/core/queries/q15/round_04.py`,
copied as `plan.py`), in configuration 7 of 8: `BLOCK_SIZE = 8192`, `num_warps = 4`.

## Extraction edits

- `_q15_kernel` is renamed `kernel`.
- The `@triton.autotune` decorator is replaced by this configuration's values,
  `CONSTANTS` and `NUM_WARPS`. The autotuner lists this configuration with
  `num_stages` 2, 4; the front end does not take `num_stages`, and the kernel
  has no loop for the software pipeliner,
  so these give one corpus kernel.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `suppkey_ptr: *i64`, `price_ptr: *fp64`, `disc_ptr: *fp64`, `shipdate_ptr: *i32`, `suppkey_to_pos_ptr: *i32`, `rev_ptr: *fp64`, `present_ptr: *i32`, `date_lo: i32`, `date_hi: i32`, `n_suppliers: i32`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 106: `_K._q15_kernel[grid](...)` in `_query_core`, with `grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)` and the autotuner's `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- The four columns are lineitem columns of `n_elements = l_suppkey.numel()` rows.
- `suppkey_to_pos` is `torch.full((max_suppkey + 1,), -1)` with `max_suppkey = max(s_suppkey)`, holding each supplier key's position in the sorted supplier keys and `-1` elsewhere. The kernel indexes it by `l_suppkey`, whose range the plan takes from TPC-H's foreign key from lineitem to supplier without establishing it, so `suppkey_ptr` is `read`; `max_suppkey` is its own size symbol.
- `totals_sorted` and `present_sorted` are `torch.zeros(n_suppliers)`, atomic accumulators (`atomic_add`, `atomic_or`): `readwrite`; the kernel checks `pos` against `[0, n_suppliers)` itself.
