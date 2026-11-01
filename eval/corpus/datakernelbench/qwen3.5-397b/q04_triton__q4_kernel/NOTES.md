# q04_triton__q4_kernel

The kernel `_q4_kernel` of the Triton plan for q04 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/full/queries/q04/round_07.py`,
copied as `plan.py`): `BLOCK_SIZE = 4096`, `num_warps = 4`.

## Extraction edits

- `_q4_kernel` is renamed `kernel`.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `priority_ptr: *i32`, `filter_mask_ptr: *i32`, `counts_ptr: *i64`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 122: `_K._q4_kernel[grid](..., BLOCK_SIZE=BLOCK_SIZE)` in `_query_core`, when `n_filtered > 0`, with `grid = (triton.cdiv(n_elements, 4096),)`, `BLOCK_SIZE = 4096`, and the default `num_warps = 4`.

## Lengths and ranges

- `o_orderpriority_codes` and `filter_mask` have `n_elements = o_orderkey.numel()` rows; the launch happens only when some row is selected, so `n_elements` is at least 1.
- `counts` is `torch.zeros(num_priorities)`, with `num_priorities` the number of categories of `o_orderpriority`, at least 1 for a non-empty orders table; the kernel adds into it atomically: `readwrite`. The plan does not check that `num_priorities` is 5, the value of TPC-H's data.
