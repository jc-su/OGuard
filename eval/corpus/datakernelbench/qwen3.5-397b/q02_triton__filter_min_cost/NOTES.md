# q02_triton__filter_min_cost

The kernel `_filter_min_cost` of the Triton plan for q02 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/full/queries/q02/round_05.py`,
copied as `plan.py`): `BLOCK_SIZE = 1024`, `num_warps = 4`.

## Extraction edits

- `_filter_min_cost` is renamed `kernel`.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `partkey_ptr: *i64`, `cost_ptr: *fp64`, `min_cost_ptr: *fp64`, `keep_mask_ptr: *i32`, `n: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 138: `_K._filter_min_cost[grid](..., BLOCK_SIZE=BLOCK_SIZE)` in `run_query`, with `grid = (triton.cdiv(n, 1024),)`, `BLOCK_SIZE = 1024`, and the default `num_warps = 4`.

## Lengths and ranges

- `pk_t` and `cost_t` are columns of the joined table `df_w`, of `n = len(df_w)` rows; the plan returns before the launches when `n == 0`.
- `min_cost_buf` is `torch.ones(max_pk) * 1e30` with `max_pk = pk_t.max() + 2`: the plan sizes it from the largest partkey, so `partkey_ptr` is `index-in` `min_cost_ptr` (partkeys are TPC-H's non-negative keys). Its length is `max_partkey + 2`, with `max_partkey` the largest partkey.
- `keep_mask` is `torch.zeros(n)`; the kernel stores every position it covers, without reading it: `write`.
