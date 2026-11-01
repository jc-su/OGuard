# q03_triton_fused_filter_revenue_kernel

Kernel `fused_filter_revenue_kernel` of the plan `logs/kernel_plans/tpch/sf10/h100/mistralai__devstral-2-123b-instruct-2512/triton/full/queries/q03/round_07.py` (DataKernelBench commit `43c361d`; `results.json` has `"correct": true` and `"output_round": 7`), copied here as `plan.py`.

Configuration: `BLOCK_SIZE=2048`, `num_warps=4` (the launch's `BLOCK_SIZE` and Triton's default `num_warps`).

## Extraction edits

- Renamed the kernel `fused_filter_revenue_kernel` to `kernel`.
- Set `CONSTANTS = {"BLOCK_SIZE": 2048}` from the launch's `BLOCK_SIZE` and `NUM_WARPS = 4`, Triton's default, since the launch leaves `num_warps` unset.
- Added `TYPES`, the Triton type of each argument's value at the plan's launch.
- Left out the host code.

## Launch

`_launch_fused_filter_agg` launches `fused_filter_revenue_kernel[grid]` at plan lines 82-89 with `BLOCK_SIZE = 2048` (line 73), `grid = (cdiv(n, 2048),)` (line 74) and `num_warps` unset; `run_query` calls it at lines 164-174.

## Lengths and ranges

- `l_orderkey_ptr` (read, int64, length `n_elements`): `col["l_orderkey"]` (line 165); `n = l_orderkey.numel()` (line 72), and every column comes from the same `col`.
- `l_extendedprice_ptr` (read, float64, length `n_elements`): `col["l_extendedprice"]` (line 166).
- `l_discount_ptr` (read, float64, length `n_elements`): `col["l_discount"]` (line 167).
- `l_shipdate_ptr` (read, int32, length `n_elements`): `l_shipdate` in days (lines 162, 168).
- `o_orderdate_ptr` (read, int32, length `n_elements`): `o_orderdate` in days (lines 161, 169).
- `o_shippriority_ptr` (read, int32, length `n_elements`): `col["o_shippriority"]` (line 170).
- `shipdate_lo` (scalar, int32): from the date parameter `shipdate_lo` (line 155).
- `orderdate_hi` (scalar, int32): from the date parameter `orderdate_hi` (line 154).
- `n_elements` (size, int32): `l_orderkey.numel()` (line 72), below 2^31 at scale factor 10, so `i32`.
- `out_lk_ptr` (write, int64, length `n_elements`): `torch.full([n], -1, dtype=torch.int64)` (line 77); the kernel only stores into it.
- `out_od_ptr` (write, int64, length `n_elements`): `torch.full([n], 9999999, dtype=torch.int64)` (line 78).
- `out_sp_ptr` (write, int64, length `n_elements`): `torch.full([n], -1, dtype=torch.int64)` (line 79).
- `out_rev_ptr` (write, float64, length `n_elements`): `torch.full([n], 0.0)` (line 80).
- size symbol `n_elements`, least value 0.
