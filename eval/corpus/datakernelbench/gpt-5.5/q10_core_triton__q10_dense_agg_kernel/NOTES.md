# q10_core_triton__q10_dense_agg_kernel

Source: `azure__gpt-5.5/triton/core/queries/q10/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q10_dense_agg_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q10_dense_agg_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_q10_dense_agg` launches the kernel once, when `n > 0`, with grid
`(triton.cdiv(n, 1024),)`, `BLOCK_SIZE=1024`, and `num_warps=4`.

## Lengths and ranges

- `custkey_ptr` (`int64`), `price_ptr`, `discount_ptr` (`float64`): the joined columns, `n_elements = c_custkey.numel()` rows each (below 2^31, so `i32`).
- `revenue_ptr`: `torch.zeros(key_range, dtype=torch.float64)`, added to with `tl.atomic_add` (`readwrite`); `flag_ptr`: `torch.zeros(key_range, dtype=torch.int8)`, written only. `key_range = max(c_custkey) - min(c_custkey) + 1` is a size symbol that the kernel does not receive, least value 1.
- `key_min`: `min(c_custkey)`, a Python `int` below 2^31 (`i32`), the offset of the dense index; `custkey_ptr` is `read` since the plan offsets its values before indexing.
- `n_elements`: size symbol with least value 1; the launcher returns before the launch when `n == 0`.
