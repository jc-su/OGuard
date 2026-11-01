# q11_core_triton__q11_dense_agg_kernel

Source: `azure__gpt-5.5/triton/core/queries/q11/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q11_dense_agg_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q11_dense_agg_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_q11_dense_aggregate` launches the kernel once, when `n > 0`, with grid
`(triton.cdiv(n, 512),)`, `BLOCK_SIZE=512`, `num_warps=4`, and `num_stages=4` (the
front end takes no `num_stages`).

## Lengths and ranges

- `ps_partkey_ptr` (`int64`), `ps_supplycost_ptr`, `ps_availqty_ptr` (`float64`): the joined partsupp columns, `n_elements = ps_partkey.numel()` rows each (below 2^31, so `i32`).
- `group_values_ptr` (`float64`), `group_counts_ptr` (`int32`): `torch.zeros(dense_size)` each, added to with `tl.atomic_add` (`readwrite`). `dense_size = max(ps_partkey) - min(ps_partkey) + 1` is a size symbol that the kernel does not receive, least value 1.
- `total_ptr`: `torch.zeros(1)`, one element, added to with `tl.atomic_add` (`readwrite`).
- `base_key`: `min(ps_partkey)`, a Python `int` below 2^31 (`i32`), the offset of the dense index; `ps_partkey_ptr` is `read` since the plan offsets its values before indexing.
- `n_elements`: size symbol with least value 1; the launcher returns before the launch when `n == 0`.
