# q08_core_triton__q8_reduce_1995_1996_kernel

Source: `azure__gpt-5.5/triton/core/queries/q08/round_02.py` (`output_round` 2),
copied as `plan.py`. The kernel is `_q8_reduce_1995_1996_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q8_reduce_1995_1996_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_query_core` launches the kernel once, when `n > 0`, with grid
`(triton.cdiv(n, 2048),)`, `BLOCK_SIZE=2048`, and no `num_warps` (4).

## Lengths and ranges

- `o_year_ptr` (`int32`), `volume_ptr` (`float64`), `is_brazil_ptr` (a `torch.bool` tensor, `*i1`): the joined columns, `n_elements = volume.numel()` rows each (below 2^31, so `i32`).
- `agg_ptr`: `torch.zeros(4, dtype=torch.float64)`, 4 elements, added to with `tl.atomic_add` (`readwrite`).
- `n_elements`: size symbol with least value 1; `_query_core` returns before the launch when `n == 0`.
