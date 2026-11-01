# q05_core_triton__q5_group_reduce_kernel

Source: `azure__gpt-5.5/triton/core/queries/q05/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q5_group_reduce_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q5_group_reduce_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_q5_group_reduce` launches the kernel once, when `n_elements > 0` and
`num_nations > 0`, with grid `(triton.cdiv(n_elements, 2048),)`, `BLOCK_SIZE=2048`,
`NUM_NATIONS=num_nations` (a `tl.constexpr`, the 5 nations of ASIA), and
`num_warps=8`.

## Lengths and ranges

- `nation_id_ptr` (`int64`), `extendedprice_ptr`, `discount_ptr` (`float64`): the joined columns, `n_elements = l_extendedprice.numel()` rows each (below 2^31, so `i32`); `nation_id_ptr` is compared with the constants only, so `read`.
- `out_ptr`: `torch.zeros(num_nations, dtype=torch.float64)`, 5 elements, added to with `tl.atomic_add` (`readwrite`).
- `n_elements`: size symbol with least value 1; the launcher returns before the launch when it is 0.
