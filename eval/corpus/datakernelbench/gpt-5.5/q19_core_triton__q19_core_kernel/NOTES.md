# q19_core_triton__q19_core_kernel

Source: `azure__gpt-5.5/triton/core/queries/q19/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q19_core_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q19_core_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_q19_core` launches the kernel once, when `n > 0`, with grid
`(triton.cdiv(n, 2048),)`, `BLOCK_SIZE=2048`, `num_warps=8`, and `num_stages=4` (the
front end takes no `num_stages`).

## Lengths and ranges

- `quantity_ptr`, `extendedprice_ptr`, `discount_ptr` (`float64`), `brand_id_ptr`, `container_grp_ptr`, `size_ptr` (`int32`): the joined columns, `n_elements = l_quantity.numel()` rows each (below 2^31, so `i32`).
- `out_ptr`: `torch.zeros(1, dtype=torch.float64)`, one element, added to with `tl.atomic_add` (`readwrite`).
- `n_elements`: size symbol with least value 1; the launcher returns before the launch when `n == 0`.
- `brand12_id`, `brand23_id`, `brand34_id`, `sm_grp`, `md_grp`, `lg_grp`: the Python `int` codes 12, 23, 34, 1, 2, 3 of `run_query`, `i32`; compared only.
