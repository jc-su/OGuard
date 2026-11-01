# q04_core_triton__max_i64_kernel

Source: `azure__gpt-5.5/triton/core/queries/q04/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_max_i64_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_max_i64_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_triton_max_i64` launches the kernel in the loop `while cur_n > 1`, each time with
grid `(triton.cdiv(cur_n, 1024),)`, `BLOCK_SIZE=1024`, and no `num_warps` (4). Every
launch of the loop has this configuration, so it is one corpus kernel. At scale
factor 10 (59,986,052 lineitem rows) the loop launches it three times, for
59,986,052, 58,581, and 58 values.

## Lengths and ranges

- `in_ptr`: `l_orderkey` at the first launch and the previous launch's `partial` afterwards, `n_elements` elements; read only.
- `n_elements`: `cur_n`, size symbol with least value 2; the loop launches only while `cur_n > 1`. Below 2^31, so `i32`.
- `out_ptr`: `torch.empty(n_blocks, dtype=torch.int64)`, one element per program of the plan's grid, `(n_elements + 1023) / 1024`; the quotient gives the buffer its own size symbol with least value 1. Written only.
