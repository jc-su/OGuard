# q12_core_cuda_q12_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q12/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q12_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q12_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`, the macro `BLOCK_SIZE`, and the device function `atomic_add_i64_nonnegative`; left out `<vector>`, the CUDA runtime include, and the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q12_launcher` launches the kernel once, when `n > 0`, with
`<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>`: block (1024, 1, 1), no
dynamic shared memory.

## Lengths and ranges

- `l_shipmode_code`, `o_orderpriority_high`, `l_commitdate`, `l_receiptdate`, `l_shipdate` (`int32`): the joined columns, `n_elements = l_shipmode_code.numel()` rows each (checked by the launcher).
- `n_elements`: size symbol with least value 1; the launch runs only when `n > 0`.
- `high_counts`, `low_counts`: the 2-element buffers that `q12_init_kernel` zeroed, which the kernel adds to with `atomicAdd` (`readwrite`).
- `receipt_lo_days`, `receipt_hi_days`: comparison thresholds only.
