# q15_core_cuda_q15_aggregate_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q15/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q15_aggregate_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q15_aggregate_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the device function `q15_find_supplier_index`; left out `<math_constants.h>` and the macro `BLOCK_SIZE` (not used by the kernel), `<vector>`, the CUDA runtime include, the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q15_launcher` launches the kernel once, when `n_line > 0` and `n_sup > 0`, with
`<<<(n_line + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>`: block
(256, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `l_suppkey` (`int64`), `l_extendedprice`, `l_discount` (`float64`), `l_shipdate_days` (`int32`): the lineitem columns, `n_line = l_suppkey.numel()` rows each (checked by the launcher).
- `s_suppkey`: the supplier keys, `n_sup = s_suppkey.numel()` rows; read only (the kernel bounds every index into it).
- `totals`: `torch::zeros({n_sup})`, added to with `atomicAdd` (`readwrite`); `present`: `torch::zeros({n_sup})` of `uint8`, written only.
- `n_line`, `n_sup`: size symbols with least value 1; the launch runs only when both are positive.
- `shipdate_lo_days`, `shipdate_hi_days`: comparison thresholds only.
