# q20_cuda_filter_supplier_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q20/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `filter_supplier_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `filter_supplier_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`; left out the other includes, the macros, the
  device functions (not used by the kernel), and the host code.

## Launch

`q20_launcher` launches the kernel once with
`<<<(ns + 255) / 256, BLOCK_SIZE, 0, stream>>>`: block (256, 1, 1), no dynamic
shared memory.

## Lengths and ranges

- `s_suppkey`: `ns` rows, the suppliers of the selected nation.
- `ns`: size symbol with least value 1; the launcher returns for `ns == 0`.
- `supp_mask`: the `max_suppkey + 1` bytes that `mark_qualifying_supp_kernel`
  wrote; read only here.
- `max_suppkey`: `int(df_supplier["s_suppkey"].max())`, size symbol with least
  value 0 (the launcher returns for `max_suppkey < 0`).
- `out_row_idx`: `torch::empty({ns})` of int64, which the kernel only stores
  into (`write`).
- `out_count`: `torch::zeros({1})` of int64, the counter that the kernel
  increments with `atomicAdd` (`readwrite`); the plan reads it back to slice
  `out_row_idx`.
