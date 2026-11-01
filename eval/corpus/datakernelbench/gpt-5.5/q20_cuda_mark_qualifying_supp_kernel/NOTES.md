# q20_cuda_mark_qualifying_supp_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q20/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `mark_qualifying_supp_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `mark_qualifying_supp_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`; left out the other includes, the macros, the
  device functions (not used by the kernel), and the host code.

## Launch

`q20_launcher` launches the kernel once with
`<<<(nps + 255) / 256, BLOCK_SIZE, 0, stream>>>`: block (256, 1, 1), no dynamic
shared memory.

## Lengths and ranges

- `ps_suppkey`, `ps_availqty`: `nps` rows each, the filtered partsupp; the
  launcher checks `ps_availqty.numel() == nps`. `ps_sum_qty`:
  `torch::zeros({nps})`.
- `nps`: size symbol with least value 1; the launcher returns for `nps == 0`.
- `supp_mask`: `torch::zeros({max_suppkey + 1})` of uint8; the kernel only
  stores 1 into it (`write`). Its length is `max_suppkey + 1`.
- `max_suppkey`: `int(df_supplier["s_suppkey"].max())`, passed as the
  kernel's `max_suppkey`; size symbol with least value 0, since the launcher
  returns for `max_suppkey < 0`.
