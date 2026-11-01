# q01_core_cuda_q1_aggregate_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q01/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q1_aggregate_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q1_aggregate_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macros `BLOCK_SIZE`, `MAX_RF`, `MAX_LS`, and `MAX_GROUPS`; left out the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q1_launcher` launches the kernel once, when `n > 0`, with
`<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>`: block (256, 1, 1),
no dynamic shared memory.

## Lengths and ranges

- `rf_ptr`, `ls_ptr`, `shipdate_ptr` (`int32`), `qty_ptr`, `price_ptr`, `disc_ptr`, `tax_ptr` (`float64`): `n_elements` rows each; the launcher sets `n = l_quantity.numel()` and checks the other six columns against it.
- `n_elements`: size symbol with least value 1; the launcher launches only when `n > 0`.
- `g_sum_qty`, `g_sum_base_price`, `g_sum_disc_price`, `g_sum_charge`, `g_sum_disc`, `g_count`: `torch::zeros({MAX_GROUPS})`, 256 elements each, which the kernel adds to with `atomicAdd` (`readwrite`).
- `shipdate_hi_days`: a comparison threshold only; the plan states no range.
