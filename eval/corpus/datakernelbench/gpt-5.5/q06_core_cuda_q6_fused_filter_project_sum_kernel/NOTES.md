# q06_core_cuda_q6_fused_filter_project_sum_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q06/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q6_fused_filter_project_sum_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q6_fused_filter_project_sum_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macro `BLOCK_SIZE`; left out `<algorithm>`, the CUDA runtime include, the host helper `ensure_cuda_contiguous`, and the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q6_launcher` launches the kernel once, when `n > 0`, with
`<<<min((n + BLOCK_SIZE - 1) / BLOCK_SIZE, max(1, 8 * multiProcessorCount)), BLOCK_SIZE, 0, stream>>>`:
block (256, 1, 1), no dynamic shared memory, a grid-stride loop over the rows.

## Lengths and ranges

- `shipdate_ptr` (`int32`), `discount_ptr`, `quantity_ptr`, `price_ptr` (`float64`): `n_elements` rows each; the launcher sets `n = extendedprice.numel()` and checks the other three columns against it.
- `n_elements`: size symbol with least value 1; the launcher returns before the launch when `n == 0`.
- `out_ptr`: `torch::zeros({1})`, one element, which the kernel adds to with `atomicAdd` (`readwrite`).
- `date_lo`, `date_hi`, `disc_lo`, `disc_hi`, `qty_hi`: comparison thresholds only.
