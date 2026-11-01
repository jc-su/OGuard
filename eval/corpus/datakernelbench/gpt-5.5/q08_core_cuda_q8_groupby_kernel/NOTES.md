# q08_core_cuda_q8_groupby_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q08/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q8_groupby_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q8_groupby_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macros `BLOCK_SIZE`, `YEAR_BASE`, and `NUM_YEARS`; left out `<vector>`, the CUDA runtime include, and the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q8_groupby_launcher` launches the kernel once, when `n > 0`, with
`<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, shmem_bytes, stream>>>`,
`shmem_bytes = 2 * NUM_YEARS * sizeof(double)` = 3216: block (256, 1, 1), dynamic
shared memory 3216 bytes (`launch.json` constant).

## Lengths and ranges

- `o_year` (`int32`), `volume` (`float64`), `is_brazil` (`bool`): the joined columns, `n_elements = volume.numel()` rows each (checked by the launcher).
- `n_elements`: size symbol with least value 1; the launch runs only when `n > 0`.
- `sums`: `torch::zeros({2, NUM_YEARS})`, 402 contiguous doubles, which the kernel adds to with `atomicAdd` (`readwrite`).
