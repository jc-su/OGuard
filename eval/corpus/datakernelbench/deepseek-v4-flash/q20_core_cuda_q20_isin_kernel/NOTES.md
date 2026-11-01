# q20_core_cuda_q20_isin_kernel

`kernel.cu` is the kernel `q20_isin_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/core/queries/q20/round_03.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `q20_isin_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cstdint>`, `#define BLOCK_SIZE 1024` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding); the plan's other kernel (`q20_filter_kernel`).
- The kernel's body and parameters are the plan's.

## Launch

`q20_isin_launcher`, called once by `_query_core` when `q20_filter_kernel` found a qualifying row, launches `q20_isin_kernel<<<(n_s + 1023) / 1024, BLOCK_SIZE>>>` with no dynamic shared memory. `launch.json` states the block (1024, 1, 1), 0 bytes of dynamic shared memory, and the grid maxima (4194304, 1, 1): `2^32 / 1024` blocks in x, past which the unsigned 32-bit row index wraps, and one block in y and z.

## Lengths and ranges

- `s_suppkey` (read, length `n_s`): `s_suppkey` of the nation's suppliers; `n_s` is `s_suppkey.numel()`.
- `s_row_idx` (read, length `n_s`): the row number of each of the nation's suppliers; the kernel only copies it.
- `qual_suppkeys` (read, length `n_qual`): `unique_qual`, the distinct qualifying supplier keys; `n_qual` is `qual_suppkeys.numel()`.
- `n_s` (size): `s_suppkey.numel()`.
- `n_qual` (size): `qual_suppkeys.numel()`.
- `out_indices` (write, length `n_s`): `torch.zeros(s_suppkey.shape[0], int64)` in `_query_core`.
- `n_out` (readwrite, length `1`): `torch::zeros({1})` in `q20_isin_launcher`; the kernel takes output positions from it with `atomicAdd`.

Least values of the size symbols: `n_s` 0, `n_qual` 1. The plan launches only when at least one partsupp row qualifies, so `n_qual >= 1`; the nation can have no suppliers.
