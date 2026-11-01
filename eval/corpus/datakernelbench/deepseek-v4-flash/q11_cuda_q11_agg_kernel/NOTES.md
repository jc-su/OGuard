# q11_cuda_q11_agg_kernel

`kernel.cu` is the kernel `q11_agg_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/full/queries/q11/round_03.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `q11_agg_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cuda_runtime.h>`, `#include <stdint.h>`, `#define BLOCK_SIZE 1024` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding).
- The kernel's body and parameters are the plan's.

## Launch

`q11_launcher` launches `q11_agg_kernel<<<(n + 1023) / 1024, 1024>>>` once, with no dynamic shared memory.
`launch.json` states the block (1024, 1, 1) and 0 bytes of dynamic shared memory.

## Lengths and ranges

- `ps_partkey` (read, length `n`): `ps_partkey` of the joined frame, one per row; `n` is `ps_partkey.numel()`.
- `ps_supplycost` (read, length `n`): a column of the joined frame, one per row.
- `ps_availqty` (read, length `n`): a column of the joined frame, one per row.
- `unique_keys_out` (write, length `n`): `torch::empty({n})` in the launcher.
- `grouped_values_out` (write, length `n`): `torch::zeros({n})` in the launcher.
- `key_count_out` (readwrite, length `1`): `torch::zeros({1}, int64)`; the kernel takes output positions from it with `atomicAdd`.
- `total_rev_ptr` (readwrite, length `1`): `torch::zeros({1})`; the kernel adds to it with `atomicAdd`.
- `n` (size): `ps_partkey.numel()`.

Least values of the size symbols: `n` 0. The plan launches on whatever rows its joins and filters give, which can be none.
