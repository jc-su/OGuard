# q12_cuda_q12_kernel

`kernel.cu` is the kernel `q12_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/full/queries/q12/round_05.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `q12_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cstdint>`, `#define BLOCK_SIZE 1024` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding).
- The kernel's body and parameters are the plan's.

## Launch

`q12_launcher` launches `q12_kernel<<<(n + 1023) / 1024, 1024>>>` once, with no dynamic shared memory.
`launch.json` states the block (1024, 1, 1) and 0 bytes of dynamic shared memory.

## Lengths and ranges

- `shipmode_code_ptr` (read, length `n_elements`): a 0/1 code of each joined row; `n_elements` is `shipmode_code.numel()`.
- `priority_high_ptr` (read, length `n_elements`): a 0/1 code of each joined row; `n_elements` is `shipmode_code.numel()`.
- `l_commitdate_ptr` (read, length `n_elements`): a date column in days, one per joined row.
- `l_receiptdate_ptr` (read, length `n_elements`): a date column in days, one per joined row.
- `l_shipdate_ptr` (read, length `n_elements`): a date column in days, one per joined row.
- `receipt_lo_days` (scalar): from the query parameter; no range.
- `receipt_hi_days` (scalar): from the query parameter; no range.
- `high0_out_ptr` (readwrite, length `1`): `torch::zeros({1}, int64)` in the launcher; thread 0 of each block adds to it with `atomicAdd`.
- `high1_out_ptr` (readwrite, length `1`): `torch::zeros({1}, int64)` in the launcher; thread 0 of each block adds to it with `atomicAdd`.
- `low0_out_ptr` (readwrite, length `1`): `torch::zeros({1}, int64)` in the launcher; thread 0 of each block adds to it with `atomicAdd`.
- `low1_out_ptr` (readwrite, length `1`): `torch::zeros({1}, int64)` in the launcher; thread 0 of each block adds to it with `atomicAdd`.
- `n_elements` (size): `shipmode_code.numel()`.

Least values of the size symbols: `n_elements` 0. The plan launches on whatever rows its joins and filters give, which can be none.
