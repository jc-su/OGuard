# q22_core_cuda_q22_filter_kernel

`kernel.cu` is the kernel `q22_filter_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/core/queries/q22/round_03.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `q22_filter_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cuda_runtime.h>`, `#define BLOCK_SIZE 1024` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding); the plan's other kernels (`q22_avg_kernel`, `q22_groupby_kernel`).
- The kernel's body and parameters are the plan's.

## Launch

`q22_filter_launcher`, called once by `_query_core`, launches `q22_filter_kernel<<<(n + 1023) / 1024, BLOCK_SIZE>>>` with no dynamic shared memory. `launch.json` states the block (1024, 1, 1) and 0 bytes of dynamic shared memory.

## Lengths and ranges

- `custkey_ptr` (read, length `n`): `c_custkey`, one per customer; `n` is `custkey.numel()`.
- `acctbal_ptr` (read, length `n`): `c_acctbal`, one per customer.
- `cntrycode_ptr` (read, length `n`): the two-digit country code of each customer.
- `in_cntrycode_mask_ptr` (read, length `n`): `in_cntrycode_mask.to(int32)`, one per customer.
- `sorted_okeys_ptr` (read, length `n_okeys`): the distinct `o_custkey` values, sorted; `n_okeys` is `sorted_okeys.numel()`.
- `n_okeys` (size): `sorted_okeys.numel()`.
- `avg_val` (scalar): `avg_sum / avg_count` of `q22_avg_kernel`, or 0; no range.
- `out_cntrycodes` (write, length `n`): `torch.zeros(n, int32)` in `_query_core`.
- `out_acctbals` (write, length `n`): `torch.zeros(n, float64)` in `_query_core`.
- `out_count` (readwrite, length `1`): `torch.zeros(1, int32)`; the kernel takes output positions from it with `atomicAdd`.
- `n` (size): `custkey.numel()`.

Least values of the size symbols: `n` 0, `n_okeys` 0. The plan launches on the whole `customer` table, which can be empty; the orders can be empty too.
