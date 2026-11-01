# q22_cuda_q22_filter_kernel

`kernel.cu` is the kernel `q22_filter_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/full/queries/q22/round_03.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `q22_filter_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cuda_runtime.h>`, `#define BLOCK_SIZE 1024` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API) and `#include <cuda.h>` (the CUDA driver API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding); and the plan's other kernels (`q22_avg_kernel`).
- The kernel's body and parameters are the plan's.

## Launch

`q22_launcher` launches `q22_filter_kernel<<<(n + 1023) / 1024, 1024>>>` once, with no dynamic shared memory, after `q22_avg_kernel`.
`launch.json` states the block (1024, 1, 1) and 0 bytes of dynamic shared memory.

## Lengths and ranges

- `custkey` (read, length `n`): `c_custkey`, one per customer; `n` is `custkey.size(0)`.
- `acctbal` (read, length `n`): `c_acctbal`, one per customer.
- `cntrycode` (read, length `n`): the country code of each customer's phone.
- `in_mask` (read, length `n`): `in_cntrycode_mask`, one per customer.
- `sorted_okeys` (read, length `okeys_size`): `torch.sort(o_custkey_unique)`; `okeys_size` is its `size(0)`.
- `okeys_size` (size): `sorted_okeys.size(0)`.
- `avg_val` (scalar): `sum_out / count_out` of `q22_avg_kernel`.
- `out_code` (write, length `n`): `torch::empty({n}, int32)`.
- `out_bal` (write, length `n`): `torch::empty({n})`.
- `out_n` (readwrite, length `1`): `torch::zeros({1}, int64)`; thread 0 of each block takes the block's output offset from it with `atomicAdd`.
- `n` (size): `custkey.size(0)`.

Least values of the size symbols: `n` 0, `okeys_size` 0. The plan launches on whatever rows its joins and filters give, which can be none.
