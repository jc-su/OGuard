# q13_cuda_q13_gather_kernel

`kernel.cu` is the kernel `q13_gather_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/full/queries/q13/round_02.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `q13_gather_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cuda_runtime.h>`, `#include <cstdint>` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding); and the plan's other kernels (`q13_groupby_kernel`, `q13_histogram_kernel`).
- The kernel's body and parameters are the plan's.

## Launch

`q13_launcher` launches `q13_gather_kernel<<<(n_cust + 1023) / 1024, 1024>>>` once, with no dynamic shared memory.
`launch.json` states the block (1024, 1, 1) and 0 bytes of dynamic shared memory.

## Lengths and ranges

- `c_custkey_ptr` (read, length `n_elements`): `c_custkey`, one per customer; `n_elements` is `c_custkey.size(0)`. The largest value is `max_custkey`, but the plan does not check that the keys are non-negative, so no index role.
- `counts_ptr` (read, length `max_custkey + 1`): `order_counts`, `torch::zeros({num_customers + 1})` with `num_customers = max(c_custkey)`, filled by `q13_histogram_kernel`.
- `c_count_ptr` (write, length `n_elements`): `torch::zeros({n_cust})`.
- `n_elements` (size): `c_custkey.size(0)`.

Least values of the size symbols: `n_elements` 0, `max_custkey` 0. The plan launches on whatever rows its joins and filters give, which can be none.
