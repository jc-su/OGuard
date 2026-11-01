# q08_cuda_q8_year_kernel

`kernel.cu` is the kernel `q8_year_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/full/queries/q08/round_01.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `q8_year_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cuda_runtime.h>`, `#include <cstdint>`, `#define MAX_GROUPS 64` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding); and the plan's other kernels (`q8_kernel`, `q8_volume_kernel`).
- The kernel's body and parameters are the plan's.

## Launch

`q8_launcher` launches `q8_year_kernel<<<(n_elements + 1023) / 1024, 1024, MAX_GROUPS * 2 * sizeof(double)>>>` once: 1024 threads and 1024 bytes of dynamic shared memory, which the kernel does not use (its shared arrays are static).
`launch.json` states the block (1024, 1, 1) and 1024 bytes of dynamic shared memory.

## Lengths and ranges

- `o_year_ptr` (read, length `n_elements`): the order year of each joined row.
- `volume_ptr` (read, length `n_elements`): the `volume` tensor that `q8_volume_kernel` wrote, `torch::empty({n_elements})`.
- `is_brazil_ptr` (read, length `n_elements`): `(li["n_name"] == nation_name).astype("int32")`, one per row.
- `sum_brazil_out` (readwrite, length `num_groups`): `torch.zeros(num_groups)`; the kernel adds to it with `atomicAdd`.
- `sum_total_out` (readwrite, length `num_groups`): `torch.zeros(num_groups)`; the kernel adds to it with `atomicAdd`.
- `unique_years` (read, length `num_groups`): `torch.unique(o_year_t)`, whose length is `num_groups`.
- `num_groups` (size): `unique_years.shape[0]`; the plan does not bound it.
- `n_elements` (size): `o_year_t.shape[0]`.

Least values of the size symbols: `num_groups` 0, `n_elements` 0. The plan launches on whatever rows its joins and filters give, which can be none.
