# q07_cuda_q7_kernel

`kernel.cu` is the kernel `q7_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/full/queries/q07/round_01.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `q7_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cuda_runtime.h>`, `#define MAX_GROUPS 8` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API) and `#include <cuda.h>` (the CUDA driver API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding).
- The kernel's body and parameters are the plan's.

## Launch

`q7_launcher`, called once by `q7_full_launcher`, launches `q7_kernel<<<(n + 1023) / 1024, 1024, MAX_GROUPS * sizeof(double)>>>`: 1024 threads and 64 bytes of dynamic shared memory.
`launch.json` states the block (1024, 1, 1) and 64 bytes of dynamic shared memory.

## Lengths and ranges

- `supp_nation` (read, length `n_elements`): the 0/1 nation code `(joined[...] == nation_b).astype("int32")` of each joined row; the kernel checks the group key itself, so no index role.
- `cust_nation` (read, length `n_elements`): the 0/1 nation code `(joined[...] == nation_b).astype("int32")` of each joined row; the kernel checks the group key itself, so no index role.
- `shipdate_days` (read, length `n_elements`): `l_shipdate` in days, one per joined row.
- `price` (read, length `n_elements`): a column of the joined frame, one per row; `n_elements` is `price.numel()`.
- `discount` (read, length `n_elements`): a column of the joined frame, one per row; `n_elements` is `price.numel()`.
- `date_lo` (scalar): from the query parameter; no range.
- `date_hi` (scalar): from the query parameter; no range.
- `revenue_groups` (readwrite, length `8`): `torch::zeros({MAX_GROUPS})`, `MAX_GROUPS = 8`, in `q7_full_launcher`; the kernel adds to it with `atomicAdd`.
- `n_elements` (size): `p.numel()`.

Least values of the size symbols: `n_elements` 0. The plan launches on whatever rows its joins and filters give, which can be none.
