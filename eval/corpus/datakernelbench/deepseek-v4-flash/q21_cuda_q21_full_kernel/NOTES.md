# q21_cuda_q21_full_kernel

`kernel.cu` is the kernel `q21_full_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/full/queries/q21/round_09.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `q21_full_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cuda_runtime.h>` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API) and `#include <cuda.h>` (the CUDA driver API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding).
- The kernel's body and parameters are the plan's.

## Launch

`q21_launcher` launches `q21_full_kernel<<<min((n_elements + 255) / 256, 65535), 256>>>` once, with no dynamic shared memory.
`launch.json` states the block (256, 1, 1) and 0 bytes of dynamic shared memory.

## Lengths and ranges

- `l1_orderkey` (read, length `n_elements`): `l_orderkey` of each row of `l1`; `n_elements` is `l1_orderkey.numel()`.
- `l1_late_int` (read, length `n_elements`): `l1_late.to(torch.int64)`, one per row.
- `supp_ids` (index-in `counts`, length `n_elements`): `supp_id` of each row of `l1`: the row number `range(len(supp_n))` of `supp_n`, joined by `s_suppkey`, so every value lies in `[0, num_suppliers)`, the length of `counts`.
- `lo_orderkeys` (read, length `n_lo`): `agg["l_orderkey"]`; `n_lo` is `lo_orderkeys.numel()`.
- `lo_distinct_supp` (read, length `n_lo`): a column of `agg`, one per order.
- `lo_late_distinct_supp` (read, length `n_lo`): a column of `agg`, one per order.
- `counts` (readwrite, length `num_suppliers`): `torch::zeros({num_suppliers})` with `num_suppliers = len(supp_n)`; the kernel adds to it with `atomicAdd`.
- `n_elements` (size): `l1_orderkey.numel()`.
- `n_lo` (size): `lo_orderkeys.numel()`.

Least values of the size symbols: `n_elements` 0, `n_lo` 0, `num_suppliers` 0. The plan launches on whatever rows its joins and filters give, which can be none.
