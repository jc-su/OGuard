# q15_cuda_q15_kernel

`kernel.cu` is the kernel `q15_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/full/queries/q15/round_04.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `q15_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cuda_runtime.h>`, `#include <cstdint>` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding).
- The kernel's body and parameters are the plan's.

## Launch

The `q15_launcher` lambda launches `q15_kernel<<<(n + 255) / 256, 256>>>` once, with no dynamic shared memory.
`launch.json` states the block (256, 1, 1) and 0 bytes of dynamic shared memory.

## Lengths and ranges

- `price_ptr` (read, length `n_elements`): a `lineitem` column, one per row; `n_elements` is `price.numel()`.
- `discount_ptr` (read, length `n_elements`): a `lineitem` column, one per row; `n_elements` is `price.numel()`.
- `shipdate_ptr` (read, length `n_elements`): `l_shipdate` in days, one per row.
- `l_suppkey_ptr` (read, length `n_elements`): `l_suppkey`, one per row.
- `sorted_supplier_ptr` (read, length `n_suppliers`): `torch.sort(s_suppkey)`; `n_suppliers` is its `numel()`.
- `date_lo` (scalar): from the query parameter; no range.
- `date_hi` (scalar): from the query parameter; no range.
- `n_elements` (size): `price.numel()`.
- `n_suppliers` (size): `sorted_supplier.numel()`.
- `out_totals` (readwrite, length `n_suppliers`): `torch.zeros(n_suppliers)` with `n_suppliers = len(s_suppkey)`; the kernel adds to it with `atomicAdd`.

Least values of the size symbols: `n_elements` 0, `n_suppliers` 0. The plan launches on whatever rows its joins and filters give, which can be none.
