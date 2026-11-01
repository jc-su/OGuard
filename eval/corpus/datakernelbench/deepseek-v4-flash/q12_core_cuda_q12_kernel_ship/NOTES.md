# q12_core_cuda_q12_kernel_ship

`kernel.cu` is the kernel `q12_kernel_ship` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/core/queries/q12/round_03.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `q12_kernel_ship` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cstdint>`, `#define BLOCK_SIZE 1024` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding); the plan's other kernel (`q12_kernel_mail`).
- The kernel's body and parameters are the plan's.

## Launch

`q12_launcher`, called once by `_query_core`, launches `q12_kernel_ship<<<(n + 1023) / 1024, BLOCK_SIZE>>>` with no dynamic shared memory, on the same arguments as the other kernel. `launch.json` states the block (1024, 1, 1) and 0 bytes of dynamic shared memory.

## Lengths and ranges

- `shipmode_ptr` (read, length `n_elements`): the 0/1 ship mode code `(joined["l_shipmode"] == shipmode2).astype("int32")` of each joined row; `n_elements` is `shipmode.numel()`. The kernel only compares it.
- `priority_ptr` (read, length `n_elements`): the 0/1 high-priority flag of each joined row; the kernel only tests it.
- `commit_ptr` (read, length `n_elements`): `l_commitdate` in days, one per joined row.
- `receipt_ptr` (read, length `n_elements`): `l_receiptdate` in days, one per joined row.
- `ship_ptr` (read, length `n_elements`): `l_shipdate` in days, one per joined row.
- `receipt_lo` (scalar): from the query parameter; no range.
- `receipt_hi` (scalar): from the query parameter; no range.
- `high_out` (readwrite, length `2`): `torch.zeros(2, int64)` in `_query_core`, passed as `unsigned long long*`; the kernel adds to it with `atomicAdd`.
- `low_out` (readwrite, length `2`): `torch.zeros(2, int64)` in `_query_core`, passed as `unsigned long long*`; the kernel adds to it with `atomicAdd`.
- `n_elements` (size): `shipmode.numel()`.

Least values of the size symbols: `n_elements` 0. The plan launches on whatever rows its joins and filters give, which can be none.
