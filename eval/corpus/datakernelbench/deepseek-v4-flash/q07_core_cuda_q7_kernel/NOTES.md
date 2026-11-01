# q07_core_cuda_q7_kernel

`kernel.cu` is the kernel `q7_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/core/queries/q07/round_02.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `q7_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cuda_runtime.h>`, `#include <cstdint>`, `#define BLOCK_SIZE 1024`, `#define NUM_GROUPS 8` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding).
- The kernel's body and parameters are the plan's.

## Launch

`q7_launcher`, called once by `_query_core`, launches `q7_kernel<<<(n + 1023) / 1024, BLOCK_SIZE>>>` with no dynamic shared memory. `launch.json` states the block (1024, 1, 1) and 0 bytes of dynamic shared memory.

## Lengths and ranges

- `supp_ptr` (read, length `n_elements`): the 0/1 nation code `(joined["supp_nation"] == nation_b).astype("int32")` of each joined row; the kernel bounds neither it nor the group it computes from it.
- `cust_ptr` (read, length `n_elements`): the 0/1 nation code `(joined["cust_nation"] == nation_b).astype("int32")` of each joined row.
- `shipdate_ptr` (read, length `n_elements`): `l_shipdate` in days, one per joined row.
- `price_ptr` (read, length `n_elements`): a column of the joined frame, one per row; `n_elements` is `price.numel()`.
- `disc_ptr` (read, length `n_elements`): a column of the joined frame, one per row.
- `shipdate_lo_days` (scalar): from the query parameter; no range.
- `shipdate_hi_days` (scalar): from the query parameter; no range.
- `out_revenue` (readwrite, length `8`): `torch.zeros(num_groups)` with `num_groups = 8` in `_query_core`; the kernel adds to it with `atomicAdd`.
- `n_elements` (size): `price.numel()`.

Least values of the size symbols: `n_elements` 0. The plan launches on whatever rows its joins and filters give, which can be none.
