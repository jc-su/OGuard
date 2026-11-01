# q01_cuda_q1_per_group_kernel

`kernel.cu` is the kernel `q1_per_group_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/full/queries/q01/round_09.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `q1_per_group_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cuda_runtime.h>` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding).
- The kernel's body and parameters are the plan's.

## Launch

`launch_q1_fused` launches `q1_per_group_kernel<<<(n + 255) / 256, 256>>>` once, with no dynamic shared memory; `run_query` calls it once.
`launch.json` states the block (256, 1, 1) and 0 bytes of dynamic shared memory.

## Lengths and ranges

- `rf_code_ptr` (read, length `n_elements`): `rf_code`, the category codes of `l_returnflag`, one per `lineitem` row; `n_elements` is `rf_code.numel()`. The codes lie in `[0, categories)`, and the plan does not bound the number of categories, so no index role.
- `ls_code_ptr` (read, length `n_elements`): `ls_code`, the category codes of `l_linestatus`, one per row; as `rf_code_ptr`, no index role.
- `qty_ptr` (read, length `n_elements`): a `lineitem` column as a tensor, one per row.
- `price_ptr` (read, length `n_elements`): a `lineitem` column as a tensor, one per row.
- `disc_ptr` (read, length `n_elements`): a `lineitem` column as a tensor, one per row.
- `tax_ptr` (read, length `n_elements`): a `lineitem` column as a tensor, one per row.
- `shipdate_ptr` (read, length `n_elements`): `l_shipdate` in days since 1970-01-01, one per row.
- `shipdate_hi_days` (scalar): computed from the query parameter `shipdate_hi`; no range.
- `out_sum_qty` (readwrite, length `65536`): `torch.zeros(MAX_GROUPS * MAX_GROUPS)` with `MAX_GROUPS = 256`, zeroed again by the launcher; the kernel adds to it with `atomicAdd`.
- `out_sum_base_price` (readwrite, length `65536`): `torch.zeros(MAX_GROUPS * MAX_GROUPS)` with `MAX_GROUPS = 256`, zeroed again by the launcher; the kernel adds to it with `atomicAdd`.
- `out_sum_disc_price` (readwrite, length `65536`): `torch.zeros(MAX_GROUPS * MAX_GROUPS)` with `MAX_GROUPS = 256`, zeroed again by the launcher; the kernel adds to it with `atomicAdd`.
- `out_sum_charge` (readwrite, length `65536`): `torch.zeros(MAX_GROUPS * MAX_GROUPS)` with `MAX_GROUPS = 256`, zeroed again by the launcher; the kernel adds to it with `atomicAdd`.
- `out_sum_disc` (readwrite, length `65536`): `torch.zeros(MAX_GROUPS * MAX_GROUPS)` with `MAX_GROUPS = 256`, zeroed again by the launcher; the kernel adds to it with `atomicAdd`.
- `out_count` (readwrite, length `65536`): `torch.zeros(65536, int64)`; the kernel adds to it with `atomicAdd`.
- `out_rf` (readwrite, length `65536`): `torch.zeros(65536, int32)`; the kernel updates it with `atomicCAS`.
- `out_ls` (readwrite, length `65536`): `torch.zeros(65536, int32)`; the kernel updates it with `atomicCAS`.
- `n_elements` (size): `rf_code.numel()`.

Least values of the size symbols: `n_elements` 0. The plan launches on whatever rows its joins and filters give, which can be none.
