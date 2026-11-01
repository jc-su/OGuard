# q21_core_cuda_q21_kernel

`kernel.cu` is the kernel `q21_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/core/queries/q21/round_04.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `q21_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cstdint>`, `#define BLOCK_SIZE 1024` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding).
- The kernel's body and parameters are the plan's.

## Launch

`q21_launcher`, called once by `_launch_q21`, launches `q21_kernel<<<(n_elements + 1023) / 1024, BLOCK_SIZE>>>` when that block count is positive, with no dynamic shared memory. `launch.json` states the block (1024, 1, 1) and 0 bytes of dynamic shared memory.

## Lengths and ranges

- `l1_orderkey_ptr` (read, length `n_elements`): `l_orderkey` of each row of `l1`; `n_elements` is `l1_orderkey.numel()`.
- `l1_suppkey_ptr` (read, length `n_elements`): `l_suppkey` of each row of `l1`; the kernel does not access it.
- `l1_late_ptr` (read, length `n_elements`): `l1["late"]`, a bool per row.
- `supp_ids_ptr` (read, length `n_elements`): `supp_id` of each row of `l1`, the row number of `supp_n`; the kernel bounds it by `num_suppliers` before indexing, so no index role.
- `lo_orderkeys_ptr` (read, length `n_lo`): `agg["l_orderkey"]`, sorted; `n_lo` is `lo_orderkeys.numel()`.
- `lo_distinct_supp_ptr` (read, length `n_lo`): a column of `agg`, one per order.
- `lo_late_distinct_supp_ptr` (read, length `n_lo`): a column of `agg`, one per order.
- `num_suppliers` (size): `len(supp_n)`, the length of `out_counts_ptr`.
- `out_counts_ptr` (readwrite, length `num_suppliers`): `torch.zeros(num_suppliers, int32)` in `_launch_q21`, zeroed again by the launcher; the kernel adds to it with `atomicAdd`.
- `n_elements` (size): `l1_orderkey.numel()`; the launcher launches only when its block count is positive.
- `n_lo` (size): `lo_orderkeys.numel()`.

Least values of the size symbols: `n_elements` 1, `num_suppliers` 0, `n_lo` 0. The launcher launches only when `l1` has rows; `agg` and the nation's suppliers can be empty.
