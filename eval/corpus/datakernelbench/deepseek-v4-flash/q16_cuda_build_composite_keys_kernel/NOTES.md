# q16_cuda_build_composite_keys_kernel

`kernel.cu` is the kernel `build_composite_keys_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/full/queries/q16/round_06.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `build_composite_keys_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cuda_runtime.h>` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding); and the plan's other kernels (`filter_partsupp_kernel`, `hash_group_counts_kernel`).
- The kernel's body and parameters are the plan's.

## Launch

`launch_build_composites`, called once by `_query_core`, launches `build_composite_keys_kernel<<<(n + 255) / 256, 256>>>` with no dynamic shared memory.
`launch.json` states the block (256, 1, 1) and 0 bytes of dynamic shared memory.

## Lengths and ranges

- `group_ids` (read, length `n`): `_gid` of each joined row, the row number of `group_df`; `n` is `group_ids.numel()`.
- `supp_ids` (read, length `n`): `ps_suppkey` of each joined row.
- `out_composites` (write, length `n`): `torch.zeros(n)` in `_query_core`.
- `out_count` (readwrite, length `1`): `torch.zeros(1, int64)`; the kernel takes output positions from it with `atomicAdd`.
- `n` (size): `group_ids.numel()`, passed by `_query_core`.

Least values of the size symbols: `n` 0. The plan launches on whatever rows its joins and filters give, which can be none.
