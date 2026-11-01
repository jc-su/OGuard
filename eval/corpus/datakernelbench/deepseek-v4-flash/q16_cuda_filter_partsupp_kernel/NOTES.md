# q16_cuda_filter_partsupp_kernel

`kernel.cu` is the kernel `filter_partsupp_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/full/queries/q16/round_06.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `filter_partsupp_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cuda_runtime.h>` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding); and the plan's other kernels (`build_composite_keys_kernel`, `hash_group_counts_kernel`).
- The kernel's body and parameters are the plan's.

## Launch

`launch_filter_partsupp` launches `filter_partsupp_kernel<<<(n + 255) / 256, 256>>>` once, with no dynamic shared memory.
`launch.json` states the block (256, 1, 1) and 0 bytes of dynamic shared memory.

## Lengths and ranges

- `ps_partkey` (read, length `n`): a `partsupp` column, one per row; `n` is `ps_partkey.numel()`.
- `ps_suppkey` (read, length `n`): a `partsupp` column, one per row; `n` is `ps_partkey.numel()`.
- `is_bad_supplier` (read, length `max_supp_key`): `torch.zeros(max_supp_key, int32)` with `max_supp_key = max(s_suppkey) + 1`, set to 1 at the complaint suppliers.
- `out_partkey` (write, length `n`): `torch.zeros(n_partsupp)`, `n_partsupp = len(df_partsupp)`, the same row count as `n`.
- `out_suppkey` (write, length `n`): `torch.zeros(n_partsupp)`, `n_partsupp = len(df_partsupp)`, the same row count as `n`.
- `out_count` (readwrite, length `1`): `torch.zeros(1, int64)`; the kernel takes output positions from it with `atomicAdd`.
- `n` (size): `ps_partkey.numel()`.
- `max_supp_key` (size): `int(df_supplier["s_suppkey"].max()) + 1`, the length of `is_bad_supplier`.

Least values of the size symbols: `n` 0, `max_supp_key` 0. The plan launches on whatever rows its joins and filters give, which can be none.
