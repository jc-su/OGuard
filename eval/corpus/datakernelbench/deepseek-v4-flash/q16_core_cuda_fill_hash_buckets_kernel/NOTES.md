# q16_core_cuda_fill_hash_buckets_kernel

`kernel.cu` is the kernel `fill_hash_buckets_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/core/queries/q16/round_05.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `fill_hash_buckets_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cuda_runtime.h>`, `#define BLOCK_SIZE 1024` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding); `#include <cuda.h>` (the CUDA driver API) and `#include <cub/cub.cuh>` (CUB's radix sort), which only the host code uses; the plan's other kernel (`count_runs_kernel`).
- The kernel's body and parameters are the plan's.

## Launch

`count_distinct_correct_launcher`, called once by `_launch_count_distinct`, launches `fill_hash_buckets_kernel<<<(n + 1023) / 1024, BLOCK_SIZE>>>` with no dynamic shared memory, then sorts with CUB and launches `count_runs_kernel`. `launch.json` states the block (1024, 1, 1) and 0 bytes of dynamic shared memory.

## Lengths and ranges

- `group_ids` (read, length `n`): `_gid` of each joined row, the row number of `group_df`; `n` is `group_ids.numel()`. The kernel only hashes it.
- `supp_ids` (read, length `n`): `ps_suppkey` of each joined row; the kernel only hashes it.
- `n` (size): `group_ids.numel()`.
- `keys` (write, length `n`): `torch::empty({n})`.
- `indices` (write, length `n`): `torch::empty({n})`; the kernel writes each row's own number.

Least values of the size symbols: `n` 1. `run_query` returns before the launch when the join is empty, so the launch has `n >= 1`.
