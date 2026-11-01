# q16_core_cuda_count_runs_kernel

`kernel.cu` is the kernel `count_runs_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/core/queries/q16/round_05.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `count_runs_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cuda_runtime.h>`, `#define BLOCK_SIZE 1024` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding); `#include <cuda.h>` (the CUDA driver API) and `#include <cub/cub.cuh>` (CUB's radix sort), which only the host code uses; the plan's other kernel (`fill_hash_buckets_kernel`).
- The kernel's body and parameters are the plan's.

## Launch

`count_distinct_correct_launcher`, called once by `_launch_count_distinct`, launches `count_runs_kernel<<<(n + 1023) / 1024, BLOCK_SIZE>>>` with no dynamic shared memory, after `fill_hash_buckets_kernel` and the CUB sort. `launch.json` states the block (1024, 1, 1) and 0 bytes of dynamic shared memory.

## Lengths and ranges

- `sorted_keys` (read, length `n`): `keys` of `fill_hash_buckets_kernel`, sorted in place by `cub::DeviceRadixSort::SortPairs`; `n` is `group_ids.numel()`.
- `sorted_indices` (index-in `group_ids`, length `n`): `indices` of `fill_hash_buckets_kernel`, the row numbers `0..n-1`, permuted by the sort with `keys`, so every value lies in `[0, n)`, the length of `group_ids`.
- `group_ids` (index-in `out_counts`, length `n`): `_gid` of each joined row, the row number `range(len(group_df))` of `group_df` joined to each row, so every value lies in `[0, num_groups)`, the length of `out_counts`.
- `supp_ids` (read, length `n`): `ps_suppkey` of each joined row; the kernel does not access it.
- `n` (size): `group_ids.numel()`.
- `out_counts` (readwrite, length `num_groups`): `torch::zeros({num_groups})` with `num_groups = len(group_df)`; the kernel adds to it with `atomicAdd`.

Least values of the size symbols: `n` 1, `num_groups` 1. `run_query` returns before the launch when the join is empty, so the launch has `n >= 1` and `num_groups >= 1`.
