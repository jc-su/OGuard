# q20_core_cuda_q20_filter_kernel

`kernel.cu` is the kernel `q20_filter_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/core/queries/q20/round_03.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `q20_filter_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cstdint>`, `#define BLOCK_SIZE 1024` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding); the plan's other kernel (`q20_isin_kernel`).
- The kernel's body and parameters are the plan's.

## Launch

`q20_filter_launcher`, called once by `_query_core`, launches `q20_filter_kernel<<<(n_ps + 1023) / 1024, BLOCK_SIZE>>>` with no dynamic shared memory. `launch.json` states the block (1024, 1, 1), 0 bytes of dynamic shared memory, and the grid maxima (4194304, 1, 1): `2^32 / 1024` blocks in x, past which the unsigned 32-bit row index wraps, and one block in y and z.

## Lengths and ranges

- `ps_suppkey` (read, length `n_ps`): `ps_suppkey` of the filtered partsupp rows; `n_ps` is `ps_suppkey.numel()`.
- `ps_availqty` (read, length `n_ps`): `ps_availqty` of the filtered partsupp rows.
- `sum_qty` (read, length `num_groups`): `sum_qty`, `torch.zeros(num_groups)` with `num_groups = unique_pairs.shape[0]`, the distinct (partkey, suppkey) pairs of partsupp and lineitem.
- `ps_group` (index-in `sum_qty`, length `n_ps`): `inverse[:Nps]` of `torch.unique` over the pairs, so every value lies in `[0, num_groups)`, the length of `sum_qty`.
- `n_ps` (size): `ps_suppkey.numel()`.
- `qualifying_suppkeys` (write, length `n_ps`): `torch.zeros(n_ps, int64)` in `_query_core`.
- `n_qualifying` (readwrite, length `1`): `torch.zeros(1, int64)`; the kernel takes output positions from it with `atomicAdd`.

Least values of the size symbols: `n_ps` 0, `num_groups` 0. The plan launches on whatever rows its joins and filters give, which can be none.
