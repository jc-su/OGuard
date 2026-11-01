# q18_core_cuda_q18_threshold_kernel

`kernel.cu` is the kernel `q18_threshold_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/core/queries/q18/round_07.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `q18_threshold_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cuda_runtime.h>`, `#include <cstdint>`, `#define BLOCK_SIZE 256` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding).
- The kernel's body and parameters are the plan's.

## Launch

`q18_launcher`, called once by `_query_core`, launches `q18_threshold_kernel<<<(n + 255) / 256, 256>>>` with no dynamic shared memory. `launch.json` states the block (256, 1, 1), 0 bytes of dynamic shared memory, and the grid maxima (16777216, 1, 1): `2^32 / 256` blocks in x, past which the unsigned 32-bit row index wraps, and one block in y and z.

## Lengths and ranges

- `sorted_keys` (read, length `n`): `l_orderkey` sorted by `torch::sort`; `n` is `orderkey.numel()`.
- `sorted_qty` (read, length `n`): `l_quantity` in the same order.
- `out_keys` (write, length `n`): `torch::empty({n})`.
- `out_sums` (write, length `n`): `torch::empty({n})`.
- `out_count` (readwrite, length `1`): `torch::zeros({1})`; the kernel takes output positions from it with `atomicAdd`.
- `qty_threshold` (scalar): from the query parameter; no range.
- `n` (size): `orderkey.numel()`.

Least values of the size symbols: `n` 0. The plan launches on the whole `lineitem` table, which can be empty.
