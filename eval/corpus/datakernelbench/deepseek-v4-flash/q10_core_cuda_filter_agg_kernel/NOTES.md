# q10_core_cuda_filter_agg_kernel

`kernel.cu` is the kernel `filter_agg_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/core/queries/q10/round_02.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `filter_agg_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#define BLOCK_SIZE 1024` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding); `#include <cuda.h>` (the CUDA driver API), which only the host code uses.
- The kernel's body and parameters are the plan's.

## Launch

`query_core_launcher`, called once by `_query_core`, launches `filter_agg_kernel<<<(n + 1023) / 1024, BLOCK_SIZE>>>` with no dynamic shared memory; `_query_core` then discards its outputs and aggregates with `torch.unique` and `scatter_add_`. `launch.json` states the block (1024, 1, 1) and 0 bytes of dynamic shared memory.

## Lengths and ranges

- `custkey_ptr` (read, length `n_elements`): `c_custkey` of each joined row; `n_elements` is `custkey.numel()`.
- `price_ptr` (read, length `n_elements`): a column of the joined frame, one per row.
- `discount_ptr` (read, length `n_elements`): a column of the joined frame, one per row.
- `out_keys_ptr` (write, length `(n_elements + 1023) / 1024`): `torch::empty({blocks})` with `blocks = (n + 1023) / 1024`, one per block of the plan's launch; the quotient is not affine in `n_elements`, so it is the size symbol `out_keys_ptr` of its own.
- `out_vals_ptr` (write, length `(n_elements + 1023) / 1024`): `torch::empty({blocks})`, of the same length as `out_keys_ptr`.
- `n_elements` (size): `custkey.numel()`.

Least values of the size symbols: `n_elements` 0, `out_keys_ptr` 0. The plan launches on whatever rows its joins and filters give, which can be none.
