# q08_core_cuda_q8_compact_kernel

`kernel.cu` is the kernel `q8_compact_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/core/queries/q08/round_01.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `q8_compact_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cstdint>`, `#include <cuda_runtime.h>`, `#define BLOCK_SIZE 1024` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding); the Thrust includes (`thrust/sort.h`, `thrust/device_ptr.h`, `thrust/execution_policy.h`), which only the host code's sort uses; the plan's other kernel (`q8_scatter_kernel`). The plan compiles its second source string, `_CUDA_SOURCE_FIXED`; its first, `_CUDA_SOURCE`, is overwritten and never compiled.
- The kernel's body and parameters are the plan's.

## Launch

`q8_launcher` of `_CUDA_SOURCE_FIXED`, called once by `_launch_q8`, launches `q8_compact_kernel<<<(max_bins + 255) / 256, 256>>>` after `q8_scatter_kernel`, with no dynamic shared memory. `launch.json` states the block (256, 1, 1), 0 bytes of dynamic shared memory, and the grid maxima (8388608, 1, 1): `2^31 / 256` blocks in x, past which the `int` index `tid` wraps, and one block in y and z.

## Lengths and ranges

- `sum_brazil_arr` (read, length `max_bins`): `sum_brazil_arr`, `torch::zeros({max_bins})` filled by `q8_scatter_kernel`; the kernel only reads it.
- `sum_total_arr` (read, length `max_bins`): `sum_total_arr`, `torch::zeros({max_bins})` filled by `q8_scatter_kernel`; the kernel only reads it.
- `year_flags` (read, length `max_bins`): `year_flags`, `torch::zeros({max_bins})` set by `q8_scatter_kernel`; the kernel only reads it.
- `out_years` (write, length `20`): `torch.zeros(max_years, int32)` with `max_years = 20` in `_launch_q8`.
- `out_mkt_share` (write, length `20`): `torch.zeros(max_years, float64)` with `max_years = 20` in `_launch_q8`.
- `num_years_dev` (readwrite, length `1`): `torch::zeros({1})`; the kernel takes output positions from it with `atomicAdd`.
- `max_bins` (size): `max_bins = 100` in `q8_launcher`, the length of the three year arrays.

Least values of the size symbols: `max_bins` 0. The plan launches with `max_bins = 100` bins.
