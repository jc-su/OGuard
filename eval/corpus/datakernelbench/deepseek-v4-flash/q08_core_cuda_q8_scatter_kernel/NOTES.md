# q08_core_cuda_q8_scatter_kernel

`kernel.cu` is the kernel `q8_scatter_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/core/queries/q08/round_01.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `q8_scatter_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cstdint>`, `#include <cuda_runtime.h>`, `#define BLOCK_SIZE 1024` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding); the Thrust includes (`thrust/sort.h`, `thrust/device_ptr.h`, `thrust/execution_policy.h`), which only the host code's sort uses; the plan's other kernel (`q8_compact_kernel`). The plan compiles its second source string, `_CUDA_SOURCE_FIXED`; its first, `_CUDA_SOURCE`, is overwritten and never compiled.
- The kernel's body and parameters are the plan's.

## Launch

`q8_launcher` of `_CUDA_SOURCE_FIXED`, called once by `_launch_q8`, launches `q8_scatter_kernel<<<(n + 1023) / 1024, BLOCK_SIZE>>>` with no dynamic shared memory, then `q8_compact_kernel`. `launch.json` states the block (1024, 1, 1) and 0 bytes of dynamic shared memory.

## Lengths and ranges

- `o_year_ptr` (read, length `n_elements`): the order year of each joined row, `li["o_orderdate"].dt.year`; `n_elements` is `o_year.numel()`. The kernel bounds neither it nor the bin it computes from it.
- `volume_ptr` (read, length `n_elements`): `price * (1 - discount)` of each joined row.
- `is_brazil_ptr` (read, length `n_elements`): `(li["n_name"] == nation_name)`, a bool per joined row.
- `sum_brazil_out` (readwrite, length `100`): `torch::zeros({max_bins})` with `max_bins = 100` in `q8_launcher`; the kernel adds to it with `atomicAdd`.
- `sum_total_out` (readwrite, length `100`): `torch::zeros({max_bins})` with `max_bins = 100`; the kernel adds to it with `atomicAdd`.
- `year_flags_out` (readwrite, length `100`): `torch::zeros({max_bins})` with `max_bins = 100`; the kernel sets its elements with `atomicExch`.
- `n_elements` (size): `o_year.numel()`.

Least values of the size symbols: `n_elements` 0. The plan launches on whatever rows its joins and filters give, which can be none.
