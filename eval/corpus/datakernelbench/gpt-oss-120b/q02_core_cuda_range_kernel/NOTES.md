# q02_core_cuda_range_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/core/queries/q02/round_01.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `range_kernel` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The entry kernel `range_kernel` is renamed `kernel` and declared `extern "C" __global__ void kernel(...)`.
- Copied unchanged: the kernel's body.
- Left out as host code: `#include <torch/extension.h>`, the launcher, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.
- The kernel's fixed-width integer types are declared for the plan by `<torch/extension.h>` through `<cstdint>`; the nvcc front end has no torch include path, so `kernel.cu` includes `<cstdint>` in its place.

## Launch

`range_launcher` launches `range_kernel<<<blocks, threads>>>` with `BLOCK_SIZE = 1024` threads, `blocks = (limit + 1023) / 1024`, and no dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 0 bytes.

## Lengths and ranges

- `out`: `torch.empty(limit, dtype=int64)` allocated by `_launch_range` with `limit = k = min(limit, len(final))`, the rows the query keeps; the kernel writes `out[idx] = idx`.
- `limit`: the same `k`, the length of `out`; the plan returns before the launch when `k <= 0`.
