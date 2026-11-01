# q08_core_cuda_q8_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/core/queries/q08/round_01.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `q8_kernel` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The entry kernel `q8_kernel` is renamed `kernel` and declared `extern "C" __global__ void kernel(...)`.
- Copied unchanged: the kernel's body.
- Left out as host code: `#include <torch/extension.h>`, `#include <c10/cuda/CUDAStream.h>`, the launcher, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.
- The kernel's fixed-width integer types are declared for the plan by `<torch/extension.h>` through `<cstdint>`; the nvcc front end has no torch include path, so `kernel.cu` includes `<cstdint>` in its place.

## Launch

`q8_launcher` launches `q8_kernel<<<blocks, threads, 0, stream>>>` with `threads = BLOCK_SIZE = 1024`, `blocks = (N + 1023) / 1024`, and no dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 0 bytes.

## Lengths and ranges

- `year`, `volume`, `is_brazil`: the order year (int32), `price * (1 - discount)`, and the supplier-nation flag (bool cast to uint8) of the joined table `li`, `N = year.numel()` rows.
- `sum_total`, `sum_brazil`: `torch::zeros({num_years})` with `num_years = max(year) - min_year + 1` and `min_year = min(year)`; the kernel adds into them atomically at `year[idx] - min_year`. The plan bounds that bin only through this relation between the column and the scalars, so `year` is `read`.
