# q09_core_cuda_compute_amount_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/core/queries/q09/round_01.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `compute_amount_kernel` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The entry kernel `compute_amount_kernel` is renamed `kernel` and declared `extern "C" __global__ void kernel(...)`.
- Copied unchanged: the kernel's body.
- Left out as host code: `#include <torch/extension.h>`, the launcher, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.
- The kernel's fixed-width integer types are declared for the plan by `<torch/extension.h>` through `<cstdint>`; the nvcc front end has no torch include path, so `kernel.cu` includes `<cstdint>` in its place.

## Launch

`compute_amount_launcher` launches `compute_amount_kernel<<<blocks, threads>>>` with `BLOCK_SIZE = 1024` threads, `blocks = (N + 1023) / 1024`, and no dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 0 bytes.

## Lengths and ranges

- `extprice`, `discount`, `supplycost`, `quantity`: columns of the joined table `joined`, `N = extprice.numel()` rows.
- `out`: `torch::empty_like(extprice)`, `N` rows, written at every row.
