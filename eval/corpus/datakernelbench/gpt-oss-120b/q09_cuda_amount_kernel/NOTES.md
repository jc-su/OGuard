# q09_cuda_amount_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/full/queries/q09/round_03.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `amount_kernel` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The entry kernel `amount_kernel` is renamed `kernel` and declared `extern "C" __global__ void kernel(...)`.
- Copied unchanged: `#include <cuda.h>`, `#include <cuda_runtime.h>`, and the kernel's body.
- Left out as host code: `#include <torch/extension.h>`, the launchers, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.

## Launch

`amount_launcher` launches `amount_kernel<<<blocks, BLOCK_SIZE>>>` with `BLOCK_SIZE = 1024` threads, `blocks = (n + 1023) / 1024`, and no dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 0 bytes.

## Lengths and ranges

- `price`, `discount`, `supplycost`, `quantity`: columns of the joined table `joined`, `n = price.numel()` rows.
- `out`: `torch::empty({n})`, which the kernel only writes.
