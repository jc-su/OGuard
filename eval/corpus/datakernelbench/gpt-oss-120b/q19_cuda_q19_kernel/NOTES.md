# q19_cuda_q19_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/full/queries/q19/round_01.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `q19_kernel` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The entry kernel `q19_kernel` is renamed `kernel` and declared `extern "C" __global__ void kernel(...)`.
- Copied unchanged: `#include <cuda.h>`, `#include <cuda_runtime.h>`, `#define BLOCK_SIZE 1024` (with its `#ifndef` guard), and the kernel's body.
- Left out as host code: `#include <torch/extension.h>`, the launchers, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.

## Launch

`q19_launcher` launches `q19_kernel<<<grid, block>>>` with `block = BLOCK_SIZE = 1024` threads, `grid = (N + 1023) / 1024`, and no dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 0 bytes.

## Lengths and ranges

- `l_quantity`, `l_extendedprice`, `l_discount`, `p_brand_id`, `p_container_grp`, `p_size`: columns of the joined table `joined`, `N = l_quantity.numel()` rows.
- The six int scalars are the plan's constants 12, 23, 34, 1, 2, 3.
- `out`: `torch::zeros({1})`, added into once per block.
