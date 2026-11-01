# q19_core_cuda_q19_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/core/queries/q19/round_01.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `q19_kernel` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The entry kernel `q19_kernel` is renamed `kernel` and declared `extern "C" __global__ void kernel(...)`.
- Copied unchanged: `#include <cuda.h>`, `#include <cuda_runtime.h>`, `#define BLOCK_SIZE 1024`, and the kernel's body.
- Left out as host code: `#include <torch/extension.h>`, the launcher, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.

## Launch

`q19_launcher` launches `q19_kernel<<<blocks, BLOCK_SIZE>>>` with `BLOCK_SIZE = 1024` threads, `blocks = (n + 1023) / 1024`, and no dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 0 bytes.

## Lengths and ranges

- `l_quantity`, `l_extendedprice`, `l_discount`, `p_brand_id`, `p_container_grp`, `p_size`: columns of the joined table (the brand and container as int32 codes), `n = l_quantity.numel()` rows.
- `brand12_id`, `brand23_id`, `brand34_id`, `sm_grp`, `md_grp`, `lg_grp`: the codes the plan compares against, any int32.
- `out`: `torch::zeros({1})`, which each block adds into atomically. The static shared array `block_sum` has `BLOCK_SIZE = 1024` entries.
