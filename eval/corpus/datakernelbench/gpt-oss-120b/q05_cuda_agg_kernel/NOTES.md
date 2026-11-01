# q05_cuda_agg_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/full/queries/q05/round_02.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `agg_kernel` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The entry kernel `agg_kernel` is renamed `kernel` and declared `extern "C" __global__ void kernel(...)`.
- Copied unchanged: `#include <cuda.h>`, `#include <cuda_runtime.h>`, `#define MAX_GROUPS 64` (with its `#ifndef` guard), and the kernel's body.
- Left out as host code: `#include <torch/extension.h>`, the launchers, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.

## Launch

`agg_launcher` launches `agg_kernel<<<grid, block>>>` with `block = BLOCK_SIZE = 1024`, `grid = (n + 1023) / 1024`, and no dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 0 bytes.

## Lengths and ranges

- `nation_id`, `price`, `discount`: columns of the joined table `final_df`, `n = nation_id.numel()` rows.
- `nation_id`: `final_df` is merged with `nations_unique`, whose `nation_id` is `range(len(nations_unique))`, so every value lies in `[0, num_groups)`: an index into `out`.
- `out`: `torch::zeros({num_groups})`, one double per nation; the kernel adds into it atomically.
- `num_groups`: `num_nations = len(nations_unique)`, the nations of the region (5 in TPC-H). The kernel's static shared array `s_out` has `MAX_GROUPS = 64` entries; the plan does not check `num_groups <= 64`.
