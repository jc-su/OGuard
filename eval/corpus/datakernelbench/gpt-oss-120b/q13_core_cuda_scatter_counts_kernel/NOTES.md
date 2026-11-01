# q13_core_cuda_scatter_counts_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/core/queries/q13/round_01.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `scatter_counts_kernel` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The entry kernel `scatter_counts_kernel` is renamed `kernel` and declared `extern "C" __global__ void kernel(...)`.
- Copied unchanged: `#include <cuda.h>`, `#include <cuda_runtime.h>`, and the kernel's body.
- Left out as host code: `#include <torch/extension.h>`, the launcher, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.

## Launch

`scatter_counts_launcher` launches `scatter_counts_kernel<<<blocks, threads>>>` with `BLOCK_SIZE = 1024` threads, `blocks = (n + 1023) / 1024`, and no dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 0 bytes.

## Lengths and ranges

- `keys`: `o_custkey` of the orders whose comment passes the NOT LIKE filter, `n = keys.numel()` rows.
- `out`: `torch::zeros({num_customers + 1}, int32)` with `num_customers = max(c_custkey)` of the customer table; the kernel adds into it atomically at `(int32_t)keys[idx]`.
- `keys`: the order keys are bounded by `max(c_custkey)` only through TPC-H's referential integrity, not by the plan, so `keys` is `read`.
