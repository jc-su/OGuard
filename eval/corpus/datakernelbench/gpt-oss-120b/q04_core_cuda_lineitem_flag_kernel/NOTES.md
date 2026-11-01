# q04_core_cuda_lineitem_flag_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/core/queries/q04/round_02.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `lineitem_flag_kernel` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The entry kernel `lineitem_flag_kernel` is renamed `kernel` and declared `extern "C" __global__ void kernel(...)`.
- Copied unchanged: `#include <cuda_runtime.h>` and the kernel's body.
- Left out as host code: `#include <torch/extension.h>`, the launcher, the other kernel, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.

## Launch

`q4_launcher` launches `lineitem_flag_kernel<<<blocks_line, BLOCK_SIZE>>>` with `BLOCK_SIZE = 1024` threads, `blocks = (n_line + 1023) / 1024`, and no dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 0 bytes.

## Lengths and ranges

- `l_orderkey`, `l_commit`, `l_receipt`: columns of `lineitem` (the dates as int32 days), `n = n_line = l_orderkey.numel()` rows.
- `flags`: `torch::zeros({max_key + 1}, int32)` with `max_key = max(max(o_orderkey), max(l_orderkey))`, its length the size symbol `max_key + 1`. The kernel ORs `flags[l_orderkey[idx]]` atomically; the plan bounds the keys only above, by that length, and establishes no lower bound (TPC-H order keys are positive), so `l_orderkey` is `read`, not an index.
