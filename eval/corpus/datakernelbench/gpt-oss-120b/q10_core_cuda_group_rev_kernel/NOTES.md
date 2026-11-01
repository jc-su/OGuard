# q10_core_cuda_group_rev_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/core/queries/q10/round_02.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `group_rev_kernel` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The entry kernel `group_rev_kernel` is renamed `kernel` and declared `extern "C" __global__ void kernel(...)`.
- Copied unchanged: `#include <cuda_runtime.h>` and the kernel's body.
- Left out as host code: `#include <torch/extension.h>`, the launcher, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.

## Launch

`group_rev_launcher` launches `group_rev_kernel<<<blocks, BLOCK_SIZE>>>` with `BLOCK_SIZE = 1024` threads, `blocks = (n + 1023) / 1024`, and no dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 0 bytes.

## Lengths and ranges

- `custkey`, `price`, `discount`: columns of the joined table `loc`, `n = custkey.numel()` rows.
- `out_rev`: `torch::zeros({max_key + 1})` with `max_key = custkey.max()`, which the kernel adds into atomically; its length is the size symbol `buckets`.
- `custkey`: the plan bounds its values only above, by the length it gives `out_rev`; it establishes no lower bound (TPC-H customer keys are positive), so `custkey` is `read`, not an index.
