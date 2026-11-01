# q10_cuda_agg_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/full/queries/q10/round_01.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `agg_kernel` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The entry kernel `agg_kernel` is renamed `kernel` and declared `extern "C" __global__ void kernel(...)`.
- Copied unchanged: `#include <cuda_runtime.h>`, and the kernel's body.
- Left out as host code: `#include <torch/extension.h>`, the launchers, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.

## Launch

`agg_launcher` launches `agg_kernel<<<blocks, threads>>>` with `threads = BLOCK_SIZE = 1024`, `blocks = (n + 1023) / 1024`, and no dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 0 bytes.

## Lengths and ranges

- `custkey`, `price`, `discount`: columns of the joined table `loc`, `n = c_custkey.shape[0]` rows.
- `revenue_out`: `torch.zeros(max_key + 1)` with `max_key = c_custkey.max()`, which the kernel adds into atomically; its length is the size symbol `buckets`.
- `custkey`: the plan bounds its values only above, by the length it gives `revenue_out`; it establishes no lower bound (TPC-H customer keys are positive), so `custkey` is `read`, not an index.
