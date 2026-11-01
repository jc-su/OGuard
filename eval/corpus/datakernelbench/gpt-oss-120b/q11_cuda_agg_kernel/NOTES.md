# q11_cuda_agg_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/full/queries/q11/round_02.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `agg_kernel` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The entry kernel `agg_kernel` is renamed `kernel` and declared `extern "C" __global__ void kernel(...)`.
- Copied unchanged: `#include <cuda.h>`, `#include <cuda_runtime.h>`, `#define BLOCK_SIZE 1024` (with its `#ifndef` guard), and the kernel's body.
- Left out as host code: `#include <torch/extension.h>`, the launchers, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.

## Launch

`agg_launcher` launches `agg_kernel<<<blocks, BLOCK_SIZE>>>` with `BLOCK_SIZE = 1024` threads, `blocks = (n + 1023) / 1024`, and no dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 0 bytes.

## Lengths and ranges

- `keys`, `supplycost`, `availqty`: columns of the joined table `joined` (`ps_partkey`, `ps_supplycost`, `ps_availqty`), `n = keys.numel()` rows.
- `groups`: `torch.zeros(max_key + 1)` with `max_key = max(ps_partkey)`, which the kernel adds into atomically; its length is the size symbol `buckets`.
- `total`: `torch::zeros({1})`, added into once per block.
- `keys`: bounded above only by the length the plan gives `groups`, with no lower bound, so it is `read`, not an index.
