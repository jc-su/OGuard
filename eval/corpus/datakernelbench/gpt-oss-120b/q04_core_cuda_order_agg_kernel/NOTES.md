# q04_core_cuda_order_agg_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/core/queries/q04/round_02.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `order_agg_kernel` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The entry kernel `order_agg_kernel` is renamed `kernel` and declared `extern "C" __global__ void kernel(...)`.
- Copied unchanged: `#include <cuda_runtime.h>` and the kernel's body.
- Left out as host code: `#include <torch/extension.h>`, the launcher, the other kernel, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.

## Launch

`q4_launcher` launches `order_agg_kernel<<<blocks_order, BLOCK_SIZE>>>` with `BLOCK_SIZE = 1024` threads, `blocks = (n_order + 1023) / 1024`, and no dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 0 bytes.

## Lengths and ranges

- `o_orderkey`, `o_orderdate`, `o_priority`: columns of `orders` (the date as int32 days, `o_orderpriority` as cuDF category codes cast to int32), `n = n_order = o_orderkey.numel()` rows.
- `flags`: `torch::zeros({max_key + 1}, int32)` with `max_key = max(max(o_orderkey), max(l_orderkey))`, its length the size symbol `max_key + 1`. The kernel reads `flags[o_orderkey[idx]]`; the plan bounds the keys only above, by that length, so `o_orderkey` is `read`, not an index.
- `counts`: `torch::zeros({num_priorities}, int64)` with `num_priorities = len(priority_categories)`, the categories of `o_orderpriority`; the kernel adds into it atomically at `o_priority[idx]`, a category code in `[0, num_priorities)`: an index into `counts`. The int64 tensor is added into through an `unsigned long long *` cast.
- `date_lo`, `date_hi`: days since epoch of the date parameters, any int32.
