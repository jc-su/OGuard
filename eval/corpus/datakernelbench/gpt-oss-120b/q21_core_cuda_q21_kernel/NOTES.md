# q21_core_cuda_q21_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/core/queries/q21/round_03.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `q21_kernel` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The entry kernel `q21_kernel` is renamed `kernel` and declared `extern "C" __global__ void kernel(...)`.
- Copied unchanged: `#include <cuda_runtime.h>`, `#include <cstdint>`, and the kernel's body.
- Left out as host code: `#include <torch/extension.h>`, the launcher, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.

## Launch

`q21_launcher` launches `q21_kernel<<<blocks, threads>>>` with `BLOCK_SIZE = 1024` threads, `blocks = (n + 1023) / 1024`, and no dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 0 bytes.

## Lengths and ranges

- `l1_orderkey`, `l1_late`, `supp_ids`: columns of `l1`, the lineitem rows of the nation's suppliers and the status-F orders (`late` as bool cast to uint8), `n = l1_orderkey.size(0)` rows.
- `supp_ids`: `supp_id = range(len(supp_n))` merged onto `l1`, so every value lies in `[0, num_suppliers)` with `num_suppliers = len(supp_n)`: an index into `counts`.
- `lo_orderkeys`, `lo_distinct_supp`, `lo_late_distinct_supp`: the aggregated table `agg`, sorted by order key, `lo_n = lo_orderkeys.size(0)` rows; the kernel binary-searches `lo_orderkeys` and reads the other two at the found position with no bound test, relying on an exact match for every `l1` order key.
- `counts`: `torch.zeros(num_suppliers, int64)` passed as `unsigned long long *`; the kernel adds into it atomically.
