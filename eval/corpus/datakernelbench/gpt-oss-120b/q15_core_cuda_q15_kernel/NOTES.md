# q15_core_cuda_q15_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/core/queries/q15/round_01.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `q15_kernel` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The entry kernel `q15_kernel` is renamed `kernel` and declared `extern "C" __global__ void kernel(...)`.
- Copied unchanged: `#include <cuda_runtime.h>`, the device function `lower_bound`, and the kernel's body.
- Left out as host code: `#include <torch/extension.h>`, the launcher, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.

## Launch

`q15_launcher` launches `q15_kernel<<<blocks, threads>>>` with `BLOCK_SIZE = 1024` threads, `blocks = (n_rows + 1023) / 1024`, and no dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 0 bytes.

## Lengths and ranges

- `suppkey`, `price`, `disc`, `shipdate`: columns of `lineitem` (`l_shipdate` as int32 days), `n_rows = suppkey.numel()` rows.
- `sorted_supp`: `torch.sort(s_suppkey)`, `n_sup = s_suppkey.numel()` entries; the kernel binary-searches it. `totals` and `counts`: `torch.zeros(n_sup)` (float64 and int64), which the kernel adds into atomically at the found position, `counts` through an `unsigned long long *` cast.
- `date_lo`, `date_hi`: days since epoch of the date parameters, any int32.
