# q03_core_cuda_filter_compute_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/core/queries/q03/round_01.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `filter_compute_kernel` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The entry kernel `filter_compute_kernel` is renamed `kernel` and declared `extern "C" __global__ void kernel(...)`.
- Copied unchanged: `#include <cuda_runtime.h>` and the kernel's body, with the comments of its parameter list.
- Left out as host code: `#include <torch/extension.h>`, the launcher, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.

## Launch

`filter_compute_launcher` launches `filter_compute_kernel<<<blocks, threads>>>` with `BLOCK_SIZE = 1024` threads, `blocks = (n + 1023) / 1024`, and no dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 0 bytes.

## Lengths and ranges

- `l_orderkey`, `l_extendedprice`, `l_discount`, `l_shipdate`, `o_orderdate`, `o_shippriority`: columns of the joined table `col` (the dates as int32 days), `n = l_orderkey.numel()` rows. The kernel does not read `l_orderkey` or `o_shippriority`.
- `mask_out`: `torch.empty(n, dtype=bool)`, passed as `unsigned char *`; `rev_out`: `torch.empty(n, dtype=float64)`. The kernel writes both at every row.
- `shipdate_lo`, `orderdate_hi`: days since epoch of the date parameters, any int32.
