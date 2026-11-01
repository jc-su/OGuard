# q01_core_cuda_q1_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/core/queries/q01/round_01.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `q1_kernel` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The entry kernel `q1_kernel` is renamed `kernel` and declared `extern "C" __global__ void kernel(...)`.
- Copied unchanged: `#include <cuda.h>`, `#include <cuda_runtime.h>`, and the kernel's body.
- Left out as host code: `#include <torch/extension.h>`, `#include <vector>`, the launcher, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.

## Launch

`q1_launcher` launches `q1_kernel<<<blocks, BLOCK_SIZE>>>` with `BLOCK_SIZE = 1024` threads, `blocks = (N + 1023) / 1024`, and no dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 0 bytes. `launch.json` also states the grid claim `[2097152, 1, 1]`: the kernel holds the row id `blockIdx.x * blockDim.x + threadIdx.x` in an `int`, which is at most 2^31 - 1 for 2097152 = 2^31 / 1024 blocks in x and negative beyond.

## Lengths and ranges

- `qty`, `price`, `disc`, `tax`, `rf`, `ls`, `shipdate`: columns of `lineitem` (the numeric columns, `l_returnflag` and `l_linestatus` as cuDF category codes cast to int32, and `l_shipdate` as int32 days), all with the `N = qty.numel()` rows of the table.
- `sum_qty`, `sum_price`, `sum_disc_price`, `sum_charge`, `sum_disc`, `cnt`: `torch::zeros({MAX_GROUPS})` with `MAX_GROUPS = 256 * 256 = 65536`; the kernel adds into them atomically. `cnt` is the int64 tensor passed as `unsigned long long *`.
- `shipdate_hi`: days since epoch of the `shipdate_hi` parameter, any int32.
- The plan does not bound the category codes: `rf` and `ls` have the role `read`. The group index `rf * 256 + ls` stays below 65536 only while each column has at most 256 categories (TPC-H has 3 and 2).
