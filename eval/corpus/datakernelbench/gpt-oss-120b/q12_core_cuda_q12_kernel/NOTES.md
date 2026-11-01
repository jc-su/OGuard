# q12_core_cuda_q12_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/core/queries/q12/round_02.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `q12_kernel` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The entry kernel `q12_kernel` is renamed `kernel` and declared `extern "C" __global__ void kernel(...)`.
- Copied unchanged: `#include <cuda_runtime.h>` and the kernel's body, with the comments of its parameter list.
- Left out as host code: `#include <torch/extension.h>`, the launcher, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.
- The plan already declares the kernel `extern "C" __global__`; the declaration keeps that and renames it.

## Launch

`q12_launcher` launches `q12_kernel<<<blocks, threads>>>` with `BLOCK_SIZE = 1024` threads, `blocks = (n + 1023) / 1024`, and no dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 0 bytes.

## Lengths and ranges

- `shipmode_code`, `priority_high`, `commitdate`, `receiptdate`, `shipdate`: columns of the joined table (the dates as int32 days), `n_elements = shipmode_code.numel()` rows.
- `shipmode_code`: `(l_shipmode == shipmode2).astype(int32)`, 0 or 1, and `high_counts` and `low_counts` have 2 entries, so it is an index into `high_counts` (and into `low_counts`, of the same length). `priority_high` is a 0/1 flag used as a value.
- `high_counts`, `low_counts`: `torch::zeros({2}, int64)` passed as `unsigned long long *`; the kernel adds into them atomically. The static shared arrays `block_high` and `block_low` have 2 entries.
- `receipt_lo`, `receipt_hi`: days since epoch of the date parameters, any int32.
