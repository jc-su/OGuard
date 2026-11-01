# q06_core_cuda_q6_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/core/queries/q06/round_01.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `q6_kernel` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The entry kernel `q6_kernel` is renamed `kernel` and declared `extern "C" __global__ void kernel(...)`.
- Copied unchanged: `#include <cuda.h>`, `#include <cuda_runtime.h>`, `#define BLOCK_SIZE 1024` (with its `#ifndef` guard), and the kernel's body.
- Left out as host code: `#include <torch/extension.h>`, the launcher, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.

## Launch

`q6_launcher` launches `q6_kernel<<<blocks, BLOCK_SIZE>>>` with `BLOCK_SIZE = 1024` threads, `blocks = (n + 1023) / 1024`, and no dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 0 bytes.

## Lengths and ranges

- `price_ptr`, `discount_ptr`, `quantity_ptr`, `shipdate_ptr`: columns of `lineitem` (`l_shipdate` as int32 days), `n_elements = price.numel()` rows.
- `out_ptr`: `torch::zeros({1})`, which each block adds into atomically.
- `date_lo`, `date_hi`: days since epoch of the date parameters, any int32; `disc_lo`, `disc_hi`, `qty_hi`: the discount and quantity bounds, any double.
- The static shared array `block_sum` has `BLOCK_SIZE = 1024` entries, one per thread of the block.
