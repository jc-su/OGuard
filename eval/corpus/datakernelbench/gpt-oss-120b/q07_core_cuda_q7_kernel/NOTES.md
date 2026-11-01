# q07_core_cuda_q7_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/core/queries/q07/round_01.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `q7_kernel` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The entry kernel `q7_kernel` is renamed `kernel` and declared `extern "C" __global__ void kernel(...)`.
- Copied unchanged: `#include <cstdint>` and the kernel's body, with the comment of its parameter list.
- Left out as host code: `#include <torch/extension.h>`, the launcher, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.

## Launch

`q7_launcher` launches `q7_kernel<<<blocks, threads>>>` with `BLOCK_SIZE = 1024` threads, `blocks = (n + 1023) / 1024`, and no dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 0 bytes.

## Lengths and ranges

- `supp_code`, `cust_code`, `shipdate_days`, `price`, `discount`: columns of the joined table `joined` (`l_shipdate` as int32 days), `n = price.numel()` rows.
- `supp_code`, `cust_code`: `(nation == nation_b).astype(int32)`, 0 or 1; the plan establishes this range only through the comparison, and no buffer has 2 entries, so both are `read`.
- `out_rev`: `torch::zeros({8})`, the 8 groups `supp * 4 + cust * 2 + year_offset`; the kernel adds into it atomically. The static shared array `block_rev` has 8 entries.
- `date_lo`, `date_hi`: days since epoch of the date parameters, any int32.
