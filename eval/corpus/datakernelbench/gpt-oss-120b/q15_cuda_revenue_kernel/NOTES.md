# q15_cuda_revenue_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/full/queries/q15/round_01.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `revenue_kernel` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The entry kernel `revenue_kernel` is renamed `kernel` and declared `extern "C" __global__ void kernel(...)`.
- Copied unchanged: `#include <cuda.h>`, `#include <cuda_runtime.h>`, and the kernel's body.
- Left out as host code: `#include <torch/extension.h>`, the launchers, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.

## Launch

`revenue_launcher` launches `revenue_kernel<<<blocks, threads>>>` with `threads = BLOCK_SIZE = 1024`, `blocks = (n + 1023) / 1024`, and no dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 0 bytes.

## Lengths and ranges

- `l_suppkey`, `l_extendedprice`, `l_discount`, `l_shipdate`: columns of `lineitem`, `n = l_suppkey.numel()` rows.
- `supp_key_to_idx`: `supp_map`, `torch.full((max_key + 1,), -1)` with `max_key = max(s_suppkey)`, holding -1 or the supplier's row index; its length is `n_sup + 1` for the launcher's `n_sup = supp_key_to_idx.size(0) - 1`.
- `supplier_totals`: `torch::zeros({n_sup})`, `max(s_suppkey)` entries, which the kernel adds into atomically.
- `l_suppkey` indexes `supp_key_to_idx`, but the plan bounds neither: the lineitem keys are bounded by `max(s_suppkey)` only through TPC-H's referential integrity. `supp_key_to_idx` holds -1 or a row index below `s_suppkey.numel()`, which is not an index range into `supplier_totals`. Both are `read`.
