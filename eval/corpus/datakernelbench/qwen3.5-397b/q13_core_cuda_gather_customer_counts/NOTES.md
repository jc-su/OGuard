# q13_core_cuda_gather_customer_counts

The kernel `gather_customer_counts` of the CUDA plan for q13 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/core/queries/q13/round_02.py`,
copied as `plan.py`).

## Extraction edits

- `gather_customer_counts` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` (host binding code and host-only headers), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 76: `gather_customer_counts<<<blocks_gather, BLOCK_SIZE>>>` in `q13_launcher`, after `count_orders_per_customer`, with `blocks_gather = (n_customers + 1023) / 1024`, a block of `BLOCK_SIZE = 1024` threads, and no dynamic shared memory.

## Lengths and ranges

- `c_custkey` is the customer table's key column, `n_customers = c_custkey.numel()` rows; `c_count` is `torch::empty({n_customers})`, written once per customer: `write`.
- `order_counts` has `num_customers + 1` elements with `num_customers = max(c_custkey)`, the counts of the previous kernel, read only here. The plan sizes it from the largest `c_custkey`, so `c_custkey` is `index-in` `order_counts` (customer keys are TPC-H's non-negative keys).
