# q13_core_cuda_count_orders_per_customer

The kernel `count_orders_per_customer` of the CUDA plan for q13 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/core/queries/q13/round_02.py`,
copied as `plan.py`).

## Extraction edits

- `count_orders_per_customer` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` (host binding code and host-only headers), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 66: `count_orders_per_customer<<<blocks_orders, BLOCK_SIZE>>>` in `q13_launcher`, with `blocks_orders = (n_orders + 1023) / 1024`, a block of `BLOCK_SIZE = 1024` threads, and no dynamic shared memory.

## Lengths and ranges

- `o_custkey` is the `o_custkey` column of the orders whose comment does not match, `n_orders` rows.
- `order_counts` is `torch::zeros({num_customers + 1})` of int64, passed as `unsigned long long*`, with `num_customers = max(c_custkey)` of the customer table, an `atomicAdd` accumulator: `readwrite`. The kernel indexes it by `o_custkey`, whose range the plan takes from TPC-H's foreign key from orders to customer without establishing it, so `o_custkey` is `read`.
