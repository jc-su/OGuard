# q21_core_cuda_q21_kernel

The kernel `q21_kernel` of the CUDA plan for q21 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/core/queries/q21/round_02.py`,
copied as `plan.py`).

## Extraction edits

- `q21_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` (host binding code and host-only headers), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- The device code that the kernel uses is copied unchanged: `binary_search_left`.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 102: `q21_kernel<<<blocks, BLOCK_SIZE>>>` in `q21_launcher`, with `blocks = (n_rows + 1023) / 1024`, a block of `BLOCK_SIZE = 1024` threads, and no dynamic shared memory.

## Lengths and ranges

- `l1_orderkey`, `l1_suppkey`, `l1_late`, and `supp_ids` are columns of `l1`, the nation's lineitems of status-F orders, `n_rows` each; `l1_suppkey` is passed but not read; `l1_late` is a bool tensor.
- `lo_orderkeys`, `lo_distinct_supp`, and `lo_late_distinct_supp` are the three columns of `agg`, sorted by orderkey, `n_orders` rows; the gate does not establish sortedness, and the binary search's reads stay inside `lo_orderkeys` whatever the order.
- `supp_id` is `range(len(supp_n))` joined onto the rows, in `[0, num_suppliers)` with `num_suppliers = len(supp_n)`: `supp_ids` is `index-in` `counts` (the kernel also skips negative ids). `counts` is `torch::zeros({num_suppliers})` of int64, passed as `unsigned long long*`, an `atomicAdd` accumulator: `readwrite`.
