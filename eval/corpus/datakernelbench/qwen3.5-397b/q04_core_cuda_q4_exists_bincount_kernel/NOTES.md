# q04_core_cuda_q4_exists_bincount_kernel

The kernel `q4_exists_bincount_kernel` of the CUDA plan for q04 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/core/queries/q04/round_02.py`,
copied as `plan.py`).

## Extraction edits

- `q4_exists_bincount_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` (host binding code and host-only headers), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 95: `q4_exists_bincount_kernel<<<blocks, BLOCK_SIZE>>>` in `q4_launcher`, when `n_filt_orders > 0 && n_unique_lineitems > 0`, with `blocks = (n_filt_orders + 1023) / 1024`, a block of `BLOCK_SIZE = 1024` threads, and no dynamic shared memory (`block_counts` is static).

## Lengths and ranges

- `filt_orderkey` and `filt_priority` are the orders in the date range, `n_filt_orders = filt_orderkey.numel()` rows; `unique_l_orderkey` is the sorted distinct `l_orderkey` of the late lineitems, `n_unique_lineitems` rows; the launch requires both to be positive.
- `filt_priority` holds the cuDF category codes of `o_orderpriority`, in `[0, num_priorities)` with `num_priorities` the number of categories (5 at TPC-H's data); `priority_counts` is `torch::zeros({num_priorities})`, an `atomicAdd` accumulator (`readwrite`), so `filt_priority` is `index-in` `priority_counts`; the kernel also checks the code itself. A non-empty orders table has at least one category.
- The kernel's static `block_counts` has `MAX_PRIORITIES = 10` entries; the plan does not check `num_priorities` against it, so `launch.json` states no size maximum.
