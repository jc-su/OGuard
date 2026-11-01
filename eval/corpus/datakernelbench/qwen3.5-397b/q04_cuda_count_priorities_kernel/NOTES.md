# q04_cuda_count_priorities_kernel

The kernel `count_priorities_kernel` of the CUDA plan for q04 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/full/queries/q04/round_03.py`,
copied as `plan.py`).

## Extraction edits

- `count_priorities_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` (host binding code and host-only headers), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 93: `count_priorities_kernel<<<blocks_ord, BLOCK_SIZE>>>` in `q4_launcher`, with `blocks_ord = (n_ord + 511) / 512`, a block of `BLOCK_SIZE = 512` threads, and no dynamic shared memory.

## Lengths and ranges

- The three orders columns have `n_ord = o_orderkey.numel()` rows, passed as `n_orders`.
- `bitmap` is the `max_key + 1` words of `mark_valid_keys_kernel`; `max_key` is the largest `o_orderkey` and `l_orderkey`, so `o_orderkey` is `index-in` `bitmap`.
- `o_priority` holds the category codes of `o_orderpriority`, in `[0, num_priorities)` where `num_priorities` is the number of categories; `counts` is `torch::zeros({num_priorities})`: `o_priority` is `index-in` `counts`, and `counts`, an `atomicAdd` accumulator, is `readwrite`.
