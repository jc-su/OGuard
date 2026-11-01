# q04_cuda_mark_valid_keys_kernel

The kernel `mark_valid_keys_kernel` of the CUDA plan for q04 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/full/queries/q04/round_03.py`,
copied as `plan.py`).

## Extraction edits

- `mark_valid_keys_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` (host binding code and host-only headers), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 81: `mark_valid_keys_kernel<<<blocks_li, BLOCK_SIZE>>>` in `q4_launcher`, with `blocks_li = (n_li + 511) / 512`, a block of `BLOCK_SIZE = 512` threads, and no dynamic shared memory.

## Lengths and ranges

- The three lineitem columns have `n_li = l_orderkey.numel()` rows, passed as `n_lineitem`.
- `bitmap` is `torch::zeros({max_key + 1})`, where the plan computes `max_key` as the largest `o_orderkey` and `l_orderkey`: every `l_orderkey` value is at most `max_key`, so `l_orderkey` is `index-in` `bitmap` (the kernel also requires `key > 0`). The kernel sets bits with `atomicOr` into the zeroed bitmap: `readwrite`.
